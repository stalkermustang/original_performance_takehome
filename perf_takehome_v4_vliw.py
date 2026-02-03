"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import copy
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        
    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)
    
    def vec_const(self, val, name=None):
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val, name=f"{name}_scalar" if name else None)
            vec_addr = self.alloc_vec(name)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        def vec_range(base):
            return list(range(base, base + VLEN))

        def slot_reads_writes(engine, slot):
            if engine == "alu":
                _, dest, a1, a2 = slot
                return [a1, a2], [dest]
            if engine == "valu":
                if slot[0] == "vbroadcast":
                    _, dest, src = slot
                    return [src], vec_range(dest)
                _, dest, a1, a2 = slot
                return vec_range(a1) + vec_range(a2), vec_range(dest)
            if engine == "load":
                op = slot[0]
                if op == "const":
                    _, dest, _val = slot
                    return [], [dest]
                if op == "load":
                    _, dest, addr = slot
                    return [addr], [dest]
                if op == "vload":
                    _, dest, addr = slot
                    return [addr], vec_range(dest)
                if op == "load_offset":
                    _, dest_base, addr_base, offset = slot
                    return [addr_base + offset], [dest_base + offset]
            if engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    return [addr, src], []
                if op == "vstore":
                    _, addr, src = slot
                    return [addr] + vec_range(src), []
            if engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dest, cond, a, b = slot
                    return [cond, a, b], [dest]
                if op == "vselect":
                    _, dest, cond, a, b = slot
                    return vec_range(cond) + vec_range(a) + vec_range(b), vec_range(dest)
                if op == "pause":
                    return [], []
            return [], []

        ops = []
        for engine, slot in slots:
            reads, writes = slot_reads_writes(engine, slot)
            ops.append(
                {
                    "engine": engine,
                    "slot": slot,
                    "reads": reads,
                    "writes": writes,
                }
            )

        n_ops = len(ops)
        dependents = [[] for _ in range(n_ops)]
        pending_deps = [0] * n_ops
        last_writer_by_addr = {}  # RAW/WAW: last writer per address
        readers_by_addr = defaultdict(list)  # WAR: outstanding readers per address

        for i, op in enumerate(ops):
            predecessors = set()
            for addr in op["reads"]:
                # RAW: this read depends on the most recent writer.
                if addr in last_writer_by_addr:
                    predecessors.add(last_writer_by_addr[addr])
            for addr in op["writes"]:
                # WAW: serialize writers to the same address.
                if addr in last_writer_by_addr:
                    predecessors.add(last_writer_by_addr[addr])
                # WAR: this write must wait for earlier reads to finish.
                for r in readers_by_addr.get(addr, []):
                    predecessors.add(r)
            for pred in predecessors:
                dependents[pred].append(i)
            pending_deps[i] = len(predecessors)
            for addr in op["reads"]:
                # Track readers for WAR edges of future writes.
                readers_by_addr[addr].append(i)
            for addr in op["writes"]:
                # New write clears pending reads; becomes last writer (RAW/WAW).
                readers_by_addr[addr] = []
                last_writer_by_addr[addr] = i

        # Greedy list scheduling: pack ready ops into per-engine slots each cycle.
        ready_ops = [i for i in range(n_ops) if pending_deps[i] == 0]
        schedule = []
        cycle_idx = 0
        while ready_ops:
            if len(schedule) <= cycle_idx:
                schedule.append({engine: [] for engine in SLOT_LIMITS})
            slots_left_by_engine = copy.copy(SLOT_LIMITS)
            ready_ops_next = []
            scheduled_ops = []
            for op_idx in ready_ops:
                engine = ops[op_idx]["engine"]
                if slots_left_by_engine[engine] > 0:
                    # Schedule op when its deps are satisfied and slot is free.
                    schedule[cycle_idx][engine].append(ops[op_idx]["slot"])
                    slots_left_by_engine[engine] -= 1
                    scheduled_ops.append(op_idx)
                else:
                    ready_ops_next.append(op_idx)

            if not scheduled_ops:
                # No available slots this cycle; advance time.
                cycle_idx += 1
                continue

            for op_idx in scheduled_ops:
                for succ in dependents[op_idx]:
                    # Release successors once all RAW/WAR/WAW deps are satisfied.
                    pending_deps[succ] -= 1
                    if pending_deps[succ] == 0:
                        ready_ops_next.append(succ)
            # Next cycle considers newly unblocked ops plus leftovers.
            ready_ops = ready_ops_next
            cycle_idx += 1

        instrs = [
            {engine: slots for engine, slots in bundle.items() if slots}
            for bundle in schedule
            if any(bundle.values())
        ]
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("valu", (op1, tmp1, val_hash_addr, self.vec_const(val1))))
            slots.append(("valu", (op3, tmp2, val_hash_addr, self.vec_const(val3))))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vector implementation using vector ALU and load/store.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        body = []  # array of slots

        zero_const = self.vec_const(0)
        one_const = self.vec_const(1)
        two_const = self.vec_const(2)
        
        # Per-chunk temporaries to reduce false dependencies in scheduling.
        n_chunks = batch_size // VLEN
        tmp_node_vals = [self.alloc_vec(f"tmp_node_val_{chunk}") for chunk in range(n_chunks)]
        tmp_addrs = [self.alloc_vec(f"tmp_addr_{chunk}") for chunk in range(n_chunks)]

        n_nodes_vec = self.alloc_vec("n_nodes_vec")
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))
        forest_values_p_vec = self.alloc_vec("forest_values_p_vec")
        self.add("valu", ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"]))

        idx_base = self.alloc_scratch("idx_base", batch_size)
        obj_val_base = self.alloc_scratch("obj_val_base", batch_size)

        for chunk_index in range(n_chunks):
            obj_index = chunk_index * VLEN
            obj_index_const = self.scratch_const(obj_index)
            # idx = mem[inp_indices_p + obj_index]
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], obj_index_const)))
            body.append(("load", ("vload", idx_base + obj_index, tmp_scalar)))
            # obj_val = mem[inp_values_p + obj_index]
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], obj_index_const)))
            body.append(("load", ("vload", obj_val_base + obj_index, tmp_scalar)))

        for round_number in range(rounds):
            for chunk_index in range(n_chunks):
                obj_index = chunk_index * VLEN
                tmp_idx = idx_base + obj_index
                tmp_obj_val = obj_val_base + obj_index
                tmp_node_val = tmp_node_vals[chunk_index]
                tmp_addr = tmp_addrs[chunk_index]
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_addr, forest_values_p_vec, tmp_idx)))
                for offset in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, offset)))
                # obj_val = myhash(obj_val ^ node_val)
                body.append(("valu", ("^", tmp_obj_val, tmp_obj_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_obj_val, tmp_node_val, tmp_addr, round_number, chunk_index))
                # idx = 2*idx + (1 if obj_val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp_node_val, tmp_obj_val, two_const)))
                body.append(("valu", ("==", tmp_node_val, tmp_node_val, zero_const)))
                body.append(("flow", ("vselect", tmp_node_val, tmp_node_val, one_const, two_const)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp_node_val)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp_addr, tmp_idx, n_nodes_vec)))
                body.append(("flow", ("vselect", tmp_idx, tmp_addr, tmp_idx, zero_const)))
        for chunk_index in range(n_chunks):
            obj_index = chunk_index * VLEN
            obj_index_const = self.scratch_const(obj_index)
            tmp_obj_val = obj_val_base + obj_index
            tmp_idx = idx_base + obj_index
            # mem[inp_values_p + obj_index] = obj_val
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], obj_index_const)))
            body.append(("store", ("vstore", tmp_scalar, tmp_obj_val)))
            # mem[inp_indices_p + obj_index] = idx
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], obj_index_const)))
            body.append(("store", ("vstore", tmp_scalar, tmp_idx)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
