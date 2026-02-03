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
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
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
        tmp1 = self.alloc_vec("tmp1")
        tmp2 = self.alloc_vec("tmp2")
        tmp3 = self.alloc_vec("tmp3")
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

        # Vector scratch registers
        tmp_idx = self.alloc_vec("tmp_idx")
        tmp_obj_val = self.alloc_vec("tmp_obj_val")
        tmp_node_val = self.alloc_vec("tmp_node_val")
        tmp_addr = self.alloc_vec("tmp_addr")

        n_nodes_vec = self.alloc_vec("n_nodes_vec")
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))
        forest_values_p_vec = self.alloc_vec("forest_values_p_vec")
        self.add("valu", ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"]))

        n_chunks = batch_size // VLEN

        for round_number in range(rounds):
            for chunk_index in range(n_chunks):
                obj_index = chunk_index * VLEN
                obj_index_const = self.scratch_const(obj_index)
                # idx = mem[inp_indices_p + obj_index]
                body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], obj_index_const)))
                body.append(("load", ("vload", tmp_idx, tmp_scalar)))
                # obj_val = mem[inp_values_p + obj_index]
                body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], obj_index_const)))
                body.append(("load", ("vload", tmp_obj_val, tmp_scalar)))
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_addr, forest_values_p_vec, tmp_idx)))
                for offset in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, offset)))
                # obj_val = myhash(obj_val ^ node_val)
                body.append(("valu", ("^", tmp_obj_val, tmp_obj_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_obj_val, tmp1, tmp2, round_number, chunk_index))
                # idx = 2*idx + (1 if obj_val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_obj_val, two_const)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("vselect", tmp3, tmp1, one_const, two_const)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_vec)))
                body.append(("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const)))
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
