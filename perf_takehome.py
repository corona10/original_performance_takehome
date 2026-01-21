"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_slot_def_use(self, engine, slot):
        """Return the set of defined (dest) and used (src) scratch addresses for a slot"""
        defs = set()
        uses = set()

        if engine == "debug":
            return defs, uses

        if engine == "alu":
            # (op, dest, a1, a2)
            op, dest, a1, a2 = slot
            defs.add(dest)
            uses.add(a1)
            uses.add(a2)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                # ("vbroadcast", dest, src)
                _, dest, src = slot
                for i in range(VLEN):
                    defs.add(dest + i)
                uses.add(src)
            else:
                # (op, dest, a1, a2) - vector operation
                op, dest, a1, a2 = slot
                for i in range(VLEN):
                    defs.add(dest + i)
                    uses.add(a1 + i)
                    uses.add(a2 + i)
        elif engine == "load":
            if slot[0] == "load":
                # ("load", dest, addr)
                _, dest, addr = slot
                defs.add(dest)
                uses.add(addr)
            elif slot[0] == "vload":
                # ("vload", dest, addr) - addr is scalar, dest is vector
                _, dest, addr = slot
                for i in range(VLEN):
                    defs.add(dest + i)
                uses.add(addr)
            elif slot[0] == "const":
                # ("const", dest, val)
                _, dest, val = slot
                defs.add(dest)
            elif slot[0] == "load_offset":
                # ("load_offset", dest, addr, offset)
                _, dest, addr, offset = slot
                defs.add(dest + offset)
                uses.add(addr + offset)
        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                _, addr, src = slot
                uses.add(addr)
                uses.add(src)
            elif slot[0] == "vstore":
                # ("vstore", addr, src) - addr is scalar, src is vector
                _, addr, src = slot
                uses.add(addr)
                for i in range(VLEN):
                    uses.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                # ("select", dest, cond, a, b)
                _, dest, cond, a, b = slot
                defs.add(dest)
                uses.add(cond)
                uses.add(a)
                uses.add(b)
            elif slot[0] == "vselect":
                # ("vselect", dest, cond, a, b) - all vectors
                _, dest, cond, a, b = slot
                for i in range(VLEN):
                    defs.add(dest + i)
                    uses.add(cond + i)
                    uses.add(a + i)
                    uses.add(b + i)
            elif slot[0] == "pause":
                pass

        return defs, uses

    def rename_registers_pass(self, slots):
        """
        Register renaming pass: assigns unique scratch addresses for each iteration
        to expose parallelism. Interleaves iterations for better VLIW packing.
        """
        # Find iteration pattern by store operations
        store_indices = [i for i, (e, s) in enumerate(slots) if e == "store"]

        if len(store_indices) < 4:
            return slots

        iter_size = store_indices[2] - store_indices[0]
        n_iters = len(slots) // iter_size

        if n_iters < 2:
            return slots

        # Analyze one iteration to find internal scratch addresses
        first_iter = slots[:iter_size]
        internal_defs = set()
        for engine, slot in first_iter:
            if engine == "debug":
                continue
            defs, _ = self.get_slot_def_use(engine, slot)
            internal_defs.update(defs)

        n_internal = len(internal_defs)
        if n_internal == 0:
            return slots

        # Calculate maximum interleave factor based on scratch space
        # Must be multiple of VLEN for vectorization
        available_scratch = SCRATCH_SIZE - self.scratch_ptr
        max_interleave = min(available_scratch // n_internal, n_iters)
        # Try to use more interleave - use 2x VLEN chunks if possible
        max_interleave = (max_interleave // VLEN) * VLEN  # Round down to VLEN multiple

        # Debug: print interleave info
        # print(f"n_internal={n_internal}, available={available_scratch}, max_interleave={max_interleave}, n_iters={n_iters}")

        if max_interleave < VLEN:
            return slots

        # Create mapping from original addr -> slot index
        internal_list = sorted(internal_defs)
        addr_to_slot = {addr: idx for idx, addr in enumerate(internal_list)}

        # Allocate scratch space for renamed registers
        base_offset = self.scratch_ptr
        self.scratch_ptr += max_interleave * n_internal

        result = []

        # Process in groups of max_interleave iterations
        for group_start in range(0, n_iters, max_interleave):
            group_size = min(max_interleave, n_iters - group_start)

            if group_size < VLEN:
                # Remaining iterations that don't fill a VLEN group - copy as-is
                for iter_idx in range(group_start, group_start + group_size):
                    iter_start = iter_idx * iter_size
                    iter_end = iter_start + iter_size
                    result.extend(slots[iter_start:iter_end])
                continue

            # Process in VLEN-sized chunks for vectorization
            n_vlen_chunks = group_size // VLEN

            for vlen_chunk in range(n_vlen_chunks):
                chunk_base_lane = vlen_chunk * VLEN

                # For each instruction in an iteration
                for instr_offset in range(iter_size):
                    # Emit VLEN scalar instructions with consecutive addresses
                    for lane_in_chunk in range(VLEN):
                        lane = chunk_base_lane + lane_in_chunk
                        iter_idx = group_start + lane
                        slot_idx = iter_idx * iter_size + instr_offset
                        engine, slot = slots[slot_idx]

                        if engine == "debug":
                            result.append((engine, slot))
                            continue

                        new_slot = self.rename_slot_addrs(engine, slot, addr_to_slot, base_offset, lane, internal_defs, max_interleave)
                        result.append((engine, new_slot))

        return result

    def rename_slot_addrs(self, engine, slot, addr_to_slot, base_offset, lane, internal_defs, max_interleave):
        """Rename internal addresses in a slot for the given lane."""
        def rename(addr):
            if addr in internal_defs:
                slot_idx = addr_to_slot[addr]
                # Layout: reg0_lane0, reg0_lane1, ..., reg1_lane0, reg1_lane1, ...
                # This makes same register across lanes contiguous for vectorization
                return base_offset + slot_idx * max_interleave + lane
            return addr

        if engine == "alu":
            op, dest, a1, a2 = slot
            return (op, rename(dest), rename(a1), rename(a2))
        elif engine == "load":
            if slot[0] == "load":
                return ("load", rename(slot[1]), rename(slot[2]))
            elif slot[0] == "const":
                return ("const", rename(slot[1]), slot[2])
            elif slot[0] == "vload":
                return ("vload", rename(slot[1]), rename(slot[2]))
            elif slot[0] == "load_offset":
                return ("load_offset", rename(slot[1]), rename(slot[2]), slot[3])
        elif engine == "store":
            if slot[0] == "store":
                return ("store", rename(slot[1]), rename(slot[2]))
            elif slot[0] == "vstore":
                return ("vstore", rename(slot[1]), rename(slot[2]))
        elif engine == "flow":
            if slot[0] == "select":
                return ("select", rename(slot[1]), rename(slot[2]), rename(slot[3]), rename(slot[4]))
            elif slot[0] == "vselect":
                return ("vselect", rename(slot[1]), rename(slot[2]), rename(slot[3]), rename(slot[4]))
            elif slot[0] == "pause":
                return slot

        return slot

    def vectorize_pass(self, slots):
        """
        Vectorization pass: converts groups of VLEN scalar operations into vector operations.
        Only vectorizes when ALL operands are contiguous (no broadcast) to ensure correctness.
        """
        # Separate debug instructions - they shouldn't block vectorization
        non_debug = [(i, e, s) for i, (e, s) in enumerate(slots) if e != "debug"]
        debug_slots = [(i, e, s) for i, (e, s) in enumerate(slots) if e == "debug"]

        result = []
        i = 0
        n = len(non_debug)
        broadcast_cache = {}  # scalar_addr -> vector_addr for reuse

        while i < n:
            orig_idx, engine, slot = non_debug[i]

            # Try to vectorize ALU operations (contiguous or broadcast operands)
            if engine == "alu" and i + VLEN <= n:
                op, dest, a1, a2 = slot
                can_vectorize = True
                a1_is_broadcast = True  # All lanes use same a1
                a2_is_broadcast = True  # All lanes use same a2

                # Check if next VLEN instructions form a vectorizable group
                for j in range(1, VLEN):
                    if i + j >= n:
                        can_vectorize = False
                        break
                    _, next_engine, next_slot = non_debug[i + j]
                    if next_engine != "alu":
                        can_vectorize = False
                        break
                    next_op, next_dest, next_a1, next_a2 = next_slot
                    # Check same operation and contiguous dest
                    if next_op != op or next_dest != dest + j:
                        can_vectorize = False
                        break
                    # Check a1: contiguous or broadcast
                    if next_a1 != a1:
                        a1_is_broadcast = False
                        if next_a1 != a1 + j:
                            can_vectorize = False
                            break
                    # Check a2: contiguous or broadcast
                    if next_a2 != a2:
                        a2_is_broadcast = False
                        if next_a2 != a2 + j:
                            can_vectorize = False
                            break

                if can_vectorize:
                    # Handle broadcast operands with caching
                    vec_a1 = a1
                    vec_a2 = a2
                    if a1_is_broadcast:
                        if a1 not in broadcast_cache:
                            # Check if we have enough scratch space
                            if self.scratch_ptr + VLEN <= SCRATCH_SIZE:
                                broadcast_cache[a1] = self.alloc_scratch(length=VLEN)
                                result.append(("valu", ("vbroadcast", broadcast_cache[a1], a1)))
                            else:
                                # No space for broadcast, skip vectorization
                                result.append((engine, slot))
                                i += 1
                                continue
                        vec_a1 = broadcast_cache[a1]
                    if a2_is_broadcast:
                        if a2 not in broadcast_cache:
                            if self.scratch_ptr + VLEN <= SCRATCH_SIZE:
                                broadcast_cache[a2] = self.alloc_scratch(length=VLEN)
                                result.append(("valu", ("vbroadcast", broadcast_cache[a2], a2)))
                            else:
                                result.append((engine, slot))
                                i += 1
                                continue
                        vec_a2 = broadcast_cache[a2]
                    result.append(("valu", (op, dest, vec_a1, vec_a2)))
                    i += VLEN
                    continue

            # Try to vectorize select operations (only when ALL operands are contiguous)
            if engine == "flow" and slot[0] == "select" and i + VLEN <= n:
                _, dest, cond, a, b = slot
                can_vectorize = True

                for j in range(1, VLEN):
                    if i + j >= n:
                        can_vectorize = False
                        break
                    _, next_engine, next_slot = non_debug[i + j]
                    if next_engine != "flow" or next_slot[0] != "select":
                        can_vectorize = False
                        break
                    _, next_dest, next_cond, next_a, next_b = next_slot
                    # Check ALL addresses contiguous
                    if (next_dest != dest + j or next_cond != cond + j or
                        next_a != a + j or next_b != b + j):
                        can_vectorize = False
                        break

                if can_vectorize:
                    result.append(("flow", ("vselect", dest, cond, a, b)))
                    i += VLEN
                    continue

            # No vectorization possible, keep original instruction
            result.append((engine, slot))
            i += 1

        # Add debug slots back at the end (they don't affect cycles)
        for _, engine, slot in debug_slots:
            result.append((engine, slot))

        return result

    def super_instruction_pass(self, slots):
        """
        Super instruction: interleave multiple iterations to fill VLIW slots.
        Find iteration boundaries and interleave independent operations.
        """
        # Find iteration pattern by looking at store operations
        store_indices = [i for i, (e, s) in enumerate(slots) if e == "store"]

        if len(store_indices) < 4:
            return slots

        # Each iteration has 2 stores, so iteration size = distance between 1st and 3rd store
        iter_size = store_indices[2] - store_indices[0]
        n_iters = len(slots) // iter_size

        if n_iters < 2:
            return slots

        # Determine how many iterations to interleave based on slot limits
        # We want to maximize parallelism while respecting slot limits
        # load: 2 slots -> can do 2 loads per cycle
        # store: 2 slots -> can do 2 stores per cycle
        # alu: 12 slots -> can do 12 ALU ops per cycle
        # flow: 1 slot -> can do 1 flow op per cycle (bottleneck for select)

        # Count operations per iteration
        ops_per_iter = defaultdict(int)
        for e, s in slots[:iter_size]:
            if e != "debug":
                ops_per_iter[e] += 1

        # Interleave factor: how many iterations can run in parallel
        # Limited by the most constrained resource
        interleave = min(
            SLOT_LIMITS["alu"] // max(ops_per_iter.get("alu", 1), 1),
            SLOT_LIMITS["load"] // max(ops_per_iter.get("load", 1), 1),
            SLOT_LIMITS["store"] // max(ops_per_iter.get("store", 1), 1),
            SLOT_LIMITS["flow"] // max(ops_per_iter.get("flow", 1), 1),
            n_iters
        )

        if interleave < 2:
            return slots

        # Reorder: for each instruction position, emit that instruction from all interleaved iterations
        result = []
        for group_start in range(0, n_iters, interleave):
            group_end = min(group_start + interleave, n_iters)
            actual_interleave = group_end - group_start

            for instr_idx in range(iter_size):
                for iter_idx in range(group_start, group_end):
                    slot_idx = iter_idx * iter_size + instr_idx
                    if slot_idx < len(slots):
                        result.append(slots[slot_idx])

        return result

    def vselect_fusion_pass(self, slots):
        """
        Convert VLEN consecutive select ops to a single vselect op.
        Supports broadcast for a/b operands.
        """
        result = []
        i = 0
        n = len(slots)
        broadcast_cache = {}

        while i < n:
            engine, slot = slots[i]

            if engine == "flow" and slot[0] == "select" and i + VLEN <= n:
                _, dest0, cond0, a0, b0 = slot
                can_fuse = True
                a_is_broadcast = True
                b_is_broadcast = True

                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "flow" or next_slot[0] != "select":
                        can_fuse = False
                        break
                    _, next_dest, next_cond, next_a, next_b = next_slot
                    # Check contiguous dest and cond
                    if next_dest != dest0 + j or next_cond != cond0 + j:
                        can_fuse = False
                        break
                    # Check a: contiguous or broadcast
                    if next_a != a0:
                        a_is_broadcast = False
                        if next_a != a0 + j:
                            can_fuse = False
                            break
                    # Check b: contiguous or broadcast
                    if next_b != b0:
                        b_is_broadcast = False
                        if next_b != b0 + j:
                            can_fuse = False
                            break

                if can_fuse:
                    # Handle broadcast operands
                    if a_is_broadcast:
                        if a0 not in broadcast_cache:
                            vec_a = self.alloc_scratch(length=VLEN)
                            result.append(("valu", ("vbroadcast", vec_a, a0)))
                            broadcast_cache[a0] = vec_a
                        a0 = broadcast_cache[a0]
                    if b_is_broadcast:
                        if b0 not in broadcast_cache:
                            vec_b = self.alloc_scratch(length=VLEN)
                            result.append(("valu", ("vbroadcast", vec_b, b0)))
                            broadcast_cache[b0] = vec_b
                        b0 = broadcast_cache[b0]
                    result.append(("flow", ("vselect", dest0, cond0, a0, b0)))
                    i += VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def strength_reduction_pass(self, slots):
        """
        Replace expensive operations with cheaper equivalents:
        - x * 2 -> x + x
        - x % 2 -> x & 1
        """
        # Build reverse map: addr -> value
        addr_to_const = {addr: val for val, addr in self.const_map.items()}

        result = []
        for engine, slot in slots:
            if engine == "alu":
                op, dest, a1, a2 = slot
                # x * 2 -> x + x
                if op == "*" and a2 in addr_to_const and addr_to_const[a2] == 2:
                    result.append(("alu", ("+", dest, a1, a1)))
                    continue
                # x % 2 -> x & 1
                if op == "%" and a2 in addr_to_const and addr_to_const[a2] == 2:
                    one_const = self.scratch_const(1)
                    result.append(("alu", ("&", dest, a1, one_const)))
                    continue
            result.append((engine, slot))
        return result

    def select_elimination_pass(self, slots):
        """
        Replace select operations with ALU operations where possible:
        - select(dest, cond, 1, 2) -> dest = 2 - cond (when cond is 0 or 1)
        - select(dest, cond, x, 0) -> dest = x * cond (when cond is 0 or 1)
        """
        addr_to_const = {addr: val for val, addr in self.const_map.items()}

        result = []
        for engine, slot in slots:
            if engine == "flow" and slot[0] == "select":
                _, dest, cond, a, b = slot
                # select(dest, cond, 1, 2) -> 2 - cond
                if a in addr_to_const and b in addr_to_const:
                    a_val, b_val = addr_to_const[a], addr_to_const[b]
                    if a_val == 1 and b_val == 2:
                        # cond=1 -> 1, cond=0 -> 2, so dest = 2 - cond
                        two_const = self.scratch_const(2)
                        result.append(("alu", ("-", dest, two_const, cond)))
                        continue
                # select(dest, cond, x, 0) -> x * cond
                if b in addr_to_const and addr_to_const[b] == 0:
                    result.append(("alu", ("*", dest, a, cond)))
                    continue
            result.append((engine, slot))
        return result

    def cse_pass(self, slots):
        """
        Common Subexpression Elimination: reuse results of identical computations.
        Tracks ALU operations and replaces redundant ones with copies.
        """
        result = []
        # Map (op, a1, a2) -> dest address for computed values
        computed = {}
        # Track which addresses have been overwritten
        valid_dests = set()

        for engine, slot in slots:
            if engine == "alu":
                op, dest, a1, a2 = slot
                key = (op, a1, a2)

                # Check if we've computed this before and the result is still valid
                if key in computed and computed[key] in valid_dests:
                    # Reuse the previous result - but we still need to write to dest
                    # For commutative ops, also check reversed operands
                    prev_dest = computed[key]
                    if prev_dest != dest:
                        # Copy from previous result (use + with 0, but that needs a zero const)
                        # Actually, we can't easily copy without adding instructions
                        # So we just skip CSE for now if dest is different
                        result.append((engine, slot))
                        computed[key] = dest
                        valid_dests.add(dest)
                    else:
                        # Same dest - this is redundant, skip it
                        pass
                else:
                    result.append((engine, slot))
                    computed[key] = dest
                    valid_dests.add(dest)

                    # For commutative ops, also store reversed key
                    if op in ["+", "*", "^", "&", "|", "=="]:
                        computed[(op, a2, a1)] = dest

                # Invalidate any previous computation that used dest as input
                # (because dest is now overwritten)
                keys_to_remove = [k for k, v in computed.items() if v == dest or dest in k[1:]]
                for k in keys_to_remove:
                    if k in computed and computed[k] != dest:
                        del computed[k]

            elif engine == "load":
                result.append((engine, slot))
                # Load overwrites dest, invalidate computations using it
                if slot[0] in ["load", "const"]:
                    dest = slot[1]
                    valid_dests.add(dest)
                elif slot[0] == "vload":
                    dest = slot[1]
                    for i in range(VLEN):
                        valid_dests.add(dest + i)

            elif engine == "store":
                result.append((engine, slot))
                # Store doesn't affect scratch, but memory writes could
                # For now, we don't track memory

            elif engine == "flow":
                result.append((engine, slot))
                if slot[0] == "select":
                    dest = slot[1]
                    valid_dests.add(dest)
                elif slot[0] == "vselect":
                    dest = slot[1]
                    for i in range(VLEN):
                        valid_dests.add(dest + i)

            else:
                result.append((engine, slot))

        return result

    def valu_fusion_pass(self, slots):
        """
        Convert VLEN consecutive scalar ALU ops to a single valu op.
        Supports broadcast: if a1 or a2 are the same across all lanes, broadcast them first.
        """
        result = []
        i = 0
        n = len(slots)
        broadcast_cache = {}  # scalar_addr -> vector_addr

        while i < n:
            engine, slot = slots[i]

            # Try to fuse VLEN consecutive ALU ops
            if engine == "alu" and i + VLEN <= n:
                op, dest0, a1_0, a2_0 = slot
                can_fuse = True
                a1_is_broadcast = True
                a2_is_broadcast = True

                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "alu":
                        can_fuse = False
                        break
                    next_op, next_dest, next_a1, next_a2 = next_slot
                    # Check same op and contiguous dest
                    if next_op != op or next_dest != dest0 + j:
                        can_fuse = False
                        break
                    # Check a1: contiguous or broadcast
                    if next_a1 != a1_0:
                        a1_is_broadcast = False
                        if next_a1 != a1_0 + j:
                            can_fuse = False
                            break
                    # Check a2: contiguous or broadcast
                    if next_a2 != a2_0:
                        a2_is_broadcast = False
                        if next_a2 != a2_0 + j:
                            can_fuse = False
                            break

                if can_fuse:
                    # Handle broadcast operands
                    if a1_is_broadcast:
                        if a1_0 not in broadcast_cache:
                            vec_a1 = self.alloc_scratch(length=VLEN)
                            result.append(("valu", ("vbroadcast", vec_a1, a1_0)))
                            broadcast_cache[a1_0] = vec_a1
                        a1_0 = broadcast_cache[a1_0]
                    if a2_is_broadcast:
                        if a2_0 not in broadcast_cache:
                            vec_a2 = self.alloc_scratch(length=VLEN)
                            result.append(("valu", ("vbroadcast", vec_a2, a2_0)))
                            broadcast_cache[a2_0] = vec_a2
                        a2_0 = broadcast_cache[a2_0]
                    result.append(("valu", (op, dest0, a1_0, a2_0)))
                    i += VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def vload_fusion_pass(self, slots):
        """
        Convert VLEN consecutive scalar loads from contiguous memory to a single vload.
        vload semantics: reads mem[scratch[addr]], mem[scratch[addr]+1], ..., mem[scratch[addr]+VLEN-1]
        So we need: contiguous dest AND contiguous addr (where scratch[addr+j] = scratch[addr] + j)
        """
        result = []
        i = 0
        n = len(slots)

        while i < n:
            engine, slot = slots[i]

            # Try to fuse VLEN consecutive loads
            if engine == "load" and slot[0] == "load" and i + VLEN <= n:
                _, dest0, addr0 = slot
                can_fuse = True

                # Check if next VLEN-1 loads form a fusable pattern
                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "load" or next_slot[0] != "load":
                        can_fuse = False
                        break
                    _, next_dest, next_addr = next_slot
                    # Check contiguous dest AND contiguous addr
                    if next_dest != dest0 + j or next_addr != addr0 + j:
                        can_fuse = False
                        break

                if can_fuse:
                    # All loads have contiguous dest and addr
                    # But vload needs a single addr that points to contiguous memory
                    # This only works if scratch[addr0+j] = base + j for some base
                    # We'll use load_offset instead for now
                    result.append(("load", ("vload", dest0, addr0)))
                    i += VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def vstore_fusion_pass(self, slots):
        """
        Convert VLEN consecutive scalar stores to contiguous memory to a single vstore.
        vstore semantics: writes scratch[src+0..VLEN-1] to mem[scratch[addr]], mem[scratch[addr]+1], ...
        We need: contiguous addr AND contiguous src
        """
        result = []
        i = 0
        n = len(slots)

        while i < n:
            engine, slot = slots[i]

            # Try to fuse VLEN consecutive stores
            if engine == "store" and slot[0] == "store" and i + VLEN <= n:
                _, addr0, src0 = slot
                can_fuse = True

                # Check if next VLEN-1 stores form a fusable pattern
                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "store" or next_slot[0] != "store":
                        can_fuse = False
                        break
                    _, next_addr, next_src = next_slot
                    # Check contiguous addr AND contiguous src
                    if next_addr != addr0 + j or next_src != src0 + j:
                        can_fuse = False
                        break

                if can_fuse:
                    result.append(("store", ("vstore", addr0, src0)))
                    i += VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """VLIW scheduling based on dependency analysis with critical path priority"""
        # Apply strength reduction before renaming
        slots = self.strength_reduction_pass(slots)

        # Eliminate select operations where possible (convert to ALU)
        slots = self.select_elimination_pass(slots)

        # CSE disabled - causes correctness issues with address calculations
        # slots = self.cse_pass(slots)

        # Apply register renaming with contiguous layout for vectorization
        slots = self.rename_registers_pass(slots)

        # vload/vstore fusion disabled - addr not contiguous across lanes
        # slots = self.vload_fusion_pass(slots)

        # Apply vectorization pass to convert VLEN scalar ops to vector ops
        slots = self.vectorize_pass(slots)

        # Filter out debug slots
        non_debug = [(i, e, s) for i, (e, s) in enumerate(slots) if e != "debug"]
        debug_slots = [(i, e, s) for i, (e, s) in enumerate(slots) if e == "debug"]

        if not non_debug:
            return []

        # Dependency analysis: which instructions must execute after which
        n = len(non_debug)
        deps = [set() for _ in range(n)]  # deps[i] = set of instruction indices that i depends on

        # Last instruction that defined each scratch address
        last_def = {}
        # Instructions that used each scratch address
        last_use = defaultdict(set)

        for idx, (orig_i, engine, slot) in enumerate(non_debug):
            defs, uses = self.get_slot_def_use(engine, slot)

            # RAW (Read After Write): depend on instruction that wrote to address I read
            for addr in uses:
                if addr in last_def:
                    deps[idx].add(last_def[addr])

            # WAW (Write After Write): depend on instruction that wrote to address I write
            for addr in defs:
                if addr in last_def:
                    deps[idx].add(last_def[addr])

            # WAR (Write After Read): depend on instructions that read address I write
            for addr in defs:
                for prev_idx in last_use[addr]:
                    deps[idx].add(prev_idx)

            # Update tracking
            for addr in defs:
                last_def[addr] = idx
                last_use[addr] = set()
            for addr in uses:
                last_use[addr].add(idx)

        # Topological sort (Kahn's algorithm) with priority for loads
        in_degree = [len(deps[i]) for i in range(n)]
        dependents = [[] for _ in range(n)]
        for idx in range(n):
            for dep in deps[idx]:
                dependents[dep].append(idx)

        ready = [i for i in range(n) if in_degree[i] == 0]
        topo_order = []

        while ready:
            # Sort by engine priority: load > alu/valu > flow > store
            def priority(idx):
                engine = non_debug[idx][1]
                if engine == "load":
                    return 0
                elif engine in ("alu", "valu"):
                    return 1
                elif engine == "flow":
                    return 2
                else:
                    return 3
            ready.sort(key=priority)
            idx = ready.pop(0)
            topo_order.append(idx)
            for dep_idx in dependents[idx]:
                in_degree[dep_idx] -= 1
                if in_degree[dep_idx] == 0:
                    ready.append(dep_idx)

        if len(topo_order) != n:
            raise RuntimeError("Cycle in dependencies!")

        # Pack instructions into bundles respecting slot limits, in topo order
        result = []
        scheduled_cycle = [-1] * n  # Cycle each instruction is scheduled in

        for idx in topo_order:
            _, engine, slot = non_debug[idx]

            # Find the latest cycle among instructions this one depends on
            if deps[idx]:
                earliest_cycle = max(scheduled_cycle[dep] for dep in deps[idx]) + 1
            else:
                earliest_cycle = 0

            # Find a cycle starting from earliest_cycle with available slot
            placed = False
            for cycle_idx in range(earliest_cycle, len(result)):
                bundle = result[cycle_idx]
                if len(bundle.get(engine, [])) < SLOT_LIMITS[engine]:
                    bundle.setdefault(engine, []).append(slot)
                    scheduled_cycle[idx] = cycle_idx
                    placed = True
                    break

            if not placed:
                # Create new bundle
                new_bundle = {engine: [slot]}
                result.append(new_bundle)
                scheduled_cycle[idx] = len(result) - 1

        # Trace: analyze slot utilization
        if True:  # Set to True to enable tracing
            print(f"\n=== Instruction Trace Analysis ===")
            print(f"Total cycles: {len(result)}")

            # Count by engine type
            engine_counts = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0}
            engine_slots_used = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0}

            for bundle in result:
                for eng in engine_counts:
                    if eng in bundle:
                        engine_counts[eng] += 1
                        engine_slots_used[eng] += len(bundle[eng])

            print(f"\nEngine utilization (cycles with at least 1 slot used):")
            for eng in ["alu", "valu", "load", "store", "flow"]:
                pct = 100 * engine_counts[eng] / len(result) if result else 0
                avg_slots = engine_slots_used[eng] / engine_counts[eng] if engine_counts[eng] > 0 else 0
                print(f"  {eng}: {engine_counts[eng]} cycles ({pct:.1f}%), avg {avg_slots:.1f}/{SLOT_LIMITS[eng]} slots")

            # Sample first 20 cycles
            print(f"\nFirst 20 cycles:")
            for i, bundle in enumerate(result[:20]):
                parts = []
                for eng in ["load", "alu", "valu", "flow", "store"]:
                    if eng in bundle:
                        parts.append(f"{eng}:{len(bundle[eng])}")
                print(f"  Cycle {i}: {', '.join(parts)}")

        return result

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
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
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
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
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

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
