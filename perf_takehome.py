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

    def iteration_interleaving_pass(self, slots):
        """
        Iteration interleaving with live range analysis for scratch reuse.
        Reorders iterations for vectorization opportunities while minimizing scratch usage
        through interval graph coloring-based register allocation.
        """
        # Find iteration pattern by store operations
        store_indices = [i for i, (e, s) in enumerate(slots) if e == "store"]

        if len(store_indices) < 4:
            return slots

        iter_size = store_indices[2] - store_indices[0]
        n_iters = len(slots) // iter_size

        if n_iters < 2:
            return slots

        # Analyze one iteration to find internal scratch addresses and live ranges
        first_iter = slots[:iter_size]
        internal_defs = set()

        # Build def-use chains for live range analysis
        addr_first_def = {}  # addr -> first instruction index that defines it
        addr_last_use = {}   # addr -> last instruction index that uses it

        for i, (engine, slot) in enumerate(first_iter):
            if engine == "debug":
                continue
            defs, uses = self.get_slot_def_use(engine, slot)
            internal_defs.update(defs)

            for addr in defs:
                if addr not in addr_first_def:
                    addr_first_def[addr] = i
            for addr in uses:
                addr_last_use[addr] = i

        n_internal = len(internal_defs)
        if n_internal == 0:
            return slots

        # Calculate live ranges: (start, end) for each internal address
        live_ranges = {}
        for addr in internal_defs:
            start = addr_first_def.get(addr, 0)
            end = addr_last_use.get(addr, iter_size - 1)
            live_ranges[addr] = (start, end)

        # Greedy interval coloring to minimize scratch usage
        # Sort addresses by live range start
        sorted_addrs = sorted(internal_defs, key=lambda a: live_ranges[a][0])

        # Assign colors (scratch slots) to addresses
        addr_to_color = {}
        color_end_time = []  # For each color, when it becomes free

        for addr in sorted_addrs:
            start, end = live_ranges[addr]

            # Find a free color (one that ended before this starts)
            assigned = False
            for color, end_time in enumerate(color_end_time):
                if end_time < start:
                    addr_to_color[addr] = color
                    color_end_time[color] = end
                    assigned = True
                    break

            if not assigned:
                # Need a new color
                addr_to_color[addr] = len(color_end_time)
                color_end_time.append(end)

        n_colors = len(color_end_time)

        # Reserve space for broadcasts (estimate ~20 broadcasts * VLEN = 160)
        broadcast_reserve = 20 * VLEN

        # Calculate maximum interleave factor based on reduced scratch space
        available_scratch = SCRATCH_SIZE - self.scratch_ptr - broadcast_reserve
        max_interleave = min(available_scratch // n_colors, n_iters) if n_colors > 0 else n_iters
        max_interleave = (max_interleave // VLEN) * VLEN  # Round down to VLEN multiple

        if max_interleave < VLEN:
            return slots

        # Allocate scratch space for renamed registers (using colors, not raw addresses)
        base_offset = self.scratch_ptr
        self.scratch_ptr += max_interleave * n_colors

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

                        new_slot = self.rename_slot_addrs_with_coloring(engine, slot, addr_to_color, base_offset, lane, internal_defs, n_colors, max_interleave)
                        result.append((engine, new_slot))

        return result

    def rename_slot_addrs_with_coloring(self, engine, slot, addr_to_color, base_offset, lane, internal_defs, n_colors, max_interleave):
        """Rename addresses using interval graph coloring for better scratch reuse."""
        def rename(addr):
            if addr in internal_defs:
                color = addr_to_color[addr]
                # Layout for vectorization: consecutive lanes must be consecutive in memory
                # [color0_lane0, color0_lane1, ..., color0_laneN, color1_lane0, ...]
                return base_offset + color * max_interleave + lane
            return addr

        if engine == "alu":
            op, dest, a1, a2 = slot
            return (op, rename(dest), rename(a1), rename(a2))
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                return ("vbroadcast", rename(dest), rename(src))
            else:
                op, dest, a1, a2 = slot
                return (op, rename(dest), rename(a1), rename(a2))
        elif engine == "load":
            if slot[0] == "load":
                _, dest, addr = slot
                return ("load", rename(dest), rename(addr))
            elif slot[0] == "vload":
                _, dest, addr = slot
                return ("vload", rename(dest), rename(addr))
            elif slot[0] == "const":
                _, dest, val = slot
                return ("const", rename(dest), val)
            elif slot[0] == "load_offset":
                _, dest, addr, offset = slot
                return ("load_offset", rename(dest), rename(addr), offset)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, src = slot
                return ("store", rename(addr), rename(src))
            elif slot[0] == "vstore":
                _, addr, src = slot
                return ("vstore", rename(addr), rename(src))
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                return ("select", rename(dest), rename(cond), rename(a), rename(b))
            elif slot[0] == "vselect":
                _, dest, cond, a, b = slot
                return ("vselect", rename(dest), rename(cond), rename(a), rename(b))
            elif slot[0] == "pause":
                return slot

        return slot

    def vectorize_pass(self, slots):
        """
        2-pass vectorization:
        Pass 1: Identify all broadcast values needed and insert vbroadcast at the beginning
        Pass 2: Convert VLEN scalar ops to vector ops using pre-computed broadcasts
        """
        # Separate debug instructions
        non_debug = [(i, e, s) for i, (e, s) in enumerate(slots) if e != "debug"]
        debug_slots = [(i, e, s) for i, (e, s) in enumerate(slots) if e == "debug"]

        n = len(non_debug)

        # Pass 1: Find all broadcast values needed
        broadcast_needed = set()
        i = 0
        while i < n:
            _, engine, slot = non_debug[i]

            if engine == "alu" and i + VLEN <= n:
                op, dest, a1, a2 = slot
                can_vectorize = True
                a1_is_broadcast = True
                a2_is_broadcast = True

                for j in range(1, VLEN):
                    if i + j >= n:
                        can_vectorize = False
                        break
                    _, next_engine, next_slot = non_debug[i + j]
                    if next_engine != "alu":
                        can_vectorize = False
                        break
                    next_op, next_dest, next_a1, next_a2 = next_slot
                    if next_op != op or next_dest != dest + j:
                        can_vectorize = False
                        break
                    if next_a1 != a1:
                        a1_is_broadcast = False
                        if next_a1 != a1 + j:
                            can_vectorize = False
                            break
                    if next_a2 != a2:
                        a2_is_broadcast = False
                        if next_a2 != a2 + j:
                            can_vectorize = False
                            break

                if can_vectorize:
                    if a1_is_broadcast:
                        broadcast_needed.add(a1)
                    if a2_is_broadcast:
                        broadcast_needed.add(a2)
                    i += VLEN
                    continue

            i += 1

        # Debug: print broadcast_needed (disabled)
        # print(f"Broadcast needed: {sorted(broadcast_needed)[:20]}... (total {len(broadcast_needed)})")
        # print(f"scratch_ptr before broadcast alloc: {self.scratch_ptr}")

        # Allocate all broadcasts upfront
        broadcast_cache = {}
        broadcast_instrs = []
        for addr in sorted(broadcast_needed):
            if self.scratch_ptr + VLEN <= SCRATCH_SIZE:
                vec_addr = self.alloc_scratch(length=VLEN)
                broadcast_cache[addr] = vec_addr
                broadcast_instrs.append(("valu", ("vbroadcast", vec_addr, addr)))

        # print(f"broadcast_cache keys: {sorted(broadcast_cache.keys())[:20]}")

        # Pass 2: Vectorize using pre-computed broadcasts
        result = list(broadcast_instrs)  # Start with all broadcasts
        i = 0
        while i < n:
            _, engine, slot = non_debug[i]

            if engine == "alu" and i + VLEN <= n:
                op, dest, a1, a2 = slot
                can_vectorize = True
                a1_is_broadcast = True
                a2_is_broadcast = True

                for j in range(1, VLEN):
                    if i + j >= n:
                        can_vectorize = False
                        break
                    _, next_engine, next_slot = non_debug[i + j]
                    if next_engine != "alu":
                        can_vectorize = False
                        break
                    next_op, next_dest, next_a1, next_a2 = next_slot
                    if next_op != op or next_dest != dest + j:
                        can_vectorize = False
                        break
                    if next_a1 != a1:
                        a1_is_broadcast = False
                        if next_a1 != a1 + j:
                            can_vectorize = False
                            break
                    if next_a2 != a2:
                        a2_is_broadcast = False
                        if next_a2 != a2 + j:
                            can_vectorize = False
                            break

                if can_vectorize:
                    # Check if broadcast is needed but not available
                    # If broadcast needed but not in cache, skip vectorization
                    if a1_is_broadcast and a1 not in broadcast_cache:
                        can_vectorize = False
                    if a2_is_broadcast and a2 not in broadcast_cache:
                        can_vectorize = False

                    if can_vectorize:
                        vec_a1 = broadcast_cache[a1] if a1_is_broadcast else a1
                        vec_a2 = broadcast_cache[a2] if a2_is_broadcast else a2
                        result.append(("valu", (op, dest, vec_a1, vec_a2)))
                        i += VLEN
                        continue

            # Try to vectorize select operations
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
                    if (next_dest != dest + j or next_cond != cond + j or
                        next_a != a + j or next_b != b + j):
                        can_vectorize = False
                        break

                if can_vectorize:
                    result.append(("flow", ("vselect", dest, cond, a, b)))
                    i += VLEN
                    continue

            result.append((engine, slot))
            i += 1

        # Add debug slots back
        for _, engine, slot in debug_slots:
            result.append((engine, slot))

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
        # Reverse map: addr -> set of keys that use this addr as operand
        addr_to_keys = defaultdict(set)
        # Reverse map: dest -> set of keys that have this dest
        dest_to_keys = defaultdict(set)

        for engine, slot in slots:
            if engine == "alu":
                op, dest, a1, a2 = slot
                key = (op, a1, a2)

                # Check if we've computed this before and the result is still valid
                if key in computed:
                    prev_dest = computed[key]
                    if prev_dest == dest:
                        # Same dest - this is redundant, skip it
                        continue
                    # Different dest - can't easily reuse, just compute again

                result.append((engine, slot))

                # Invalidate previous computations that wrote to dest
                for old_key in list(dest_to_keys.get(dest, [])):
                    if old_key in computed:
                        del computed[old_key]
                    dest_to_keys[dest].discard(old_key)

                # Invalidate computations that used dest as input
                for old_key in list(addr_to_keys.get(dest, [])):
                    if old_key in computed:
                        old_dest = computed[old_key]
                        dest_to_keys[old_dest].discard(old_key)
                        del computed[old_key]
                    addr_to_keys[dest].discard(old_key)

                # Store new computation
                computed[key] = dest
                dest_to_keys[dest].add(key)
                addr_to_keys[a1].add(key)
                addr_to_keys[a2].add(key)

                # For commutative ops, also store reversed key
                if op in ["+", "*", "^", "&", "|", "=="]:
                    rev_key = (op, a2, a1)
                    computed[rev_key] = dest
                    dest_to_keys[dest].add(rev_key)
                    addr_to_keys[a1].add(rev_key)
                    addr_to_keys[a2].add(rev_key)

            elif engine == "load":
                result.append((engine, slot))
                # Load overwrites dest, invalidate computations
                if slot[0] in ["load", "const", "load_offset"]:
                    dest = slot[1]
                    for old_key in list(addr_to_keys.get(dest, [])):
                        if old_key in computed:
                            old_dest = computed[old_key]
                            dest_to_keys[old_dest].discard(old_key)
                            del computed[old_key]
                        addr_to_keys[dest].discard(old_key)
                elif slot[0] == "vload":
                    dest = slot[1]
                    for i in range(VLEN):
                        d = dest + i
                        for old_key in list(addr_to_keys.get(d, [])):
                            if old_key in computed:
                                old_dest = computed[old_key]
                                dest_to_keys[old_dest].discard(old_key)
                                del computed[old_key]
                            addr_to_keys[d].discard(old_key)

            elif engine == "store":
                result.append((engine, slot))

            elif engine == "flow":
                result.append((engine, slot))
                if slot[0] == "select":
                    dest = slot[1]
                    for old_key in list(addr_to_keys.get(dest, [])):
                        if old_key in computed:
                            old_dest = computed[old_key]
                            dest_to_keys[old_dest].discard(old_key)
                            del computed[old_key]
                        addr_to_keys[dest].discard(old_key)
                elif slot[0] == "vselect":
                    dest = slot[1]
                    for i in range(VLEN):
                        d = dest + i
                        for old_key in list(addr_to_keys.get(d, [])):
                            if old_key in computed:
                                old_dest = computed[old_key]
                                dest_to_keys[old_dest].discard(old_key)
                                del computed[old_key]
                            addr_to_keys[d].discard(old_key)

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

        This pass looks for patterns like:
          alu: ('+', addr0, base, const0)  where scratch[const0] = 0
          alu: ('+', addr1, base, const1)  where scratch[const1] = 1
          ...
          load: ('load', dest0, addr0)
          load: ('load', dest1, addr1)
          ...

        And converts them to:
          vload: ('vload', dest0, base)

        This eliminates VLEN address calculations AND VLEN scalar loads.
        """
        result = []
        i = 0
        n = len(slots)

        # Build reverse const map: scratch addr -> const value
        addr_to_const = {addr: val for val, addr in self.const_map.items()}

        while i < n:
            engine, slot = slots[i]

            # Try to fuse VLEN consecutive loads that follow VLEN address calculations
            if engine == "alu" and slot[0] == "+" and i + 2 * VLEN <= n:
                # Check if this is the start of a fusable pattern:
                # VLEN alu additions followed by VLEN loads
                op, addr0, base0, const_addr0 = slot

                # The const address must contain a small integer offset (0, 1, 2, ...)
                if const_addr0 not in addr_to_const:
                    result.append((engine, slot))
                    i += 1
                    continue

                offset0 = addr_to_const[const_addr0]

                can_fuse = True
                alu_slots = [slot]

                # Check VLEN-1 more ALU ops with pattern: addr = base + const_j where const_j = offset0 + j
                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "alu" or next_slot[0] != "+":
                        can_fuse = False
                        break
                    _, next_addr, next_base, next_const_addr = next_slot
                    # Must have same base and contiguous addr output
                    if next_base != base0 or next_addr != addr0 + j:
                        can_fuse = False
                        break
                    # Must have const that is offset0 + j
                    if next_const_addr not in addr_to_const:
                        can_fuse = False
                        break
                    if addr_to_const[next_const_addr] != offset0 + j:
                        can_fuse = False
                        break
                    alu_slots.append(next_slot)

                if not can_fuse:
                    result.append((engine, slot))
                    i += 1
                    continue

                # Now check VLEN loads following the ALU ops
                load_start = i + VLEN
                if load_start + VLEN > n:
                    result.append((engine, slot))
                    i += 1
                    continue

                load_e, load_s = slots[load_start]
                if load_e != "load" or load_s[0] != "load":
                    result.append((engine, slot))
                    i += 1
                    continue

                _, dest0, load_addr0 = load_s
                # The load addr must match the alu output
                if load_addr0 != addr0:
                    result.append((engine, slot))
                    i += 1
                    continue

                # Check VLEN-1 more loads
                for j in range(1, VLEN):
                    idx = load_start + j
                    if idx >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[idx]
                    if next_engine != "load" or next_slot[0] != "load":
                        can_fuse = False
                        break
                    _, next_dest, next_load_addr = next_slot
                    if next_dest != dest0 + j or next_load_addr != addr0 + j:
                        can_fuse = False
                        break

                if can_fuse:
                    # We can fuse!
                    # But vload needs base address that points to mem[base + offset0]
                    # If offset0 != 0, we need to compute base + offset0 first
                    if offset0 == 0:
                        # Perfect - base0 already points to the right memory location
                        result.append(("load", ("vload", dest0, base0)))
                    else:
                        # Need to add offset0 to base first
                        # Keep the first alu to compute addr0 = base0 + offset0
                        result.append(("alu", ("+", addr0, base0, const_addr0)))
                        result.append(("load", ("vload", dest0, addr0)))
                    i = load_start + VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def simple_vload_fusion_pass(self, slots):
        """
        Convert VLEN consecutive scalar loads to vload, but ONLY when the
        addresses were computed by a valu that produces contiguous memory addresses.

        Pattern:
          valu: ('+', addr, base_vec, offset_vec)  where offset_vec contains 0,1,2,...,VLEN-1
          load: ('load', dest0, addr)
          load: ('load', dest1, addr+1)
          ...

        vload(dest, addr) reads: mem[scratch[addr] + i] for i in 0..VLEN-1
        This matches the loads only if scratch[addr+i] = scratch[addr] + i,
        which is guaranteed when valu computed addr+i = base + i.
        """
        result = []
        i = 0
        n = len(slots)

        # Build reverse const map: scratch addr -> const value
        addr_to_const = {addr: val for val, addr in self.const_map.items()}

        # Track valu '+' outputs where offset is contiguous constants (0,1,2,...)
        safe_vload_addrs = set()

        while i < n:
            engine, slot = slots[i]

            # Track valu '+' that produce contiguous address patterns
            if engine == "valu" and slot[0] == "+":
                _, dest, base_vec, offset_vec = slot
                # Check if offset_vec..offset_vec+VLEN-1 contain exactly 0, 1, ..., VLEN-1
                is_contiguous_offset = True
                for k in range(VLEN):
                    addr = offset_vec + k
                    if addr not in addr_to_const or addr_to_const[addr] != k:
                        is_contiguous_offset = False
                        break
                if is_contiguous_offset:
                    safe_vload_addrs.add(dest)

            # Try to fuse VLEN consecutive loads
            if engine == "load" and slot[0] == "load" and i + VLEN <= n:
                _, dest0, addr0 = slot

                # Only convert if addr0 was produced by a safe valu pattern
                if addr0 in safe_vload_addrs:
                    can_fuse = True

                    # Check contiguous dest and addr
                    for j in range(1, VLEN):
                        if i + j >= n:
                            can_fuse = False
                            break
                        next_engine, next_slot = slots[i + j]
                        if next_engine != "load" or next_slot[0] != "load":
                            can_fuse = False
                            break
                        _, next_dest, next_addr = next_slot
                        if next_dest != dest0 + j or next_addr != addr0 + j:
                            can_fuse = False
                            break

                    if can_fuse:
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

        This pass looks for patterns like:
          alu: ('+', addr0, base, const0)  where scratch[const0] = 0
          alu: ('+', addr1, base, const1)  where scratch[const1] = 1
          ...
          store: ('store', addr0, src0)
          store: ('store', addr1, src1)
          ...

        And converts them to:
          vstore: ('vstore', base, src0)
        """
        result = []
        i = 0
        n = len(slots)

        # Build reverse const map: scratch addr -> const value
        addr_to_const = {addr: val for val, addr in self.const_map.items()}

        while i < n:
            engine, slot = slots[i]

            # Try to fuse VLEN consecutive stores that follow VLEN address calculations
            if engine == "alu" and slot[0] == "+" and i + 2 * VLEN <= n:
                # Check if this is the start of a fusable pattern:
                # VLEN alu additions followed by VLEN stores
                _, addr0, base0, const_addr0 = slot

                # The const address must contain a small integer offset (0, 1, 2, ...)
                if const_addr0 not in addr_to_const:
                    result.append((engine, slot))
                    i += 1
                    continue

                offset0 = addr_to_const[const_addr0]

                can_fuse = True

                # Check VLEN-1 more ALU ops with pattern: addr = base + const_j where const_j = offset0 + j
                for j in range(1, VLEN):
                    if i + j >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[i + j]
                    if next_engine != "alu" or next_slot[0] != "+":
                        can_fuse = False
                        break
                    _, next_addr, next_base, next_const_addr = next_slot
                    # Must have same base and contiguous addr output
                    if next_base != base0 or next_addr != addr0 + j:
                        can_fuse = False
                        break
                    # Must have const that is offset0 + j
                    if next_const_addr not in addr_to_const:
                        can_fuse = False
                        break
                    if addr_to_const[next_const_addr] != offset0 + j:
                        can_fuse = False
                        break

                if not can_fuse:
                    result.append((engine, slot))
                    i += 1
                    continue

                # Now check VLEN stores following the ALU ops
                store_start = i + VLEN
                if store_start + VLEN > n:
                    result.append((engine, slot))
                    i += 1
                    continue

                store_e, store_s = slots[store_start]
                if store_e != "store" or store_s[0] != "store":
                    result.append((engine, slot))
                    i += 1
                    continue

                _, store_addr0, src0 = store_s
                # The store addr must match the alu output
                if store_addr0 != addr0:
                    result.append((engine, slot))
                    i += 1
                    continue

                # Check VLEN-1 more stores
                for j in range(1, VLEN):
                    idx = store_start + j
                    if idx >= n:
                        can_fuse = False
                        break
                    next_engine, next_slot = slots[idx]
                    if next_engine != "store" or next_slot[0] != "store":
                        can_fuse = False
                        break
                    _, next_store_addr, next_src = next_slot
                    if next_store_addr != addr0 + j or next_src != src0 + j:
                        can_fuse = False
                        break

                if can_fuse:
                    # We can fuse!
                    # vstore needs base address that points to mem[base + offset0]
                    if offset0 == 0:
                        # Perfect - base0 already points to the right memory location
                        result.append(("store", ("vstore", base0, src0)))
                    else:
                        # Need to add offset0 to base first
                        result.append(("alu", ("+", addr0, base0, const_addr0)))
                        result.append(("store", ("vstore", addr0, src0)))
                    i = store_start + VLEN
                    continue

            result.append((engine, slot))
            i += 1

        return result

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """VLIW scheduling based on dependency analysis with critical path priority"""
        # Strength reduction disabled - all ALU ops are same cost (1 cycle)
        # slots = self.strength_reduction_pass(slots)

        # Eliminate select operations where possible (convert to ALU)
        slots = self.select_elimination_pass(slots)

        # Apply iteration interleaving with register allocation for vectorization
        # This reorders iterations so that VLEN ops are grouped together
        slots = self.iteration_interleaving_pass(slots)

        # Apply optimization passes repeatedly until no change
        prev_len = -1
        iteration = 0
        while len(slots) != prev_len:
            prev_len = len(slots)
            iteration += 1

            # Apply vload/vstore/vselect fusion
            slots = self.vload_fusion_pass(slots)
            slots = self.vstore_fusion_pass(slots)
            slots = self.vselect_fusion_pass(slots)

            # Apply vectorization pass
            slots = self.vectorize_pass(slots)

            # CSE pass
            slots = self.cse_pass(slots)

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
        # Pipeline init_vars loads using tmp1/tmp2 alternately for parallelism
        # Cycle 0: const tmp1, 0
        # Cycle 1: load v0, tmp1 | const tmp2, 1
        # Cycle 2: load v1, tmp2 | const tmp1, 2
        # ... (8 cycles instead of 14)
        n_vars = len(init_vars)
        self.add("load", ("const", tmp1, 0))  # First const alone
        for i in range(n_vars):
            curr_tmp = tmp1 if i % 2 == 0 else tmp2
            next_tmp = tmp2 if i % 2 == 0 else tmp1
            if i + 1 < n_vars:
                # Pack: load current var | const for next
                bundle = {"load": [
                    ("load", self.scratch[init_vars[i]], curr_tmp),
                    ("const", next_tmp, i + 1)
                ]}
                self.instrs.append(bundle)
            else:
                # Last load alone
                self.add("load", ("load", self.scratch[init_vars[i]], curr_tmp))

        # Pre-allocate i_const values in CONTIGUOUS scratch locations for vload fusion
        # This allows vload to load batch items with a single instruction
        # IMPORTANT: Allocate these BEFORE zero_const/one_const/two_const so they're truly contiguous
        i_const_base = self.alloc_scratch("i_const_base", batch_size)
        # Pack const loads - 2 per cycle (SLOT_LIMITS["load"] = 2)
        for i in range(0, batch_size, 2):
            bundle = {"load": [("const", i_const_base + i, i)]}
            if i + 1 < batch_size:
                bundle["load"].append(("const", i_const_base + i + 1, i + 1))
            self.instrs.append(bundle)
        # Register all in const_map
        for i in range(batch_size):
            self.const_map[i] = i_const_base + i

        # Use the contiguous allocation for 0, 1, 2 as well
        zero_const = i_const_base + 0  # = self.const_map[0]
        one_const = i_const_base + 1   # = self.const_map[1]
        two_const = i_const_base + 2   # = self.const_map[2]

        # Pre-load hash constants (packed) - avoids individual loads during body construction
        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            if val1 not in self.const_map:
                hash_consts.append(val1)
            if val3 not in self.const_map:
                hash_consts.append(val3)
        # Load hash constants in pairs
        for i in range(0, len(hash_consts), 2):
            addr1 = self.alloc_scratch()
            self.const_map[hash_consts[i]] = addr1
            if i + 1 < len(hash_consts):
                addr2 = self.alloc_scratch()
                self.const_map[hash_consts[i + 1]] = addr2
                self.instrs.append({"load": [("const", addr1, hash_consts[i]), ("const", addr2, hash_consts[i + 1])]})
            else:
                self.add("load", ("const", addr1, hash_consts[i]))

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
                i_const = i_const_base + i  # Use contiguous scratch address
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

    def test_dump_instructions(self):
        """Dump instructions to file for analysis"""
        from problem import Tree, Input, build_mem_image
        import random
        random.seed(123)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)

        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

        with open("instructions_dump.txt", "w") as f:
            f.write(f"Total bundles: {len(kb.instrs)}\n\n")
            for i, bundle in enumerate(kb.instrs[:1000]):
                f.write(f"=== Cycle {i} ===\n")
                for engine, slots in bundle.items():
                    for slot in slots:
                        f.write(f"  {engine}: {slot}\n")
                f.write("\n")
        print("Dumped to instructions_dump.txt")


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
