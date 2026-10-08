#!/usr/bin/env python3
"""gen_consts.py - builds src/noid_consts.h from the output of
`cargo run --release -- consts` (tools/noidvec).

    python3 tools/gen_consts.py consts.txt src/noid_consts.h
"""
import sys

lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sec, cur = {}, None
for l in lines:
    if l.startswith('IV '):
        sec['IV'] = l.split()[1:]
        continue
    if not l.startswith('0x'):
        cur = l
        sec[cur] = []
        continue
    sec[cur].append(l)


def u(v):
    v = int(v, 16)
    return "{0x%016xULL,0x%016xULL}" % (v & ((1 << 64) - 1), v >> 64)


o = []
o.append("// noid_consts.h - Poseidon2b PoW constants for ParanO(1)d (NOID), generated from\n"
         "// the official parano1d v2.0.2 Rust crates (noid_core / noid_poseidon2b, Apache-2.0).\n"
         "// All values are in the FLAT (GCM polynomial) basis unless noted. {lo, hi} 64-bit halves.\n"
         "#pragma once\n#include <stdint.h>\n")
o.append("typedef struct { uint64_t lo, hi; } noid_u128;\n")
o.append("#define NOID_N_ROUNDS 66\n#define NOID_F_ROUNDS 8\n#define NOID_P_ROUNDS 58\n")
o.append("// TOWER->FLAT: column i = tower_to_flat(1 << i)\nstatic const noid_u128 NOID_T2F[128] = {\n"
         + ",\n".join("  " + u(x) for x in sec['T2F']) + "\n};\n")
o.append("// FLAT->TOWER: column i = flat_to_tower(1 << i)\nstatic const noid_u128 NOID_F2T[128] = {\n"
         + ",\n".join("  " + u(x) for x in sec['F2T']) + "\n};\n")
o.append("// capacity IV of the POWHDR__ domain (state[2], state[3])\nstatic const noid_u128 NOID_IV[2] = { %s, %s };\n"
         % (u(sec['IV'][0]), u(sec['IV'][1])))
rc = sec['RC']
o.append("// round constants RC[lane][round]\nstatic const noid_u128 NOID_RC[4][66] = {\n"
         + ",\n".join("  {" + ", ".join(u(x) for x in rc[i * 66:(i + 1) * 66]) + "}" for i in range(4)) + "\n};\n")
o.append("static const noid_u128 NOID_MDS_FULL[4][4] = {\n"
         + ",\n".join("  {" + ", ".join(u(x) for x in sec['MDSF'][i * 4:(i + 1) * 4]) + "}" for i in range(4)) + "\n};\n")
o.append("static const noid_u128 NOID_MDS_PARTIAL[4][4] = {\n"
         + ",\n".join("  {" + ", ".join(u(x) for x in sec['MDSP'][i * 4:(i + 1) * 4]) + "}" for i in range(4)) + "\n};\n")
open(sys.argv[2], 'w').write("\n".join(o))
