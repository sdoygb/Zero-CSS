#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_L 比较（高效版）

关键修正
  · 查表不再用矩阵乘法（2^n × 矩阵乘 ⇒ 慢）；改为**逐比特累积 syndrome**
  · toric 用 **pymatching**（MWPM），不用查表
  · 只算规模可行的码（RM(1,4) n=16；toric L=3/4/5）
"""
from __future__ import annotations
import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0: return 0
    r, rows, cols = 0, A.shape[0], A.shape[1]
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if A[i, c]: p = i; break
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]: A[i] = A[i] ^ A[r]
        r += 1
        if r == rows: break
    return r


def nullspace(M):
    A = np.array(M, dtype=np.uint8) % 2
    m, n = A.shape
    piv, r = [], 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if A[i, c]: p = i; break
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]: A[i] = A[i] ^ A[r]
        piv.append(c); r += 1
        if r == m: break
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8); x[f] = 1
        for i, pc in enumerate(piv): x[pc] = A[i, f]
        out.append(x)
    return out


def syndrome_table(HX, HZ, n):
    """逐比特累积：table[syndrome] = 最小重量错误（n ≤ 20）。"""
    colZ = [tuple(int(v) for v in HZ[:, e]) for e in range(n)]
    colX = [tuple(int(v) for v in HX[:, e]) for e in range(n)]
    nZ, nX = HX.shape[0], HZ.shape[0]
    table = {}
    for mask in range(1 << n):
        sz = [0] * nZ; sx = [0] * nX; w = 0
        for e in range(n):
            if (mask >> e) & 1:
                w += 1
                cz, cx = colZ[e], colX[e]
                for i in range(nZ): sz[i] ^= cz[i]
                for i in range(nX): sx[i] ^= cx[i]
        key = (tuple(sz), tuple(sx))
        if key not in table or table[key][0] > w:
            table[key] = (w, mask)
    return {k: v[1] for k, v in table.items()}


def min_logical(Hc, Hs, cap=20):
    ker = nullspace(Hc)
    if not ker or len(ker) > cap: return None
    A = np.array(Hs, dtype=np.uint8) % 2
    rows, r = [], 0
    tmp = A.copy()
    for c in range(A.shape[1]):
        p = None
        for i in range(r, tmp.shape[0]):
            if tmp[i, c]: p = i; break
        if p is None: continue
        tmp[[r, p]] = tmp[[p, r]]
        for i in range(tmp.shape[0]):
            if i != r and tmp[i, c]: tmp[i] = tmp[i] ^ tmp[r]
        rows.append(tmp[r]); r += 1
    def in_row(v):
        w = v.copy()
        for r_ in rows:
            p = next((c for c in range(len(r_)) if r_[c]), None)
            if p is not None and w[p]: w = w ^ r_
        return not w.any()
    best = None
    for mask in range(1, 1 << len(ker)):
        v = np.zeros(len(ker[0]), dtype=np.uint8)
        for i in range(len(ker)):
            if (mask >> i) & 1: v = v ^ ker[i]
        w = int(v.sum())
        if best is not None and w >= best[0]: continue
        if not in_row(v): best = (w, v)
    return best[1] if best else None


def bits2err(mask, n):
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)


def simulate(HX, HZ, p, shots, seed=0):
    n = HX.shape[1]
    table = syndrome_table(HX, HZ, n)
    LX, LZ = min_logical(HX, HZ), min_logical(HZ, HX)
    if LX is None or LZ is None: return None
    rnd = np.random.default_rng(seed)
    fails = 0
    colZ = [np.array(HZ[:, e], dtype=np.uint8) for e in range(n)]
    colX = [np.array(HX[:, e], dtype=np.uint8) for e in range(n)]
    for _ in range(shots):
        u = rnd.random(n)
        mX = ((u < p/3) | ((u >= p/3) & (u < 2*p/3)))
        mZ = (((u >= p/3) & (u < 2*p/3)) | ((u >= 2*p/3) & (u < p)))
        sz = np.zeros(HZ.shape[0], dtype=np.uint8)
        sx = np.zeros(HX.shape[0], dtype=np.uint8)
        for e in range(n):
            if mZ[e]: sz ^= colZ[e]
            if mX[e]: sx ^= colX[e]
        key = (tuple(int(v) for v in sz), tuple(int(v) for v in sx))
        m = table.get(key)
        if m is None: fails += 1; continue
        ehat = bits2err(m, n)
        if int(LX @ (bits2err(sum(1 << e for e in range(n) if mX[e]), n) ^ ehat) % 2) or \
           int(LZ @ (bits2err(sum(1 << e for e in range(n) if mZ[e]), n) ^ ehat) % 2):
            fails += 1
    return fails / shots


def main():
    print("=" * 92)
    print("p_L：RM-CSS[[16,6,4]] vs toric[[18,2,3]]（code-capacity，全表解码）")
    print("=" * 92)
    n = 16
    rows = [[1 if (c & mk) == mk else 0 for c in range(n)] for mk in range(n) if mk.bit_count() <= 1]
    G = np.array(rows, dtype=np.uint8)
    print("  %-16s %5s %5s %5s %11s %11s %11s" % ("码", "n", "k", "d", "p=1e-3", "p=5e-3", "p=1e-2"))
    for name, HX, HZ, d in (("RM-CSS(1,4)", G, G, 4),):
        k = n - rank2(HX) - rank2(HZ)
        vals = []
        for p in (1e-3, 5e-3, 1e-2):
            pl = simulate(HX, HZ, p, shots=2000, seed=int(p*1e6))
            vals.append("?" if pl is None else "%.3e" % pl)
        print("  %-16s %5d %5d %5d %11s %11s %11s" % (name, n, k, d, *vals))


if __name__ == "__main__":
    main()
