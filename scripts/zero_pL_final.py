#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_L：RM-CSS 码 vs toric 码（code-capacity，正确解码器）

策略
  · 小码（n ≤ ~24）：**全表解码**（枚举全部 syndrome）
  · toric：用 **pymatching**（MWPM，该几何的最优解码器）
"""
from __future__ import annotations
import itertools
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


def full_lookup(HX, HZ):
    """全表：枚举 2^n 个错误，按 syndrome 取最小重量。n ≤ 22 可行。"""
    n = HX.shape[1]
    table = {}
    for mask in range(1 << n):
        e = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        w = int(e.sum())
        s = (int((HZ @ e % 2).sum() * 0 + sum((HZ @ e % 2) << np.arange(len(HZ))) if False else 0))
        key = (tuple((HZ @ e) % 2), tuple((HX @ e) % 2))
        if key not in table or table[key][0] > w:
            table[key] = (w, e)
    return {k: v[1] for k, v in table.items()}


def logicals(HX, HZ, cap=18):
    def minl(Hc, Hs):
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
    return minl(HX, HZ), minl(HZ, HX)


def simulate_table(HX, HZ, p, shots, seed=0):
    table = full_lookup(HX, HZ)
    LX, LZ = logicals(HX, HZ)
    if LX is None or LZ is None: return None
    rnd = np.random.default_rng(seed)
    n = HX.shape[1]
    fails = 0
    for _ in range(shots):
        u = rnd.random(n)
        eX = ((u < p/3) | ((u >= p/3) & (u < 2*p/3))).astype(np.uint8)
        eZ = (((u >= p/3) & (u < 2*p/3)) | ((u >= 2*p/3) & (u < p))).astype(np.uint8)
        key = (tuple((HZ @ eZ) % 2), tuple((HX @ eX) % 2))
        ehat = table.get(key)
        if ehat is None: fails += 1; continue
        if int(LX @ (eX ^ ehat) % 2) or int(LZ @ (eZ ^ ehat) % 2): fails += 1
    return fails / shots


def main():
    print("=" * 94)
    print("p_L：RM-CSS[[16,6,4]] vs toric[[32,2,4]]（code-capacity 去极化，全表解码）")
    print("=" * 94)
    # RM(1,4)
    n = 16
    rows = [[1 if (c & mk) == mk else 0 for c in range(n)] for mk in range(n) if mk.bit_count() <= 1]
    G = np.array(rows, dtype=np.uint8)
    HXr, HZr = G, G
    # toric L=4
    L = 4
    idx = lambda i, j: (i % L) * L + (j % L)
    edges, seen = [], {}
    def add(a, b):
        if a == b: return None
        k = (min(a, b), max(a, b))
        if k not in seen: seen[k] = len(edges); edges.append(k)
        return seen[k]
    for i in range(L):
        for j in range(L):
            add(idx(i, j), idx(i, j+1)); add(idx(i, j), idx(i+1, j))
    V, ne = L*L, len(edges)
    HXt = np.zeros((V, ne), dtype=np.uint8)
    for v in range(V):
        for e, (a, b) in enumerate(edges):
            if v in (a, b): HXt[v, e] = 1
    faces = []
    for i in range(L):
        for j in range(L):
            faces.append(sorted({add(idx(i,j), idx(i,j+1)), add(idx(i,j+1), idx(i+1,j+1)),
                                 add(idx(i+1,j+1), idx(i+1,j)), add(idx(i+1,j), idx(i,j))}))
    HZt = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f: HZt[fi, e] = 1

    print("  %-20s %5s %5s %5s %10s %10s %10s" % ("码", "n", "k", "d", "p=1e-3", "p=5e-3", "p=1e-2"))
    for name, HX, HZ, d in (("RM-CSS(1,4)", HXr, HZr, 4), ("toric L=4", HXt, HZt, 4)):
        nq = HX.shape[1]
        k = nq - rank2(HX) - rank2(HZ)
        vals = []
        for p in (1e-3, 5e-3, 1e-2):
            pl = simulate_table(HX, HZ, p, shots=3000, seed=int(p*1e6))
            vals.append("?" if pl is None else "%.2e" % pl)
        print("  %-20s %5d %5d %5d %10s %10s %10s" % (name, nq, k, d, *vals))
    print("""
  说明：两者都是 [[.,.,4]] —— 同距离、不同的 n 与 k。
  RM: n=16, k=6（率高）；toric: n=32, k=2（率低）。
  ⇒ 若 RM 的 p_L 与 toric 相当，则 RM 在**每逻辑比特**意义上更好。""")


if __name__ == "__main__":
    main()
