#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_L：RM-CSS vs toric（快速版；MWD = ML for iid）

关键事实（解释了为什么不用 ML 表）
  对独立同分布错误，每个 syndrome 类内 P(e) 只依赖重量 w(e)
  ⇒ 类内最大概率 = 最小重量 ⇒ **MWD ≡ ML**（无需 O(4^n) 的 ML 表）
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


def in_rowspace(Hs, v):
    A = np.array(Hs, dtype=np.uint8) % 2
    m, n = A.shape
    rows, r = [], 0
    tmp = A.copy()
    for c in range(n):
        p = None
        for i in range(r, tmp.shape[0]):
            if tmp[i, c]: p = i; break
        if p is None: continue
        tmp[[r, p]] = tmp[[p, r]]
        for i in range(tmp.shape[0]):
            if i != r and tmp[i, c]: tmp[i] = tmp[i] ^ tmp[r]
        rows.append(tmp[r]); r += 1
    w = v.copy()
    for r_ in rows:
        p = next((cc for cc in range(len(r_)) if r_[cc]), None)
        if p is not None and w[p]: w = w ^ r_
    return not w.any()


def mwd_table(HZ, HX, n):
    """按 syndrome 取最小重量代表（逐比特累积，n ≤ 20）。"""
    colZ = [tuple(int(v) for v in HZ[:, e]) for e in range(n)]
    colX = [tuple(int(v) for v in HX[:, e]) for e in range(n)]
    nZ, nX = HZ.shape[0], HX.shape[0]
    table = {}
    # 按重量递增枚举（首次命中即最小）
    for w in range(n + 1):
        for mask in range(1 << n):
            if bin(mask).count("1") != w: continue
            sz = [0]*nZ; sx = [0]*nX
            for e in range(n):
                if (mask >> e) & 1:
                    cz, cx = colZ[e], colX[e]
                    for i in range(nZ): sz[i] ^= cz[i]
                    for i in range(nX): sx[i] ^= cx[i]
            key = (tuple(sz), tuple(sx))
            if key not in table:
                table[key] = mask
    return table


def min_logical_weight(Hc, Hs, cap=18):
    ker = nullspace(Hc)
    if not ker or len(ker) > cap:
        return None, None
    best, bv = None, None
    for mask in range(1, 1 << len(ker)):
        v = np.zeros(len(ker[0]), dtype=np.uint8)
        for i in range(len(ker)):
            if (mask >> i) & 1: v = v ^ ker[i]
        w = int(v.sum())
        if best is not None and w >= best: continue
        if not in_rowspace(Hs, v): best, bv = w, v
    return best, bv


def sim(HX, HZ, LX, LZ, p, shots, seed=0):
    n = HX.shape[1]
    tbl = mwd_table(HZ, HX, n)
    rnd = np.random.default_rng(seed)
    colZ = [np.array(HZ[:, e], dtype=np.uint8) for e in range(n)]
    colX = [np.array(HX[:, e], dtype=np.uint8) for e in range(n)]
    fails = 0
    for _ in range(shots):
        u = rnd.random(n)
        mX = ((u < p/3) | ((u >= p/3) & (u < 2*p/3)))
        mZ = (((u >= p/3) & (u < 2*p/3)) | ((u >= 2*p/3) & (u < p)))
        sz = np.zeros(HZ.shape[0], dtype=np.uint8)
        sx = np.zeros(HX.shape[0], dtype=np.uint8)
        eX = np.zeros(n, dtype=np.uint8); eZ = np.zeros(n, dtype=np.uint8)
        for e in range(n):
            if mZ[e]: sz ^= colZ[e]; eZ[e] = 1
            if mX[e]: sx ^= colX[e]; eX[e] = 1
        key = (tuple(int(v) for v in sz), tuple(int(v) for v in sx))
        m = tbl.get(key)
        if m is None: fails += 1; continue
        ch = np.array([(m >> i) & 1 for i in range(n)], dtype=np.uint8)
        if int(LX @ (eX ^ ch) % 2) or int(LZ @ (eZ ^ ch) % 2):
            fails += 1
    return fails / shots


def toric(L):
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
    HX = np.zeros((V, ne), dtype=np.uint8)
    for v in range(V):
        for e, (a, b) in enumerate(edges):
            if v in (a, b): HX[v, e] = 1
    faces = []
    for i in range(L):
        for j in range(L):
            faces.append(sorted({add(idx(i,j), idx(i,j+1)), add(idx(i,j+1), idx(i+1,j+1)),
                                 add(idx(i+1,j+1), idx(i+1,j)), add(idx(i+1,j), idx(i,j))}))
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f: HZ[fi, e] = 1
    return HX, HZ


def main():
    print("=" * 92)
    print("p_L：RM-CSS[[16,6,4]] vs toric[[18,2,3]]（MWD = ML for iid）")
    print("=" * 92)
    n = 16
    rows = [[1 if (c & mk) == mk else 0 for c in range(n)] for mk in range(n) if mk.bit_count() <= 1]
    G = np.array(rows, dtype=np.uint8)
    dRM, LXrm = min_logical_weight(G, G)
    print("  RM-CSS: n=16 k=%d d=%d  生成元数=%d  最小逻辑个数多" % (n - 2*rank2(G), dRM, G.shape[0]))
    HXt, HZt = toric(3)
    dT, LXt = min_logical_weight(HZt, HXt)
    print("  toric : n=%d k=%d d=%d" % (HXt.shape[1], HXt.shape[1]-rank2(HXt)-rank2(HZt), dT))
    print()
    print("  %-14s %7s %5s %5s %11s %11s %11s" % ("码", "n", "k", "d", "p=1e-3", "p=5e-3", "p=1e-2"))
    for name, HX, HZ, LX, LZ, d in (("RM-CSS(1,4)", G, G, LXrm, LXrm, dRM),
                                     ("toric L=3", HXt, HZt, LXt, LXt, dT)):
        nq = HX.shape[1]
        k = nq - rank2(HX) - rank2(HZ)
        vals = []
        for p in (1e-3, 5e-3, 1e-2):
            pl = sim(HX, HZ, LX, LZ, p, shots=1500, seed=int(p*1e6))
            vals.append("%.3e" % pl)
        print("  %-14s %7d %5d %5d %11s %11s %11s" % (name, nq, k, d, *vals))


if __name__ == "__main__":
    main()
