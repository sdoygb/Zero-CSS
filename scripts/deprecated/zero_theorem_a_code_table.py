#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 定理 A 码：参数表（纯 Zero，辛向量实现，可扩规模）

定理 A（已核验，见 zero_theorem_a_symplectic.py）
  复形 (V,E,F) 上，顶点稳定子 A_v=∏_{e∋v}X_e 与面稳定子 B_f=∏_{e∈∂f}Z_e 全对易
  ⟺ |star(v) ∩ ∂f| 恒为偶（对本构造：恒真，因为交集 ∈ {0,2}）。

  Zero 依据：顶点=(C,E)（Z0）；面=**闭合词**（零和 ⇒ 闭合边序列）。

码参数（本文件计算）
  n = 2|E|（每边两个量子比特）
  稳定子数 = |V| + |F|；rank(A 组) = V−1（星形行和为 0，且一般秩 = V−1）
                     rank(B 组) = |F|（面边界线性无关，在 2-流形样复形上）
  k = n − (V−1) − F
  d = min(最小 X 逻辑重量, 最小 Z 逻辑重量)
      X 逻辑：顶点星形的"对偶闭合"（割）；Z 逻辑：闭合词（面边界的组合 = 非平凡圈）

用法: python3 zero_theorem_a_code_table.py
"""
from __future__ import annotations

import itertools

import numpy as np


def grid(L, periodic=False):
    def vid(i, j):
        return (i % L) * L + (j % L) if periodic else i * (L + 1) + j
    edges, seen = [], {}

    def add(a, b):
        if a == b:
            return None
        k = (min(a, b), max(a, b))
        if k not in seen:
            seen[k] = len(edges)
            edges.append(k)
        return seen[k]

    ni = nj = L if periodic else L + 1
    for i in range(ni):
        for j in range(L):
            add(vid(i, j), vid(i, j + 1))
    for i in range(L):
        for j in range(nj):
            add(vid(i, j), vid(i + 1, j))
    faces = []
    for i in range(L):
        for j in range(L):
            f = [add(vid(i, j), vid(i, j + 1)), add(vid(i, j + 1), vid(i + 1, j + 1)),
                 add(vid(i + 1, j + 1), vid(i + 1, j)), add(vid(i + 1, j), vid(i, j))]
            if all(x is not None for x in f):
                faces.append(sorted(set(f)))
    nv = L * L if periodic else (L + 1) ** 2
    return nv, edges, faces


def gf2_rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def kernel(M):
    m, n = M.shape
    A = M.copy() % 2
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, pc in enumerate(piv):
            x[pc] = A[i, f]
        out.append(x)
    return out


def min_w(basis, cap=18):
    k = len(basis)
    if k == 0:
        return None
    if k > cap:
        return "—"
    best = 10 ** 9
    for mask in range(1, 1 << k):
        x = np.zeros(len(basis[0]), dtype=np.uint8)
        for i in range(k):
            if mask >> i & 1:
                x ^= basis[i]
        w = int(x.sum())
        if 0 < w < best:
            best = w
    return best


def main() -> None:
    print("=" * 82)
    print("Zero 定理 A 码：参数表（n=2E；k=n−(V−1)−F；d=min(割, 圈)）")
    print("=" * 82)
    print(f"    {'复形(格数)':16s} {'V':>4s} {'E':>4s} {'F':>4s} {'n':>5s} "
          f"{'rankX':>6s} {'rankZ':>6s} {'k':>5s} {'d_Z':>5s} {'d_X':>5s} {'d':>4s} {'k/n':>6s}")
    for name, (L, per) in [("方格 %dx%d" % (L, L), (L, False)) for L in (2, 3, 4, 5)] + \
                          [("环面 %dx%d" % (L, L), (L, True)) for L in (2, 3, 4, 5)]:
        nv, edges, faces = grid(L, per)
        ne, nf = len(edges), len(faces)
        n = 2 * ne
        # X 组：顶点星形（n 列，X 部分）
        HX = np.zeros((nv, n), dtype=np.uint8)
        for v in range(nv):
            for e, (a, b) in enumerate(edges):
                if v in (a, b):
                    HX[v, e] = 1
        # Z 组：面边界（Z 部分，列偏移 ne）
        HZ = np.zeros((nf, n), dtype=np.uint8)
        for fi, f in enumerate(faces):
            for e in f:
                HZ[fi, ne + e] = 1
        rX, rZ = gf2_rank(HX), gf2_rank(HZ)
        k = n - rX - rZ
        # Z 逻辑 = ker(HX) 去掉 stabilizer 部分 → 用 ker(HX) 的最小重量（X 型的逻辑）
        dX = min_w(kernel(HX))
        dZ = min_w(kernel(HZ))
        dd = min([x for x in (dX, dZ) if isinstance(x, int)], default="—")
        print(f"    {name:16s} {nv:4d} {ne:4d} {nf:4d} {n:5d} "
              f"{rX:6d} {rZ:6d} {k:5d} {str(dZ):>5s} {str(dX):>5s} {str(dd):>4s} {k/n:6.3f}")

    print("\n结论（纯 Zero）")
    print("  · 顶点（来自 Z0 的 (C,E)）＋ 面（来自 Z0③ 的闭合词/零和）⇒ 合法量子码（定理 A）；")
    print("  · 参数由复形欧拉数决定：k = 2E − (V−1) − F；")
    print("  · 平面方格：k 随格数增长（开边界 ⇒ 边界自由度为逻辑自由度）；")
    print("  · 环面：k = 2E − (V−1) − F 为拓扑量（与 L 无关的部分 = 2）；")
    print("  · 注意：'平面 k=1'是**标准表面码**的结论（其 X/Z 边界条件不同），")
    print("          **不能**搬到本构造——本构造是均匀权重星形/面，参数如实表中所示。")


if __name__ == "__main__":
    main()
