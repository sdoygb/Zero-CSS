#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#1 的两个关键引理：面边界 ∈ 圈空间；自环不影响零和码

引理 4（面边界是圈）
  Zero 的面 = 闭合词（Z0③ 零和 ⇒ 闭合边序列）。
  ⇒ 其边集 ∂f 满足：每个顶点在 ∂f 中度数为偶（引理 3 第二步）。
  ⇒ 等价地：∂f ∈ ker(δ) = 圈空间（**每一条闭合词的边集都是圈，或圈的并**）。

引理 5（自环）
  若允许自环 e=(v,v)：Z0 的词计数 w_e 对 v 贡献 2w_e（偶），故**零和不受自环影响**；
  但在关联矩阵中，自环给对角元 +2 = 0 (mod 2) ⇒ 不出现在 KX 中。
  ⇒ 自环不改变 r_X = V−1（若不连通则改）；码参数量 n 增加 1 而非 2。

核验
  S1  对每个面 f，检验 ∂f 在圈空间里（顶点度数为偶）。
  S2  r_X = V−1 的成立条件：连通 vs 不连通；自环的影响。
"""
from __future__ import annotations

from collections import Counter

import numpy as np


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


def incidence(nv, edges, include_selfloops=False):
    """关联矩阵：自环（a==b）默认跳过（其 GF(2) 贡献为 0）。"""
    rows = []
    for v in range(nv):
        row = []
        for (a, b) in edges:
            if a == b:
                row.append(0)          # 自环：2 = 0 (mod 2)
            else:
                row.append(1 if v in (a, b) else 0)
        rows.append(row)
    return np.array(rows, dtype=np.uint8)


def main() -> None:
    print("=" * 84)
    print("#1 关键引理：面边界 ∈ 圈空间；自环无关")
    print("=" * 84)

    print("\n[S1] 每个面的边界是否在圈空间（顶点度数全偶）")
    for name, (L, per) in {"方格 3x3": (3, False), "方格 5x5": (5, False),
                           "环面 3x3": (3, True), "环面 5x5": (5, True)}.items():
        nv, edges, faces = grid(L, per)
        bad = []
        for fi, f in enumerate(faces):
            deg = Counter()
            for e in f:
                a, b = edges[e]
                deg[a] += 1
                deg[b] += 1
            if any(d % 2 for d in deg.values()):
                bad.append(fi)
        print(f"    {name:12s} 面数={len(faces):3d}  非圈的面 = {len(bad)} "
              f"{'✅ 全部是圈' if not bad else '❌ ' + str(bad[:3])}")

    print("\n[S2] r_X = V−1 的成立条件")
    tests = {
        "连通（方格 3x3）": (lambda: grid(3, False)),
        "连通（环面 3x3）": (lambda: grid(3, True)),
        "不连通（两个分离三角）": (lambda: (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], [])),
        "带自环（三角 + 自环）": (lambda: (3, [(0, 1), (1, 2), (2, 0), (0, 0)], [])),
    }
    for name, mk in tests.items():
        nv, edges, faces = mk()
        KX = incidence(nv, edges)
        rX = gf2_rank(KX)
        comps = count_components(nv, edges)
        print(f"    {name:24s} V={nv} E={len(edges)}  r_X={rX}  "
              f"V−1={nv-1}  r_X==V−1? {rX == nv-1}   连通分量数={comps}")

    print("\n[S2'] 不连通时的正确公式：r_X = V − (连通分量数)")
    for name, mk in tests.items():
        nv, edges, faces = mk()
        rX = gf2_rank(incidence(nv, edges))
        comps = count_components(nv, edges)
        print(f"    {name:24s} r_X={rX}  V−c={nv-comps}  相等? {rX == nv-comps}")


def count_components(nv, edges):
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in edges:
        if a == b:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(v) for v in range(nv)})


if __name__ == "__main__":
    main()
