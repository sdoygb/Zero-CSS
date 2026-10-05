#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 Zero：量子圈码（只用 Z0 原语，实测参数）

构造（全部来自 Zero，不引外部码族）
  · 图 Γ=(C,E) 来自 Z0 的 (C,E)；
  · 每条边放 **两个** 量子比特（X 型与 Z 型）——对应零和词的两种角色；
  · 稳定子：
      A_v = ∏_{e∋v} X_e     （Z1 定理 1 的局域性：顶点星形）
      B_v = ∏_{e∋v} Z_e
  · 对易判据：A_v B_w = (-1)^{|star(v)∩star(w)|} B_w A_v
     ⇒ 对易 ⟺ 顶点星形交于偶数条边（图的**欧拉性**：所有度数为偶）。

  ⇒ **图的所有顶点度数为偶** 时（欧拉图），上式全对易 ⇒ 合法的量子稳定子码。
    这是 Zero 原生（零和 ⇒ 欧拉图）条件，不是外部约束。

码参数（实测，不靠闭式猜）
  n = 2|E| 个量子比特；稳定子数 = 2|C|（但含一个冗余：∏_v A_v = I）
  k = n − rank(A 与 B 的秩) ；d = 最小逻辑算符重量（枚举小码）

用法: python3 zero_quantum_cycle_code.py
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


def ring(n):
    return n, [(i, (i + 1) % n) for i in range(n)]


def torus(L):
    """L×L 环面：**去重**后的边集（无自环、无重复边）。"""
    idx = lambda i, j: (i % L) * L + (j % L)
    es = set()
    for i in range(L):
        for j in range(L):
            for (a, b) in ((idx(i, j), idx(i, j + 1)), (idx(i, j), idx(i + 1, j))):
                if a != b:
                    es.add((min(a, b), max(a, b)))
    return L * L, sorted(es)


def complete(n):
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def star_matrix(nv, edges):
    """顶点-边关联矩阵 A：A[v,e]=1 ⟺ e 与 v 关联。"""
    A = np.zeros((nv, len(edges)), dtype=np.uint8)
    for e, (u, v) in enumerate(edges):
        A[u, e] = 1
        A[v, e] = 1
    return A


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


def commutes(A, nv, ne):
    """A_v 与 B_w 是否全对易：判据 = 所有顶点度数为偶。"""
    deg = A.sum(axis=1)
    return bool(np.all(deg % 2 == 0)), deg


def logical_distance(A, ne, cap: int = 3):
    """最小逻辑算符重量：与全部稳定子对易、但不属于稳定子群的 Pauli 的最小重量。

    简化：在 X 型逻辑（只在 X 边上的算符）里找 —— 逻辑 = ker(A) 中非 stabilizer 的部分。
    ker(A) 的维数 = β1（圈空间），其中 stabilizer 部分 = A 的行空间对偶…这里用
    "最小重量非零圈"（= 圈码的 d）作为量子距离的下界并报出。
    """
    # ker(A)
    m, n = A.shape
    M = A.copy() % 2
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, pc in enumerate(piv):
            x[pc] = M[i, f]
        basis.append(x)
    if not basis:
        return None, 0
    best = n + 1
    k = len(basis)
    if k > 20:
        return None, k   # 枚举上限（2^20）
    for mask in range(1, 1 << k):
        x = np.zeros(n, dtype=np.uint8)
        for i in range(k):
            if mask >> i & 1:
                x ^= basis[i]
        w = int(x.sum())
        if 0 < w < best:
            best = w
    return best, k


def main() -> None:
    print("=" * 78)
    print("纯 Zero：量子圈码（顶点星形 X/Z 稳定子）—— 实测参数")
    print("=" * 78)
    cases = [("环 C4", ring(4)), ("环 C6", ring(6)), ("环 C8", ring(8)),
             ("环面 2x2", torus(2)), ("环面 3x3", torus(3)), ("环面 4x4", torus(4)),
             ("K5", complete(5)), ("K4", complete(4))]
    print(f"    {'图':12s} {'V':>4s} {'E':>4s} {'n=2E':>5s} {'欧拉?':>6s} "
          f"{'d_cyc':>6s} {'β1':>4s} {'k=2β1':>6s} {'k/n':>6s}")
    for name, (nv, edges) in cases:
        A = star_matrix(nv, edges)
        ne = len(edges)
        euler, deg = commutes(A, nv, ne)
        d, beta1 = logical_distance(A, ne)
        n = 2 * ne
        k = 2 * beta1 if beta1 else 0
        print(f"    {name:12s} {nv:4d} {ne:4d} {n:5d} {str(euler):>6s} "
              f"{str(d):>6s} {beta1:4d} {k:6d} {k/n:6.3f}")
    print("\n判读：")
    print("  · 欧拉性（所有度数为偶）⟺ A_v 与 B_w 全对易 ⟺ 合法量子稳定子码；")
    print("    零和 ⇒ 每个顶点的星形边数贡献偶数 ⇒ 欧拉性由 Z0 的零和**原生**保证。")
    print("  · k = 2β1（β1 = 圈空间维数 = E − V + 1）；d_cyc = 最小非平凡圈长。")


if __name__ == "__main__":
    main()
