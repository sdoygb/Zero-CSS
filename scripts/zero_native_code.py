#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 原生码：零和词 = 图的循环空间

待检验的命题
  Z0③ 的"零和"（sum w_i = 0）在 GF(2) 上就是**闭合条件**。
  于是：图 Γ 的**圈空间**（cycle space）就是 Zero 原生的码空间，
  而"传递矩阵／关联矩阵"就是它的校验矩阵 H —— **零和即校验，无需外部码族**。

  Z1 定理 1 的**局域补偿移动** T_{ex} = x + e_j - e_i 保零和，
  正是这个码的**对数算符（logical operators）**——它们改变构型但保持全部校验。

检验（每条可判）
  T1 [码参数]  对若干图，算 n=|E|、rank(H)、k=n-rank、d=最短非平凡圈长，
               得 [[n,k,d]]，与"Zero 原生"是否自洽。
  T2 [简并类]  在**旋转类**（Zero 的类）下，权重 1/2 层错误的类大小分布与 fail(2)；
               并与 **syndrome 类**对照——两者是否同一物。
  T3 [压缩-分辨率] 记录越细（旋转类 → 词），fail(2) 与账本纯度 q 如何此消彼长。

用法: python3 zero_native_code.py
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np


# ---------- 图与关联矩阵（Z0 的 (C,E)） ----------
def cycle_space_basis(nv: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """关联矩阵 B（nv × ne，GF(2)），ker(B) = 圈空间。返回 B。"""
    B = np.zeros((nv, len(edges)), dtype=np.uint8)
    for e, (u, v) in enumerate(edges):
        B[u, e] = 1
        B[v, e] = 1
    return B


def gf2_rank(M: np.ndarray) -> int:
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def cycle_basis(H: np.ndarray) -> list[tuple]:
    """H 的零空间基（GF(2)）：返回码字（长度 n 的 0/1 向量元组）。"""
    m, n = H.shape
    M = H.copy() % 2
    pivots = []
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if M[i, c]:
                p = i
                break
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, pc in enumerate(pivots):
            x[pc] = M[i, f]
        basis.append(tuple(int(v) for v in x))
    return basis


def min_weight_nonzero(basis: list[tuple], n: int) -> int:
    """码的最小权重（枚举 span；规模小才可行）。"""
    best = n + 1
    k = len(basis)
    for mask in range(1, 1 << k):
        x = np.zeros(n, dtype=np.uint8)
        for i in range(k):
            if mask >> i & 1:
                x ^= np.array(basis[i], dtype=np.uint8)
        w = int(x.sum())
        if 0 < w < best:
            best = w
    return best


def rot_class(x: tuple) -> tuple:
    n = len(x)
    return min(tuple(x[(i + k) % n] for i in range(n)) for k in range(n))


def main() -> None:
    print("=" * 78)
    print("Zero 原生码：零和词 = 图的循环空间")
    print("=" * 78)

    # 测试图：(a) 环 C_n；(b) 三角柱；(c) K4
    graphs = {}
    for n in (4, 6, 8, 10):
        graphs[f"环 C{n}"] = (n, [(i, (i + 1) % n) for i in range(n)])
    graphs["K4"] = (4, [(i, j) for i in range(4) for j in range(i + 1, 4)])

    rows = []
    for name, (nv, edges) in graphs.items():
        H = cycle_space_basis(nv, edges)
        n = len(edges)
        r = gf2_rank(H)
        k = n - r
        basis = cycle_basis(H)
        d = min_weight_nonzero(basis, n) if k <= 12 else -1
        rows.append((name, n, k, d))
        print(f"\n=== {name}: 顶点={nv} 边={n}  rank(H)={r}  ⟹ 码 [[{n},{k},{d}]]")
        # 权重 1/2 层的简并（只在码字 span 上做：错误 = 码字）
        if 0 < k <= 8:
            cw = []
            for mask in range(1, 1 << k):
                x = np.zeros(n, dtype=np.uint8)
                for i in range(k):
                    if mask >> i & 1:
                        x ^= np.array(basis[i], dtype=np.uint8)
                cw.append(tuple(int(v) for v in x))
            lo = [c for c in cw if sum(c) <= 2]
            if lo:
                by_syn = defaultdict(list)
                for c in lo:
                    by_syn[tuple((H @ np.array(c)) % 2)].append(c)
                by_orb = defaultdict(list)
                for c in lo:
                    by_orb[rot_class(c)].append(c)
                vA = Counter(len(v) for v in by_syn.values())
                vB = Counter(len(v) for v in by_orb.values())
                vA_arr = np.array(list(vA.elements()), float)
                fail2 = 1 - np.mean(1 / vA_arr) if len(vA_arr) else float("nan")
                sizes = np.array([len(v) for v in by_orb.values()], float)
                w = sizes / sizes.sum()
                q = float((w ** 2).sum())
                print(f"    低权重码字数={len(lo)}  syndrome 类分布={dict(sorted(vA.items()))}")
                print(f"                        旋转轨道分布={dict(sorted(vB.items()))}")
                print(f"                        fail(2)={fail2:.4f}  q=Σω²={q:.4f}")
    print("\n--- 汇总 ---")
    for name, n, k, d in rows:
        print(f"  {name:10s} [[{n},{k},{d}]]")
    print("\n判读：若 k>0 且 d>1，则 Zero 的零和条件**原生**给出一个[[n,k,d]]码，"
          "且局域补偿移动 = 保持校验的移动（= 对数算符）。")


if __name__ == "__main__":
    main()
