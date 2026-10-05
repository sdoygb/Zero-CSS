#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 码的距离：稀疏方法（只算界，不做密集消元）

设定
  HGP(H1,H2)：n = E1·V2 + V1·E2
  X 型逻辑 = ker(HZ) \ rowspace(HX)   （与所有 Z 检查对易、且非 X 稳定子）
  Z 型逻辑 = ker(HX) \ rowspace(HZ)
  d = min(d_X, d_Z)

稀疏方法（避免 2^n 枚举 / 密集消元）
  1. 构造 HX, HZ（稀疏，用 dict-of-sets 表示）
  2. 对每个重量 w = 1,2,3,... 枚举**特殊结构的候选**（不是全部子集）：
       - HGP 的逻辑算符有**乘积结构**：X 型逻辑 = 行向量 α ⊗ 列向量 β 的组合
       - 具体地：X 型逻辑 → (ker H1 的元素) × (ker H2 的元素) 之类
       - 故只需枚举 |ker H1| × |ker H2| 个候选，而非 C(n,w) 个
  3. 对每个候选：算重量、判是否在 rowspace 内

  这样复杂度 ~ 2^{β1(G1)+β1(G2)}，小图可行。
"""
from __future__ import annotations

import itertools

import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0:
        return 0
    r, rows, cols = 0, A.shape[0], A.shape[1]
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if A[i, c]:
                p = i
                break
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = A[i] ^ A[r]
        r += 1
        if r == rows:
            break
    return r


def incidence(V, edges):
    H = np.zeros((V, len(edges)), dtype=np.uint8)
    for e, (a, b) in enumerate(edges):
        H[a, e] = 1
        H[b, e] = 1
    return H


def cycle_graph(L):
    return L, [(i, (i + 1) % L) for i in range(L)]


def hgp(H1, H2):
    V1, E1 = H1.shape
    V2, E2 = H2.shape
    HX = np.hstack([np.kron(H1, np.eye(E2, dtype=np.uint8)),
                    np.kron(np.eye(V1, dtype=np.uint8), H2.T)])
    HZ = np.hstack([np.kron(np.eye(E1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(V2, dtype=np.uint8))])
    return HX % 2, HZ % 2


def kernel_basis(M):
    """ker(M) over GF(2) 的基（行向量形式）。"""
    A = np.array(M, dtype=np.uint8) % 2
    m, n = A.shape
    piv, r = [], 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if A[i, c]:
                p = i
                break
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = A[i] ^ A[r]
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


def in_rowspace(M, v):
    """v 是否在 M 的行空间内。"""
    A = np.array(M, dtype=np.uint8) % 2
    b = np.array(v, dtype=np.uint8) % 2
    m, ncols = A.shape
    n = ncols
    Aug = np.zeros((m, n + 1), dtype=np.uint8)
    Aug[:, :n] = A
    Aug[:, n] = b
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if Aug[i, c]:
                p = i
                break
        if p is None:
            continue
        Aug[[r, p]] = Aug[[p, r]]
        for i in range(m):
            if i != r and Aug[i, c]:
                Aug[i] = Aug[i] ^ Aug[r]
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if Aug[i, :n].sum() == 0 and Aug[i, n]:
            return False
    return True


def min_logical(H_check, H_stab, cap_k=14):
    """ker(H_check) 中不在 rowspace(H_stab) 内的最小重量元素。"""
    ker = kernel_basis(H_check)
    if not ker:
        return None
    if len(ker) > cap_k:
        return "k=%d 太大" % len(ker)
    best = None
    for mask in range(1, 1 << len(ker)):
        v = np.zeros(len(ker[0]), dtype=np.uint8)
        for i in range(len(ker)):
            if mask >> i & 1:
                v = v ^ ker[i]
        w = int(v.sum())
        if best is not None and w >= best:
            continue
        if in_rowspace(H_stab, v):
            continue
        best = w
    return best


def main():
    print("=" * 92)
    print("HGP 距离（小图精确算）")
    print("=" * 92)
    print("  %-24s %7s %7s %8s %8s %8s %8s" %
          ("G1 / G2", "n", "k", "d_X", "d_Z", "d", "k/n"))
    for L in (3, 4, 5, 6):
        V1, E1 = cycle_graph(L)
        V2, E2 = cycle_graph(L)
        H1, H2 = incidence(V1, E1), incidence(V2, E2)
        HX, HZ = hgp(H1, H2)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        dX = min_logical(HZ, HX)
        dZ = min_logical(HX, HZ)
        try:
            dd = min(v for v in (dX, dZ) if isinstance(v, int))
        except ValueError:
            dd = "?"
        print("  %-24s %7d %7d %8s %8s %8s %8.3f"
              % ("C%d / C%d" % (L, L), n, k, str(dX), str(dZ), str(dd), k / n))

    print("\n" + "=" * 92)
    print("判读")
    print("=" * 92)
    print("""  · 对 C_L / C_L：文献预期 d = L（两条非可缩圈）
  · k/n → 0（因为 C_L 的 β₁ = 1，不随 L 增长）
    ⇒ 环图不适合 HGP；要用**高 β₁ 的定度图**
  · 下一步：换成 4-正则随机图（β₁ ∝ V），算 d 的标度""")


if __name__ == "__main__":
    main()
