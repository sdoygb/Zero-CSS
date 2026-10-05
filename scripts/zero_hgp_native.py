#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP（hypergraph product）：Zero 原生的**常码率**构造

一直以来的配方（配方 A）
  边 = 比特，顶点 = X 检查，面 = Z 检查
  ⇒ k = E − rank(H_X) − rank(H_Z) = β₁ − rank(H_Z)
  ⇒ 面数 ∝ E 会把 β₁ 吃满 ⇒ k = O(1)（平坦复形上）

另一种配方（配方 B）：**Hypergraph Product**（Tillich–Zémor）
  取两个图 G1=(V1,E1), G2=(V2,E2)，各给一个经典码（圈空间）
    H1 = G1 的关联矩阵（|V1| × |E1|）
    H2 = G2 的关联矩阵（|V2| × |E2|）
  CSS 检查：
    H_X = [ H1 ⊗ I_{E2}  |  I_{V1} ⊗ H2ᵀ ]
    H_Z = [ I_{E1} ⊗ H2  |  H1ᵀ ⊗ I_{V2} ]
  参数（Tillich–Zémor）：
    n = E1·V2 + V1·E2
    k = k1·k2 + k1ᵀ·k2ᵀ      （k = dim ker H，kᵀ = dim ker Hᵀ）
  对**圈空间**（k = β₁，kᵀ = V − rank = c 分量数）：连通时 kᵀ = 1
    ⇒ k = β₁(G1)·β₁(G2) + 1

  关键：β₁ ∝ E（定度图）⇒ **k ∝ n** ⇒ 常数码率！

检验
  H1 对若干图对，算 n、k、k/n，核对对易条件
  H2 与配方 A（环面/双曲）对比
"""
from __future__ import annotations

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
    """关联矩阵 (V × E)。"""
    H = np.zeros((V, len(edges)), dtype=np.uint8)
    for e, (a, b) in enumerate(edges):
        H[a, e] = 1
        H[b, e] = 1
    return H


def cycle_graph(L):
    return L, [(i, (i + 1) % L) for i in range(L)]


def random_regular(V, d, seed):
    """d-正则随机图（配对模型，简单化：允许少量重边）。"""
    rnd = np.random.default_rng(seed)
    if (V * d) % 2:
        V += 1
    stubs = [v for v in range(V) for _ in range(d)]
    rnd.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a != b:
            edges.append((int(min(a, b)), int(max(a, b))))
    return V, list(dict.fromkeys(edges))


def hgp(H1, H2):
    """Hypergraph product 的两组 CSS 检查矩阵。"""
    V1, E1 = H1.shape
    V2, E2 = H2.shape
    I_E2 = np.eye(E2, dtype=np.uint8)
    I_V1 = np.eye(V1, dtype=np.uint8)
    I_E1 = np.eye(E1, dtype=np.uint8)
    I_V2 = np.eye(V2, dtype=np.uint8)
    HX = np.hstack([np.kron(H1, I_E2), np.kron(I_V1, H2.T)])
    HZ = np.hstack([np.kron(I_E1, H2), np.kron(H1.T, I_V2)])
    return HX % 2, HZ % 2


def main():
    print("=" * 94)
    print("HGP（配方 B）：Zero 原生的常码率构造")
    print("=" * 94)
    print("  %-26s %7s %7s %8s %9s %9s" % ("G1 / G2", "n", "k", "k/n", "对易", "顶点度"))
    cases = []
    for L in (5, 8, 12, 20):
        cases.append(("环 C%d / 环 C%d" % (L, L), cycle_graph(L), cycle_graph(L)))
    for d in (3, 4):
        for V in (20, 40):
            cases.append(("%d-正则 V=%d / 自身" % (d, V),
                          random_regular(V, d, seed=V + d),
                          random_regular(V, d, seed=V + d)))
    for name, (V1, E1), (V2, E2) in cases:
        H1, H2 = incidence(V1, E1), incidence(V2, E2)
        HX, HZ = hgp(H1, H2)
        n = HX.shape[1]
        rX, rZ = rank2(HX), rank2(HZ)
        k = n - rX - rZ
        comm = not ((HX @ HZ.T) % 2).any()
        deg = "%d/%d" % (len(E1) * 2 / V1, len(E2) * 2 / V2)
        print("  %-26s %7d %7d %8.3f %9s %9s"
              % (name, n, k, k / n, comm, deg))

    print("\n" + "=" * 94)
    print("对比：配方 A（拓扑码）vs 配方 B（HGP）")
    print("=" * 94)
    print("  %-34s %9s %9s %8s" % ("构造", "n", "k", "k/n"))
    for L in (5, 10, 20):
        print("  %-34s %9d %9d %8.3f" % ("配方A: 环面 L=%d" % L, 2 * L * L, 2, 2 / (2 * L * L)))
    print()
    print("  （配方B 的大规模值用闭式 k = beta1(G1)*beta1(G2)+1，见 zero_hgp_summary.py）")


if __name__ == "__main__":
    main()
