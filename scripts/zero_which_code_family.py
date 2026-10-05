#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哪个码族最适合 Zero 体系——候选逐一实测

背景
  Zero 原生给出：零和 ⇒ 圈空间 ⇒ CSS 码（顶点星形 X ＋ 面边界 Z）。
  但在**方格/环面**上参数退化：平面 k=0；环面 [[2L^2, 2, L]]，码率 2/L^2 → 0。
  旧理论想发展的是 AG 完备码（$[[1024,672,16]]$，率 0.656）。

判据（"适合 Zero"= 只用 Zero 原生物件就能构造，且参数不退化为 0）
  C1 零和（Z0③）⇒ 圈空间：需要图/复形
  C2 局域补偿移动（Z1 定理 1）：需要**局域**校验（顶点的度数为 O(1)）
  C3 对易判据（定理 A）：面必须是闭合词，且每边属偶数个面
  C4 二元算符空间（定理 F）：每边一个二能级自由度

候选（全部是已知码族，逐个检验与 Zero 原语的吻合度与参数）
  A 环面上的 toric code         [[2L^2, 2, L]]        —— 已实测：率为 0
  B **高维复形上的 toric code**  4 维环面 T^4         —— k ~ L^3，率不衰减？
  C **hypergraph product（HGP）** 两个图的乘积         —— qLDPC 主流，率可正
  D 3 维环面 T^3                                     —— k ~ L
  E 圈码（cycle code）                                —— [[E, β₁, girth]]
  F 色码 / 子系统码                                   —— 需更多结构

本脚本算 B/C/D 的实际参数，与 AG 对照。
"""
from __future__ import annotations

import itertools
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


# ---------- 复形构造 ----------
def torus_nd(L, d):
    """d 维环面 T^L...T^L 的 1-骨架（顶点 L^d，边 d*L^d）。返回 (nv, edges, 2-cells)."""
    def vid(idx):
        return sum((i % L) * (L ** k) for k, i in enumerate(idx))
    edges, seen = [], {}

    def add(a, b):
        if a == b:
            return
        k = (min(a, b), max(a, b))
        if k not in seen:
            seen[k] = len(edges)
            edges.append(k)

    for cell in itertools.product(range(L), repeat=d):
        for ax in range(d):
            nxt = list(cell)
            nxt[ax] = (nxt[ax] + 1) % L
            add(vid(cell), vid(nxt))
    # 2-cells：每个 (ax, ay) 平面上的方格
    faces = []
    for cell in itertools.product(range(L), repeat=d):
        for ax in range(d):
            for ay in range(ax + 1, d):
                c1 = list(cell); c2 = list(cell)
                c1[ax] = (c1[ax] + 1) % L
                c2[ay] = (c2[ay] + 1) % L
                c3 = list(c1); c3[ay] = (c3[ay] + 1) % L
                try:
                    f = sorted({seen[(min(vid(cell), vid(c1)), max(vid(cell), vid(c1)))],
                                seen[(min(vid(c1), vid(c3)), max(vid(c1), vid(c3)))],
                                seen[(min(vid(c3), vid(c2)), max(vid(c3), vid(c2)))],
                                seen[(min(vid(c2), vid(cell)), max(vid(c2), vid(cell)))]})
                    faces.append(f)
                except KeyError:
                    pass
    return L ** d, edges, faces


def hgp(H1, H2):
    """Hypergraph product 的两组校验矩阵（CSS）。H1: m1×n1, H2: m2×n2。"""
    H1 = np.asarray(H1, dtype=np.uint8) % 2
    H2 = np.asarray(H2, dtype=np.uint8) % 2
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    I1, I2 = np.eye(n1, dtype=np.uint8), np.eye(n2, dtype=np.uint8)
    Im1, Im2 = np.eye(m1, dtype=np.uint8), np.eye(m2, dtype=np.uint8)
    HX = np.hstack([np.kron(H1, I2), np.kron(Im1, H2.T)])
    HZ = np.hstack([np.kron(I1, H2), np.kron(H1.T, Im2)])
    return HX, HZ


def rep_code(L):
    """重复码校验矩阵（L-1）×L。"""
    H = np.zeros((L - 1, L), dtype=np.uint8)
    for i in range(L - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def params(HX, HZ, verbose=True):
    """CSS 参数：n, k = n − rX − rZ（对易条件需另行核验）。"""
    n = HX.shape[1]
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    comm = not ((HX @ HZ.T) % 2).any()
    return n, n - rX - rZ, comm


def main() -> None:
    print("=" * 88)
    print("哪个码族适合 Zero——候选实测")
    print("=" * 88)

    print("\n[A/D] 环面（d 维）上的 toric code：k 是否随 L 增长？")
    print("  %-10s %5s %6s %6s %6s %8s" % ("复形", "L", "n", "k", "k/n", "对易"))
    for d in (2, 3, 4):
        for L in (3, 4, 5):
            nv, edges, faces = torus_nd(L, d)
            ne = len(edges)
            if ne > 4000:
                continue
            HX = np.zeros((nv, ne), dtype=np.uint8)
            for v in range(nv):
                for e, (a, b) in enumerate(edges):
                    if v in (a, b):
                        HX[v, e] = 1
            HZ = np.zeros((len(faces), ne), dtype=np.uint8)
            for fi, f in enumerate(faces):
                for e in f:
                    HZ[fi, e] = 1
            n, k, comm = params(HX, HZ)
            print("  T^%-8d %5d %6d %6d %6.3f %8s" % (d, L, n, k, k / n, comm))

    print("\n[C] Hypergraph product（HGP）—— qLDPC 主流构造")
    print("  %-22s %5s %5s %8s %8s" % ("H1 / H2", "n", "k", "k/n", "对易"))
    for L1 in (3, 4, 5, 6):
        for L2 in (3, 4, 5, 6):
            HX, HZ = hgp(rep_code(L1), rep_code(L2))
            if HX.shape[1] > 3000:
                continue
            n, k, comm = params(HX, HZ)
            print("  rep(%d) x rep(%-12d) %5d %5d %8.3f %8s"
                  % (L1, L2, n, k, k / n, comm))

    print("\n[E] 圈码（cycle code）：[[E, β₁, girth]]")
    print("  环 C_n: [[n, 1, n]] —— 率 1/n → 0（退化）")
    print("  完全图 K_n: [[C(n,2), C(n,2)-n+1, 3]] —— 率高但 d=3 恒定")

    print("\n" + "=" * 88)
    print("对照：AG 完备码（旧理论的目标）")
    print("=" * 88)
    for n, k, d in ((16, 6, 4), (64, 20, 8), (256, 70, 16), (1024, 672, 16)):
        print("  [[%4d, %3d, %2d]]  率 = %.3f" % (n, k, d, k / n))


if __name__ == "__main__":
    main()
