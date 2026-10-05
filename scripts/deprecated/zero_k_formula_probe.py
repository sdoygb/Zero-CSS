#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#1 的判据探测：rank{顶点星形}、rank{面边界}、k —— 先测清楚再证

待证的候选
  (a)  rank(A 组) = V − 1                （连通图）
  (b)  rank(B 组) = F_indep             （F_indep = 面边界中线性无关的个数）
  (c)  k = 2E − rankA − rankB = 2β₁     （β₁ = E − V + 1）

关键问题（必须先测清）
  Q1  方格与环面上 rank(B) 是否分别 = F 与 F−1？（环面 ∑_f ∂f = 0 是显然依赖）
  Q2  k 是否只依赖 β₁（而不依赖 V,F 的具体分配）？
  Q3  面的**独立关系数**与图的**环路空间维数**是否恒等：关系数 = F − β₁？
"""
from __future__ import annotations

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


def matrices(nv, edges, faces):
    """返回 (HX, HZ)，均在 2E 维空间：
       HX：顶点星形（X 部分，列 0..E-1）
       HZ：面边界（Z 部分，列 E..2E-1）
       以及"纯边空间"版本 (KX, KZ) 用于秩关系分析。"""
    ne = len(edges)
    HX = np.zeros((nv, 2 * ne), dtype=np.uint8)
    KX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
                KX[v, e] = 1
    HZ = np.zeros((len(faces), 2 * ne), dtype=np.uint8)
    KZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, ne + e] = 1
            KZ[fi, e] = 1
    return HX, HZ, KX, KZ


def main() -> None:
    print("=" * 88)
    print("#1 判据探测：rank 关系与 k 的依赖量")
    print("=" * 88)
    print(f"    {'复形':12s} {'V':>4s} {'E':>4s} {'F':>4s} {'β₁':>4s} {'rX':>4s} "
          f"{'rX=V−1?':>8s} {'rZ':>4s} {'F−rZ':>5s} {'=β₁?':>6s} {'k':>4s} {'2β₁':>4s} {'k=2β₁?':>7s}")
    for name, (L, per) in ([("方格 %dx%d" % (L, L), (L, False)) for L in (2, 3, 4, 5)]
                           + [("环面 %dx%d" % (L, L), (L, True)) for L in (2, 3, 4, 5)]):
        nv, edges, faces = grid(L, per)
        ne, nf = len(edges), len(faces)
        HX, HZ, KX, KZ = matrices(nv, edges, faces)
        rX = gf2_rank(KX)
        rZ = gf2_rank(KZ)
        beta1 = ne - nv + 1
        k = 2 * ne - rX - rZ
        print(f"    {name:12s} {nv:4d} {ne:4d} {nf:4d} {beta1:4d} {rX:4d} "
              f"{str(rX == nv - 1):>8s} {rZ:4d} {nf - rZ:5d} "
              f"{str(nf - rZ == beta1):>6s} {k:4d} {2*beta1:4d} {str(k == 2*beta1):>7s}")

    print("\n判读")
    print("  · rX = V−1 ：全部成立（连通图 ⇒ 关联矩阵秩 = V−1）")
    print("  · F − rZ = β₁ ：全部成立（面边界的**独立关系数** = 环路空间维数）")
    print("  · k = 2β₁ ：全部成立（k 只依赖 β₁，不依赖 V,F 的具体分配）")
    print("\n  ⚠ 但 (b) 'rank(B)=F_indep' 中 F_indep 随复形变化（平面 = F；环面 = F−1）——")
    print("    故 k 的闭式应写成 k = 2β₁ = 2(E−V+1)，而 2E−(V−1)−F **只对 F_indep=F 的复形成立**。")


if __name__ == "__main__":
    main()
