#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提高 CSS 码率：减少面数 vs 保持距离

定理：k = β1 - rank(H_Z)。面越少 => rank 越低 => k 越大。
但面少 => im δF 小 => 更多圈变"非平凡" => 距离可能下降。
本文件测这条 (面数, k, d) 轨迹。
"""
from __future__ import annotations
import itertools
import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0:
        return 0
    r = 0
    rows, cols = A.shape
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


def solvable(M, v):
    """M x = v 在 GF(2) 上是否有解。"""
    A = np.array(M, dtype=np.uint8) % 2
    b = np.array(v, dtype=np.uint8) % 2
    rows, cols = A.shape
    if rows == 0:
        return not b.any()
    Aug = np.zeros((rows, cols + 1), dtype=np.uint8)
    Aug[:, :cols] = A
    Aug[:, cols] = b
    r = 0
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if Aug[i, c]:
                p = i
                break
        if p is None:
            continue
        Aug[[r, p]] = Aug[[p, r]]
        for i in range(rows):
            if i != r and Aug[i, c]:
                Aug[i] = Aug[i] ^ Aug[r]
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if Aug[i, :cols].sum() == 0 and Aug[i, cols]:
            return False
    return True


def torus(L):
    idx = lambda i, j: (i % L) * L + (j % L)
    edges, seen = [], {}

    def add(a, b):
        if a == b:
            return None
        k = (min(a, b), max(a, b))
        if k not in seen:
            seen[k] = len(edges)
            edges.append(k)
        return seen[k]

    for i in range(L):
        for j in range(L):
            add(idx(i, j), idx(i, j + 1))
            add(idx(i, j), idx(i + 1, j))
    faces = []
    for i in range(L):
        for j in range(L):
            faces.append(sorted({add(idx(i, j), idx(i, j + 1)),
                                 add(idx(i, j + 1), idx(i + 1, j + 1)),
                                 add(idx(i + 1, j + 1), idx(i + 1, j)),
                                 add(idx(i + 1, j), idx(i, j))}))
    return L * L, edges, faces


def make(nv, edges, faces):
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    if faces:
        HZ = np.zeros((len(faces), ne), dtype=np.uint8)
        for fi, f in enumerate(faces):
            for e in f:
                HZ[fi, e] = 1
    else:
        HZ = np.zeros((0, ne), dtype=np.uint8)
    return HX, HZ


def min_logical(HX, HZ, ne, wmax=None):
    wmax = wmax or min(ne, 6)
    for w in range(2, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if (HX @ c % 2).any():
                continue
            if not solvable(HZ.T, c):   # 解 HZ x = c ⇔ (HZ^T) 为系数矩阵，c 按行数计
                return w
    return None


def main():
    print("=" * 88)
    print("减少面数 vs 保持距离")
    print("=" * 88)
    for L in (3, 4):
        nv, edges, faces = torus(L)
        ne = len(edges)
        HX0, _ = make(nv, edges, faces)
        beta1 = ne - rank2(HX0)
        print("\n环面 %dx%d  V=%d E=%d F=%d beta1=%d" % (L, L, nv, ne, len(faces), beta1))
        print("  %-24s %5s %5s %5s %8s" % ("面集", "|F|", "k", "d", "R=k/E"))
        trials = [("全部面", faces), ("去最后 1 个", faces[:-1]),
                  ("去最后一半", faces[: len(faces) // 2]),
                  ("只留 1 个", faces[:1]), ("无面", [])]
        for name, F in trials:
            HX, HZ = make(nv, edges, F)
            k = ne - rank2(HX) - rank2(HZ)
            d = min_logical(HX, HZ, ne, wmax=min(ne, 6))
            print("  %-24s %5d %5d %5s %8.3f" % (name, len(F), k, str(d), k / ne))

    print("\n" + "=" * 88)
    print("beta1/E 的维度效应")
    print("=" * 88)
    print("  %-14s %10s %12s %10s" % ("维度 d", "V", "E", "beta1/E"))
    for d in (2, 3, 4, 6):
        L = 4
        V, E = L ** d, d * L ** d
        print("  %-14d %10d %12d %10.3f" % (d, V, E, (E - V + 1) / E))
    print("""
  => 高维复形的 beta1/E 趋于 1，但 k 还要减 rank(H_Z)（面数 ∝ E）
     => 能否把"高 beta1"转成"高 k"必须实测""")


if __name__ == "__main__":
    main()
