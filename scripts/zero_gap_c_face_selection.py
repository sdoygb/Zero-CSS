#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (c)：面的选择规则——候选方案与后果（先测，不猜）

背景
  面 = Zero 的闭合词（Z0③ 零和）。但"取哪些闭合作面"此前是**输入**，
  而这个输入同时决定 k 与 d（定理 B：k = β₁ − rank(H_Z)；定理 D：d_X = 最短非平凡圈）。

候选规则（都只用 Zero 的原生物件，不引入外部几何）
  R1 全部最短闭合词（最短圈 = 局部最小闭合）——例如环面上的 4-圈
  R2 循环轨道代表（L1′ 的旋转类）——每条轨道取一个代表
  R3 全部闭合词（L1 的精确词层，不粗粒化）
  R4 圈空间的一组基（生成元）——"极大"选择
  R5 闭合词按长度分层后的某一层

判据（对每个候选算 k 与 d_X，看哪个方案给出有意义的码）
  期望：
    · 若取全部最短圈 ⇒ 环面上应给 k=2（拓扑自由度）
    · 若取一组基   ⇒ rank(H_Z) = β₁ − k 可用；k 取决于基的选法
    · 若取全部闭合词 ⇒ 所有圈都是面组合 ⇒ k=0
"""
from __future__ import annotations

import itertools
from collections import defaultdict

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


def gf2_solve(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    Aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    r, pivots = 0, []
    for c in range(n):
        p = next((i for i in range(r, m) if Aug[i, c]), None)
        if p is None:
            continue
        Aug[[r, p]] = Aug[[p, r]]
        for i in range(m):
            if i != r and Aug[i, c]:
                Aug[i] ^= Aug[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if Aug[i, :n].sum() == 0 and Aug[i, n]:
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


def build(nv, edges, faces):
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    return HX, HZ


def all_short_cycles(ne, HX, maxw):
    """枚举全部长度 ≤ maxw 的圈（作为边集）。"""
    out = []
    for w in range(3, maxw + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if not ((HX @ c) % 2).any():
                out.append(frozenset(sub))
    return out


def cycle_basis(HX):
    """ker(HX) 的一组基（边集形式）。"""
    m, n = HX.shape
    A = HX.copy() % 2
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
        out.append(frozenset(int(i) for i in np.nonzero(x)[0]))
    return out


def k_and_dX(HX, HZ, ne, wmax=6):
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = ne - rX - rZ
    if k <= 0:
        return k, None
    for w in range(2, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if ((HX @ c) % 2).any():
                continue
            if not gf2_solve(HZ.T % 2, c):
                return k, w
    return k, ">%d" % wmax


def main() -> None:
    print("=" * 90)
    print("缺口 (c)：面的选择规则——候选方案的后果")
    print("=" * 90)

    for L in (3, 4):
        nv, edges, faces = torus(L)
        ne = len(edges)
        HX, _ = build(nv, edges, faces)
        print(f"\n{'='*90}\n环面 {L}x{L}：V={nv} E={ne} β₁={ne-nv+1}\n{'='*90}")

        # R1：全部最短圈（4-圈）
        short = all_short_cycles(ne, HX, 4)
        print(f"  最短圈（长度 ≤4）个数 = {len(short)}")
        if short:
            HZ1 = np.zeros((len(short), ne), dtype=np.uint8)
            for fi, f in enumerate(short):
                for e in f:
                    HZ1[fi, e] = 1
            k1, d1 = k_and_dX(HX, HZ1, ne)
            print(f"  [R1] 面 = 全部最短圈（{len(short)} 个）: rank(H_Z)={gf2_rank(HZ1)} "
                  f"k={k1} d_X={d1}")

        # R2：循环轨道代表（L1′ 旋转类）
        def rot_class(es):
            es = sorted(es)
            n = len(es)
            return min(tuple(sorted((e + s) % ne for e in es)) for s in range(ne))
        orbits = defaultdict(list)
        for c in short:
            orbits[rot_class(c)].append(c)
        reps = [v[0] for v in orbits.values()]
        print(f"  L1′ 旋转类数 = {len(orbits)}（每条轨道取一个代表 ⇒ {len(reps)} 个面）")
        if reps:
            HZ2 = np.zeros((len(reps), ne), dtype=np.uint8)
            for fi, f in enumerate(reps):
                for e in f:
                    HZ2[fi, e] = 1
            k2, d2 = k_and_dX(HX, HZ2, ne)
            print(f"  [R2] 面 = 旋转轨道代表（{len(reps)} 个）: rank(H_Z)={gf2_rank(HZ2)} "
                  f"k={k2} d_X={d2}")

        # R3：全部闭合词（直到某长度）
        allc = all_short_cycles(ne, HX, 6)
        if allc:
            HZ3 = np.zeros((len(allc), ne), dtype=np.uint8)
            for fi, f in enumerate(allc):
                for e in f:
                    HZ3[fi, e] = 1
            k3, d3 = k_and_dX(HX, HZ3, ne)
            print(f"  [R3] 面 = 全部长度≤6 的圈（{len(allc)} 个）: rank(H_Z)={gf2_rank(HZ3)} "
                  f"k={k3} d_X={d3}")

        # R4：圈空间的一组基
        basis = cycle_basis(HX)
        HZ4 = np.zeros((len(basis), ne), dtype=np.uint8)
        for fi, f in enumerate(basis):
            for e in f:
                HZ4[fi, e] = 1
        k4, d4 = k_and_dX(HX, HZ4, ne)
        print(f"  [R4] 面 = 圈空间一组基（{len(basis)} 个）: rank(H_Z)={gf2_rank(HZ4)} "
              f"k={k4} d_X={d4}")

    print("\n" + "=" * 90)
    print("判读：哪个候选给「有意义的码」（k>0 且 d 随 L 增长）")
    print("=" * 90)
    print("""  期望：环面应给 k=2（拓扑自由度）；且 d_X = L（定理 D 已算）。
  若某候选给出 k=2 且 d_X=L ⇒ 该候选与"标准胞腔化（全部方格）"一致。
  若某候选给出 k=0 ⇒ 该候选把逻辑自由度"填满"了（面太多）。""")


if __name__ == "__main__":
    main()
