#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#1：先测清 r_Z 的规律（不猜）

已知（上一轮实测）
  r_X := rank(顶点星形矩阵) = V − 1        （连通图）
  k   := 2E − r_X − r_Z
  ⇒ 要证 k 的闭式，**只需**搞清 r_Z := rank(面边界矩阵)。

本脚本测
  M1  r_Z 与 F、β₁、E−V+1、E−F 等量的关系（方格／环面／其他图）。
  M2  面边界的**关系数** rel := F − r_Z 与下列量对照：
        · β₁ = E − V + 1（环路空间维数，= 圈空间的维数）
        · "边-面关联"的补：∑_f |∂f| 与 2E 的关系（每边属几个面）
  M3  换一族复形（三角剖分、柱面、K4 上的面集）看规律是否稳定。
  M4  给出 k 的**实测闭式**候选并逐例核验。
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


def cylinder(L):
    """柱面：L 层环，竖向开边界。"""
    idx = lambda i, j: i * L + (j % L)
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
    for i in range(L - 1):
        for j in range(L):
            add(idx(i, j), idx(i + 1, j))
    faces = []
    for i in range(L - 1):
        for j in range(L):
            faces.append(sorted({add(idx(i, j), idx(i, j + 1)),
                                 add(idx(i, j + 1), idx(i + 1, j + 1)),
                                 add(idx(i + 1, j + 1), idx(i + 1, j)),
                                 add(idx(i + 1, j), idx(i, j))}))
    return L * L, edges, faces


def report(name, nv, edges, faces):
    ne, nf = len(edges), len(faces)
    KX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                KX[v, e] = 1
    KZ = np.zeros((nf, ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            KZ[fi, e] = 1
    rX, rZ = gf2_rank(KX), gf2_rank(KZ)
    beta1 = ne - nv + 1
    rel = nf - rZ
    k = 2 * ne - rX - rZ
    # 每边属几个面
    cnt = Counter()
    for f in faces:
        for e in f:
            cnt[e] += 1
    return dict(name=name, V=nv, E=ne, F=nf, beta1=beta1, rX=rX, rZ=rZ,
                rel=rel, k=k, edge_face=sorted(set(cnt.values())))


def main() -> None:
    print("=" * 92)
    print("#1：r_Z 的规律探测")
    print("=" * 92)
    rows = []
    for L in (2, 3, 4, 5):
        rows.append(report(f"方格 {L}x{L}", *grid(L, False)))
    for L in (2, 3, 4, 5):
        rows.append(report(f"环面 {L}x{L}", *grid(L, True)))
    for L in (3, 4, 5):
        rows.append(report(f"柱面 {L}", *cylinder(L)))

    print(f"    {'复形':12s} {'V':>4s} {'E':>4s} {'F':>4s} {'β₁':>4s} {'rX':>4s} "
          f"{'rZ':>4s} {'rel':>4s} {'k':>5s} {'每边面数':>10s}  {'k−2β₁':>7s} {'rel−β₁':>7s}")
    for d in rows:
        print(f"    {d['name']:12s} {d['V']:4d} {d['E']:4d} {d['F']:4d} {d['beta1']:4d} "
              f"{d['rX']:4d} {d['rZ']:4d} {d['rel']:4d} {d['k']:5d} "
              f"{str(d['edge_face']):>10s}  {d['k']-2*d['beta1']:7d} {d['rel']-d['beta1']:7d}")

    print("\n观察与检验")
    # 检验候选闭式
    cands = {
        "k = 2β₁": lambda d: 2 * d["beta1"],
        "k = 2E−(V−1)−F": lambda d: 2 * d["E"] - (d["V"] - 1) - d["F"],
        "k = 2E−(V−1)−rZ": lambda d: 2 * d["E"] - (d["V"] - 1) - d["rZ"],
    }
    for cname, f in cands.items():
        ok = all(f(d) == d["k"] for d in rows)
        bad = [d["name"] for d in rows if f(d) != d["k"]]
        print(f"    {cname:22s} 全例成立? {str(ok):>5s}"
              + ("" if ok else f"   反例: {bad[:4]}"))
    # β₁ 的两种算法对照
    print("\n    β₁ 与 rel 的关系（逐例）")
    for d in rows:
        note = ""
        if d["edge_face"] == [1, 2]:
            note = "（含边界边 ⇒ 开边界）"
        elif d["edge_face"] == [2]:
            note = "（每边恰属 2 面 ⇒ 闭曲面样）"
        elif d["edge_face"] == [1]:
            note = "（每边恰属 1 面）"
        print(f"      {d['name']:12s} β₁={d['beta1']:3d}  rel=F−rZ={d['rel']:3d}  "
              f"k={d['k']:4d}  {note}")


if __name__ == "__main__":
    main()
