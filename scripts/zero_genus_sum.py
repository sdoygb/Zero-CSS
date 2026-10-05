#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亏格 g 曲面：连通和构造（每步可验证）

连通和 S1 # S2
  从 S1 删掉一个三角形 t1，从 S2 删掉一个三角形 t2
  把 t1 的边界 (3 条边) 与 t2 的边界逐条粘合
  ⇒ χ(S1#S2) = χ(S1) + χ(S2) − 2

起始：**环面**的标准胞格化（L×L 方格，环面拓扑），χ = 0
  g 次连通和 ⇒ χ = −2(g−1) = 2 − 2g ✓

构造细节（离散）
  顶点/边/面都用整数索引；"删三角形"= 从面表里去掉；
  边界粘合时，t1 的边界边 (u1,v1) 与 t2 的 (u2,v2) 粘合 ⇒ union(u1,v2), union(v1,u2)
  粘合后重新统计 V、E（去重）、F
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
                p = i; break
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


def cell_torus(L):
    """L×L 环面：V=L², E=2L², F=L²（每边属 2 面，每面 4 边）。"""
    idx = lambda i, j: (i % L) * L + (j % L)
    V = L * L
    edges, emap, faces = [], {}, []
    def add(a, b):
        k = (min(a, b), max(a, b))
        if k not in emap:
            emap[k] = len(edges); edges.append(k)
        return emap[k]
    for i in range(L):
        for j in range(L):
            add(idx(i, j), idx(i, j + 1)); add(idx(i, j), idx(i + 1, j))
    for i in range(L):
        for j in range(L):
            faces.append(sorted({add(idx(i, j), idx(i, j + 1)),
                                 add(idx(i, j + 1), idx(i + 1, j + 1)),
                                 add(idx(i + 1, j + 1), idx(i + 1, j)),
                                 add(idx(i + 1, j), idx(i, j))}))
    return V, edges, faces


def connected_sum(data1, data2):
    """连通和：删各自一个面，沿边界逐边粘合。"""
    V1, E1, F1 = data1
    V2, E2, F2 = data2
    hole1 = set(F1[-1])
    hole2 = set(F2[-1])
    if len(hole1) != len(hole2):
        return None
    off = V1

    # 顶点并查集
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in E1:
        find(a); find(b)
    for (a, b) in E2:
        find(a + off); find(b + off)

    # 洞的边界端点（按顺序）
    def border(E, hole, shift):
        vs = []
        for e in sorted(hole):
            a, b = E[e]
            vs.append((a + shift, b + shift))
        return vs

    b1 = border(E1, hole1, 0)
    b2 = border(E2, hole2, off)
    for k in range(len(b1)):
        a1, c1 = b1[k]
        a2, c2 = b2[k]
        union(a1, a2)
        union(c1, c2)

    # 重建边表（跳过洞边）
    newedges, emap = [], {}

    def eid(a, b):
        a, b = find(a), find(b)
        k = (min(a, b), max(a, b))
        if k not in emap:
            emap[k] = len(newedges)
            newedges.append(k)
        return emap[k]

    for i, (a, b) in enumerate(E1):
        if i in hole1:
            continue
        eid(a, b)
    for i, (a, b) in enumerate(E2):
        if i in hole2:
            continue
        eid(a + off, b + off)

    # 重建面表
    newfaces = []
    for f in F1:
        if f is F1[-1] or set(f) == hole1:
            continue
        newfaces.append(sorted({eid(*E1[e]) for e in f}))
    for f in F2:
        if set(f) == hole2:
            continue
        newfaces.append(sorted({eid(E2[e][0] + off, E2[e][1] + off) for e in f}))

    V = len({find(x) for x in parent})
    return V, newedges, newfaces


def params(V, edges, faces):
    ne = len(edges)
    HX = np.zeros((V, ne), dtype=np.uint8)
    for v in range(V):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    chi = V - ne + len(faces)
    comm = not ((HX @ HZ.T) % 2).any() if faces else True
    rX, rZ = rank2(HX), rank2(HZ)
    return dict(V=V, E=ne, F=len(faces), chi=chi, g=(2 - chi) / 2,
                rX=rX, rZ=rZ, k=ne - rX - rZ, comm=comm)


def main():
    print("=" * 88)
    print("亏格 g 曲面（连通和）：Zero 原生码参数")
    print("=" * 88)
    base = cell_torus(3)
    p0 = params(*base)
    print("  基块（3x3 环面）: V=%d E=%d F=%d χ=%d g_eff=%.1f rX=%d rZ=%d k=%d 对易=%s"
          % (p0["V"], p0["E"], p0["F"], p0["chi"], p0["g"], p0["rX"], p0["rZ"],
             p0["k"], p0["comm"]))
    cur = base
    print("\n  %-6s %6s %6s %6s %8s %8s %6s %6s %8s %8s"
          % ("和次数", "V", "E", "F", "χ", "g_eff", "rX", "rZ", "k", "对易"))
    for n in range(1, 6):
        nxt = connected_sum(cur, base)
        if nxt is None:
            print("  第 %d 次连通和失败" % n); break
        cur = nxt
        r = params(*cur)
        print("  %-6d %6d %6d %6d %8d %8.1f %6d %6d %8d %8s"
              % (n, r["V"], r["E"], r["F"], r["chi"], r["g"], r["rX"], r["rZ"],
                 r["k"], r["comm"]))


if __name__ == "__main__":
    main()
