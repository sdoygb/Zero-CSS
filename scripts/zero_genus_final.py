#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亏格 g 曲面：面-边关联直接给出（最可靠表述）

构造
  4g 边形，边 0..4g-1
  扇形三角化：三角形 T_k = (角 0, 角 k, 角 k+1)，k = 1..4g-2
     T_k 的边：多边形边 (k,k+1)，以及两条对角线 (0,k)、(0,k+1)
     对角线按 (0,k) 标签共享：k=2..4g-2 的 (0,k) 各一条
  边配对（标准亏格 g 词 a1b1a1⁻¹b1⁻¹…）：
     位置 p 与 p' = p+2 (p%4<2) 或 p-2 配对 —— 对**多边形边**成立
  ⇒ 复形：V 由粘合决定、E = 多边形边(2g) + 对角线(4g-3)、F = 4g-2
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


def build(g):
    m = 4 * g
    # 1) 多边形边的粘合：位置 p <-> p'，反向 ⇒ 顶点配对
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for p in range(m):
        if p % 2:
            continue                       # 只处理偶数位置，避免重复
        q = p + 2 if (p % 4) < 2 else p - 2
        q %= m
        # 边 p: 角 p -> p+1；边 q: 角 q -> q+1；反向粘合
        union(p % m, (q + 1) % m)
        union((p + 1) % m, q % m)
    Vmap = {}
    for i in range(m):
        r = find(i)
        if r not in Vmap:
            Vmap[r] = len(Vmap)
    V = len(Vmap)

    # 2) 边表
    edges, emap = [], {}

    def eid(a, b):
        a, b = Vmap[find(a)], Vmap[find(b)]
        if a == b:
            return None
        k = (min(a, b), max(a, b))
        if k not in emap:
            emap[k] = len(edges)
            edges.append(k)
        return emap[k]

    # 多边形边（粘合后只剩一半）
    for p in range(m):
        eid(p, (p + 1) % m)
    # 对角线 (0, k), k = 2..m-2
    for k in range(2, m - 1):
        eid(0, k)

    # 3) 三角形 T_k = (0, k, k+1)，k=1..m-2
    faces = []
    for k in range(1, m - 1):
        a, b, c = 0, k, k + 1
        es = []
        for (u, v) in ((a, b), (b, c), (c, a)):
            e = eid(u, v)
            if e is None:
                es = None
                break
            es.append(e)
        if es:
            faces.append(sorted(set(es)))
    return V, edges, faces


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
    print("=" * 90)
    print("亏格 g 曲面（4g 边形 + 扇形三角化）：Zero 原生码参数")
    print("=" * 90)
    print("  %-6s %6s %6s %6s %8s %8s %6s %6s %8s %8s"
          % ("g", "V", "E", "F", "χ", "g_eff", "rX", "rZ", "k", "对易"))
    for g in (1, 2, 3, 4, 6, 8, 10):
        V, edges, faces = build(g)
        r = params(V, edges, faces)
        print("  %-6d %6d %6d %6d %8d %8.2f %6d %6d %8d %8s"
              % (g, r["V"], r["E"], r["F"], r["chi"], r["g"], r["rX"], r["rZ"],
                 r["k"], r["comm"]))
    print("""
  检验：g_eff 应 ≈ g；对易应 = True
  g=1 应给 V=1, E=2, F=2（退化但正确），k = E − rX − rZ""")


if __name__ == "__main__":
    main()
