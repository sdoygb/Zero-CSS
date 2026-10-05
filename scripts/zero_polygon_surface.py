#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亏格 g 曲面：4g 边形 + 对边粘合 + 扇形三角化（每步可核）

构造（教科书，计数完全确定）
  取 4g 边形，边按 a₁b₁a₁⁻¹b₁⁻¹a₂b₂a₂⁻¹b₂⁻¹… 粘合
    ⇒ 所有顶点粘成 1 个（V = 1），边配成 2g 条（E = 2g）
  再把多边形**扇形三角化**成 4g−2 个三角形（F = 4g−2）
  欧拉数：χ = 1 − 2g + (4g−2) = 2g − 1  ← 不对，应为 2−2g

  ⚠ 检查：4g 边形的三角化需要 4g−2 个三角形，每个三角形 3 条**对角线或边**
     边数 = 4g（多边形边）+ (4g−3)（对角线）= 8g−3
     但这些边在粘合后要合并 ⇒ 复杂。

  改用**更简单且确定**的构造：g 个方格的"环面链"用**正确的**边配对。
    第 i 块方格：顶点 (i,0),(i,1),(i,2),(i,3)，边 (i,0)-(i,1), (i,1)-(i,2),
                 (i,2)-(i,3), (i,3)-(i,0)
    环面粘合（单块）：(i,0)-(i,1) 与 (i,2)-(i,3) 配； (i,1)-(i,2) 与 (i,3)-(i,0) 配
      ⇒ 顶点：(i,0)=(i,2)=(i,3)? 需仔细。
"""
from __future__ import annotations

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


def build_surface(V, edges, faces):
    """从 (顶点数, 边表, 面表) 构造 HX/HZ 并算参数。"""
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


def poly_surface(g):
    """4g 边形 + 对边粘合 + 扇形三角化。

    顶点：多边形的 4g 个角，粘合后 V 由配对决定（先不粘顶点，只粘边）
    边：多边形的 4g 条边 + 扇形三角化的 (4g−3) 条对角线
    面：4g−2 个三角形
    边配对：第 k 条边与第 k+2g 条边配（对边）
    """
    m = 4 * g
    # 顶点 0..m-1（多边形角），边 i 连接角 i 与角 i+1
    # 边配对：(i, i+1) ~ (i+2g, i+2g+1)
    # 先做顶点粘合：边配对 (0,1)~(2g,2g+1) ⇒ 0~2g, 1~2g+1
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

    # 标准亏格 g 词：a1 b1 a1^-1 b1^-1 a2 b2 ...
    #   边位置 p 配对：p <-> (p+2 if p%4 < 2 else p-2)  ⇒ 即 p 与 p^2 的翻转
    for p in range(0, m, 2):
        q = p + 2 if (p % 4) < 2 else p - 2
        # 边 p: (p, p+1) ~ 边 q: (q, q+1)，且方向相反
        a1, b1 = p % m, (p + 1) % m
        a2, b2 = q % m, (q + 1) % m
        union(a1, b2)          # 反向粘合：p 的起点 ~ q 的终点
        union(b1, a2)
    V = len({find(i) for i in range(m)})
    # 边：多边形的 m 条边（粘合后 m/2 = 2g 条）+ 对角线 (4g-3) 条
    edges = []
    emap = {}
    for i in range(m):
        a, b = find(i), find((i + 1) % m)
        key = frozenset([a, b])
        if key not in emap:
            emap[key] = len(edges)
            edges.append((a, b))
    # 扇形三角化：三角形 (0, k, k+1), k=1..m-2
    diag = {}
    for k in range(1, m - 1):
        a, b = find(0), find(k)
        key = frozenset([a, b])
        if key not in diag:
            diag[key] = len(edges)
            edges.append((a, b))
    faces = []
    for k in range(1, m - 1):
        f = []
        for (u, v) in ((0, k), (k, k + 1), (k + 1, 0)):
            key = frozenset([find(u), find(v)])
            e = emap.get(key, diag.get(key))
            if e is None:
                # 多边形边 (k,k+1) 可能已被粘合成对角线
                e = diag.get(key)
            if e is not None:
                f.append(e)
        if len(f) == 3:
            faces.append(sorted(set(f)))
    return V, edges, faces


def main() -> None:
    print("=" * 92)
    print("4g 边形 + 对边粘合 + 扇形三角化")
    print("=" * 92)
    print("  %-6s %6s %6s %6s %8s %8s %6s %6s %8s %8s"
          % ("g", "V", "E", "F", "χ", "g_eff", "rX", "rZ", "k", "对易"))
    for g in (1, 2, 3, 4, 5, 8, 12):
        V, edges, faces = poly_surface(g)
        r = build_surface(V, edges, faces)
        print("  %-6d %6d %6d %6d %8d %8.1f %6d %6d %8d %8s"
              % (g, r["V"], r["E"], r["F"], r["chi"], r["g"], r["rX"], r["rZ"],
                 r["k"], r["comm"]))
    print("""
  检验：g_eff 应 = g（构造正确），对易应 = True
  若 k 随 g 线性增长 ⇒ 亏格提高码率成立""")


if __name__ == "__main__":
    main()
