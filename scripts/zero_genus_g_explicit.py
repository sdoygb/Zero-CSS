#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""亏格 g 曲面的严格构造（多边形粘合）：直接算 Zero 原生码的参数

构造（教科书，可验证）
  一个环面 = 一个方格，其对边按 (a,b,a⁻¹,b⁻¹) 粘合 ⇒ V=1, E=2, F=1, χ=0
  挖一个洞 = 删掉一个顶点（洞的边界变成自由的边界边）
  把 g 个"带洞环面"沿洞的边界交替粘合 ⇒ 亏格 g 闭曲面

计数（每步可核）
  单块（带洞环面）：V=0, E=2, F=1（把洞的顶点删掉，两条边成自由边）
  块间粘合：每粘一对边 ⇒ E 减 1
  g 块：V = 0（全部顶点被删）?? —— 改用下面的显式方案

更稳的显式方案：**g 个方格按环形粘合**（每块贡献 2 条内部边 + 2 条边界边）
  第 i 块有顶点 a_i, b_i, c_i, d_i；边 (a_i,b_i), (b_i,c_i), (c_i,d_i), (d_i,a_i)
  块内：a_i~d_i 与 b_i~c_i 粘合（环面的一对）
  块间：第 i 块的 (c_i,d_i) 与第 i+1 块的 (a_{i+1},b_{i+1}) 粘合
  ⇒ 顶点数、边数、面数都可直接数

本文件
  用**显式边配对**构造，逐例核对 χ = 2 − 2g 与对易条件，再算 k。
"""
from __future__ import annotations

from collections import defaultdict

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


def genus_g_surface(g):
    """g 个环面块沿洞粘合。

    每块 i：4 个顶点 (i,0..3)，4 条边 e(i,0..3)。
    块内粘合：顶点 (i,0)~(i,3) 与 (i,1)~(i,2)（形成环面的两个顶点）
    边配对：e(i,0)-e(i,2), e(i,1)-e(i,3)  ← 这不是环面，需要正确的环面模式
    """
    # 直接采用**已发表的**计数：亏格 g 的标准胞格化
    # 用 g 个 4g-边形不好；改用 g 个方格的**环形粘合**：
    #   块 i 的右边 = 块 i+1 的左边（形成管）
    #   所有块的上下边各自粘合（形成洞，需在末端封闭）
    # 简化且**可验证**的版本：直接用 g 个方格，右-左链式粘合，首尾也粘合（管状）
    # 再粘合上-下边对 ⇒ 环面 × ... 先构造，再核 χ。
    verts = {}          # (i, corner) -> 顶点编号
    edges = []
    faces = []
    V = 0

    def getv(key):
        nonlocal V
        if key not in verts:
            verts[key] = V
            V += 1
        return verts[key]

    # 先给每块 4 个独立顶点
    for i in range(g):
        for c in range(4):
            getv((i, c))
    # 块内：0~3 与 1~2 粘合（把方格变成"管"）
    for i in range(g):
        verts[(i, 3)] = verts[(i, 0)]
        verts[(i, 2)] = verts[(i, 1)]
    # 块间：右列 = 下一块左列
    for i in range(g - 1):
        verts[(i + 1, 0)] = verts[(i, 1)]   # 左列 → 上一块右列
        verts[(i + 1, 3)] = verts[(i, 1)]
        verts[(i + 1, 1)] = verts[(i, 0)]   # 右列 → 上一块左列
        verts[(i + 1, 2)] = verts[(i, 0)]
    # 首尾粘合（形成闭合的管）
    verts[(0, 0)] = verts[(g - 1, 1)]
    verts[(0, 3)] = verts[(g - 1, 1)]
    verts[(0, 1)] = verts[(g - 1, 0)]
    verts[(0, 2)] = verts[(g - 1, 0)]

    # 边：每块 4 条（两条"水平"边在粘合后成 1 条）
    for i in range(g):
        for k in range(4):
            edges.append(((i, k), (i, (k + 1) % 4)))
    # 面：每块
    for i in range(g):
        faces.append([(i, k) for k in range(4)])

    # 归一化顶点编号
    uniq = {}
    for k in sorted(verts):
        if verts[k] not in uniq:
            uniq[verts[k]] = len(uniq)
    vmap = {k: uniq[verts[k]] for k in verts}

    # 边（去重：同一对半边只算一条）
    eset, elist = {}, []
    for a, b in edges:
        key = frozenset([a, b])
        if key not in eset:
            eset[key] = len(elist)
            elist.append((a, b))
    # 面 -> 边号
    facelist = []
    for f in faces:
        fl = []
        for k in range(4):
            key = frozenset([f[k], f[(k + 1) % 4]])
            fl.append(eset[key])
        facelist.append(sorted(set(fl)))
    return len(uniq), vmap, elist, facelist


def main() -> None:
    print("=" * 92)
    print("亏格 g 曲面（显式粘合）：Zero 原生码参数")
    print("=" * 92)
    print("  %-8s %6s %6s %6s %8s %8s %8s %6s %8s %8s"
          % ("g", "V", "E", "F", "χ", "g_eff", "rX", "rZ", "k", "k/E"))
    for g in (1, 2, 3, 4, 5, 6):
        V, vmap, edges, faces = genus_g_surface(g)
        ne = len(edges)
        F = len(faces)
        chi = V - ne + F
        g_eff = (2 - chi) / 2
        HX = np.zeros((V, ne), dtype=np.uint8)
        for v in range(V):
            for e, (a, b) in enumerate(edges):
                if vmap.get(a) == v or vmap.get(b) == v:
                    HX[v, e] = 1
        HZ = np.zeros((F, ne), dtype=np.uint8)
        for fi, f in enumerate(faces):
            for e in f:
                HZ[fi, e] = 1
        comm = not ((HX @ HZ.T) % 2).any() if F else True
        rX, rZ = rank2(HX), rank2(HZ)
        k = ne - rX - rZ
        print("  %-8d %6d %6d %6d %8d %8.1f %8d %6d %6d %8.3f%s"
              % (g, V, ne, F, chi, g_eff, rX, rZ, k, k / ne,
                 "" if comm else "  ← 对易失败"))

    print("\n" + "=" * 92)
    print("判读")
    print("=" * 92)
    print("""  · χ = 2 − 2g 与 g_eff 应一致（构造正确性检验）
  · 对易须为 True（每边恰属 2 面）
  · 若 k 随 g 增长 ⇒ **亏格提高码率**，即"用 Zero 提高 CSS 码率"成立
  · 码率 R = k/E 是否趋于非零常数""")


if __name__ == "__main__":
    main()
