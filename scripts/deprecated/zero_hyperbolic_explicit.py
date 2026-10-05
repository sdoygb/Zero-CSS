#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""显式双曲复形上的 Zero 原生码：直接算 k（不猜公式）

构造（显式、可验证）
  取亏格 g 的**闭曲面三角化**：每面 3 条边、每边恰属 2 面（⇒ 定理 A 对易满足）。
  方法：对每个面给 3 条"半边"，随机两两配对 ⇒ 每边恰属 2 面 ⇒ 闭曲面。
  然后将"面角"按 6 个一组粘合成顶点（q=6，即每顶点 6 个面 ⇒ {3,6} 平坦）
  或按其它 q 粘合（q>6 ⇒ 双曲）。

  对每个构造，直接算：
    n = 2E（物理比特 = 边 … 注意这里用 n=E 的口径，见下）
    k = E − rank(H_X) − rank(H_Z)
  并报欧拉数 χ = V − E + F 与亏格 g = (2−χ)/2，核对是否双曲。
"""
from __future__ import annotations

import random

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


def triangulated_surface(F, q, seed=0):
    """F 个三角形、每顶点 q 个面角的闭曲面三角化。

    返回 (nv, edges, faces)；edges[e] = (halfedge_a, halfedge_b) 表示该边由两个
    半边粘合，halfedge = (face, k)，k=0,1,2 为三角形内的边号。
    faces[f] = 该面的三条边号（排序后）。
    """
    rnd = random.Random(seed)
    half = [(f, k) for f in range(F) for k in range(3)]
    rnd.shuffle(half)
    edges, eid = [], {}
    for i in range(0, len(half), 2):
        a, b = half[i], half[i + 1]
        e = len(edges)
        edges.append((a, b))
        eid[a] = e
        eid[b] = e
    # 顶点：把面角 (f,k)（三角形第 k 个角）按 q 个一组粘合
    corners = [(f, k) for f in range(F) for k in range(3)]
    rnd.shuffle(corners)
    vid, nv = {}, 0
    for i in range(0, len(corners), q):
        grp = corners[i:i + q]
        if len(grp) < q:
            # 余数并入上一组（保持 q 的近似；报告实际分组）
            for c in grp:
                vid[c] = nv - 1
            continue
        for c in grp:
            vid[c] = nv
        nv += 1
    faces = [sorted({eid[(f, k)] for k in range(3)}) for f in range(F)]
    return nv, edges, faces, vid, eid


def build(nv, edges, faces, vid, eid):
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for f in range(len(faces)):
        for k in range(3):
            v = vid[(f, k)]
            for kk in (k, (k + 1) % 3):        # 该角的两条邻边
                HX[v, eid[(f, kk)]] = 1
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    return HX, HZ


def analyze(name, F, q, seed=0):
    nv, edges, faces, vid, eid = triangulated_surface(F, q, seed)
    nv = len(set(vid.values()))
    ne = len(edges)
    chi = nv - ne + len(faces)
    g = (2 - chi) / 2
    HX, HZ = build(nv, edges, faces, vid, eid)
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = ne - rX - rZ
    comm = not ((HX @ HZ.T) % 2).any()
    print("  %-16s F=%3d q=%d  V=%3d E=%3d  χ=%4d g=%5.1f  rX=%3d rZ=%3d  k=%4d  k/E=%6.3f  对易=%s"
          % (name, F, q, nv, ne, chi, g, rX, rZ, k, k / ne, comm))
    return dict(F=F, q=q, V=nv, E=ne, g=g, rX=rX, rZ=rZ, k=k, comm=comm)


def main() -> None:
    print("=" * 100)
    print("显式双曲三角化上的 Zero 原生码")
    print("=" * 100)
    print("\n[平坦对照 q=6] 每顶点 6 个面角 ⇒ χ=0 ⇒ 环面")
    for F in (8, 18, 32, 50):
        analyze("flat q=6", F, 6)
    print("\n[双曲 q=7] 每顶点 7 个面角 ⇒ χ<0 ⇒ 高亏格")
    for F in (14, 28, 56, 112):
        try:
            analyze("hyp q=7", F, 7)
        except Exception as e:
            print("   F=%d 失败：%s" % (F, e))
    print("\n[双曲 q=8]")
    for F in (16, 48, 96):
        try:
            analyze("hyp q=8", F, 8)
        except Exception as e:
            print("   F=%d 失败：%s" % (F, e))

    print("\n" + "=" * 100)
    print("判读")
    print("=" * 100)
    print("""  · q=6（平坦/环面）：预期 k 恒定 ⇒ 码率 → 0
  · q>6（双曲）：若 k 随 F 增长 ⇒ 码率不退化
  · 注意：本构造的"顶点粘合"是随机的，只保证计数（V,E,F,q），
    必须核对对易列（每边恰属 2 面 ⇒ 应为 True）。""")


if __name__ == "__main__":
    main()
