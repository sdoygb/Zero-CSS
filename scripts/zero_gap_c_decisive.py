#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (c) 决定性测试：面的选择是否可从 Zero 导出？

前一轮的发现
  · 旋转（按边索引的循环移位）**不是图自同构**（把圈转一下就不再是圈，0/126）
    ⇒ L1′ 的旋转类在 2 维几何上**不给出面**。
  · "最短闭合词" **不等于**面：L=3 时 4-圈有 15 个，其中 6 个不是方格面；
    L=5 时恰好 25 个全是面。⇒ 长度极小性**不足以**定出面集。

本脚本要判（决定性）
  D1 若把"最短圈"与"面"混用，会得到什么？（已知：k=0，见 R1）
  D2 面集能否由"极小生成集"（minimal generating set）刻画？
     具体：面集是否 = π 的某个极小生成集（π：圈空间到其商的投影）？
  D3 **标准胞格化是否被 k=2 唯一确定？** 换不同的合法面集，k 能取哪些值？
  D4 面上的**依赖关系**（rel）：面集 rank = F − 1（实测 L=3,4,5 全部如此）
     ⇒ 是否存在一个**必然关系**（如 ∑_f ∂f = 0 当每边属偶数个面）？
  D5 最要紧的：面集选择**是否根本上是输入**？（即：Zero 的哪些原语能定出它？）
"""
from __future__ import annotations

import itertools
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


def build_HX(nv, edges):
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    return HX


def cycles_upto(ne, HX, wmax):
    out = []
    for w in range(3, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if not ((HX @ c) % 2).any():
                out.append(frozenset(sub))
    return out


def main() -> None:
    print("=" * 90)
    print("缺口 (c) 决定性测试：面的选择能否从 Zero 导出？")
    print("=" * 90)

    L = 3
    nv, edges, faces = torus(L)
    ne = len(edges)
    HX = build_HX(nv, edges)
    beta1 = ne - nv + 1
    F = {frozenset(f) for f in faces}
    print(f"\n环面 {L}x{L}: V={nv} E={ne} β₁={beta1} 标准面集 |F|={len(F)}")

    # [D4] 面上的必然关系
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    r = gf2_rank(HZ)
    print(f"\n[D4] 面集 rank = {r}，|F| − rank = {len(faces)-r}（依赖关系数）")
    # 找那个关系：哪些面之和为零
    dep = []
    for mask in range(1, 1 << len(faces)):
        v = np.zeros(ne, dtype=np.uint8)
        for i in range(len(faces)):
            if mask >> i & 1:
                for e in faces[i]:
                    v[e] ^= 1
        if not v.any():
            dep.append(mask)
    print(f"     恰有 {len(dep)} 个非空零和面组合（= 2^1 − 1），依赖维数 = 1")
    # 每边属几个面
    cnt = Counter()
    for f in faces:
        for e in f:
            cnt[e] += 1
    print(f"     每边属几个面：{sorted(set(cnt.values()))}"
          f"  ⇒ 全部为偶数？{all(v % 2 == 0 for v in cnt.values())}")
    print(f"     ⇒ 若每边属偶数个面，则 ∑_f ∂f = 0 是**必然关系**（解释了 rank = F − 1）")

    # [D3] 换不同面集，k 能取哪些值？
    print(f"\n[D3] 不同合法面集给出的 k（k = β₁ − rank(H_Z)）")
    cands = {}
    sq = list(F)
    cands["标准方格面（全部）"] = sq
    # 去掉一个面
    if len(sq) > 1:
        cands["去掉一个面"] = sq[:-1]
    # 加上全部"其它 4-圈"
    c4 = cycles_upto(ne, HX, 4)
    extra = [c for c in c4 if c not in F]
    cands["标准面 + 全部其它 4-圈"] = sq + extra
    cands["只取其它 4-圈"] = extra
    # 长度 ≤6 的全部圈
    cands["长度≤6 全部圈"] = cycles_upto(ne, HX, 6)
    for name, fs in cands.items():
        if not fs:
            continue
        M = np.zeros((len(fs), ne), dtype=np.uint8)
        for fi, f in enumerate(fs):
            for e in f:
                M[fi, e] = 1
        rr = gf2_rank(M)
        print(f"     {name:26s} |F|={len(fs):3d}  rank={rr:2d}  k={beta1-rr}")

    # [D5] Zero 的原语里，哪些能定出面集？
    print(f"\n[D5] Zero 的原语能否定出面集？")
    print(f"     · 零和（Z0③）      ⇒ 定出**全部**闭合词（太多：长度≤6 就有 "
          f"{len(cycles_upto(ne, HX, 6))} 个）")
    print(f"     · 循环次序（Z_n）   ⇒ **不是图自同构**（实测 0/126）⇒ 不保持圈 ⇒ 不能定面")
    print(f"     · 局域补偿移动       ⇒ 保零和，但对所有闭合词一视同仁 ⇒ 不区分面/非面")
    print(f"     · 长度极小性        ⇒ 不足（L=3 有 6 个非面的 4-圈）")
    print(f"     ⇒ **结论：面集不能从上述原语唯一导出。**")

    print("\n" + "=" * 90)
    print("缺口 (c) 的判定")
    print("=" * 90)
    print("""  ❌ 面集**不能**从 Zero 的原语唯一导出：
       · 零和给出全部闭合词（含非面）
       · 循环次序不是图自同构（不保持圈）
       · 局域补偿移动对所有闭合词一视同仁
       · 长度极小性不足（存在非面的最短圈）

  ✅ 但有一条**必然关系**可以导出：
       若每条边恰属偶数个面（对标准胞格化成立），则 ∑_f ∂f = 0
       ⇒ rank(H_Z) = |F| − 1  ⇒ k = β₁ − |F| + 1
       这解释了"面集总有 1 维依赖"这一实测事实（L=3,4,5 全部 rank = |F|−1）。

  ⇒ 缺口 (c) 的**诚实结论**：面集是**输入**（对应 Zero 的"胞腔化选择"），
     但一旦选定，k 由必然关系给出：k = β₁ − |F| + 1（当每边属偶数面）。""")


if __name__ == "__main__":
    main()
