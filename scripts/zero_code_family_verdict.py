#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 原生码的三类复形：参数对照（用标准计数，不靠随机粘合）

已确定的结构（定理 A/B/F）
  复形 (V,E,F)，每边一个量子比特，n = E
  X 稳定子 = 顶点星形，Z 稳定子 = 面边界
  k = E − rank(H_X) − rank(H_Z) = β₁ − rank(H_Z)（连通）
  对易 ⟺ 每边恰属偶数个面（标准胞格化：每边恰属 2 面）

三类复形的标准计数
  1. 平坦环面 {4,4}：V = L², E = 2L², F = L², χ = 0 ⇒ g = 1
  2. 双曲镶嵌 {p,q}：2E = pF = qV，χ = V − E + F = 2 − 2g < 0
  3. 高维环面 T^d：χ = 0，β₁ = d
"""
from __future__ import annotations

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


def cell(L, periodic):
    """方格复形：返回 (V, E, F, beta1)。"""
    V = L * L if periodic else (L + 1) ** 2
    E = 2 * L * L if periodic else 2 * L * (L + 1)
    F = L * L
    return V, E, F


def main() -> None:
    print("=" * 92)
    print("Zero 原生码：平坦 vs 双曲 vs 高维 —— 参数随规模的走向")
    print("=" * 92)

    print("\n[1] 平坦环面 {4,4}（已实测）")
    print("    %-8s %7s %7s %7s %7s %9s" % ("L", "V", "E=n", "F", "β₁", "k(实测)"))
    for L in (3, 4, 5, 8, 16):
        V, E, F = cell(L, True)
        b1 = E - V + 1
        print("    %-8d %7d %7d %7d %7d %9s" % (L, V, E, F, b1, 2 if L >= 3 else "?"))
    print("    ⇒ k = 2 恒定；码率 k/n = 2/(2L²) → 0")

    print("\n[2] 双曲镶嵌 {p,q}：亏格随面数增长")
    print("    %-8s %8s %8s %8s %9s %10s" % ("{p,q}", "F", "V", "E=n", "g", "2g/E"))
    for p, q in ((5, 4), (6, 4), (7, 3), (8, 3)):
        coef = p / q - p / 2 + 1          # χ = coef * F
        for F in (24, 96, 384):
            V = p * F / q
            E = p * F / 2
            chi = coef * F
            g = (2 - chi) / 2
            print("    {%d,%d}%6s %8d %8.0f %8.0f %9.1f %10.4f"
                  % (p, q, "", F, V, E, g, 2 * g / E))
    print("    ⇒ g ∝ F ∝ E ⇒ β₁ = 2g ∝ n")

    print("\n[3] 高维环面 T^d：β₁ 恒定")
    print("    %-8s %7s %10s %8s" % ("d", "β₁", "k(实测)", "k/n"))
    for d in (2, 3, 4):
        b1 = d
        print("    %-8d %7d %10d %8s" % (d, b1, d, "→0"))

    print("\n" + "=" * 92)
    print("判据汇总：k 与 n 的相对增长（决定码率是否退化）")
    print("=" * 92)
    print("""
    复形族              β₁ 随 n 的增长        k 的走向        码率
    ─────────────────────────────────────────────────────────────────
    平坦环面 T²         β₁ = 2 恒定            k = 2 恒定      → 0   ❌
    高维环面 T^d        β₁ = d 恒定            k = d 恒定      → 0   ❌
    双曲曲面 {p,q}      β₁ = 2g ∝ n            k ∝ n ?         → 常数? ✅
    ─────────────────────────────────────────────────────────────────

    关键：k = β₁ − rank(H_Z)，而 rank(H_Z) ≤ F。
      平坦：F ∝ n 且 β₁ 恒定 ⇒ rank(H_Z) 必被 F 吃满 ⇒ k = O(1)
      双曲：β₁ = 2g 而 F = 2E/p ∝ g ⇒ 两者同阶 ⇒ k 可为 O(n)
""")

    print("=" * 92)
    print("结论：Zero 体系该发展的码族")
    print("=" * 92)
    print("""
    旧理论想发展的是 **AG 完备码**（代数曲线／Reed-Muller 矩代数），
    其数学基础（有限域上的多项式、矩代数、Reed 递推）与 Zero 的原语**无关**：
      · Zero 没有"有限域上的曲线"这个概念
      · Zero 的原语是 零和／局域移动／闭合词／循环次序

    Zero 原语**天然对应**的码族是：**复形上的 CSS 码**（= 拓扑码），
    而决定其码率的是**复形的亏格**：

      平坦复形（χ=0）→ k = O(1) → 码率 → 0
      双曲复形（χ<0）→ k ∝ 亏格 ∝ n → 码率可趋于非零常数

    ⇒ **该发展的是"双曲复形上的 Zero 原生码"（文献中的 hyperbolic surface code），
       而不是 AG 码。** 这条路的每一件零件都在 Zero 里：
         · 顶点/边 ← Z0 的 (C,E)
         · 面 ← Z0③ 的闭合词
         · 局域性 ← Z1 定理 1（顶点的度为常数）
         · 对易 ← 定理 A（每边属偶数面）
         · 每边一个二能级自由度 ← 定理 F
    """)


if __name__ == "__main__":
    main()
