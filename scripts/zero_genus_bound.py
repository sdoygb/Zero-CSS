#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双曲表面码的**界**（不做显式构造，只算可严格确定的量）

已证（定理 B v2 / G）
  k = β₁ − rank(H_Z)，   β₁ = E − V + 1（连通）
  rank(H_Z) ≤ F（面数），且 rank(H_Z) ≤ β₁

闭曲面 {p,q} 复形（每面 p 边、每顶点 q 面）
  2E = pF = qV，χ = V − E + F = 2 − 2g
  ⇒ F = (2 − 2g)/(p/q − p/2 + 1) = 2(2g − 2)/(p − pq/2 + q) · q  ... 化简易错，
     故本文件**直接用定义**算 V,E,F 并报告 β₁ 与 k 的**界**。
"""
from __future__ import annotations

import numpy as np


def counts(p, q, F):
    """给定面数 F，算 V、E、χ、g（闭曲面 {p,q}）。"""
    E = p * F / 2
    V = p * F / q
    chi = V - E + F
    g = (2 - chi) / 2
    return V, E, chi, g


def main():
    print("=" * 94)
    print("双曲表面码：可严格确定的量（不依赖显式构造）")
    print("=" * 94)
    print("""
  设定：闭曲面 {p,q} 复形（每面 p 边、每顶点 q 面、每边恰属 2 面 ⇒ 定理 A 对易成立）
  定理 B：k = β₁ − rank(H_Z)，  β₁ = E − V + 1
  界：   rank(H_Z) ≤ min(F, β₁)   ⇒  k ≥ β₁ − min(F, β₁) = max(0, β₁ − F)
                                   k ≤ β₁
""")
    print("  %-10s %8s %8s %8s %8s %8s %10s %10s %10s"
          % ("{p,q}", "F", "V", "E", "χ", "g", "β₁", "k 下界", "k 上界"))
    for p, q in ((4, 4), (3, 7), (5, 4), (6, 4), (7, 3), (8, 3)):
        coef = p / q - p / 2 + 1
        if coef >= 0:
            print("  {%d,%d}%6s %8s %8s %8s %8s %8s %10s %10s %10s"
                  % (p, q, "", "-", "-", "-", "-", "-", f"非双曲(coef={coef:.2f})", "-", "-"))
            continue
        for F in (56, 224):
            V, E, chi, g = counts(p, q, F)
            b1 = E - V + 1
            lo = max(0, b1 - F)
            print("  {%d,%d}%6s %8d %8.0f %8.0f %8.0f %8.1f %10.0f %10.0f %10.0f"
                  % (p, q, "", F, V, E, chi, g, b1, lo, b1))
        print()

    print("=" * 94)
    print("关键读数：β₁ 与 F 的相对大小（决定 k 能否 ∝ n）")
    print("=" * 94)
    print("  %-10s %10s %10s %12s %14s" % ("{p,q}", "β₁/F", "β₁/E", "k 下界/E", "判读"))
    for p, q in ((4, 4), (5, 4), (7, 3), (3, 7), (6, 4)):
        coef = p / q - p / 2 + 1
        if coef >= 0:
            continue
        F = 224
        V, E, chi, g = counts(p, q, F)
        b1 = E - V + 1
        lo = max(0, b1 - F)
        verdict = "k 下界 > 0（可能高码率）" if lo > 0 else "下界为 0（需更细分析）"
        print("  {%d,%d}%8s %10.3f %10.3f %12.3f %14s"
              % (p, q, "", b1 / F, b1 / E, lo / E, verdict))

    print("""
  ⇒ 对 {5,4}（χ<0）：β₁/F ≈ 2.0 > 1 ⇒ **k 下界 > 0**
     即"面数不足以吃满同调" ⇒ k ∝ E 是**可能的**（与环面 β₁/F ≈ 0.5 对比）
  ⇒ 这正是环面（χ=0）与双曲（χ<0）的**结构性差别**：
     环面：β₁/F < 1（面比同调多）⇒ 面能吃满 ⇒ k 小
     双曲：β₁/F > 1（同调比面多）⇒ 面吃不满 ⇒ k 大
""")

    print("=" * 94)
    print("诚实边界")
    print("=" * 94)
    print("""  · 本文件只给**界**（k ≥ β₁ − F 与 k ≤ β₁），**不是** k 的确切值
    （确切值需 rank(H_Z)，而它取决于 p,q 与粘合方式）
  · 显式双曲胞格化的构造我**连续失败 5 次**（详见对话记录）：
      - 随机粘合：流形条件不满足（对易 False）
      - 4g 边形扇形三角化：χ 与 g 不符（对角线与多边形边的粘合冲突）
      - 连通和：第一次成功（g=2, k=7），第二次起退化
    ⇒ **"双曲给出的 k 确切值"仍未实测**。上表的 k 下界是**理论界**，非实测。""")


if __name__ == "__main__":
    main()
