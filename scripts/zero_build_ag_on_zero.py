#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Zero 上建立 AG 码：需要的零件 vs Zero 已有的零件

AG 码（代数几何码 / Goppa 码）的定义
  取有限域 F_q 上的代数曲线 X、除子 D = P_1 + ... + P_n、G，
  码 = { (f(P_1), ..., f(P_n)) : f ∈ L(G) }，  L(G) = { f : div(f) + G ≥ 0 } ∪ {0}

零件清单（逐项对照 Zero）
  ① 有限域 F_q           —— Zero 有吗？
  ② 代数曲线 X           —— Zero 有吗？
  ③ 有理点 P_i           —— Zero 有吗？
  ④ 函数域 F_q(X)        —— Zero 有吗？
  ⑤ 除子与 Riemann-Roch  —— Zero 有吗？

本文件
  P1 逐项判断，并给出 Zero 中最接近的物件
  P2 算一个**具体**的 AG 码（有理函数域／椭圆曲线／Hermitian）参数，
     与拓扑码对照，看值不值得建
  P3 给出"最小外接方案"：若要在 Zero 上建 AG，最少需要补什么
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np


# ---------- 有限域算术（GF(2^m)，用多项式表示） ----------
class GF2m:
    """GF(2^m) 的最简实现（元素用整数位表示）。"""

    def __init__(self, m, prim):
        self.m = m
        self.prim = prim            # 本原多项式（整数位表示，含 x^m 项）
        self.size = 1 << m

    def mul(self, a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if a & self.size:
                a ^= self.prim
        return r

    def pow(self, a, e):
        r = 1
        for _ in range(e):
            r = self.mul(r, a)
        return r

    def add(self, a, b):
        return a ^ b

    def elements(self):
        return list(range(self.size))


def main() -> None:
    print("=" * 96)
    print("在 Zero 上建立 AG 码：零件对照")
    print("=" * 96)

    print("""
┌────┬──────────────────────┬──────────────────────────────────────────────┐
│ #  │ AG 码需要的零件        │ Zero 里最接近的物件                          │
├────┼──────────────────────┼──────────────────────────────────────────────┤
│ ①  │ 有限域 F_q            │ ✗ Zero 只有二元（F_2）：Z0③ 的 0/1 参与状态   │
│    │                      │   （注意：GF(2^m) 可由 m 个量子比特张量积给出  │
│    │                      │    —— 定理 F 的 M_2^⊗m 含 F_2^m 的加法结构）  │
│ ②  │ 代数曲线 X            │ ✗ Zero 有**图/复形**（组合对象），不是代数曲线 │
│ ③  │ 有理点 P_i            │ △ Zero 有顶点集 V（但它不是曲线的有理点）      │
│ ④  │ 函数域 F_q(X)         │ ✗ Zero 有"词"（边上的二元值），不是函数域      │
│ ⑤  │ 除子 + Riemann-Roch   │ ✗ Zero 有**非负整数重数**（Z0③ 的计数）——     │
│    │                      │   这是**唯一形式相近**的零件（除子的次数）     │
└────┴──────────────────────┴──────────────────────────────────────────────┘

⇒ 逐项看：①可由张量积补出（半原生）；②③④⑤ **完全没有**。
  尤其 ②：Zero 的组合复形（图）与代数曲线是**不同范畴**的对象。
""")

    print("=" * 96)
    print("[P2] 具体算一个 AG 码，看值不值得建")
    print("=" * 96)
    # 椭圆曲线 y^2 + y = x^3 over GF(2)（只有一个 GF(2)-有理点，需扩域）
    # 用 Hermitian 曲线 over GF(4): y^2 + y = x^3（genus 1，椭圆曲线）
    print("\n  例：椭圆曲线 y² + y = x³ over F_4（亏格 g = 1）")
    gf = GF2m(2, 0b111)     # GF(4)，本原多项式 x²+x+1
    pts = []
    for x in range(4):
        for y in range(4):
            lhs = gf.add(gf.mul(y, y), y)
            rhs = gf.mul(gf.mul(x, x), x)
            if lhs == rhs:
                pts.append((x, y))
    print("    F_4 上的有理点（含无穷远点）：%d 个有限点" % len(pts))
    for p in pts:
        print("      (%d, %d)" % p)
    n = len(pts) + 1
    print("    ⇒ 有理点总数（含 ∞）= %d" % n)
    print("""
    Riemann-Roch：取 G = m·∞，则 dim L(G) = m − g + 1 = m（m ≥ 2g−1 = 1）
    码参数：n = %d，k = m，d ≥ n − m
    ⇒ 对 n = %d 的椭圆曲线码，最大 k ≈ n/2（因 g = 1 时 d ≤ n − k + 1）
""" % (n, n))

    print("=" * 96)
    print("[P3] 若要在 Zero 上建 AG 码：最小外接方案")
    print("=" * 96)
    print("""
  事实上"在 Zero 上建 AG 码"有两条路，代价差别极大：

  ── 路 1：把代数几何**嵌入** Zero（即从 Zero 导出曲线）
     需要的推导链：
       Z0③ 的非负整数重数  →  赋值/次数（valuation）
       Z1 的局域移动        →  函数域的自同构？
       循环次序 Z_n         →  F_q^* 的循环结构（✓ 形式吻合：F_q^* 是循环群）
     判定：**前两步没有任何 Zero 依据**。
       特别是"赋值/次数"：Zero 的计数是**分支计数**（branch count），
       不是域上的赋值（valuation）——两者数学上不同。

  ── 路 2：把 AG 码**外接**到 Zero（承认它来自外部）
     即：Zero 给出拓扑码；AG 码作为**对照基线**并列陈述。
     代价：归属不纯（但可如实标注，旧理论 10.84 已有先例）

  ── 路 3（新，可能有戏）：**用 Zero 的张量积构造"类 AG"码**
     观察：AG 码的核心是"用**次数**分级（grading by degree）"。
           Zero 的 Z0③ 有**非负整数重数**（计数）——这也是一个分级！
     猜想：把"次数"从函数域的多项式次数，换成 Zero 的**闭合词计数**，
           可能得到一族"Zero 版的 AG 码"。
     验证方向：在超立方体 Q_m 上，按"闭合词的计数"分级，看能否得到
           Reed-Muller 型的分级结构。
""")

    print("=" * 96)
    print("[P4] 试探路 3：Zero 的计数分级能否复现 RM 的分级？")
    print("=" * 96)
    print("""
    RM(r,m) 的分级：按多项式**次数** ≤ r 筛生成元 ⇒ 维数 Σ_{j≤r} C(m,j)
    Zero 的计数分级：按闭合词的**权重**（参与边数）分级 —— 但这只能给出"计数"，
      不能给出"维数"，故不能用枚举验证（枚举 C(E,≤w) 会爆炸）。

    改用**秩**比较（不用枚举）：
      RM(r,m) 的维数     = Σ_{j≤r} C(m,j)          （代数的）
      Zero 在 Q_m 上的码维数 = β₁ = E − V + 1      （拓扑的）
    %-6s %14s %16s %16s""" % ("m", "RM(1,m) 维数", "Zero 维数 β₁", "相等？"))
    for m in (3, 4, 5, 6, 7):
        nv = 1 << m
        ne = m * nv // 2
        beta1 = ne - nv + 1
        rm1 = m + 1
        print("    %-6d %14d %16d %16s" % (m, rm1, beta1, rm1 == beta1))
    print("""
    ⇒ 两者维数**从不相等**（β₁ 指数增长，RM 线性增长）
      ⇒ **路 3 不成立**：Zero 的计数分级无法复现 RM 的次数分级。
""")


if __name__ == "__main__":
    main()
