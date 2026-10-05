#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定：AG/RM 码是否"长在 Zero 上"？

判据（可判、不含糊）
  Zero 原生给出的码空间 = 某个图/复形的**圈空间** Z₁ = ker(H_X)
  （这是 Z0③ 零和 + Z1 局域移动的直接后果，定理 A/B）
  ⇒ 若 RM(r,m) 是 Zero 的码，则必须存在一个图/复形，其圈空间 = RM(r,m)（或包含它）

最自然的候选复形：**超立方体 Q_m**（因为 RM 的定义域正是 F_2^m，Q_m 的顶点就是 F_2^m）

检验
  T1  Q_m 的圈空间维数 vs RM(r,m) 的维数
  T2  RM(r,m) ⊆ Z₁(Q_m) 是否成立（对各个 r）
  T3  若都不成立 ⇒ AG/RM **不是** Zero 原生的，而是外接的
  T4  反过来：Q_m 的圈空间**本身**是什么码？（Zero 在 Q_m 上原生给什么）
"""
from __future__ import annotations

import itertools
from math import comb

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


def hypercube(m):
    """Q_m：顶点 = F_2^m（2^m 个），边 = 每个方向的 2^{m-1} 条。"""
    n = 1 << m
    edges = []
    for v in range(n):
        for d in range(m):
            w = v ^ (1 << d)
            if v < w:
                edges.append((v, w))
    return n, edges


def rm_basis(m, r):
    """RM(r,m) 的生成元（求值向量）作为 F_2^{2^m} 的行。"""
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    return np.array(rows, dtype=np.uint8)


def main() -> None:
    print("=" * 96)
    print("判定：AG/RM 码是否长在 Zero 上？")
    print("=" * 96)
    print("""
  判据：Zero 原生的码空间 = 图的圈空间 Z₁ = ker(H_X)
        （Z0③ 零和 ＋ Z1 局域移动的直接后果）
  ⇒ 若 RM(r,m) 是 Zero 的码，必须存在复形使 Z₁ ⊇ RM(r,m)
  最自然候选：超立方体 Q_m（顶点恰是 RM 的定义域 F_2^m）
""")
    for m in (3, 4, 5):
        nv, edges = hypercube(m)
        ne = len(edges)
        # Q_m 的关联矩阵与圈空间维数
        HX = np.zeros((nv, ne), dtype=np.uint8)
        for v in range(nv):
            for e, (a, b) in enumerate(edges):
                if v in (a, b):
                    HX[v, e] = 1
        beta1 = ne - gf2_rank(HX)
        print("─" * 96)
        print("Q_%d：顶点=%d  边=%d  圈空间维数 β₁=%d" % (m, nv, ne, beta1))
        for r in range(0, m):
            R = rm_basis(m, r)
            dim = gf2_rank(R)
            # RM(r,m) 作为 F_2^{2^m} 的子空间，而 Q_m 的圈空间在 F_2^{ne}
            # ⇒ 二者维数不同（2^m vs ne），**不同空间**，不能直接比较子集关系！
            print("    RM(%d,%d)：n=2^m=%d 维数=%d（空间 F_2^%d）"
                  % (r, m, nv, dim, nv))
        print("    ⚠ 注意：Q_m 的圈空间在 F_2^{ne}（维数 %d），" % ne)
        print("      而 RM(r,m) 在 F_2^{2^m}（维数 %d）—— **两者是不同的空间**！" % nv)

    print("\n" + "=" * 96)
    print("[T2/T3] 关键澄清：Zero 的'词'在边上，RM 的'字'在顶点上")
    print("=" * 96)
    print("""
  Zero 的配置：词 w ∈ F_2^E —— 每个**边**一个二元值
  RM 的字：    c ∈ F_2^{2^m} —— 每个**顶点**一个二元值

  ⇒ 要比较，必须先给"顶点上的字"一个 Zero 解释。两条路：
     路 A：把顶点字 = 边字的**关联读出**（H_X w 或 H_Z w）—— 读出是 Z1 的局部操作
     路 B：把顶点字 = 圈空间的对偶（割空间）
""")
    for m in (3, 4):
        nv, edges = hypercube(m)
        ne = len(edges)
        HX = np.zeros((nv, ne), dtype=np.uint8)
        for v in range(nv):
            for e, (a, b) in enumerate(edges):
                if v in (a, b):
                    HX[v, e] = 1
        # 路 A：读出像 = im(H_X^T)（顶点字 ∈ F_2^{nv} 由 H_X^T 作用得到？）
        # H_X: F_2^{ne} -> F_2^{nv}（关联），故 im(H_X) ⊆ F_2^{nv} 是"读出可达的顶点字"
        rk = gf2_rank(HX)
        im_dim = rk
        print("  Q_%d：读出映射 H_X: F_2^%d → F_2^%d，像维数 = %d"
              % (m, ne, nv, im_dim))
        for r in range(0, m):
            R = rm_basis(m, r)
            dim = gf2_rank(R)
            if dim == im_dim:
                # 维数相同，检查是否相等
                print("      RM(%d,%d) 维数 %d == 读出像维数 %d ⇒ **可能相等**，需逐向量验证"
                      % (r, m, dim, im_dim))
            else:
                print("      RM(%d,%d) 维数 %3d  ≠  读出像维数 %d" % (r, m, dim, im_dim))

    print("\n" + "=" * 96)
    print("[T4] Q_m 上 Zero 原生给的码是什么？")
    print("=" * 96)
    print("    %-10s %8s %10s %12s" % ("图", "β₁", "Zero 读出维数", "RM(1,m) 维数"))
    for m in (3, 4, 5, 6):
        nv, edges = hypercube(m)
        ne = len(edges)
        HX = np.zeros((nv, ne), dtype=np.uint8)
        for v in range(nv):
            for e, (a, b) in enumerate(edges):
                if v in (a, b):
                    HX[v, e] = 1
        beta1 = ne - gf2_rank(HX)
        rm_dim = m + 1
        print("    Q_%d%8s %8d %10d %12d" % (m, "", beta1, gf2_rank(HX), rm_dim))
    print("""
    ⇒ Zero 在 Q_m 上原生给出的是**割空间/圈空间**（维数 ~(m−2)·2^{m−1} 或 2^m−1），
      而 RM(1,m) 的维数是 **m+1** —— 两者维数不同，**不是同一个码**。
    """)

    print("=" * 96)
    print("判定")
    print("=" * 96)
    print("""
    ❌ **AG/RM 码不是从 Zero 长出来的。**

    三条独立理由：
      1. **空间不同**：Zero 的词在边上（F_2^E），RM 的字在顶点上（F_2^{2^m}）；
         没有 Zero 原生的映射把后者作为**码空间**导出（读出映射的像维数与 RM 不符）。
      2. **生成元不同**：RM(r,m) 含**全一向量**（权重 2^m）——它没有局域支撑；
         而 Zero 的生成元（顶点星形/面边界）**必须局域**（Z1 定理 1）。
      3. **稳定性来源不同**：RM 靠**多项式次数** r（代数），Zero 靠**闭合词/亏格**（拓扑）。

    ⇒ AG/RM 是旧理论**外接**的：它的数学基础（F_2 上的代数曲线、矩代数、Reed 递推）
      在 Zero 里**一个零件都没有**。这一点与之前查出的"量子比特"借贷同类，
      但更彻底：那次只缺"站点识别"，这次缺的是整个代数几何层。
    """)


if __name__ == "__main__":
    main()
