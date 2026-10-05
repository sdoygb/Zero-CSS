#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计：CSS 结构里哪些是 Zero 给的，哪些是我搬来的？

把"从 Zero 推导 CSS"拆成可判的几步，逐步检查**是否真的用到 Zero**。

步 1（Z0→经典码）  Z0③ 零和 sum w_i = 0 ⇒ 闭合成边序列 ⇒ 圈空间 = 经典线性码。
                    → 检验：圈空间是否是线性码；其参数是什么。
步 2（Pauli 从哪来？）CSS 需要"同一比特上 X 与 Z 反对易"。
                    Zero 的 Z2（±）是否**导出** X、Z 两个对易的算子类？
                    → 检验：Zero 有没有给出"寄存器"这个物件（可承载错误的自由度）。
步 3（对易判据）   H_X H_Z^T = 0 需要"面边界为偶"。
                    → 检验：把面换成**非闭合**词，对易是否仍成立？若仍成立，则 Zero 的
                      "闭合"条件**没起作用**（判据不依赖它）。
步 4（检查算子从哪来）顶点星形/面边界是**选定**的还是**导出**的？
                    → 检验：换一组检查（如顶点 X + 顶点 Z）能否也"合法"，若能，
                      则 CSS 结构不是 Zero 强制的。

判据：每一步都要能回答"若去掉 Zero 的这条，结论还成立吗？"
"""
from __future__ import annotations

import itertools

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


def main() -> None:
    L = 3
    nv, edges, faces = torus(L)
    ne = len(edges)
    print("=" * 84)
    print("审计：CSS 结构里哪些来自 Zero？（复形：环面 %dx%d）" % (L, L))
    print("=" * 84)

    # ---------- 步 1：Z0 零和 ⇒ 经典码 ----------
    print("\n[步 1] Z0③ 零和 ⇒ 圈空间，是否是**线性**码？")
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    # 枚举全部零和向量，检查对 GF(2) 加法封闭
    words = [w for w in itertools.product((0, 1), repeat=ne)
             if not ((HX @ np.array(w, dtype=np.uint8)) % 2).any()]
    closed_add = True
    for a, b in itertools.combinations(words[:60], 2):
        c = tuple(x ^ y for x, y in zip(a, b))
        if c not in set(words):
            closed_add = False
            break
    print(f"    零和向量个数 = {len(words)} = 2^{gf2_rank(np.array(words, dtype=np.uint8).T)}"
          f"  → 线性？ {closed_add}")
    print(f"    经典码参数：n={ne}, k_classical={ne - gf2_rank(HX)}, d={min(sum(w) for w in words if any(w))}")
    print(f"    → **这一步确实由 Z0③ 给出**：零和 = 校验约束，圈空间 = 经典线性码 ✅")

    # ---------- 步 2：Pauli 从哪来 ----------
    print("\n[步 2] CSS 需要 X、Z 两类算子。Zero 给出'寄存器'了吗？")
    print("    Zero 的 Z0 给的是：**构型（词）**、计数、局域补偿移动。")
    print("    Zero 的 Z2（± 号）给的是：符号，不是'可承载错误的自由度'。")
    print("    ⇒ '每条边一个量子比特' 是**搬进来的**，Zero 里没有'量子比特'这个物件。")
    print("    ⇒ 判定：**未从 Zero 导出** ❌（本步是借用）")

    # ---------- 步 3：对易判据是否依赖'闭合' ----------
    print("\n[步 3] 把'面'换成**非闭合**词，对易还成立吗？（若成立 ⇒ 闭合没起作用）")
    # 造一个非闭合的"面"：路径 0-1-2（三顶点两边的路径）
    path_edges = [e for e, (a, b) in enumerate(edges) if a in (0, 1) and b in (1, 2)][:2]
    print(f"    取一个非闭合词（路径）：边 {path_edges}")
    HZ_bad = np.zeros((1, ne), dtype=np.uint8)
    for e in path_edges:
        HZ_bad[0, e] = 1
    prod = (HX @ HZ_bad.T) % 2
    nz = int(prod.sum())
    print(f"    H_X H_Z^T 非零元素 = {nz}"
          f"  ⇒ 对易{'成立（闭合没起作用）' if nz == 0 else '**不成立**（闭合是必要的）'}")
    print("    → 判定：" + ("闭合**确实必要** ✅" if nz else "闭合不必要 ❌"))

    # ---------- 步 4：检查算子是选定的还是导出的 ----------
    print("\n[步 4] 检查算子是**选定**的还是**导出**的？")
    # 变体 A：顶点 X + 顶点 Z（都放顶点）
    HZ_v = HX.copy()     # Z 检查也放顶点
    prodA = (HX @ HZ_v.T) % 2
    # 变体 B：顶点 X + 面 Z（本构造）
    HZ_f = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ_f[fi, e] = 1
    prodB = (HX @ HZ_f.T) % 2
    print(f"    变体 A（顶点 X + 顶点 Z）：H_X H_Z^T 非零元素 = {int(prodA.sum())} "
          f"⇒ {'合法' if not prodA.any() else '**不合法**（相邻顶点反对易）'}")
    print(f"    变体 B（顶点 X + 面   Z）：H_X H_Z^T 非零元素 = {int(prodB.sum())} "
          f"⇒ {'合法' if not prodB.any() else '不合法'}")
    kA = ne - gf2_rank(HX) - gf2_rank(HZ_v) if not prodA.any() else None
    kB = ne - gf2_rank(HX) - gf2_rank(HZ_f)
    print(f"    变体 B 的参数：k = {kB}")
    print("    → 变体 A 不合法 ⇒ '顶点 X + 面 Z' **不是任意选择**，是被对易条件**约束**出来的")
    print("      但：'面'这个物件本身是**输入**（取哪些闭合作面由我们定）⇒ 部分导出、部分选定")

    print("\n" + "=" * 84)
    print("审计结论")
    print("=" * 84)
    print("""  由 Zero 导出（真的）：
    · 零和 ⇒ 圈空间 ⇒ **经典**线性码（步 1 ✅）
    · 面必须闭合，否则对易失败 ⇒ 这是 Z0③ 的零和条件（步 3 ✅）
    · 顶点/面检查的**搭配方式**被对易条件约束（步 4：顶点X+顶点Z 不合法）

  借用（不是从 Zero 导出）：
    · 量子比特（每条边一个寄存器）——步 2 ❌
    · Pauli 代数（X、Z 及其反对易）——借用，未从 Zero 的 Z2(±) 导出
    · 稳定子形式、码空间维数公式 dim C = n − r——教科书结果
    · CSS 这一"结构名"本身

  ⇒ **严格地说：不是"从 Zero 推导出 CSS"。**
     准确表述是：**"Zero 的零和约束给出一族经典码，并在配上量子比特与 Pauli 代数后，
     其自然提升是一个合法 CSS 码（toric code）。"**""")
    print("\n  要变成真推导，需要补的工作：")
    print("    (a) 从 Zero 的 Z2（± 号）导出 Pauli 的 X/Z 两类算子；")
    print("    (b) 说明'边 = 一个自由度'在 Zero 里由什么物件承担；")
    print("    (c) 把'面'的选择规则从 Zero 导出（而不是任取闭合作面）。")


if __name__ == "__main__":
    main()
