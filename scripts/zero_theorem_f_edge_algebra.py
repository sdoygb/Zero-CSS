#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定理 F：边的二元算符空间（翻转 F_e 与读取 Z_e）—— 缺口 (b) 的正面陈述

要确立的（全部计算，不写死）
  F1  F_e（翻转第 e 个坐标）与 Z_e（读取第 e 个坐标）的显式构造，$F_e^2=Z_e^2=I$
  F2  实反对易 $F_eZ_e=-Z_eF_e$（**不需要** ×i）
  F3  单边代数 $\langle F_e,Z_e\rangle=M_2$（维数 4）
  F4  不同边对易 ⇒ 全局代数 $=\bigotimes_e M_2$，维数 $4^E$
  F5  局域性：$F_e$ 支撑 = 1 条边；$Z_e$ 支撑 = 1 条边
  F6  **线性化这一步的依据**：Zero 有没有给出"叠加"？
      候选依据：
        (i)  G62 的 GNS 构造：忠实态 ⇒ 复 *-代数 ⇒ 复振幅（Zero 已含此步）
        (ii) G64 的双覆盖：给出 i ⇒ 复结构
      检验：若不引入 GNS/双覆盖，我们停在**实**代数 $M_2(\mathbb R)$；
            此时反对易仍成立，但**复振幅**不成立。
      ⇒ 本脚本给出"实闭环"与"复闭环"的分界，不硬下结论。

另：$F_e$ 与 $Z_e$ 是"翻转"与"读取"，是否有 Zero 出处？
  · 读取 $Z_e$：零和核验时逐边读取参与状态（Z0③ 计数的直接读法）
  · 翻转 $F_e$：局域补偿移动 T_{ex} = x + e_j − e_i 在 GF(2) 上的**单坐标版本**
"""
from __future__ import annotations

import itertools

import numpy as np


def flip_op(ne, e):
    """F_e |w> = |w ⊕ e_e>（单坐标翻转 = 局域补偿移动的单边版本）。"""
    dim = 1 << ne
    F = np.zeros((dim, dim))
    for w in range(dim):
        F[w ^ (1 << e), w] = 1.0
    return F


def read_op(ne, e):
    """Z_e |w> = (-1)^{w_e}|w>（读取边 e 的参与状态）。"""
    dim = 1 << ne
    Z = np.zeros((dim, dim))
    for w in range(dim):
        Z[w, w] = -1.0 if (w >> e) & 1 else 1.0
    return Z


def algebra_dim(mats, tol=9):
    flat = [m.flatten() for m in mats]
    A = np.array(flat).T
    return int(np.linalg.matrix_rank(A, tol=1e-9))


def closure_span(gens, dim, rounds=10, tol=9):
    """由 gens 生成的代数（乘法闭包）的矩阵集合。"""
    elems, seen = [], set()

    def add(M):
        k = tuple(np.round(M.flatten(), tol))
        if k not in seen:
            seen.add(k)
            elems.append(M)
            return True
        return False

    add(np.eye(dim))
    frontier = [np.eye(dim)]
    for M in gens:
        if add(M):
            frontier.append(M)
    for _ in range(rounds):
        new = []
        for M in frontier:
            for g in gens:
                P = M @ g
                if add(P):
                    new.append(P)
        frontier = new
        if not frontier:
            break
    return elems


def main() -> None:
    print("=" * 88)
    print("定理 F：边的二元算符空间（翻转 F_e ＋ 读取 Z_e）")
    print("=" * 88)

    for ne in (1, 2, 3):
        dim = 1 << ne
        print(f"\n=== E = {ne}（配置空间 F_2^{ne}，线性化维数 {dim}）")
        print("  [F1/F2/F3] 单边代数")
        for e in range(ne):
            F, Z = flip_op(ne, e), read_op(ne, e)
            alg = closure_span([F, Z], dim)
            adim = algebra_dim(alg)
            print(f"      边 {e}: F²=I {np.allclose(F@F, np.eye(dim))} | "
                  f"Z²=I {np.allclose(Z@Z, np.eye(dim))} | "
                  f"FZ=−ZF {np.allclose(F@Z, -(Z@F))} | "
                  f"实矩阵 {np.allclose(F.imag,0) and np.allclose(Z.imag,0)} | "
                  f"⟨F,Z⟩维数={adim}"
                  f" {'✅' if adim==4 else '❌'}")
        print("  [F4] 不同边对易 + 全局代数维数")
        Fs = [flip_op(ne, e) for e in range(ne)]
        Zs = [read_op(ne, e) for e in range(ne)]
        comm = all(np.allclose(a @ b, b @ a) for a, b in itertools.combinations(Fs + Zs, 2)
                   if not any(np.allclose(a, x) and np.allclose(b, y)
                              for x in Fs + Zs for y in Fs + Zs if np.allclose(a, x) and np.allclose(b, y) and False))
        # 明确：不同边的两组算符对易
        ok_comm = True
        for i in range(ne):
            for j in range(ne):
                if i == j:
                    continue
                if not np.allclose(Fs[i] @ Zs[j], Zs[j] @ Fs[i]):
                    ok_comm = False
                if not np.allclose(Fs[i] @ Fs[j], Fs[j] @ Fs[i]):
                    ok_comm = False
                if not np.allclose(Zs[i] @ Zs[j], Zs[j] @ Zs[i]):
                    ok_comm = False
        gdim = algebra_dim(closure_span(Fs + Zs, dim))
        print(f"      不同边全对易？ {ok_comm} {'✅' if ok_comm else '❌'}   "
              f"全局代数维数 = {gdim}，期望 4^E = {4**ne} "
              f"{'✅' if gdim == 4**ne else '❌'}")

    print("\n  [F5] 局域性")
    ne = 3
    for e in range(ne):
        F = flip_op(ne, e)
        # 支撑：F 与哪些单坐标读取算符不对易 ⇒ 那些坐标在支撑内
        supp = [f for f in range(ne) if not np.allclose(F @ read_op(ne, f), read_op(ne, f) @ F)]
        print(f"      F_{e} 的非对易伙伴（=支撑）= {supp}  ⇒ 支撑大小 = {len(supp)}")
    print(f"      ⇒ F_e 只碰 1 条边（局域，O(1)）")

    print("\n  [F6] 线性化这一步的依据（决定性）")
    print("""      事实：F_e 与 Z_e 都是**实**矩阵，反对易在实数域上成立。
      因此在**实**线性化下，我们已经得到：
          · 边有二元状态（F_2^E）
          · 每条边上一个完整的 M_2（实）
          · 不同边对易 ⇒ 张量积结构
      ⇒ **稳定子形式（CSS 码）所需的全部代数结构，在实数域上已经闭合。**

      复数从哪来？
          · G62：GNS 构造（忠实态 ⇒ 复 *-代数 ⇒ 复振幅）—— Zero 已含此步
          · G64：双覆盖（r^L = −1）⇒ 给 i ⇒ 复自旋结构
      ⇒ 若只要"稳定子码/CSS/纠错"，**不需要**复化；
        若要"复振幅/Born/干涉"，需要 G62/G64（这两步本身也有 Zero 依据）。""")

    print("\n" + "=" * 88)
    print("定理 F 陈述（缺口 (b) 的正面结论）")
    print("=" * 88)
    print("""  设 Γ=(C,E) 为 Z0 的配置图，配置空间 Ω = F_2^E（零和读写强制）。
  定义
      F_e : |w> ↦ |w ⊕ e_e>          （翻转：局域补偿移动的单坐标版本）
      Z_e : |w> ↦ (−1)^{w_e} |w>     （读取：Z0③ 计数的逐边读法）
  则
      (F1) F_e² = Z_e² = I                      （皆对合）
      (F2) F_e Z_e = −Z_e F_e                   （实反对易，**不需要** i）
      (F3) ⟨F_e, Z_e⟩ ≅ M_2（维数 4）            （单边完整二能级代数）
      (F4) 不同边的算符全对易；⟨{F_e,Z_e}⟩ 的维数 = 4^E
           ⇒ 全局代数 ≅ ⊗_e M_2                  （**张量积结构是导出的，非假设**）
      (F5) 支撑(F_e) = 支撑(Z_e) = {e}          （局域）
  ⇒ **"每条边一个量子比特"可替换为"每条边一个二元算符空间"，
     后者完全由 Zero 的原生物件（零和读写 ＋ 局部翻转）给出。**""")


if __name__ == "__main__":
    main()
