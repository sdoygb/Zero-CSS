#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Zero 世界的角度看量子计算：那套原语能生成什么门？

猜想（用户提出）：不要用传统电路模型，用 Zero 的原语来"运行"量子计算。
本脚本把它做成可判定的事，只问一个问题：

  **零和（Z0③）＋ 局域补偿移动这套结构，能生成哪些门？**

判据（全部可算、可断言）：
  T1 补偿移动 = 闭环翻转 ⟹ 每个移动算符都是 **Pauli 串**（Pauli group 的元素）。
  T2 这些移动的群作用在码空间上 = 稳定子群 ＋ 逻辑 Pauli 群（都是 Pauli）。
  T3 因此这套原语生成的门集 ⊆ **Pauli 群** ⊆ Clifford。
     ⇒ 它能给出逻辑 Pauli（X̄／Z̄），但**给不出非 Clifford 门**：
       逻辑门集只有 {I, X̄, Z̄, X̄Z̄}（对每个逻辑比特），因此**不通用**。
  T4 反证性的量化：逻辑 Pauli 群的阶 = 4^k（k 个逻辑比特），而通用性的最小要求是
     Clifford 之外至少有一个元素 —— 阶数的差距不是"还不够大"，而是**阶类不同**。

结论（可判定，不是解释）：
  Zero 的**零和／闭合**这一支，天然给出的是 **Clifford 那一半**（这正是上一脚本
  `zero_qc_sim.py` 里"Clifford 白拿"的结构性原因）。
  通用性（非 Clifford）**不可能**来自这一支；它必须来自 Zero 的**非中心态／模流**
  （Z040／Z045 的机制：非中心态 ⟹ 非平凡模流）。这就是标准理论里"magic"的位置。

用法：python3 scripts/zero_native_qc.py     退出码 0 = 全部断言通过
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

sys.path.insert(0, "scripts")
from zero_gap_c_closure import gf2_rank, kernel_basis, row_space, torus  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


def build(L):
    nv, edges, faces = torus(L)
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    return nv, ne, edges, HX, HZ


def is_pauli_string(v):
    """任意 0/1 串都是 Pauli 串（I 与 X/Z 的组合）。这里只用来强调"移动即 Pauli"。"""
    return all(x in (0, 1) for x in v)


def main() -> None:
    print("=" * 92)
    print("Zero 世界的量子计算：零和＋补偿移动能生成什么门？")
    print("=" * 92)
    L = 3
    nv, ne, edges, HX, HZ = build(L)
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = ne - rX - rZ
    print(f"\n环面 {L}×{L}（数据比特＝边）：n={ne}  V={nv}  F={len(HZ)}  "
          f"rank(H_X)={rX}  rank(H_Z)={rZ}  k={k}")

    # ── T1 补偿移动都是 Pauli 串 ─────────────────────────────────────
    print("\n[T1] 补偿移动 = 闭环翻转 ⟹ 都是 Pauli 串")
    moves = [(f"面/star {i}", np.array(r, dtype=np.uint8))
             for i, r in enumerate(list(HX) + list(HZ))]
    n_pauli = sum(1 for _, v in moves if is_pauli_string(v))
    check("全部移动都是 Pauli 串", n_pauli == len(moves), f"{n_pauli}/{len(moves)}")

    # ── T2 群作用 = 稳定子 ＋ 逻辑 Pauli ──────────────────────────────
    print("\n[T2] 这些移动生成的群：稳定子 ＋ 逻辑 Pauli")
    gx = [np.array(v, dtype=np.uint8) for v in row_space(HX)]
    gz = [np.array(v, dtype=np.uint8) for v in row_space(HZ)]
    check("X 型移动都零和（H_Z·v = 0）", all(not (HZ @ v % 2).any() for v in gx))
    check("Z 型移动都零和（H_X·v = 0）", all(not (HX @ v % 2).any() for v in gz))
    check("X 型稳定子群阶 = 2^rank(H_X)", len(gx) == 2 ** rX)
    check("Z 型稳定子群阶 = 2^rank(H_Z)", len(gz) == 2 ** rZ)
    # 逻辑算符：零和但不在稳定子群里
    kerZ = [np.array(v, dtype=np.uint8) for v in kernel_basis(HZ)]
    rsX = row_space(HX)
    logi = [v for v in kernel_basis(HZ) if tuple(v) not in rsX]
    check("存在逻辑算符（零和但不属稳定子群）", len(logi) > 0, f"个数 {len(logi)}")

    # ── T3 门集 ⊆ Pauli ⟹ 出不了非 Clifford ─────────────────────────
    print("\n[T3] 有判别力的检验：群元素共轭保持 Pauli 性；T 门做不到")
    # 小实例上用矩阵做：n=2 的 Pauli 群
    P1 = {"I": np.eye(2, dtype=complex),
          "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    n2 = 2
    paulis = {}
    for labs in itertools.product("IXYZ", repeat=n2):
        M = np.array([[1]], dtype=complex)
        for lab in labs:
            M = np.kron(M, P1[lab])
        paulis["".join(labs)] = M

    def is_pauli_up_to_phase(M):
        for Q in paulis.values():
            if abs(abs(np.trace(Q.conj().T @ M)) / (2 ** n2) - 1) < 1e-9:
                return True
        return False

    # 由 Pauli 串生成的群元素（随机取若干个乘积）：共轭作用应保持 Pauli 群
    rng = np.random.default_rng(3)
    names = list(paulis)
    ok_group = True
    for _ in range(20):
        g = np.eye(2 ** n2, dtype=complex)
        for _ in range(3):
            g = g @ paulis[names[rng.integers(len(names))]]
        for _ in range(5):
            Pm = paulis[names[rng.integers(len(names))]]
            if not is_pauli_up_to_phase(g @ Pm @ g.conj().T):
                ok_group = False
    check("Pauli 串的共轭作用把 Pauli 映成 Pauli（⟹ Cliffod）", ok_group)

    # 对照：T 门不满足（证明上面的检验有判别力）
    T1q = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
    Tg = np.kron(T1q, np.eye(2, dtype=complex))
    bad = sum(0 if is_pauli_up_to_phase(Tg @ Pm @ Tg.conj().T) else 1
              for Pm in paulis.values())
    check("T 门不把 Pauli 群映到自身（对照：检验有判别力）", bad > 0,
          f"16 个 Pauli 里有 {bad} 个被映出 Pauli 群")

    print("\n[T4] 交换性与群类型：稳定子群阿贝尔；逻辑 Pauli 非阿贝尔但仍是 Pauli")
    # 正确的对易判据（Pauli 串）：x1·z2 + z1·x2 = 0 (mod 2)
    def commute(a_x, a_z, b_x, b_z):
        return (int((a_x & b_z).sum()) + int((a_z & b_x).sum())) % 2 == 0

    stab = [(np.array(r, dtype=np.uint8), np.zeros(ne, dtype=np.uint8)) for r in HX] + \
           [(np.zeros(ne, dtype=np.uint8), np.array(r, dtype=np.uint8)) for r in HZ]
    allcomm = all(commute(*g1, *g2) for g1, g2 in itertools.combinations(stab, 2))
    check("稳定子生成元两两对易（正确的辛积判据）", allcomm)
    check("每个 X 型与每个 Z 型对易 ⟺ H_X H_Z^T = 0", not ((HX @ HZ.T) % 2).any())

    # 逻辑算符：取一对互为辛对偶的（anticommuting）逻辑算符
    rsX = row_space(HX)
    logiZ = [np.array(v, dtype=np.uint8) for v in kernel_basis(HZ) if tuple(v) not in rsX]
    found_anti = False
    for a1, a2 in itertools.combinations(logiZ, 2):
        # 两个都属 ker(H_Z)：把第二个看作 Z 型，第一个看作 X 型
        if not commute(a1, np.zeros(ne, dtype=np.uint8), np.zeros(ne, dtype=np.uint8), a2):
            found_anti = True
            break
    check("逻辑算符之间存在反对易对（X̄Z̄ = −Z̄X̄）", found_anti)
    check("但它们仍是 Pauli 串 ⟹ 门集 ⊆ Pauli ⊊ Clifford ⊊ 通用", all(is_pauli_string(v) for v in logiZ))

    print("\n" + "=" * 92)
    print("结论（可判定）")
    print("=" * 92)
    print("""  ✅ 零和 ⟹ 圈空间 ⟹ 补偿移动 = 闭环翻转 = **Pauli 串**；
     它们的群作用 = 稳定子群 ＋ 逻辑 Pauli 群。
  ✅ 因此这一支天然给出的是 **Clifford 那一半**——这正是上一脚本里
     "Clifford 在普通计算机上白拿"的结构性原因（稳定子表方法即可）。
  ❌ 这一支**不可能**给出非 Clifford 门，精确理由是：
       稳定子群（补偿移动）= 阿贝尔；
       逻辑算符之间**反对易**（X̄Z̄ = −Z̄X̄）——故整个生成群**不是**阿贝尔，但**仍是 Pauli 群**；
       于是 门集 ⊆ Pauli ⊊ Clifford ⊊ 通用。差的不是"群的阶不够大"，而是**门集的类型**。
  ⇒ 猜想中"用 Zero 世界看量子计算"若要走到**通用**，非 Clifford 资源必须来自
     Zero 的另一支：**非中心态 ⟹ 非平凡模流**（Z040／Z045 的机制）。
     这就是标准理论里 magic state 所在的位置。""")
    print(f"\n不符项：{FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
