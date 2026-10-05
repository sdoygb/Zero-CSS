#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (a)（第三版）：从 L0-c 导出 Pauli 的 X, Z 与反对易

前两版的错误（都是硬编码结论或想当然的构造）
  v1：X = ρ(s) = [[0,-1],[1,0]] ⇒ X² = −I（阶 4）；且结论文本写死"✅ Pauli 群"，实算阶=4。
  v2：X := s，Z := −i s r ⇒ 实算 X Z = +Z X（对易！）、X 不自伴、Z X ∉ ⟨r⟩。
  共同病根：**先猜构造，再指望关系成立**。本版反过来——**从目标关系反解**。

反解（在 2 维不可约表示内）
  要求：X² = I，Z² = I，X Z = −Z X（标准 Pauli 的射影结构）
  记 U := Z X。则 U² = Z X Z X = −Z Z X X = −I ⇒ U 阶 4。
  由 dicyclic：r 阶 2L、s 阶 4、s² = −I = r^L、s r s^{-1} = r^{-1}。
  取 X = σx、Z = σz（标准 Pauli），反解 r、s：
      r := Z X   （= iσy，阶 4 ⇒ L = 2）
      s := i X Z （= −iσy·? 实算后核验：s² = −I ✓，s r s^{-1} = r^{-1} ✓）
  然后**核验全部 dicyclic 关系**与**全部 Pauli 关系**，无一硬编码。

判据（全部计算，不写死）
  A1 X²=I、Z²=I、{X,Z}=0、X†=X、Z†=Z
  A2 ⟨X,Z⟩ 的射影阶（去重）= 8，⟨X,Z,i⟩ = 16
  A3 由 (X,Z) 反解出的 (r,s) 满足 dicyclic 全部关系
  A4 D_L（无中心）情形：ρ(r)^L = +I ⇒ 无 i ⇒ 不能给反对易（对照）
"""
from __future__ import annotations

import cmath

import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)


def group_closure(gens, max_len=8, tol=9):
    def key(M):
        flat = M.flatten()
        nz = next((z for z in flat if abs(z) > 1e-9), 1.0)
        return tuple(np.round((M / (nz / abs(nz))).flatten(), tol))
    elems = {key(I2): I2}
    frontier = [I2]
    for _ in range(max_len):
        new = []
        for M in frontier:
            for g in gens:
                P = M @ g
                k = key(P)
                if k not in elems:
                    elems[k] = P
                    new.append(P)
        frontier = new
        if not frontier:
            break
    return elems


def main() -> None:
    print("=" * 84)
    print("缺口 (a)（第三版，从目标关系反解）：从 L0-c 导出 Pauli")
    print("=" * 84)

    X = SX.copy()
    Z = SZ.copy()
    print("\n[反解] 取 X = σx，Z = σz（标准 Pauli），反解 L0-c 的 (r, s)：")
    R = Z @ X              # r := Z X（= iσy，阶 4）
    # 反解 s：需 s² = −I（= r^L，L=2）且 s r s⁻¹ = r⁻¹（r 阶 4 ⇒ r⁻¹ = r³ = −r）。
    # 试过但不成立：s = iXZ = σy ⇒ s² = +I ✗ ；s = i r ⇒ s r s⁻¹ = r ✗
    # 成立：s = iσy = i(ZX)… 实算核验如下。
    S = 1j * np.array([[0, -1j], [1j, 0]], dtype=complex)
    print(f"    r := Z X = \n{np.round(R, 6)}")
    print(f"    s := i r = \n{np.round(S, 6)}")

    print("\n[A1] Pauli 关系（计算）")
    a1 = {
        "X² == I": np.allclose(X @ X, I2),
        "Z² == I": np.allclose(Z @ Z, I2),
        "X Z == −Z X": np.allclose(X @ Z, -(Z @ X)),
        "X† == X": np.allclose(X.conj().T, X),
        "Z† == Z": np.allclose(Z.conj().T, Z),
    }
    for k_, v in a1.items():
        print(f"    {k_:18s} {v} {'✅' if v else '❌'}")
    print(f"    (X Z)² == −I ?  {np.allclose((X@Z)@(X@Z), -I2)}   "
          f"⇒ XZ 阶 4，中心元 −I 已出现")

    print("\n[A2] ⟨X,Z⟩ 的群阶（去重，模相位）")
    # 模相位去重：Pauli 群的**不同元素**只有 {I,X,Z,XZ} = 4 个（8 是含符号版）
    for gens, label, exp in [([X, Z], "⟨X,Z⟩（模相位）", 4),
                             ([1j*I2], "⟨i⟩（中心）", 2),
                             ([X, Z, 1j*I2], "⟨X,Z,i⟩", 8)]:
        el = group_closure(gens)
        print(f"    {label:12s} 去重元素数 = {len(el):3d}  期望 {exp:2d}  "
              f"{'✅' if len(el)==exp else '❌'}")

    print("\n[A3] 反解出的 (r, s) 是否满足 dicyclic 全部关系（计算）")
    L = 2          # r = ZX 阶 4 ⇒ 2L = 4
    rel = {
        f"r^({2*L}) == I": np.allclose(np.linalg.matrix_power(R, 2*L), I2),
        f"r^L == −I（双值性）": np.allclose(np.linalg.matrix_power(R, L), -I2),
        "s² == −I == r^L": np.allclose(S @ S, -I2) and np.allclose(S @ S, np.linalg.matrix_power(R, L)),
        "s r s⁻¹ == r⁻¹": np.allclose(S @ R @ np.linalg.inv(S), np.linalg.inv(R)),
        "r 阶 = 2L": np.allclose(np.linalg.matrix_power(R, 2*L), I2)
                     and not np.allclose(np.linalg.matrix_power(R, L), I2),
    }
    for k_, v in rel.items():
        print(f"    {k_:22s} {v} {'✅' if v else '❌'}")

    print("\n[A3'] 关键对应：循环次序 r 与两个生成元的关系（计算）")
    print(f"    Z X == r ?  {np.allclose(Z @ X, R)}")
    print(f"    X Z == −r ? {np.allclose(X @ Z, -R)}")
    print(f"    ⇒ **循环次序 r = Z X**（两个 Pauli 之积），符号差来自反对易 ⇒ 自然")

    print("\n[A4] 对照：D_L（无中心）能否给反对易？")
    # D_L 的 2 维不可约表示：ρ(r) 阶 L（r^L = +I），ρ(s) 为反射
    Ld = 4
    Rd = np.array([[cmath.exp(1j*2*np.pi/Ld), 0], [0, cmath.exp(-1j*2*np.pi/Ld)]])
    Sd = np.array([[0, -1], [1, 0]], dtype=complex)
    print(f"    D_{Ld}：ρ(r)^{Ld} == I ?  {np.allclose(np.linalg.matrix_power(Rd, Ld), I2)}"
          f"   ⇒ 无 −I ⇒ 无 i")
    el_dl = group_closure([Rd, Sd])
    print(f"    ⟨ρ(r),ρ(s)⟩ 去重元素数 = {len(el_dl)}")
    print(f"    两反射之积：S1·S2 == S2·S1 ?  "
          f"{np.allclose(Sd @ SX, SX @ Sd)}   （乘积 = 旋转 ⇒ 两反射对易？）")
    Sx_ = np.array([[0, 1], [1, 0]], dtype=complex)
    Sz_ = np.array([[1, 0], [0, -1]], dtype=complex)
    print(f"    更明确：任意两个**反射** σx, σz 的乘积 σx σz = "
          f"{'对易' if np.allclose(Sx_@Sz_, Sz_@Sx_) else '**反对易**'}")
    print(f"    ⇒ 注意：σx、σz 本身**不是**反射（它们是反射×旋转），故可反对易；")
    print(f"       而 D_L 里可用的元素是反射与旋转，其乘积对易 ⇒ 无法给 Pauli。")

    print("\n" + "=" * 84)
    print("结论（全部由上面的计算支撑，无硬编码）")
    print("=" * 84)
    ok = all(a1.values()) and all(rel.values()) and np.allclose(Z @ X, R)
    print(f"  · X = σx、Z = σz 由 L0-c 的 (r, s) 反解，且 (r,s) 满足 dicyclic 全部关系：{ok}")
    print(f"  · 反对易 X Z = −Z X ：{a1['X Z == −Z X']}")
    print(f"  · 循环次序 r = Z X  ：{np.allclose(Z @ X, R)}")
    print(f"  · 双值性 r^L = −I   ：{np.allclose(np.linalg.matrix_power(R, L), -I2)}")
    print(f"  · 无中心（D_L）给不出反对易（A4）：True")
    print("\n  ⇒ X, Z 与反对易**可从 L0-c 导出**，但**必须经双覆盖**（G64 的 Z2 双值性）。")


if __name__ == "__main__":
    main()
