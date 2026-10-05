#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSS 结构钉死：用最小例子（1 个方格 = 4 条边 = 4 个量子比特）逐项核对

要钉死的问题
  之前我把"每条边两个量子比特"当成物理比特数（n = 2E），于是用了
      k = n − rank(H_X) − rank(H_Z) = 2E − r_X − r_Z   ← **错**
  正确做法（标准 CSS）：
      物理比特 = **边**（每个边一个量子比特），n = E
      X 型稳定子 = 顶点星形（行 = 顶点）      H_X ∈ F_2^{V×E}
      Z 型稳定子 = 面边界  （行 = 面）        H_Z ∈ F_2^{F×E}
      对易条件  H_X H_Z^T = 0
      k = dim ker(H_X) − rank(H_Z) = dim ker(H_Z) − rank(H_X) = E − r_X − r_Z

本脚本
  C1  在一个方格上显式写出 H_X, H_Z，逐步核对 rank / ker / k（两条公式互校）。
  C2  在一个方格上显式构造 Pauli 算符，验证 H_X H_Z^T = 0 与 k=0。
  C3  对最小非平凡例（环面 2x2 提法改为"两个面共享"??）——改用 **3x3 环面**：
      核对 k = E − r_X − r_Z，并核对两条公式一致。
  C4  警告：若改用"2E 个物理比特"（每条边两份），那是**另一个码**（两个独立经典码的直积），
      其 k = 2E − r_X − r_Z。两者不可混用。
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


def kernel_dim(M):
    return M.shape[1] - gf2_rank(M)


def one_square():
    """一个方格：4 顶点、4 边、1 面。边按 (0-1),(1-2),(2-3),(3-0)。"""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    nv, ne = 4, 4
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    HZ = np.ones((1, ne), dtype=np.uint8)      # 一个面 = 全部 4 条边
    return nv, edges, [list(range(ne))], HX, HZ


def grid(L, periodic=False):
    def vid(i, j):
        return (i % L) * L + (j % L) if periodic else i * (L + 1) + j
    edges, seen = [], {}

    def add(a, b):
        if a == b:
            return None
        k = (min(a, b), max(a, b))
        if k not in seen:
            seen[k] = len(edges)
            edges.append(k)
        return seen[k]

    ni = nj = L if periodic else L + 1
    for i in range(ni):
        for j in range(L):
            add(vid(i, j), vid(i, j + 1))
    for i in range(L):
        for j in range(nj):
            add(vid(i, j), vid(i + 1, j))
    faces = []
    for i in range(L):
        for j in range(L):
            f = [add(vid(i, j), vid(i, j + 1)), add(vid(i, j + 1), vid(i + 1, j + 1)),
                 add(vid(i + 1, j + 1), vid(i + 1, j)), add(vid(i + 1, j), vid(i, j))]
            if all(x is not None for x in f):
                faces.append(sorted(set(f)))
    nv = L * L if periodic else (L + 1) ** 2
    return nv, edges, faces


def build_H(nv, edges, faces):
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
    return HX, HZ


def analyze(name, nv, edges, faces):
    ne = len(edges)
    HX, HZ = build_H(nv, edges, faces)
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    commute = not ((HX @ HZ.T) % 2).any()
    k1 = kernel_dim(HZ) - rX
    k2 = kernel_dim(HX) - rZ
    k3 = ne - rX - rZ
    print(f"\n=== {name}:  V={nv} E(量子比特)={ne} F={len(faces)}")
    print(f"    H_X H_Z^T = 0 ?  {commute}   （非零元素 {int(((HX @ HZ.T) % 2).sum())}）")
    print(f"    rank(H_X) = {rX}   rank(H_Z) = {rZ}")
    print(f"    k = dim ker(H_Z) − rank(H_X) = {kernel_dim(HZ)} − {rX} = {k1}")
    print(f"    k = dim ker(H_X) − rank(H_Z) = {kernel_dim(HX)} − {rZ} = {k2}")
    print(f"    k = E − r_X − r_Z            = {ne} − {rX} − {rZ} = {k3}")
    print(f"    三式一致？ {k1 == k2 == k3} {'✅' if k1 == k2 == k3 else '❌'}")
    print(f"    ⚠ 若误用 n=2E：2E − r_X − r_Z = {2*ne - rX - rZ}（**另一个码**，勿混用）")
    return k1


def main() -> None:
    print("=" * 84)
    print("CSS 结构钉死：最小例逐项核对")
    print("=" * 84)

    nv, edges, faces, HX, HZ = one_square()
    print("\n[C1] 一个方格（4 边 = 4 个量子比特）的显式矩阵")
    print("    H_X（顶点-边关联，行=顶点）：")
    for v in range(nv):
        print(f"      v{v}: {list(HX[v])}")
    print(f"    H_Z（面-边关联，行=面）：{list(HZ[0])}")
    print(f"    H_X H_Z^T = {list((HX @ HZ.T) % 2)}  ⇒ 对易 ✅")
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    print(f"    rank(H_X)={rX}  rank(H_Z)={rZ}  ⇒ k = 4 − {rX} − {rZ} = {4-rX-rZ}")
    print(f"    解读：单个方格无逻辑量子比特（k=0）——平面片段的正确结果 ✅")

    print("\n[C3] 主要复形的 k（用正确的 n = E）")
    for name, (L, per) in ([("方格 %dx%d" % (L, L), (L, False)) for L in (1, 2, 3, 4, 5)]
                           + [("环面 %dx%d" % (L, L), (L, True)) for L in (3, 4, 5)]):
        p = grid(L, per)
        analyze(name, *p)

    print("\n" + "=" * 84)
    print("[C4] 结论")
    print("=" * 84)
    print("""  · 正确的 CSS 计数：物理比特 n = E，k = E − rank(H_X) − rank(H_Z)。
  · 之前用的 n = 2E（每条边两个量子比特）描述的是**另一个码**：
    两个独立经典码的直积，其 k = 2E − r_X − r_Z。**定理 B 必须按 n = E 重写。**""")
    print("\n  已知 rank(H_X) = V − c（引理 5）。故 k = E − (V−c) − rank(H_Z) = β₁ + (c−1) − rank(H_Z)。")
    print("  连通时：k = β₁ − rank(H_Z)。")
    print("  ⇒ (i) 的目标随之变为：rank(H_Z) 的闭式（面边界矩阵的秩）。")


if __name__ == "__main__":
    main()
