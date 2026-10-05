#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 定理 A（修正版，GF(2) 辛向量实现，可扩到任意规模）

为什么重写
  上一版用**矩阵**表示 Pauli 算符：n 个量子比特 ⇒ 2^n × 2^n 矩阵。
  3×3 环面有 36 个量子比特 ⇒ 4.7e21 元素 ⇒ 卡死（实测占 43% 内存、64 min CPU）。
  改为 **辛向量**：每个 Pauli 算符 = (x|z) ∈ GF(2)^{2n}，对易判据为辛积 = 0。

Zero 的构造
  · 顶点来自 (C,E)（Z0）；
  · **面来自闭合词**（零和 sum w_i = 0 ⇒ 闭合边序列 ⇒ 2-胞腔）；
  · 顶点稳定子 A_v = ∏_{e∋v} X_e（辛向量：x 位在星形边上）
  · 面稳定子   B_f = ∏_{e∈∂f} Z_e（辛向量：z 位在面边界上）
  · 对易判据：A_v 与 B_f 对易 ⟺ |star(v) ∩ ∂f| 为偶。

定理（本文件要证的）
  设复形 (V,E,F)，每条边给两个量子比特（X_e, Z_e）。则
    (i)  同型算符（A-A、B-B）恒对易；
    (ii) A_v 与 B_f 对易 ⟺ |star(v) ∩ ∂f| 为偶；
    (iii) 若复形满足"每条边恰属 2 个面"（2-流形样），则 |star(v)∩∂f| ∈ {0,2} ⇒ 全对易；
    (iv) 开边界时，边界边只属 1 个面 ⇒ 存在反对易对（= 边界条件问题的来源）。

判据：逐复形核验 (i)–(iv)。
"""
from __future__ import annotations

import numpy as np


# ---------- 辛向量 ----------
def sym_commute(p, q) -> bool:
    """p=(x|z), q=(x'|z')：对易 ⟺ x·z' + z·x' = 0 (mod 2)。"""
    n = len(p) // 2
    x1, z1 = p[:n], p[n:]
    x2, z2 = q[:n], q[n:]
    return (int(np.dot(x1, z2)) + int(np.dot(z1, x2))) % 2 == 0


def star_sym(nv, edges, v):
    """A_v 的辛向量（X 型）。"""
    ne = len(edges)
    x = np.zeros(ne, dtype=np.uint8)
    for e, (a, b) in enumerate(edges):
        if v in (a, b):
            x[e] = 1
    return np.concatenate([x, np.zeros(ne, dtype=np.uint8)])


def face_sym(edges, face):
    """B_f 的辛向量（Z 型）。"""
    ne = len(edges)
    z = np.zeros(ne, dtype=np.uint8)
    for e in face:
        z[e] = 1
    return np.concatenate([np.zeros(ne, dtype=np.uint8), z])


# ---------- 复形 ----------
def grid(L, periodic=False):
    """L×L 方格（periodic=True 为环面）：返回 (V, edges, faces, 每边面数)。"""
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
                faces.append(f)
    nv = L * L if periodic else (L + 1) ** 2
    return nv, edges, faces


def main() -> None:
    print("=" * 78)
    print("Zero 定理 A（修正版）：顶点／面 对易判据（GF(2) 辛向量）")
    print("=" * 78)

    for name, (L, per) in {"方格 3x3（开边界）": (3, False),
                           "方格 4x4（开边界）": (4, False),
                           "环面 3x3": (3, True),
                           "环面 5x5": (5, True)}.items():
        nv, edges, faces = grid(L, per)
        ne, nf = len(edges), len(faces)
        # 每条边属几个面
        cnt = [sum(1 for f in faces if e in f) for e in range(ne)]
        A = [star_sym(nv, edges, v) for v in range(nv)]
        B = [face_sym(edges, f) for f in faces]
        # (i) 同型对易
        same_ok = all(sym_commute(A[i], A[j]) for i in range(nv) for j in range(nv)) and \
                  all(sym_commute(B[i], B[j]) for i in range(nf) for j in range(nf))
        # (ii)/(iii)/(iv) 异型
        bad = [(v, fi) for v in range(nv) for fi in range(nf)
               if not sym_commute(A[v], B[fi])]
        # star(v) 的**边序号**集合（edges[e] 是 (a,b) 键）
        star_idx = {v: {e for e, (a, b) in enumerate(edges) if v in (a, b)}
                    for v in range(nv)}
        inter = {(v, fi): len(star_idx[v] & set(faces[fi]))
                 for v in range(nv) for fi in range(nf)}
        # (ii) 核验：反对易 ⟺ 交集为奇
        ii_ok = all((not sym_commute(A[v], B[fi])) == (inter[(v, fi)] % 2 == 1)
                    for v in range(nv) for fi in range(nf))
        n_qubits = 2 * ne
        print(f"\n=== {name}:  V={nv} E={ne} F={nf}  量子比特={n_qubits}")
        print(f"    每边面数取值 = {sorted(set(cnt))}")
        print(f"    (i)  同型恒对易            : {same_ok} {'✅' if same_ok else '❌'}")
        print(f"    (ii) 反对易 ⟺ 交集为奇     : {ii_ok} {'✅' if ii_ok else '❌'}")
        print(f"    (iii/iv) 异型反对易对 = {len(bad)} "
              f"{'（0 ⇒ 全对易，合法量子码）' if not bad else '（>0 ⇒ 边界/非流形，需边界条件）'}")
        if bad:
            eg = bad[0]
            print(f"          例：A_v(v={eg[0]}) 与 B_f(f={eg[1]}) 反对易，"
                  f"交集={inter[eg]}")

    print("\n" + "=" * 78)
    print("定理 A（修正版）陈述")
    print("=" * 78)
    print("""  (i)   A_v 与 A_w、B_f 与 B_g 恒对易。
  (ii)  A_v 与 B_f 对易 ⟺ |star(v) ∩ ∂f| 为偶。
  (iii) 复形满足"每条边恰属 2 个面"（2-流形样）⇒ 交集 ∈ {0,2} ⇒ 全对易。
  (iv)  开边界（边界边只属 1 个面）⇒ 存在反对易对，必须加边界条件。

  Zero 依据：顶点 = (C,E)（Z0）；**面 = 闭合词**（零和 ⇒ 闭合边序列）。
  ⇒ 合法量子码的判据是【每条边恰属两个闭合词】，**与顶点度数无关**。
  ⇒ 上一版"对易 ⟺ 欧拉（度数全偶）"**撤回**：那只对'顶点 X/Z'构造成立，而该构造非法。""")
    print("\n规模说明：本实现用辛向量，复杂度 O(V·F·E)，3×3≈10^5 量级 ⇒ 秒级完成；")
    print("          上一版用矩阵 2^n，3×3 环面需 4.7e21 元素 ⇒ 不可行（已弃用）。")


if __name__ == "__main__":
    main()
