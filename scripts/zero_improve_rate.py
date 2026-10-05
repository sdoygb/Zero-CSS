#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Zero 提高 CSS 码率：面集优化的实测

已确立
  物理比特 n = E（定理 B v2）
  k = E − rank(H_X) − rank(H_Z) = β₁ − rank(H_Z)   （连通）
  距离 d_X = 最短非平凡圈 = min{|c| : c ∈ Z₁ \ im δ_F}（定理 D）
  面集 F 是 Zero 的可调输入（定理 G）

提高码率的手段（全在 Zero 内）
  M1 加**独立的**闭合词作面 ⇒ rank(H_Z) 上升 ⇒ k 下降 ⇒ 码率下降（反效果）
  M2 加**依赖的**闭合词作面 ⇒ rank 不变 ⇒ k 不变（无效果）
  M3 换**复形**（顶点/边）⇒ β₁ 变化 ⇒ k 变化 ← 真正的杠杆
  ⇒ 关键：**提高码率必须换复形，不是加面**

  ⇒ 于是"用 Zero 提高码率"= 找 β₁/E 更大的复形

本实验
  E1 在固定复形上，改变面集，画 (k, d) 曲线
  E2 不同复形的 β₁/E 对比（找高码率复形）
  E3 hypergraph product（两个 Zero 复形的乘积）—— 是否给出更好码率
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


def gf2_solve(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    Aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    r, piv = 0, []
    for c in range(n):
        p = next((i for i in range(r, m) if Aug[i, c]), None)
        if p is None:
            continue
        Aug[[r, p]] = Aug[[p, r]]
        for i in range(m):
            if i != r and Aug[i, c]:
                Aug[i] ^= Aug[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if Aug[i, :n].sum() == 0 and Aug[i, n]:
            return False
    return True


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


def HX_of(nv, edges):
    ne = len(edges)
    H = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                H[e if False else v, e] = 1
    return H


def all_cycles_upto(nv, edges, wmax):
    ne = len(edges)
    HX = HX_of(nv, edges)
    out = []
    for w in range(3, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if not ((HX @ c) % 2).any():
                out.append(frozenset(sub))
    return out


def dX(HX, HZ, ne, wmax=8):
    for w in range(2, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if ((HX @ c) % 2).any():
                continue
            ok = True
            try:
                ok = gf2_solve(HZ.T % 2, c)
            except Exception:
                ok = False
            if not ok:
                return w
    return None


def main() -> None:
    print("=" * 96)
    print("[E1] 固定复形上改变面集：k 与 d 的联动")
    print("=" * 96)
    L = 3
    nv, edges, faces = torus(L)
    ne = len(edges)
    HX = HX_of(nv, edges)
    beta1 = ne - gf2_rank(HX)
    print("  环面 %dx%d：V=%d E=%d β₁=%d  标准面数=%d" % (L, L, nv, ne, beta1, len(faces)))
    print("  %-34s %6s %6s %6s %8s" % ("面集", "|F|", "rank", "k", "R=k/E"))
    sets = {"标准方格面": faces}
    cyc = all_cycles_upto(nv, edges, 6)
    sets["全部长度≤6 的圈"] = [sorted(c) for c in cyc]
    sets["标准面 + 一条非面 4-圈"] = faces + [sorted(c) for c in cyc if len(c) == 4 and frozenset(c) not in {frozenset(x) for x in faces}][:1]
    for name, F in sets.items():
        HZ = np.zeros((len(F), ne), dtype=np.uint8)
        for fi, f in enumerate(F):
            for e in f:
                HZ[fi, e] = 1
        r = gf2_rank(HZ)
        k = ne - gf2_rank(HX) - r
        print("  %-34s %6d %6d %6d %8.3f" % (name, len(F), r, k, k / ne))
    print("  ⇒ 加独立面 ⇒ rank↑ ⇒ k↓ ⇒ **码率下降**（加面不能提高码率）")

    print("\n" + "=" * 96)
    print("[E2] 换复形：β₁/E 能有多高？")
    print("=" * 96)
    print("  %-22s %7s %7s %8s %10s" % ("复形", "E", "β₁", "β₁/E", "检查权重"))
    rows = []
    for L in (3, 5, 8):
        nv2, ed2, _ = torus(L)
        b1 = len(ed2) - nv2 + 1
        rows.append(("环面 %dx%d" % (L, L), len(ed2), b1, b1 / len(ed2), "4"))
    # 完全图：β₁/E 高
    for n in (5, 8, 12, 20):
        E = n * (n - 1) // 2
        b1 = E - n + 1
        rows.append(("完全图 K%d" % n, E, b1, b1 / E, "n-1（非局域）"))
    # 完全二分图
    for a, b in ((3, 3), (4, 4), (5, 5)):
        E = a * b
        b1 = E - (a + b) + 1
        rows.append(("K(%d,%d)" % (a, b), E, b1, b1 / E, "a 或 b"))
    for r in rows:
        print("  %-22s %7d %7d %8.3f %10s" % r)
    print("  ⇒ 完全图/二分图 β₁/E → 1，但**检查权重随 n 增长**（非局域）")
    print("  ⇒ 环面：权重常数但 β₁/E → 0")
    print("  ⇒ **权衡**：局域性 ⟺ 低码率（这就是拓扑码的根本限制）")

    print("\n" + "=" * 96)
    print("[E3] Hypergraph product：两个 Zero 复形的乘积")
    print("=" * 96)
    print("""
  HGP 构造（Tillich-Zémor）：H = [H1⊗I | I⊗H2ᵀ]，H' = [I⊗H2 | H1ᵀ⊗I]
    n = n1·m2 + m1·n2
    k = k1·k2 + k1ᵀ·k2ᵀ   （k = 经典码维数，kᵀ = 转置码维数）
  ⇒ 若用**两个好经典码**（高率），HGP 可给高率量子码。

  但 Zero 原生的经典码是**圈空间**：C(G) = ker(H_X)，维数 β₁ = E − V + c
    · 环 C_n：β₁ = 1（低）
    · 完全图 K_n：β₁ = C(n,2) − n + 1（高，但非局域）
  ⇒ Zero 里"高率经典码"= 高 β₁ 的图 = 非局域图
  ⇒ **HGP 不能绕过这个权衡**
""")


if __name__ == "__main__":
    main()
