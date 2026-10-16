#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (c) 续攻：面集被"闭曲面胞腔化"条件钉死到什么程度？

定理 G（TheoG_face_selection.md）的结论：
  ❌ 面集不能从零和／循环次序／局域补偿移动／长度极小性四个原语唯一导出；
  ✅ 但若每边恰属偶数个面，则 ∑_f ∂f = 0 ⇒ rank(H_Z)=|F|−1 ⇒ k=β₁−|F|+1。
  §3 实测：放松到任意面集，k 可在 [0, β₁] 之间调 —— 故"面集是输入"。

本脚本问一个更紧的问题（定理 G 没做的）：
  若把"面集"限制为**闭曲面胞腔化**（每边恰属 2 个面、每顶点的面构成盘），
  面集还能不能变？k 还能不能变？d 能不能变？

三组判据（全部可判、可复算）：
  T1 Euler 钉死 |F|：V−E+F=2−2g ⇒ F=E−V+2−2g；环面 g=1 ⇒ F=E−V=9（L=3）
     ⇒ k=β₁−(F−1)=2g，**与取哪个胞腔化无关**。
  T2 穷举 4-圈子集，筛"每边恰属 2 个面"的合法胞腔化，逐一算 k：
     · 全为 k=2 ⟹ 面集选择**不影响 k**（缺口对 k 闭合）。
     · 若标准方格面是**唯一**合法者 ⟹ 面集被条件**唯一**定出（缺口正式闭合）。
  T3 合法胞腔化之间 d_X／d_Z 是否变化 ⟹ 缺口到底还剩多大（只影响 d？）。

用法：python3 scripts/zero_gap_c_closure.py
退出码 0 = 全部判据通过（断言见 check()）
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter

import numpy as np

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


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


def row_space(M):
    """GF(2) 行空间（返回集合，便于判成员）。"""
    M = M.copy() % 2
    rows = [tuple(r) for r in M]
    basis = []
    for r in rows:
        v = list(r)
        for b in basis:
            v = [x ^ y for x, y in zip(v, b)]
        if any(v):
            p = v.index(1)
            basis.append(v)
    span = {tuple([0] * M.shape[1])}
    for b in basis:
        span |= {tuple(x ^ y for x, y in zip(s, b)) for s in span}
    return span


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


def build_HX(nv, edges):
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    return HX


def cycles_upto(ne, HX, wmax):
    out = []
    for w in range(3, wmax + 1):
        for sub in itertools.combinations(range(ne), w):
            c = np.zeros(ne, dtype=np.uint8)
            for e in sub:
                c[e] = 1
            if not ((HX @ c) % 2).any():
                out.append(frozenset(sub))
    return out


def codespace_params(faces, ne, HX, beta1):
    """给定面集：rank(H_Z)、k、d_X、d_Z（精确枚举，小规模）。

    X-逻辑：ker(H_Z) 中不属于 rowspace(H_X) 的最小重量向量。
    Z-逻辑：ker(H_X) 中不属于 rowspace(H_Z) 的最小重量向量。
    """
    F = len(faces)
    HZ = np.zeros((F, ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    rHZ = gf2_rank(HZ)
    k = beta1 - rHZ
    kerHX = kernel_basis(HX)
    kerHZ = kernel_basis(HZ)
    rsX = row_space(HX)
    rsZ = row_space(HZ)
    dX = min_wt_outside(kerHZ, rsX)
    dZ = min_wt_outside(kerHX, rsZ)
    return rHZ, k, dX, dZ


def kernel_basis(M):
    """GF(2) 零空间的基（M x = 0）。"""
    M = M.copy() % 2
    rows, cols = M.shape
    pivots, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for fc in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            v[pc] = M[i, fc] % 2
        basis.append(v)
    return basis


def min_wt_outside(kbasis, span):
    """在 kbasis 张成的空间里，找不属于 span 的最小重量向量。"""
    if not kbasis:
        return None
    best = None
    n = len(kbasis)
    for mask in range(1, 1 << n):
        v = np.zeros(len(kbasis[0]), dtype=np.uint8)
        for i in range(n):
            if mask >> i & 1:
                v ^= kbasis[i]
        if tuple(v) in span:
            continue
        w = int(v.sum())
        if best is None or w < best:
            best = w
    return best


def main() -> None:
    print("=" * 92)
    print("缺口 (c) 续攻：闭曲面胞腔化条件把面集钉死到什么程度？")
    print("=" * 92)
    L = 3
    nv, edges, faces = torus(L)
    ne = len(edges)
    HX = build_HX(nv, edges)
    beta1 = ne - nv + 1
    std = [frozenset(f) for f in faces]
    print(f"\n环面 {L}×{L}:  V={nv}  E={ne}  β₁=E−V+1={beta1}  标准面集 |F|={len(std)}")

    # ── T1 Euler 钉死 |F| ────────────────────────────────────────────────
    print("\n[T1] Euler 关系钉死 |F|（闭曲面 g=1 ⇒ χ=0）")
    chi = nv - ne + len(std)
    print(f"     标准胞腔化：χ = V−E+F = {nv}−{ne}+{len(std)} = {chi}")
    check("标准胞腔化的 χ=0（环面）", chi == 0)
    F_pred = ne - nv + 2 - 2 * 1
    print(f"     χ=0 ⇒ F = E−V = {ne-nv}（预测）")
    check("标准 |F| 与 Euler 预测一致", len(std) == F_pred, f"{len(std)} vs {F_pred}")
    k_pred = beta1 - (F_pred - 1)
    print(f"     引理 G1（每边偶数面）⇒ rank(H_Z)=F−1 ⇒ k = β₁−(F−1) = {k_pred} = 2g ✅")
    check("Euler + 引理 G1 预测 k=2g=2", k_pred == 2)

    # ── T2 穷举 4-圈子集，筛合法胞腔化 ───────────────────────────────────
    print("\n[T2] 穷举全部 4-圈子集，筛「每边恰属 2 个面」的合法闭曲面胞腔化")
    c4 = cycles_upto(ne, HX, 4)
    print(f"     4-圈总数 = {len(c4)}（标准面 {len(std)} 个 + 其它 {len(c4)-len(std)} 个）")
    check("4-圈数 = 15（定理 G 记录值）", len(c4) == 15, f"实测 {len(c4)}")

    def edge_counts(fs):
        c = Counter()
        for f in fs:
            for e in f:
                c[e] += 1
        return c

    legal = []
    for r in range(1, len(c4) + 1):
        for combo in itertools.combinations(c4, r):
            c = edge_counts(combo)
            if all(v == 2 for v in c.values()) and len(c) == ne:
                legal.append(combo)
    print(f"     「每边恰属 2 个面且覆盖全部边」的子集数 = {len(legal)}")
    for i, fs in enumerate(legal):
        if set(fs) == set(std):
            print(f"       其中第 {i+1} 个 = 标准方格面集 ✅")

    std_is_legal = any(set(fs) == set(std) for fs in legal)
    check("标准方格面集合法（每边恰属 2 个面）", std_is_legal)
    print(f"\n     逐一算 k（合法胞腔化之间）:")
    ks, ds = [], []
    for i, fs in enumerate(legal):
        rhz, k, dX, dZ = codespace_params(list(fs), ne, HX, beta1)
        ks.append(k)
        ds.append((dX, dZ))
        is_std = "  ← 标准方格面" if set(fs) == set(std) else ""
        print(f"       #{i+1}: |F|={len(fs):2d} rank(H_Z)={rhz:2d} k={k}  d_X={dX} d_Z={dZ}{is_std}")
    check("合法胞腔化全为 k=2", set(ks) == {2}, f"实测 k 值集 {sorted(set(ks))}")
    check("合法胞腔化 |F| 全为 E−V=9", all(len(fs) == ne - nv for fs in legal))
    check("标准方格面是唯一合法胞腔化（缺口闭合）", len(legal) == 1,
          f"实测 {len(legal)} 个合法者")

    # ── T3 d 是否随胞腔化变 ─────────────────────────────────────────────
    print("\n[T3] 缺口还剩多大：合法胞腔化之间 d 是否变化？")
    check("合法胞腔化的 d_X 全相同", len({a for a, _ in ds}) == 1, f"d_X ∈ {sorted({a for a,_ in ds})}")
    check("合法胞腔化的 d_Z 全相同", len({b for _, b in ds}) == 1, f"d_Z ∈ {sorted({b for _,b in ds})}")
    if std_is_legal:
        rhz, k, dX, dZ = codespace_params(std, ne, HX, beta1)
        print(f"     标准胞腔化：k={k} d_X={dX} d_Z={dZ}（环面 L=3 的 toric code [[18,2,3]] 预期）")
        check("标准胞腔化 k=2", k == 2)
        check("标准胞腔化 d_X=3", dX == 3, f"实测 {dX}")
        check("标准胞腔化 d_Z=3", dZ == 3, f"实测 {dZ}")

    # ── 对照：放松到「每边偶数」但仍非胞腔化 ─────────────────────────────
    print("\n[对照] 放松条件后的 k（说明「每边恰属 2」这一条有多紧）")
    for name, sel in [("只取标准面（去掉一个）", std[:-1]),
                      ("标准面 + 全部其它 4-圈", std + [c for c in c4 if c not in set(std)])]:
        M = np.zeros((len(sel), ne), dtype=np.uint8)
        for fi, f in enumerate(sel):
            for e in f:
                M[fi, e] = 1
        print(f"     {name:26s} |F|={len(sel):3d}  rank={gf2_rank(M):2d}  k={beta1-gf2_rank(M)}")

    print("\n" + "=" * 92)
    print(f"结果：合法闭曲面胞腔化 {len(legal)} 个；k 值集 {sorted(set(ks))}；d 值集 {sorted(set(ds))}")
    print(f"不符项：{FAIL}")
    print("=" * 92)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
