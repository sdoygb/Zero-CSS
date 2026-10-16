#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (c)：面集的唯一性是不是 L=3 的巧合？——DFS 精确搜索

上一脚本（zero_gap_c_closure.py）在环面 3×3 上穷举得出结论：
  满足「每边恰属 2 个面且覆盖全部边」的 4-圈子集**唯一** = 标准方格面。
本脚本用深度优先（每次选一个"未满 2"的边，尝试覆盖它的所有 4-圈）把同一判定推到 L=4、5，
因为 2^n 穷举在 L≥4 不可行。

关键：Euler 已一般地钉死 |F| = E−V+2−2g，且 k = 2g（与胞腔化无关）。
所以这里只判「面集本身是否唯一」，以及「不同面集下 d 是否变化」。

用法：python3 scripts/zero_gap_c_closure_dfs.py [Lmax]
"""
from __future__ import annotations

import itertools
import sys

L_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4


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
    HX = [[0] * ne for _ in range(nv)]
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v][e] = 1
    return HX


def cycles_of_weight(ne, HX, w):
    out = []
    for sub in itertools.combinations(range(ne), w):
        c = [0] * ne
        for e in sub:
            c[e] = 1
        if all(sum(HX[v][e] * c[e] for e in range(ne)) % 2 == 0 for v in range(len(HX))):
            out.append(frozenset(sub))
    return out


def dfs_cellulations(nv, ne, cand):
    """枚举 cand 的子集，使每条边恰属 2 个面，且每条边至少被覆盖。"""
    inc = {e: [] for e in range(ne)}
    for i, f in enumerate(cand):
        for e in f:
            inc[e].append(i)
    missing = [e for e in range(ne) if not inc[e]]
    if missing:
        return None, missing

    counts = [0] * ne
    chosen = []
    sols = []

    def rec():
        # 选一条"未满 2"的边，尝试用覆盖它的面补齐
        target = None
        for e in range(ne):
            if counts[e] < 2:
                target = e
                break
        if target is None:
            sols.append(list(chosen))
            return
        for i in inc[target]:
            if i in chosen:
                continue
            f = cand[i]
            if any(counts[e] >= 2 for e in f):
                continue
            for e in f:
                counts[e] += 1
            chosen.append(i)
            rec()
            chosen.pop()
            for e in f:
                counts[e] -= 1

    rec()
    return sols, None


def main() -> None:
    print("=" * 88)
    print("缺口 (c)：面集唯一性是 L=3 的巧合吗？（DFS 精确搜索）")
    print("=" * 88)
    for L in range(3, L_MAX + 1):
        nv, edges, faces = torus(L)
        ne = len(edges)
        HX = build_HX(nv, edges)
        beta1 = ne - nv + 1
        std = {frozenset(f) for f in faces}
        c4 = cycles_of_weight(ne, HX, 4)
        others = [c for c in c4 if c not in std]
        print(f"\n环面 {L}×{L}: V={nv} E={ne} β₁={beta1}  4-圈={len(c4)}"
              f"（标准面 {len(std)} + 其它 {len(others)}）")
        sols, missing = dfs_cellulations(nv, ne, list(c4))
        if missing is not None:
            print(f"     有边不被任何 4-圈覆盖：{missing[:5]}… ⇒ 4-圈胞腔化不存在")
            continue
        uniq = {frozenset(x) for x in sols}          # 去重：DFS 会重复到达同一解
        print(f"     满足「每边恰属 2 个面」的 4-圈胞腔化个数 = {len(uniq)}"
              f"（DFS 原始到达次数 {len(sols)}，去重后）")
        sols = [sorted(u) for u in uniq]
        std_idx = None
        for j, s in enumerate(sols):
            fs = {list(c4)[i] for i in s}
            if fs == std:
                std_idx = j + 1
        print(f"     其中标准方格面集出现于第 {std_idx} 个" if std_idx else "     ⚠ 标准面集不在解中")
        # |F| 与 Euler 预测
        f_pred = ne - nv + 2 - 2 * 1
        sizes = {len(s) for s in sols}
        print(f"     |F| 取值集 = {sorted(sizes)}；Euler 预测 F=E−V={f_pred}"
              f"  ⇒ {'一致 ✅' if sizes == {f_pred} else '不一致 ✗'}")
        if len(sols) == 1:
            print("     ⇒ **唯一**：面集被胞腔化条件唯一确定（k=2g、d 也随之确定）")
        else:
            print(f"     ⇒ **不唯一**：有 {len(sols)} 种胞腔化；k 仍被 Euler 钉死为 2，但 d 可能不同")


if __name__ == "__main__":
    main()
