#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引理 3 的无条件证明：面 = 闭合词 ⇒ 边界是 2-正则重图 ⇒ 星-面交集 ∈ {0,2}

Zero 依据
  面 = Z0③ 的**闭合词**（零和 sum w_i = 0）＝ 图上一条**闭合边序列**
      v_0 —e_1— v_1 —e_2— ... —e_k— v_k = v_0 .
  闭合 ⇒ 序列中**每个顶点出现的次数为偶**（进一次出一次）⇒ 该子（重）图的度数全偶。

断言（引理 3 无条件版）
  设 f 为闭合词，∂f 为其边集（重边计重数）。则对任意顶点 v：
      |star(v) ∩ ∂f| ∈ {0, 2} .
  一般地：|star(v) ∩ ∂f| = deg_{∂f}(v)，而 deg_{∂f}(v) ∈ {0} ∪ 2ℕ。

核验
  R1  构造闭合词（环、8 字、含重边的闭合词），检查其每个顶点度数 ∈ {0} ∪ 2ℕ。
  R2  构造**非闭合**的边序列（路径），检查它**违反**该性质 ⇒ 说明"闭合"是必要的。
  R3  在方格／环面上，逐 (v,f) 核验 |star(v)∩∂f| = deg_{∂f}(v) 且取值为 {0,2}。
  R4  反例搜索：随机闭合词中是否可能出现 deg ≥ 4？—— 出现（如"8 字"处 deg=4），
      故引理 3 的严格形式应为 "∈ {0} ∪ 2ℕ"，{0,2} 是本文件所用复形（简单方格/环面）的特殊情形。
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def degree_in(edges_subset):
    """给定边集（重边计重数），返回各顶点度数。"""
    deg = Counter()
    for (a, b) in edges_subset:
        deg[a] += 1
        deg[b] += 1
    return deg


def is_closed_walk(walk):
    """walk = [(v0,v1),(v1,v2),...]：闭合 ⟺ 首尾相接。"""
    if not walk:
        return True
    return all(walk[i][1] == walk[i + 1][0] for i in range(len(walk) - 1)) and \
           walk[0][0] == walk[-1][1]


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


def main() -> None:
    print("=" * 78)
    print("引理 3 的无条件证明：闭合词 ⇒ 度数全偶 ⇒ 交集 ∈ {0} ∪ 2ℕ")
    print("=" * 78)

    print("\n[R1] 闭合词 ⇒ 每个顶点度数为偶")
    closed = {
        "三角 (0-1-2-0)": [(0, 1), (1, 2), (2, 0)],
        "四边环 (0-1-2-3-0)": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "8 字 (0-1-2-0 与 0-3-4-0)": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
        "重边闭合 (0-1-0)": [(0, 1), (1, 0)],
    }
    for name, w in closed.items():
        deg = degree_in(w)
        allo = all(d % 2 == 0 for d in deg.values())
        print(f"    {name:28s} 闭合={is_closed_walk(w)}  度数={dict(sorted(deg.items()))}  "
              f"全偶={allo} {'✅' if allo else '❌'}")

    print("\n[R2] 非闭合序列 ⇒ 违反（说明'闭合'是必要条件）")
    open_walks = {
        "路径 (0-1-2)": [(0, 1), (1, 2)],
        "路径 (0-1-2-3)": [(0, 1), (1, 2), (2, 3)],
    }
    for name, w in open_walks.items():
        deg = degree_in(w)
        allo = all(d % 2 == 0 for d in deg.values())
        print(f"    {name:28s} 闭合={is_closed_walk(w)}  度数={dict(sorted(deg.items()))}  "
              f"全偶={allo} {'（端点度数为奇 ⇒ 违反）' if not allo else ''}")

    print("\n[R3] 方格／环面：|star(v)∩∂f| == deg_{∂f}(v) 且取值为 {0,2}")
    for name, (L, per) in {"方格 3x3": (3, False), "环面 3x3": (3, True),
                           "环面 5x5": (5, True)}.items():
        nv, edges, faces = grid(L, per)
        star = {v: {e for e, (a, b) in enumerate(edges) if v in (a, b)} for v in range(nv)}
        ok_eq, vals = True, set()
        for f in faces:
            sub = [edges[e] for e in f]
            deg = degree_in(sub)
            for v in range(nv):
                lhs = len(star[v] & set(f))
                rhs = deg.get(v, 0)
                if lhs != rhs:
                    ok_eq = False
                vals.add(lhs)
        print(f"    {name:12s} V={nv:3d} E={len(edges):3d} F={len(faces):3d}  "
              f"交集==度数? {ok_eq} {'✅' if ok_eq else '❌'}   取值集合={sorted(vals)}")

    print("\n[R4] 一般闭合词可能出现 deg ≥ 4（故严格形式是 {0} ∪ 2ℕ）")
    deg8 = degree_in(closed["8 字 (0-1-2-0 与 0-3-4-0)"])
    print(f"    '8 字'的度数 = {dict(sorted(deg8.items()))} ⇒ 顶点 0 的 deg = {deg8[0]} = 4")
    print(f"    ⟹ 引理 3 严格形式：|star(v)∩∂f| ∈ {{0}} ∪ 2ℕ；")
    print(f"       本文件所用复形（简单方格／环面）中面边界为**简单圈** ⇒ 退化为 {{0,2}}。")

    print("\n" + "=" * 78)
    print("引理 3（无条件版）陈述")
    print("=" * 78)
    print("""  设 f 为 Zero 的闭合词（Z0③：sum w_i = 0 的闭合边序列），∂f 为其边集。
  则对任意 v ∈ V：
      |star(v) ∩ ∂f| = deg_{∂f}(v)  ∈  {0} ∪ 2ℕ 。
  特别地，若 ∂f 是**简单圈**（本文件的方格／环面），则
      |star(v) ∩ ∂f| ∈ {0, 2} ⇒ A_v 与 B_f 对易（引理 0.6，符号 = +1）。""")
    print("\n证明：闭合 ⇒ 序列中每顶点进出配对 ⇒ 度数为偶（R1 核验）；")
    print("      非闭合 ⇒ 端点度数为奇（R2 核验）⇒ '闭合'是必要结构。")


if __name__ == "__main__":
    main()
