#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 距离：结构公式（先在小例验证公式，再用于 4-正则）

候选公式（文献标准结果的组合形式）
  HGP 的距离由三部分取最小：
    (1) 最短的非平凡 X 逻辑 ~ 图 G1 的最短"非割"圈  ⇒ girth(G1)
    (2) 同类，来自 G2                                    ⇒ girth(G2)
    (3) 由"同时绕两个方向"的算符                        ⇒ 最小割

  对 HGP(C_L, C_L)：三项都是 L ⇒ d = L（与我的实测 3,4 一致）

本文件
  V1 在 HGP(C_L,C_L) 上验证公式（与已实测的 d=3,4 对照）
  V2 用于 4-正则图，给出 d 的**估计**（并标注哪些是严格、哪些是估计）
"""
from __future__ import annotations

import itertools
from collections import deque

import numpy as np


def girth(V, edges):
    """BFS 求 girth（简单图，忽略重边）。"""
    adj = {v: set() for v in range(V)}
    for (a, b) in edges:
        if a != b:
            adj[a].add(b); adj[b].add(a)
    best = 10 ** 9
    for s in range(V):
        dist = {s: 0}; par = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; par[w] = u; q.append(w)
                elif par[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def min_cut_separating(V, edges, s_set, t_set):
    """最小边割，要求把 s_set 与 t_set 分开（简单实现：枚举割集的大小，用小图）。"""
    # 用最小 s-t 割（对 s_set 中任一点与 t_set 中任一点取最小）
    best = 10 ** 9
    adj = {v: {} for v in range(V)}
    for i, (a, b) in enumerate(edges):
        if a == b:
            continue
        adj[a][b] = adj[a].get(b, 0) + 1
        adj[b][a] = adj[b].get(a, 0) + 1
    for s in s_set:
        for t in t_set:
            if s == t:
                continue
            # Edmonds-Karp（小图）
            cap = {u: dict(adj[u]) for u in adj}
            flow = 0
            while True:
                par = {s: None}
                q = deque([s])
                while q and t not in par:
                    u = q.popleft()
                    for w, c in cap[u].items():
                        if c > 0 and w not in par:
                            par[w] = u; q.append(w)
                if t not in par:
                    break
                # 找瓶颈
                path, cur = [], t
                while par[cur] is not None:
                    path.append((par[cur], cur)); cur = par[cur]
                b = min(cap[a][b_] for a, b_ in path)
                for a, b_ in path:
                    cap[a][b_] -= b; cap[b_][a] = cap[b_].get(a, 0) + b
                flow += b
            best = min(best, flow)
    return best


def main():
    print("=" * 92)
    print("HGP 距离公式验证（小例）")
    print("=" * 92)
    print("  公式 d = min(girth(G1), girth(G2), 最小割)")
    print("  %-18s %8s %8s %8s %8s %8s" %
          ("构造", "girth1", "girth2", "最小割", "公式d", "实测d"))
    known = {(3, 3): 3, (4, 4): 4}
    for L in (3, 4, 5, 6):
        V, E = L, [(i, (i + 1) % L) for i in range(L)]
        g1 = girth(V, E)
        # C_L 的最小割（把一条非可缩圈与另一条分开）= 2（两条边的割）
        mc = 2
        dform = min(g1, g1, mc)
        print("  %-18s %8d %8d %8d %8d %8s"
              % ("C%d / C%d" % (L, L), g1, g1, mc, dform,
                 str(known.get((L, L), "?"))))

    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    print("""  · HGP(C_L,C_L)：girth = L，但**最小割 = 2** ⇒ 公式给 d = 2
    而实测 d = L = 3, 4。⇒ **公式的第一版错了**（最小割那一项不该是 2）
  ⇒ 需要修正：X 逻辑与 Z 逻辑的最小重量是**不同**的项
     · d_X = 最短"非面"圈 ~ girth（对 HGP 是 min(g1, g2) 的组合）
     · d_Z = 最小割 **在乘积空间**中，不是原图的最小割
  ⇒ 我的公式把两个不同的量混了。下面改用**乘积空间**上的正确构造。""")


if __name__ == "__main__":
    main()
