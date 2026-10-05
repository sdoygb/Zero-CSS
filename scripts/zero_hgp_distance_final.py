#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 距离 = min(girth(G1), girth(G2))：在 4-正则图上算，得出完整参数

公式来源与验证
  · HGP(G1,G2) 的 X 逻辑 = "G1 的圈 ⊗ G2 的圈"型张量结构
    ⇒ 最小重量 = min(girth(G1), girth(G2))
  · 验证：HGP(C_L,C_L) 给 d = L（girth = L），与我实测 d=3,4 一致，
    且 = toric code [[2L²,2,L]] 的标准值
  · HGP(H, H)（同图自身）⇒ d = girth(H)

本文件：用 4-正则随机图（girth 尽量大）算完整 [[n,k,d]]
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np


def girth(V, edges):
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


def rand_regular_best_girth(V, d, seed, tries=200):
    rnd = np.random.default_rng(seed)
    best, bg = None, -1
    for t in range(tries):
        stubs = [v for v in range(V) for _ in range(d)]
        rnd.shuffle(stubs)
        es, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = int(stubs[i]), int(stubs[i + 1])
            if a == b:
                ok = False; break
            es.add((min(a, b), max(a, b)))
        if not ok or len(es) != V * d // 2:
            continue
        E = sorted(es)
        g = girth(V, E)
        if g > bg:
            bg, best = g, E
    return V, best, bg


def hgp_params(V1, E1, V2, E2):
    """闭式参数：n = E1*V2 + V1*E2；k = beta1_1*beta1_2 + 1；d = min(g1,g2)。"""
    n = len(E1) * V2 + V1 * len(E2)
    b1 = len(E1) - V1 + 1
    b2 = len(E2) - V2 + 1
    k = b1 * b2 + 1
    d = min(girth(V1, E1), girth(V2, E2))
    return n, k, d


def main():
    print("=" * 96)
    print("HGP(4-正则, 4-正则)：完整 [[n,k,d]]")
    print("=" * 96)
    print("  %-8s %7s %7s %9s %7s %8s %7s %8s"
          % ("V", "E", "girth", "n", "k", "k/n", "d", "d/sqrt(n)"))
    for V in (10, 12, 14, 16, 20, 24, 30, 40):
        Vg, E, g = rand_regular_best_girth(V, 4, seed=V * 31)
        if E is None:
            print("  %-8d 未找到合法 4-正则图" % V); continue
        n, k, d = hgp_params(Vg, E, Vg, E)
        print("  %-8d %7d %7d %9d %7d %8.4f %7d %8.3f"
              % (V, len(E), g, n, k, k / n, d, d / math.sqrt(n)))

    print("\n" + "=" * 96)
    print("对照：三种构造的完整参数（同数量级的 n）")
    print("=" * 96)
    print("  %-30s %9s %7s %9s %8s %8s" % ("构造", "n", "k", "k/n", "d", "d/√n"))
    print("  %-30s %9d %7d %9.4f %8d %8.3f" % ("配方A: 环面 L=16", 512, 2, 2/512, 16, 16/math.sqrt(512)))
    V, E, g = rand_regular_best_girth(16, 4, seed=16*31)
    n, k, d = hgp_params(V, E, V, E)
    print("  %-30s %9d %7d %9.4f %8d %8.3f" % ("配方B: HGP(4-正则 V=16)", n, k, k/n, d, d/math.sqrt(n)))
    print("  %-30s %9d %7d %9.4f %8d %8.3f" % ("AG: RM(3,10)", 1024, 672, 0.656, 16, 16/math.sqrt(1024)))

    print("\n" + "=" * 96)
    print("判读")
    print("=" * 96)
    print("""  · d = girth(4-正则图) ~ 2·log_3(V)（随机正则图的 girth 是 Θ(log V)）
    · n ∝ V² ⇒ d ~ log n ⇒ **d/√n → 0**
      ⇒ HGP 的"码率 × 距离"取舍：率高但距离只有对数
  · 对照 AG：n=1024, k=672, d=16 ⇒ d/√n = 0.50
  · 对照环面：n=512, k=2, d=16 ⇒ 率高不了
  ⇒ **没有一方全面占优**：HGP 率高距离低，AG 两者兼顾但非局域""")


if __name__ == "__main__":
    main()
