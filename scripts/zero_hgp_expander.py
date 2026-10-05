#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP + 高 girth 定度图：码率与距离"""
from __future__ import annotations
import numpy as np
from zero_hgp_dist2 import rank2, rref, nullspace, span_elements, incidence, hgp


def girth(V, edges):
    """BFS 求 girth（简单图，忽略重边）。"""
    adj = {v: set() for v in range(V)}
    for (a, b) in edges:
        adj[a].add(b); adj[b].add(a)
    best = 10**9
    for s in range(V):
        dist = {s: 0}; par = {s: None}
        stack = [s]
        while stack:
            u = stack.pop(0)
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; par[w] = u; stack.append(w)
                elif par[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def rand_regular_girth(V, d, seed, tries=60):
    """随机 d-正则图，取 girth 最大者。"""
    rnd = np.random.default_rng(seed)
    best, bg = None, -1
    for _ in range(tries):
        stubs = [v for v in range(V) for _ in range(d)]
        rnd.shuffle(stubs)
        es = set()
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = int(stubs[i]), int(stubs[i + 1])
            if a == b:
                ok = False; break
            es.add((min(a, b), max(a, b)))
        if not ok:
            continue
        E = sorted(es)
        if len(E) != V * d // 2:
            continue
        g = girth(V, E)
        if g > bg:
            bg, best = g, E
    return V, best, bg


def min_logical_enum(H_check, H_stab, cap=22):
    ker = nullspace(H_check)
    if len(ker) > cap:
        return "核维数 %d 过大" % len(ker)
    A, _ = rref(H_stab)
    rows = [A[i] for i in range(A.shape[0])]
    def in_row(v):
        w = v.copy()
        for r_ in rows:
            p = next((c for c in range(len(r_)) if r_[c]), None)
            if p is not None and w[p]:
                w = w ^ r_
        return not w.any()
    elems = span_elements(ker, cap=cap)
    best = None
    for v in elems:
        w = int(v.sum())
        if w == 0 or (best is not None and w >= best):
            continue
        if not in_row(v):
            best = w
    return best


def main():
    print("=" * 96)
    print("HGP + 高 girth 4-正则图")
    print("=" * 96)
    print("  %-26s %7s %7s %8s %7s %7s %9s %9s" %
          ("G（4-正则）", "V", "E", "girth", "n", "k", "k/n", "d"))
    for V in (8, 10, 12, 14, 16):
        Vg, E, g = rand_regular_girth(V, 4, seed=V + 7)
        if E is None:
            print("  V=%d：未找到" % V); continue
        H = incidence(Vg, E)
        beta1 = len(E) - Vg + 1
        HX, HZ = hgp(H, H)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        d = min_logical_enum(HX, HZ, cap=22)
        print("  %-26s %7d %7d %8d %7d %7d %9.4f %9s"
              % ("4-正则 V=%d" % V, Vg, len(E), g, n, k, k / n, str(d)))

    print("\n" + "=" * 96)
    print("对照（闭式，已验证 2 例）")
    print("=" * 96)
    print("  %-30s %10s %10s %9s" % ("构造", "n", "k", "k/n"))
    for d in (3, 4, 6):
        for V in (100, 1000):
            E = d * V // 2
            b1 = E - V + 1
            n = E * V + V * E
            k = b1 * b1 + 1
            print("  %-30s %10d %10d %9.5f" % ("HGP(%d-正则 V=%d)" % (d, V), n, k, k / n))


if __name__ == "__main__":
    main()
