#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 距离：用 BFS 上界（不枚举核空间）＋ 与 AG 码对比"""
from __future__ import annotations
import numpy as np
from zero_hgp_dist2 import rank2, incidence, hgp


def min_weight_logical_upper(H_check, H_stab, trials=4000, seed=1):
    """随机采样核空间元素，取最小非平凡重量 ⇒ d 的**上界**（构造性）。"""
    ker = __import__("zero_hgp_dist2").nullspace(H_check)
    if not ker:
        return None
    import random as _rnd
    _rnd.seed(seed)
    A = __import__("zero_hgp_dist2").rref(H_stab)[0]
    rows = [A[i] for i in range(A.shape[0])]

    def in_row(v):
        w = v.copy()
        for r_ in rows:
            p = next((c for c in range(len(r_)) if r_[c]), None)
            if p is not None and w[p]:
                w = w ^ r_
        return not w.any()

    best = None
    K = len(ker)
    for _ in range(trials):
        mask = _rnd.getrandbits(K)
        v = np.zeros(len(ker[0]), dtype=np.uint8)
        for i in range(K):
            if (mask >> i) & 1:
                v = v ^ ker[i]
        w = int(v.sum())
        if w == 0:
            continue
        if best is not None and w >= best:
            continue
        if not in_row(v):
            best = w
    return best


def main():
    print("=" * 94)
    print("HGP 距离上界（随机采样核空间）＋ 最终对比")
    print("=" * 94)
    print("  %-24s %8s %8s %8s %8s %10s" %
          ("构造", "n", "k", "k/n", "d 上界", "√n"))
    rows = []
    # C_L / C_L
    for L in (4, 5, 8, 12):
        V, E = L, [(i, (i + 1) % L) for i in range(L)]
        H = incidence(V, E)
        HX, HZ = hgp(H, H)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        d = min_weight_logical_upper(HX, HZ, trials=3000, seed=L)
        rows.append(("HGP(C%d, C%d)" % (L, L), n, k, k / n, d, np.sqrt(n)))
    # 4-正则
    for V0 in (10, 12):
        rnd = np.random.default_rng(V0)
        stubs = [v for v in range(V0) for _ in range(4)]
        rnd.shuffle(stubs)
        es = sorted({(min(int(stubs[i]), int(stubs[i+1])), max(int(stubs[i]), int(stubs[i+1])))
                     for i in range(0, len(stubs), 2) if stubs[i] != stubs[i+1]})
        H = incidence(V0, es)
        HX, HZ = hgp(H, H)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        d = min_weight_logical_upper(HX, HZ, trials=3000, seed=V0)
        rows.append(("HGP(4-正则 V=%d)" % V0, n, k, k / n, d, np.sqrt(n)))
    for name, n, k, r, d, sn in rows:
        print("  %-24s %8d %8d %8.4f %8s %10.1f"
              % (name, n, k, r, str(d), sn))

    print("\n" + "=" * 94)
    print("三条路的最终对比")
    print("=" * 94)
    print("""
  ┌──────────────────────┬────────────┬──────────────┬────────────┬──────────┐
  │ 构造                  │ 码率 k/n    │ 距离 d        │ 检查权重    │ Zero 原生 │
  ├──────────────────────┼────────────┼──────────────┼────────────┼──────────┤
  │ 配方A: 环面 toric     │ → 0        │ ~√n          │ 4（常数）  │ ✓        │
  │ 配方A: 双曲           │ ~0.05–0.5  │ ~log n       │ p（常数）  │ ✓        │
  │ 配方B: HGP(定度图)    │ 0.23–0.67  │ ~√n 或 log n │ d（常数）  │ ✓        │
  │ AG/RM                 │ 0.656      │ 常数（16）    │ 8…2^m     │ ✗        │
  └──────────────────────┴────────────┴──────────────┴────────────┴──────────┘

  **结论**：配方 B（HGP）把"Zero 原生 + 常数检查权重"下的码率从 →0 提到 **0.23–0.67**，
  与 AG 码同一档（0.656）。差别只剩**距离标度**：AG 的 d 是常数但可设大（16）；
  HGP 的 d 取决于图的 girth（随机正则图 girth ~ log V ⇒ d 小）。
""")


if __name__ == "__main__":
    main()
