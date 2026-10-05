#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_L 比较：AG/RM 码 vs 拓扑码（code-capacity 去极化 + 查表解码）

方法（自洽、可复现）
  1. 给定 CSS 码 (H_X, H_Z)、逻辑算符代表 L_X, L_Z
  2. 预建**查表**：对全部低重量错误 e（重量 ≤ w_max），
     表 key = (X 型 syndrome, Z 型 syndrome) -> 最小重量代表 ê
  3. 采样：每个比特以概率 p 独立施加 X/Y/Z（去极化）
     总错误 e = e_X ⊕ e_Z；syndrome = (H_Z e_X, H_X e_Z)
     解码：ê = table[syndrome]；残差 = e ⊕ ê
     失败 ⟺ 残差含非平凡逻辑（L_X·e_Z + L_Z·e_X ≠ 0）
  4. 统计 p_L，扫 p

对照：同一物理错误率 p 下的不同码。
"""
from __future__ import annotations
import itertools
from math import comb
import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0:
        return 0
    r, rows, cols = 0, A.shape[0], A.shape[1]
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if A[i, c]:
                p = i; break
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = A[i] ^ A[r]
        r += 1
        if r == rows:
            break
    return r


def nullspace(M):
    A = np.array(M, dtype=np.uint8) % 2
    m, n = A.shape
    piv, r = [], 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if A[i, c]:
                p = i; break
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = A[i] ^ A[r]
        piv.append(c); r += 1
        if r == m: break
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8); x[f] = 1
        for i, pc in enumerate(piv):
            x[pc] = A[i, f]
        out.append(x)
    return out


def build_decoder(HX, HZ, w_max=3, nx=None):
    """查表：syndrome -> 最小重量代表。返回 (table, LX, LZ)。"""
    n = HX.shape[1]
    table = {}
    for w in range(0, w_max + 1):
        for sub in itertools.combinations(range(n), w):
            e = np.zeros(n, dtype=np.uint8)
            for i in sub:
                e[i] = 1
            # X 型错误 e 的 syndrome = H_Z e（与 Z 检查对易性）
            s = (tuple(int(v) for v in (HZ @ e) % 2),
                 tuple(int(v) for v in (HX @ e) % 2))
            if s not in table:
                table[s] = e.copy()
    # 逻辑算符：ker(HX) \ rowspace(HZ) 的最小重量元素（X 型）
    LX = min_logical_of(HX, HZ)
    LZ = min_logical_of(HZ, HX)
    return table, LX, LZ


def min_logical_of(H_check, H_stab, cap=16):
    ker = nullspace(H_check)
    if not ker or len(ker) > cap:
        return None
    A = np.array(H_stab, dtype=np.uint8) % 2
    rows, r = [], 0
    tmp = A.copy()
    for c in range(A.shape[1]):
        p = None
        for i in range(r, tmp.shape[0]):
            if tmp[i, c]:
                p = i; break
        if p is None: continue
        tmp[[r, p]] = tmp[[p, r]]
        for i in range(tmp.shape[0]):
            if i != r and tmp[i, c]:
                tmp[i] = tmp[i] ^ tmp[r]
        rows.append(tmp[r]); r += 1

    def in_row(v):
        w = v.copy()
        for r_ in rows:
            p = next((c for c in range(len(r_)) if r_[c]), None)
            if p is not None and w[p]:
                w = w ^ r_
        return not w.any()

    best = None
    for mask in range(1, 1 << len(ker)):
        v = np.zeros(len(ker[0]), dtype=np.uint8)
        for i in range(len(ker)):
            if (mask >> i) & 1:
                v = v ^ ker[i]
        w = int(v.sum())
        if best is not None and w >= best[0]:
            continue
        if not in_row(v):
            best = (w, v)
    return best[1] if best else None


def build_rm_css(m, r):
    """RM-CSS：HX = HZ = RM(r,m) 的生成元（自正交）。"""
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    G = np.array(rows, dtype=np.uint8)
    return G, G


def build_toric(L):
    """toric code 的 HX/HZ（边 = 比特）。"""
    idx = lambda i, j: (i % L) * L + (j % L)
    edges, seen = [], {}
    def add(a, b):
        if a == b: return None
        k = (min(a, b), max(a, b))
        if k not in seen:
            seen[k] = len(edges); edges.append(k)
        return seen[k]
    for i in range(L):
        for j in range(L):
            add(idx(i, j), idx(i, j + 1)); add(idx(i, j), idx(i + 1, j))
    V = L * L
    ne = len(edges)
    HX = np.zeros((V, ne), dtype=np.uint8)
    for v in range(V):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    faces = []
    for i in range(L):
        for j in range(L):
            faces.append(sorted({add(idx(i, j), idx(i, j+1)), add(idx(i, j+1), idx(i+1, j+1)),
                                 add(idx(i+1, j+1), idx(i+1, j)), add(idx(i+1, j), idx(i, j))}))
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    return HX, HZ


def simulate(HX, HZ, p, shots=20000, seed=0, w_max=3):
    """code-capacity 去极化模拟。"""
    table, LX, LZ = build_decoder(HX, HZ, w_max=w_max)
    if LX is None or LZ is None:
        return None, "逻辑算符未找到"
    rnd = np.random.default_rng(seed)
    n = HX.shape[1]
    fails = 0
    # 每个比特：无错 (1-p)，X/Y/Z 各 p/3
    for _ in range(shots):
        u = rnd.random(n)
        eX = np.zeros(n, dtype=np.uint8); eZ = np.zeros(n, dtype=np.uint8)
        m1 = u < p / 3                      # X
        m2 = (u >= p/3) & (u < 2*p/3)       # Y（X 与 Z 同时）
        m3 = (u >= 2*p/3) & (u < p)         # Z
        eX[m1 | m2] = 1
        eZ[m2 | m3] = 1
        s = (tuple(int(v) for v in (HZ @ eZ) % 2),
             tuple(int(v) for v in (HX @ eX) % 2))
        ehat = table.get(s)
        if ehat is None:
            fails += 1
            continue
        # 残差
        rX = eX ^ ehat; rZ = eZ ^ ehat
        # 逻辑判据：与逻辑算符的内积
        if int(LX @ rX % 2) or int(LZ @ rZ % 2):
            fails += 1
    return fails / shots, None


def main():
    print("=" * 92)
    print("p_L 比较：RM-CSS 码 vs toric 码（code-capacity 去极化）")
    print("=" * 92)
    print("  %-20s %7s %5s %5s %10s %10s %10s" %
          ("码", "n", "k", "d", "p=1e-3", "p=5e-3", "p=1e-2"))
    cases = []
    for m, r in ((4, 1), (6, 2)):
        try:
            HX, HZ = build_rm_css(m, r)
            n = HX.shape[1]
            k = n - rank2(HX) - rank2(HZ)
            cases.append(("RM(%d,%d)" % (r, m), HX, HZ, 1 << (r + 1)))
        except Exception as e:
            print("  RM(%d,%d) 构造失败: %s" % (r, m, e))
    for L in (4,):
        HX, HZ = build_toric(L)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        cases.append(("toric L=%d" % L, HX, HZ, L))
    for name, HX, HZ, d in cases:
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        row = []
        for p in (1e-3, 5e-3, 1e-2):
            pl, err = simulate(HX, HZ, p, shots=4000, w_max=3)
            row.append("?" if pl is None else ("%.2e" % pl))
        print("  %-20s %7d %5d %5d %10s %10s %10s" % (name, n, k, d, *row))


if __name__ == "__main__":
    main()
