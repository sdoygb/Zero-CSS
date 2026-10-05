#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 解码的可分解性（干净版）

推导
  X 型错误 a = (a_A, a_B)，其中
    a_A[(e,w)]  e∈E1, w∈V2     （前 E1*V2 位）
    a_B[(v,f)]  v∈V1, f∈E2     （后 V1*E2 位）
  与 Z 检查对易 ⟺ H_Z a = 0，展开得
    (H2 y_e)_v  =  (H1 z_v)_e      ∀ (e,v)        … (*)
  其中 y_e := a_A[e, :] ∈ F_2^V2，z_v := a_B[v, :] ∈ F_2^E2

  取 z 为**单点**（z_v = δ_{v,v0}·H1 的第 e0 列）⇒ (*) 右边在 e=e0 时为 1、其余为 0
  ⇒ 对每对 (e,v)： (H2 y_e)_v = [e=e0 且 v 与 e0 关联]

  本文件验证 (*) 对全部 ker(H_Z) 元素成立。
"""
from __future__ import annotations
import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0: return 0
    r, rows, cols = 0, A.shape[0], A.shape[1]
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if A[i, c]: p = i; break
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]: A[i] = A[i] ^ A[r]
        r += 1
        if r == rows: break
    return r


def nullspace(M):
    A = np.array(M, dtype=np.uint8) % 2
    m, n = A.shape
    piv, r = [], 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if A[i, c]: p = i; break
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]: A[i] = A[i] ^ A[r]
        piv.append(c); r += 1
        if r == m: break
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8); x[f] = 1
        for i, pc in enumerate(piv): x[pc] = A[i, f]
        out.append(x)
    return out


def incidence(V, edges):
    H = np.zeros((V, len(edges)), dtype=np.uint8)
    for e, (a, b) in enumerate(edges):
        H[a, e] = 1; H[b, e] = 1
    return H


def cycle_graph(L):
    return L, [(i, (i + 1) % L) for i in range(L)]


def hgp(H1, H2):
    V1, E1 = H1.shape
    V2, E2 = H2.shape
    HX = np.hstack([np.kron(H1, np.eye(E2, dtype=np.uint8)),
                    np.kron(np.eye(V1, dtype=np.uint8), H2.T)]) % 2
    HZ = np.hstack([np.kron(np.eye(E1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(V2, dtype=np.uint8))]) % 2
    return HX, HZ


def check_relation(v, H1, H2):
    """验证 (*)：(H2 y_e)_v = (H1 z_v)_e 对全部 (e,v)。"""
    V1, E1 = H1.shape
    V2, E2 = H2.shape
    aA = v[:E1 * V2].reshape(E1, V2)      # aA[e, w]
    aB = v[E1 * V2:].reshape(V1, E2)      # aB[v, f]
    lhs = (aA @ H2.T) % 2                 # lhs[e, v] = Σ_w aA[e,w] H2[v,w]
    rhs = (aB @ H1.T) % 2                 # rhs[v, e] = Σ_f aB[v,f] H1[e,f]
    return np.array_equal(lhs % 2, rhs.T % 2)


def main():
    print("=" * 92)
    print("HGP 解码的分解关系 (*)：(H2 y_e)_v = (H1 z_v)_e")
    print("=" * 92)
    print("  %-12s %7s %9s %12s %14s" % ("图", "n", "dim ker(HZ)", "验证数", "关系成立?"))
    for L in (3, 4, 5):
        V, E = cycle_graph(L)
        H = incidence(V, E)
        HX, HZ = hgp(H, H)
        n = HX.shape[1]
        ker = nullspace(HZ)
        if len(ker) > 16:
            print("  %-12s %7d %9d %12s %14s" % ("C%d/C%d" % (L, L), n, len(ker), "-", "维数过大"))
            continue
        ok = 0; tot = 0
        for mask in range(1, 1 << len(ker)):
            v = np.zeros(n, dtype=np.uint8)
            for i in range(len(ker)):
                if (mask >> i) & 1: v = v ^ ker[i]
            tot += 1
            if check_relation(v, H, H): ok += 1
        print("  %-12s %7d %9d %12d %14s" % ("C%d/C%d" % (L, L), n, len(ker), tot,
                                             "✅ 全部" if ok == tot else "❌ %d/%d" % (ok, tot)))
    print("""
  ⇒ 关系 (*) 若全部成立，则 HGP 的解码可以**按 (e,v) 逐对分解**：
     每个 (e,v) 对给一个独立约束 (H2 y_e)_v = (H1 z_v)_e
     ⇒ 最小重量搜索可从 n 维降到 V1·E1 个独立子问题""")


if __name__ == "__main__":
    main()
