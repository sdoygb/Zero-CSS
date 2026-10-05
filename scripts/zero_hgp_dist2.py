#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGP 距离：枚举核空间（小图精确）"""
from __future__ import annotations
import itertools
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


def rref(M):
    A = np.array(M, dtype=np.uint8) % 2
    m, n = A.shape
    piv, r = [], 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if A[i, c]:
                p = i; break
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = A[i] ^ A[r]
        piv.append(c); r += 1
        if r == m:
            break
    return A[:r], piv


def nullspace(M):
    m, n = M.shape
    A, piv = rref(M)
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8); x[f] = 1
        for i, pc in enumerate(piv):
            x[pc] = A[i, f]
        out.append(x)
    return out


def span_elements(basis, cap=20):
    if len(basis) > cap:
        return None
    out = []
    for mask in range(1 << len(basis)):
        v = np.zeros(len(basis[0]), dtype=np.uint8)
        for i in range(len(basis)):
            if mask >> i & 1:
                v = v ^ basis[i]
        out.append(v)
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


def min_logical(H_check, H_stab):
    """ker(H_check) \\ rowspace(H_stab) 最小重量。"""
    ker = nullspace(H_check)
    stab = span_elements(nullspace(H_stab.T)) if False else None
    # rowspace(H_stab) = ker(H_stab^T)^\perp 的补 —— 直接用 rref 判定成员
    A, piv = rref(H_stab)
    rows = [A[i] for i in range(A.shape[0])]

    def in_row(v):
        w = v.copy()
        for r_ in rows:
            # 消去 w 在 r_ 主元位上的分量
            p = next((c for c in range(len(r_)) if r_[c]), None)
            if p is not None and w[p]:
                w = w ^ r_
        return not w.any()

    elems = span_elements(ker)
    if elems is None:
        return None
    best = None
    for v in elems:
        w = int(v.sum())
        if w == 0:
            continue
        if best is not None and w >= best:
            continue
        if not in_row(v):
            best = w
    return best


def main():
    print("=" * 90)
    print("HGP 距离（枚举核空间）")
    print("=" * 90)
    print("  %-16s %7s %7s %7s %7s %7s %8s" %
          ("G1/G2", "n", "k", "d_X", "d_Z", "d", "k/n"))
    for L in (3, 4, 5, 6, 7):
        V1, E1 = cycle_graph(L); V2, E2 = cycle_graph(L)
        H1, H2 = incidence(V1, E1), incidence(V2, E2)
        HX, HZ = hgp(H1, H2)
        n = HX.shape[1]
        k = n - rank2(HX) - rank2(HZ)
        dX = min_logical(HZ, HX)
        dZ = min_logical(HX, HZ)
        ds = [v for v in (dX, dZ) if isinstance(v, int)]
        print("  %-16s %7d %7d %7s %7s %7s %8.3f"
              % ("C%d / C%d" % (L, L), n, k, str(dX), str(dZ),
                 str(min(ds) if ds else "?"), k / n))


if __name__ == "__main__":
    main()
