#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核实 AG/RM 码：参数是否真的那么好？低权重逻辑有多少个？

疑点
  RM-CSS 码的**参数**（n,k,d）确实好，但：
    (1) 检查算子权重 2^(m-r) … 2^m（非局域）
    (2) d 是最小逻辑重量，但**低权重逻辑的个数**决定解码难度
    (3) 稳定子生成元数 = dim RM(r,m)，可能远少于 n（码很"稀疏"）
  本文件算 (1)(2)(3) 的定量值。

RM(r,m) 基础事实
  n = 2^m，dim = Σ_{j≤r} C(m,j)，d = 2^(m-r)
  对偶：RM(r,m)^⊥ = RM(m-r-1,m)
  自正交条件：r < m-r-1 ⇔ 2r < m-1
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


def rm_gens(m, r):
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    return np.array(rows, dtype=np.uint8)


def main():
    print("=" * 94)
    print("核实 AG/RM：参数、检查权重、生成元数")
    print("=" * 94)
    print("  %-12s %7s %6s %5s %7s %9s %10s %9s" %
          ("码", "n", "dim", "k", "d", "检查权重", "生成元数", "k/n"))
    for m in (4, 6, 8, 10, 12):
        for r in (1, 2, 3):
            if 2 * r >= m - 1 or r >= m:
                continue
            n = 1 << m
            dim = sum(comb(m, j) for j in range(r + 1))
            k = n - 2 * dim
            if k <= 0:
                continue
            d = 1 << (r + 1)
            G = rm_gens(m, r)
            w = G.sum(axis=1)
            print("  %-12s %7d %6d %5d %7d %9s %10d %9.3f"
                  % ("RM(%d,%d)" % (r, m), n, dim, k, d,
                     "%d..%d" % (w.min(), w.max()), G.shape[0], k / n))

    print("\n" + "=" * 94)
    print("关键疑点 1：稳定子生成元数 vs 码长（检查非常稀疏）")
    print("=" * 94)
    print("  %-12s %7s %10s %10s %12s" % ("码", "n", "生成元数", "生成元/n", "每个生成元的权重"))
    for m, r in ((6, 2), (8, 2), (10, 3), (12, 3)):
        n = 1 << m
        G = rm_gens(m, r)
        w = G.sum(axis=1)
        print("  %-12s %7d %10d %10.4f %12s"
              % ("RM(%d,%d)" % (r, m), n, G.shape[0], G.shape[0] / n,
                 "%.1f" % w.mean()))

    print("\n" + "=" * 94)
    print("关键疑点 2：低权重逻辑算符的**个数**（决定解码难度）")
    print("=" * 94)
    for m, r in ((4, 1), (5, 1), (6, 1), (4, 2)):
        if 2 * r >= m - 1:
            continue
        n = 1 << m
        Gr = rm_gens(m, r)          # 小码
        Gbig = rm_gens(m, m - r - 1)  # 大码 = 对偶
        target = 1 << (r + 1)
        # 数 RM(m-r-1,m) 中重量 = target 的向量（= 最小逻辑的个数下界）
        # 用小规模枚举
        if n > 64:
            print("  RM(%d,%d): n=%d 太大，跳过" % (r, m, n))
            continue
        cnt = 0
        rk = rank2(Gbig)
        if rk > 20:
            print("  RM(%d,%d): 维数 %d 太大" % (r, m, rk))
            continue
        # 枚举 span
        basis = []
        A = Gbig.copy() % 2
        # 取行空间的一组基
        rows = []
        tmp = A.copy()
        rr = 0
        for c in range(n):
            p = None
            for i in range(rr, tmp.shape[0]):
                if tmp[i, c]:
                    p = i; break
            if p is None: continue
            tmp[[rr, p]] = tmp[[p, rr]]
            for i in range(tmp.shape[0]):
                if i != rr and tmp[i, c]:
                    tmp[i] = tmp[i] ^ tmp[rr]
            rows.append(tmp[rr]); rr += 1
        for mask in range(1 << len(rows)):
            v = np.zeros(n, dtype=np.uint8)
            for i in range(len(rows)):
                if (mask >> i) & 1:
                    v = v ^ rows[i]
            if int(v.sum()) == target:
                cnt += 1
        print("  RM(%d,%d): n=%d  最小逻辑(d=%d) 的个数 = %d"
              % (r, m, n, target, cnt))


if __name__ == "__main__":
    main()
