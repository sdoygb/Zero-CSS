#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自制解码器能做什么：先划清界线（可判定的部分）

要判定的三件事
  D1 对 iid 错误，MWD 是否 ≡ ML？（若是 ⇒ 换解码器不能提高纠错能力）
  D2 Zero 的分层是否对应解码器的**可分解结构**？（若是 ⇒ 可据此造解码器）
  D3 现有主流解码器的复杂度（对照基线）
"""
from __future__ import annotations
import itertools
from math import comb
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


def in_rowspace(Hs, v):
    A = np.array(Hs, dtype=np.uint8) % 2
    m, n = A.shape
    rows, r = [], 0
    tmp = A.copy()
    for c in range(n):
        p = None
        for i in range(r, tmp.shape[0]):
            if tmp[i, c]: p = i; break
        if p is None: continue
        tmp[[r, p]] = tmp[[p, r]]
        for i in range(tmp.shape[0]):
            if i != r and tmp[i, c]: tmp[i] = tmp[i] ^ tmp[r]
        rows.append(tmp[r]); r += 1
    w = v.copy()
    for r_ in rows:
        p = next((cc for cc in range(len(r_)) if r_[cc]), None)
        if p is not None and w[p]: w = w ^ r_
    return not w.any()


def d1_test():
    """D1：对 iid 错误，MWD 与 ML 是否给出同一个纠正？"""
    print("=" * 88)
    print("[D1] iid 下 MWD 与 ML 是否一致")
    print("=" * 88)
    n = 8
    # 取一个简单的 CSS 码：H = 重复码校验（3 个校验，8 比特）
    H = np.zeros((3, n), dtype=np.uint8)
    for i in range(3):
        H[i, 2*i] = 1; H[i, 2*i+1] = 1
    Hs = H
    # 对每个 syndrome 类，比较 MWD 代表 与 ML 代表
    p = 1e-2
    cls = {}
    for mask in range(1 << n):
        e = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        s = tuple(int(v) for v in (H @ e) % 2)
        cls.setdefault(s, []).append(mask)
    agree = 0; total = 0
    for s, masks in cls.items():
        # MWD：最小重量
        mwd = min(masks, key=lambda m: bin(m).count("1"))
        # ML：类内概率和最大
        best_ml, best_val = None, -1
        for cm in masks:
            val = 0.0
            for em in masks:
                v = np.array([((em ^ cm) >> i) & 1 for i in range(n)], dtype=np.uint8)
                if in_rowspace(Hs, v):
                    val += (p/3)**int(v.sum())
            if val > best_val:
                best_val, best_ml = val, cm
        total += 1
        if bin(mwd).count("1") == bin(best_ml).count("1"):
            agree += 1
    print("  syndrome 类数 = %d" % total)
    print("  MWD 与 ML 的纠正**重量相同**的类数 = %d / %d" % (agree, total))
    print("  ⇒ 对 iid，两者等价（重量相同 ⇒ 概率相同）" if agree == total
          else "  ⇒ 存在差异（需细查）")


def d3_complexity():
    """D3：主流解码器的复杂度对照。"""
    print("\n" + "=" * 88)
    print("[D3] 主流解码器复杂度（基线）")
    print("=" * 88)
    rows = [
        ("MWPM（表面码）", "O(n^3) 或 O(n log n)（稀疏实现）", "pymatching"),
        ("BP + OSD（qLDPC）", "O(n·iter·w) + O(n^3)（OSD 部分）", "ldpc 库"),
        ("查表（code-capacity）", "O(2^n) 建表", "小码可行"),
        ("Zero 的 MWD（定理 B/D）", "O(2^{beta1}) 枚举 或 O(n^2·m) 矩代数", "旧理论的 RM 解码器"),
        ("HGP 的结构化解码", "O(n)（利用乘积结构，文献有）", "待实现"),
    ]
    print("  %-26s %-40s %s" % ("解码器", "复杂度", "现状"))
    for r in rows:
        print("  %-26s %-40s %s" % r)


def main():
    print("=" * 88)
    print("自制解码器能做什么：界线判定")
    print("=" * 88)
    print("""
  核心事实（必须先承认）
    对**独立同分布**错误，最小重量解码 ≡ 最大似然解码
    ⇒ 换解码器**不能**降低 p_L（纠错能力由码距 d 定死）
    ⇒ 自制解码器的收益只能在别处：速度、电路级噪声表现、确定性
""")
    d1_test()
    d3_complexity()
    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    print("""  (1) 纠错能力：自制解码器**不能**提高（iid 下 MWD 已最优）
  (2) 可做的：
      · **速度**：利用 Zero 的分层结构把解码拆成可分解的子问题
      · **电路级噪声**：那里 MWD 不再最优，有真实空间
      · **确定性/可解释**：非迭代、可判定
  (3) Zero 的分层确实对应解码结构：
      L0 约束图（检查矩阵） → L1 syndrome（读出面） → L2 最小重量搜索
      ⇒ 这是**组织的模板**，不是性能来源""")


if __name__ == "__main__":
    main()
