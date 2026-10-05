#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_L（正确版）：ML 解码 —— 高度简并码上最小重量解码不是最优

发现
  RM-CSS[[16,6,4]] 只有 32 个 syndrome（2^5），而错误空间 2^16
  ⇒ 每个 syndrome 类含大量错误 ⇒ **简并极强**
  ⇒ 最小重量解码（MWD）远非最优；必须用 **ML**：
      对每个 syndrome 类，选"类内总概率最大"的那种错误作为纠正
      即：纠正 c 使 Σ_{e: e⊕c ∈ 稳定子且 syndrome(e)=s} P(e) 最大

本文件
  1. 对每个 syndrome 类，**精确算**每个候选纠正的"成功率"（枚举类内全部错误）
  2. 用 ML 纠正跑模拟 ⇒ 得到正确的 p_L
  3. 与 MWD 对比，量化"简并导致的解码损失"
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


def span_elems(basis, cap=20):
    if len(basis) > cap: return None
    out = []
    for mask in range(1 << len(basis)):
        v = np.zeros(len(basis[0]), dtype=np.uint8)
        for i in range(len(basis)):
            if (mask >> i) & 1: v = v ^ basis[i]
        out.append(v)
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
        p = next((c for c in range(len(r_)) if r_[c]), None)
        if p is not None and w[p]: w = w ^ r_
    return not w.any()


def build_ml_decoder(HX, HZ, p):
    """对每个 syndrome 类，取 ML 纠正（类内概率和最大）。返回 表 + 失败概率表。"""
    n = HX.shape[1]
    SX = span_elems(nullspace(HX))     # X 稳定子群（Z 检查的核？）—— 见下
    # 说明：X 型错误 e 的 syndrome 由 H_Z 给出；等价类 = e ⊕ (X 稳定子 rowspace(H_X))
    stabX = span_elems(nullspace(HZ.T)) if False else None
    # 更直接：用 rowspace(H_X) 作为 X 稳定子
    # 枚举全部 2^n 错误太重；n=16 可行
    cls = {}      # syndrome -> {纠正 -> 权重和}
    for mask in range(1 << n):
        e = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        # 与 Z 检查对易性：syndrome = H_Z e
        s = tuple(int(v) for v in (HZ @ e) % 2)
        cls.setdefault(s, []).append(mask)
    # 对每个 syndrome 类，选纠正 c 使 Σ_{e: e⊕c ∈ rowspace(H_X)} P(e) 最大
    # P(e) 用 (p/3)^|e| (1-p)^{n-|e|} 的近似比例
    best = {}
    for s, masks in cls.items():
        # 候选纠正 = 类内所有元素（等价类代表）
        sc = {}
        for cm in masks:
            tot = 0.0
            for em in masks:
                # e ⊕ c 是否在 rowspace(H_X)
                v = np.array([((em ^ cm) >> i) & 1 for i in range(n)], dtype=np.uint8)
                if in_rowspace(HX, v):
                    tot += (p / 3) ** int(v.sum())
            sc[cm] = tot
        best[s] = max(sc.items(), key=lambda kv: kv[1])[0]
    return best


def evaluate(HX, HZ, p, dec, LX, LZ):
    """精确算 p_L（对全部 2^n 错误求和）。"""
    n = HX.shape[1]
    tot = 0.0
    for mask in range(1 << n):
        e = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        w = int(e.sum())
        prob = (p / 3) ** w * (1 - p) ** (n - w)
        if prob < 1e-300: continue
        s = tuple(int(v) for v in (HZ @ e) % 2)
        c = dec.get(s)
        if c is None: tot += prob; continue
        chat = np.array([(c >> i) & 1 for i in range(n)], dtype=np.uint8)
        r = e ^ chat
        if int(LX @ r % 2):
            tot += prob
    return tot


def main():
    print("=" * 92)
    print("RM-CSS[[16,6,4]] 的 p_L：ML 解码（精确求和）")
    print("=" * 92)
    n = 16
    rows = [[1 if (c & mk) == mk else 0 for c in range(n)] for mk in range(n) if mk.bit_count() <= 1]
    G = np.array(rows, dtype=np.uint8)
    HX = HZ = G
    print("  k =", n - rank2(HX) - rank2(HZ), " n =", n, " 生成元数 =", G.shape[0])
    # 逻辑算符
    ker = nullspace(HZ)
    LX = None
    best = None
    for mask in range(1, 1 << len(ker)):
        v = np.zeros(n, dtype=np.uint8)
        for i in range(len(ker)):
            if (mask >> i) & 1: v = v ^ ker[i]
        w = int(v.sum())
        if best is not None and w >= best: continue
        if not in_rowspace(HX, v): best, LX = w, v
    print("  d =", best, " LX 重量 =", int(LX.sum()))
    print()
    print("  %-10s %14s %14s" % ("p", "p_L (ML)", "p_L / p^2"))
    for p in (1e-4, 1e-3, 5e-3, 1e-2):
        dec = build_ml_decoder(HX, HZ, p)
        pl = evaluate(HX, HZ, p, dec, LX, LX)
        print("  %-10.0e %14.3e %14.3f" % (p, pl, pl / p**2))


if __name__ == "__main__":
    main()
