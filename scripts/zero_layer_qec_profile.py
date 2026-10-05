#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 分层下的纠错：四层各自的"量子表现"

动机
  Zero 是分层的（L0 底层／L1 历史层／L1′ 全局闭合类层／L2 演化层；R 读出面）。
  我先前只用 L1′ 的**旋转类**当纠错口径，判它"有损"。
  但旧理论的 syndrome 用的是**精确错误向量**——那是 L1 历史层。
  ⇒ 必须**逐层**测同一件纠错任务，看每层给出什么。

四层口径（同一组错误，不同层的"可区分性"）
  L0 底层        ：只问**零和 / 校验约束**是否被违反（是/否）——最粗
  L1 历史层      ：**精确错误向量**本身是否可区分（旧理论口径）
  L1′ 全局闭合类层：**旋转类**（循环移位轨道）是否可区分
  L2 演化层      ：**局域补偿移动**（保零和）能否把两个错误互相变换——动力学可达性
  R 读出面       ：**syndrome**（H·e）是否可区分（码的权威读出）

判据
  对每一层定义"同层不可区分"关系，算：
    - 类数（该层能把 N 个错误分成几类）
    - fail(2) = 1 − <1/v>
    - 与 R 读出面（syndrome）的一致性：同层同类 ⇒ 是否同 syndrome？（越接近真纠错越好）

用法: python3 zero_layer_qec_profile.py
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np


def build(nv, edges):
    H = np.zeros((nv, len(edges)), dtype=np.uint8)
    for e, (u, v) in enumerate(edges):
        H[u, e] = 1
        H[v, e] = 1
    return H


def rot(x):
    n = len(x)
    return min(tuple(x[(i + s) % n] for i in range(n)) for s in range(n))


def local_moves(x, n):
    """局域补偿移动的**可达集**：对 x 做一次 T_{ex}（保零和），返回可达向量集合。

    在 GF(2) 上：交换一位 0->1 与一位 1->0（保持重量），即 x ^ e_i ^ e_j。
    """
    out = set()
    ones = [i for i in range(n) if x[i]]
    zeros = [i for i in range(n) if not x[i]]
    for i in ones:
        for j in zeros:
            y = list(x)
            y[i], y[j] = 0, 1
            out.add(tuple(y))
    return out


def profile(errs, key, label, H=None):
    """给定"同层不可区分"的 key 函数，算类数与 fail(2)。"""
    groups = defaultdict(list)
    for e in errs:
        groups[key(e)].append(e)
    v = np.array([len(g) for g in groups.values()], float)
    fail2 = 1 - np.mean(1 / v)
    # 与 syndrome 的一致性（若给了 H）
    purity = None
    if H is not None:
        agree = 0
        for g in groups.values():
            syns = {tuple(int(t) for t in (H @ np.array(m)) % 2) for m in g}
            if len(syns) == 1:
                agree += 1
        purity = agree / len(groups)
    print(f"    {label:28s} 类数={len(groups):5d}  fail(2)={fail2:.4f}  "
          f"类内 syndrome 唯一比例={'-' if purity is None else f'{purity*100:.1f}%'}")
    return len(groups), fail2, purity


def main() -> None:
    print("=" * 78)
    print("Zero 分层下的纠错：四层 + 读出面的逐个画像")
    print("=" * 78)
    cases = {
        "环 C8（圈码 [[8,1,8]]）": (8, [(i, (i + 1) % 8) for i in range(8)]),
        "K4（圈码 [[6,3,3]]）": (4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
    }
    for name, (nv, edges) in cases.items():
        H = build(nv, edges)
        n = len(edges)
        errs = [tuple(1 if i in s else 0 for i in range(n))
                for w in (1, 2) for s in itertools.combinations(range(n), w)]
        print(f"\n=== {name}  n={n}  权重≤2 错误数={len(errs)}")
        profile(errs, lambda e: "any", "L0 只问'是否违反校验'", H)          # 全同类
        profile(errs, lambda e: e,    "L1 精确向量（历史层）", H)
        profile(errs, rot,            "L1′ 旋转类（闭合类层）", H)
        # L2：以"局域补偿可达"为同层（用单步可达做闭包近似）
        def l2key(e):
            reach = local_moves(e, n) | {e}
            return frozenset(reach)
        profile(errs, l2key,          "L2 局域移动可达（演化层）", H)
        profile(errs, lambda e: tuple(int(t) for t in (H @ np.array(e)) % 2),
                "R 读出面 syndrome（权威）", H)


if __name__ == "__main__":
    main()
