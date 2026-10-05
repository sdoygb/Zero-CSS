#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 的分层与码结构的精确关系（补完 L2 / 稳定子）

已建立
  L1（精确词）= R（syndrome）⇒ 纠错住在历史层。
  L1′（旋转类）是记录粗粒化 ⇒ 有损（100% 混类）。
  L2（局域补偿移动）达 96.6% 唯一 ⇒ 疑似"重量类"/最小权重解码。

本轮要判（三条）
  Q1  L2 的局域补偿移动生成的群是否 = **整个对称群/重量保持群**？
      若是 ⇒ L2 的类 = **Hamming 重量类** ⇒ L2 提供的是"重量分辨率"。
  Q2  L2 的类与**最小权重解码（MWD）**的关系：
      同一 syndrome 下，MWD 取最小重量代表；L2 类是否就是"同重量"？
  Q3  **稳定子在哪里**？把译码问题写成 (syndrome 决定陪集, 陪集内取最小重量)：
      稳定子 = 零 syndrome 的陪集 = **L0 约束的核**；
      逻辑 = 非零 syndrome 的陪集结构。给出 Zero 分层下的对应表。

用法: python3 zero_layers_vs_code.py
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


def syndrome(H, e):
    return tuple(int(t) for t in (H @ np.array(e, dtype=np.uint8)) % 2)


def local_move_orbit(x, n):
    """局域补偿移动（交换一个 1 与一个 0）生成的轨道 = 同重量字集合。"""
    w = sum(x)
    return frozenset(tuple(1 if i in s else 0 for i in range(n))
                     for s in itertools.combinations(range(n), w))


def main() -> None:
    print("=" * 78)
    print("Zero 分层 ↔ 码结构：L2 / 稳定子 / 最小权重解码")
    print("=" * 78)
    nv, edges = 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]
    H = build(nv, edges)
    n = len(edges)
    print(f"\n对象：K4 圈码  n={n}  稳定子生成元数={nv}  rank={np.linalg.matrix_rank(H.astype(float))}")

    allw = list(itertools.product((0, 1), repeat=n))
    # Q1: L2 轨道是否 = 重量类
    by_weight = defaultdict(set)
    for x in allw:
        by_weight[sum(x)].add(x)
    l2_ok = True
    for x in allw:
        if local_move_orbit(x, n) != frozenset(by_weight[sum(x)]):
            l2_ok = False
            break
    print(f"\n[Q1] L2 轨道（局域补偿移动）== Hamming 重量类？  {l2_ok}")
    print(f"     重量类大小: {[len(by_weight[w]) for w in sorted(by_weight)]}"
          f"  （= C({n},w)）")

    # Q2: 同 syndrome 内，L2 能否分辨不同重量？
    by_syn = defaultdict(list)
    for x in allw:
        by_syn[syndrome(H, x)].append(x)
    print(f"\n[Q2] syndrome 类数 = {len(by_syn)}（含全零 syndrome = 稳定子陪集）")
    multi_w = sum(1 for v in by_syn.values()
                  if len({sum(x) for x in v}) > 1)
    print(f"     其中'含多个不同重量'的 syndrome 类 = {multi_w} / {len(by_syn)}"
          f"  ⇒ L2（重量）{'能' if multi_w==0 else '不能'}单独决定 syndrome")

    # Q3: 稳定子 = 零 syndrome 陪集
    stab = [x for x in allw if syndrome(H, x) == tuple([0] * nv)]
    print(f"\n[Q3] 零 syndrome 陪集（= 稳定子 span）大小 = {len(stab)} = 2^rank")
    print(f"     稳定子非零最小重量 = {min(sum(x) for x in stab if any(x))}")
    # 逻辑：非零 syndrome 的陪集数
    print(f"     非零 syndrome 陪集数 = {len(by_syn) - 1} ⇒ k = log2(陪集总数) = "
          f"{np.log2(len(by_syn)):.2f}")

    print("\n--- Zero 分层 ↔ 码结构 对应表 ---")
    rows = [
        ("L0 底层", "零和/校验约束", "稳定子群（零 syndrome 的核）", "约束的来源"),
        ("L1 历史层", "精确错误向量", "syndrome（等价于错误向量）", "**纠错信息所在层**"),
        ("L1′ 闭合类层", "旋转类（循环移位轨道）", "记录粗粒化（有损）", "不是纠错层"),
        ("L2 演化层", "局域补偿移动", "Hamming 重量 / 最小权重解码", "**MWD 所在层**"),
        ("R 读出面", "syndrome 读出", "= L1 的等价表示", "读出接口"),
    ]
    print(f"    {'层':12s} {'Zero 对象':20s} {'码结构对应':28s} {'角色'}")
    for a, b, c, d in rows:
        print(f"    {a:12s} {b:20s} {c:28s} {d}")


if __name__ == "__main__":
    main()
