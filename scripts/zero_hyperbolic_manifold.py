#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""显式双曲三角化：用流形判据筛选合法构造，再算 k

前两次失败的原因
  随机粘合"面角 → 顶点"时没检查**流形条件**，得到的是伪流形 ⇒
  对易列全 False（构造非法），却当成 Zero 的结果。

正确做法
  三角形复形是闭曲面 ⟺
    (i)  每边恰属 2 个三角形（边的配对，构造即保证）
    (ii) 每个顶点的 **link 是一个简单圈**（流形条件）
  生成随机构造后**逐个检验 (ii)**，只保留通过的。

参数
  F 个三角形 ⇒ 3F 个面角；顶点数 V 由"每顶点 q 个角"决定 ⇒ V = 3F/q
  χ = V − E + F，E = 3F/2
  闭曲面：χ = 2 − 2g
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def rank2(M):
    A = np.array(M, dtype=np.uint8) % 2
    if A.size == 0:
        return 0
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if A[i, c]:
                p = i
                break
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


def link_is_cycle(lk):
    """lk: 某顶点的邻接顶点多重集（应构成一个简单圈：每点度 2、连通）。"""
    deg = defaultdict(int)
    adj = defaultdict(set)
    for a, b in lk:
        deg[a] += 1
        deg[b] += 1
        adj[a].add(b)
        adj[b].add(a)
    if not deg:
        return False
    if any(d != 2 for d in deg.values()):
        return False
    # 连通性
    start = next(iter(adj))
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(adj)


def try_build(F, q, seed, max_try=400):
    """尝试构造 F 个三角形、每顶点 q 个面角的闭曲面三角化。"""
    for attempt in range(max_try):
        rnd = random.Random(seed * 10007 + attempt)
        # 边配对
        half = [(f, k) for f in range(F) for k in range(3)]
        rnd.shuffle(half)
        eid, edges = {}, []
        for i in range(0, len(half), 2):
            a, b = half[i], half[i + 1]
            e = len(edges)
            edges.append((a, b))
            eid[a] = e
            eid[b] = e
        # 顶点：面角 -> 顶点（每 q 个一组）
        corners = [(f, k) for f in range(F) for k in range(3)]
        rnd.shuffle(corners)
        vid = {}
        V = 0
        ok = True
        for i in range(0, len(corners), q):
            grp = corners[i:i + q]
            if len(grp) < q:
                ok = False
                break
            for c in grp:
                vid[c] = V
            V += 1
        if not ok:
            continue
        # 流形检验：每个顶点的 link 是简单圈
        # 三角形 (f,0),(f,1),(f,2) 的角；角 (f,k) 的对边是 (f,k)，相邻角 (f,k+1),(f,k-1)
        links = defaultdict(list)
        for f in range(F):
            c0, c1, c2 = vid[(f, 0)], vid[(f, 1)], vid[(f, 2)]
            # 每个角的 link 边 = 该角所在的两条邻边连接的两个顶点
            links[c0].append((c1, c2))
            links[c1].append((c2, c0))
            links[c2].append((c0, c1))
        if all(link_is_cycle(lk) for lk in links.values()):
            return V, edges, vid, eid, attempt
    return None


def main() -> None:
    print("=" * 96)
    print("显式双曲三角化：流形判据筛选 + 直接算 k")
    print("=" * 96)
    print("""
  三角形复形：E = 3F/2，V = 3F/q，χ = V − E + F = 2 − 2g
    q = 6 ⇒ χ = 0（平坦/环面）
    q > 6 ⇒ χ < 0（双曲）
""")
    for q in (6, 7, 8):
        for F in (12, 24, 48, 96):
            if (3 * F) % 2:
                continue
            r = try_build(F, q, seed=F + q)
            if r is None:
                print("  q=%d F=%3d ：未找到合法构造（%d 次尝试）" % (q, F, 400))
                continue
            V, edges, vid, eid, attempt = r
            ne = len(edges)
            # 校验边数
            if ne != 3 * F // 2:
                print("  q=%d F=%3d ：边数异常 %d" % (q, F, ne))
                continue
            chi = V - ne + F
            g = (2 - chi) / 2
            # 构造 HX, HZ
            HX = np.zeros((V, ne), dtype=np.uint8)
            for f in range(F):
                for k in range(3):
                    HX[vid[(f, k)], eid[(f, k)]] = 1
                    HX[vid[(f, (k + 1) % 3)], eid[(f, k)]] = 1
            HZ = np.zeros((F, ne), dtype=np.uint8)
            for f in range(F):
                for k in range(3):
                    HZ[f, eid[(f, k)]] = 1
            comm = not ((HX @ HZ.T) % 2).any()
            rX, rZ = rank2(HX), rank2(HZ)
            k = ne - rX - rZ
            print("  q=%d F=%3d  尝试#%-4d V=%3d E=%3d χ=%4d g=%5.1f  rX=%3d rZ=%3d  "
                  "k=%4d  R=k/E=%6.3f  对易=%s"
                  % (q, F, attempt, V, ne, chi, g, rX, rZ, k, k / ne, comm))

    print("\n" + "=" * 96)
    print("判读")
    print("=" * 96)
    print("""  · q=6（平坦，χ=0）：预期 k 恒定（与环面一致）
  · q>6（双曲，χ<0）：若 k 随 F 增长 ⇒ 码率不退化 ⇒ Zero 原生高码率码存在
  · 必须核对：对易=True（流形条件）+ χ 符号正确""")


if __name__ == "__main__":
    main()
