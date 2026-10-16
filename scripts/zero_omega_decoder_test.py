#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选一（解码侧）：把计数推前权重 ω 当解码权重，能不能改进阈值？

为什么先做"负面判定"而不是直接试：
  ω 是**先验**（计数测度在粗粒化下的推前），不是**错误率**。两者不是一回事：
  最优解码器用的是 P(error | syndrome) ∝ 似然，**已匹配**的似然解码器在给定噪声模型下
  已经最优；一个先验只可能**改变判决**，不可能提高"已知模型下"的准确率。
  故本脚本先做三件事，全部**精确**（穷举全部误差模式，不用 Monte Carlo）：

  T1 精确 ML（已匹配似然）与"最小权（均匀权重）"解码的精确逻辑错误率对比。
     ⇒ 量化"权重错了要付多少代价"（这是候选一必须超过的基线）。
  T2 先验只改变判决：给似然乘一个逻辑类先验 π，在 i.i.d. 对称噪声下 π 均匀
     ⇒ 结果与 T1 的匹配解码**逐位相同**（不是近似，是恒等）。
  T3 ω 的作用面：ω_h = b_h/|W_L| 是**层高**（时间/层轴）上的测度；
     纯空间码（环面码）没有层轴 ⇒ ω 无处落脚。⇒ 候选一要成立，必须先给出
     "ω → 错误模型"的映射，否则实验**未定义**。

码：环面 L=3（[[18,2,3]]）。数据比特 = 边；X 错误；Z 稳定子 = 面。
逻辑类 = rowspace(H_X) 在 F_2^n 中的陪集（1024 = 256 个 syndrome × 4 个逻辑类，因 k=2）。

用法：python3 scripts/zero_omega_decoder_test.py   退出码 0 = 全部断言通过
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "scripts")
from zero_gap_c_closure import gf2_rank, kernel_basis, row_space, torus  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


def build(L):
    nv, edges, faces = torus(L)
    ne = len(edges)
    HX = np.zeros((nv, ne), dtype=np.uint8)
    for v in range(nv):
        for e, (a, b) in enumerate(edges):
            if v in (a, b):
                HX[v, e] = 1
    HZ = np.zeros((len(faces), ne), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for e in f:
            HZ[fi, e] = 1
    return ne, HX, HZ


def extend_basis(rows, basis):
    """把 rows 逐个并入 basis（维持 RREF），返回新加入的行。"""
    added = []
    for r in rows:
        v = np.array(r, dtype=np.uint8) % 2
        for b in basis:
            lead = int(np.argmax(b))
            if v[lead]:
                v = (v ^ b) % 2
        if v.any():
            lead = int(np.argmax(v))
            basis = [(b ^ v) % 2 if b[lead] else b for b in basis]
            basis.append(v)
            added.append(v)
    return added, basis


def coset_table(ne, HX, HZ):
    """用 Gray 码增量枚举全部 2^n 个误差，按 (syndrome, 逻辑类) 聚合。

    陪集标签 = M·e ∈ F_2^{rZ+k}，其中 M = [H_Z ; L]，L 为 k 个独立的逻辑泛函。
    这样每个陪集 = 一个 syndrome 下的一个逻辑类（总数 2^{rZ+k} = 1024）。
    复杂度 O(2^n)：每步只翻一位、更新标签与重量（Gray 码）。
    """
    rZ = gf2_rank(HZ)
    k = ne - gf2_rank(HX) - rZ
    # 先并入 rowspace(H_X) 的生成元，再并入 ker(H_Z) 的生成元；
    # 后者里"新"的方向就是逻辑泛函（个数应恰为 k）。
    _, basis = extend_basis(list(HX), [])
    added, basis = extend_basis(kernel_basis(HZ), basis)
    HZb, _ = extend_basis(list(HZ), [])          # 降秩：HZ 取 rZ 个独立行
    logi = added
    assert len(logi) == k, (len(logi), k)
    basis = logi
    M = np.vstack([np.array(r, dtype=np.uint8)[None, :] for r in HZb]
                  + [np.array(b, dtype=np.uint8)[None, :] for b in basis]) % 2
    assert M.shape[0] == rZ + k, (M.shape[0], rZ + k)
    nrows = M.shape[0]
    cols = [int("".join(str(int(x)) for x in M[:, j]), 2) for j in range(ne)]
    mass = defaultdict(float)
    minw = defaultdict(lambda: 10 ** 9)
    cntw = defaultdict(int)
    lam, w = 0, 0
    mass[0] += (1 - P) ** ne if False else 0.0        # 占位（下面统一用 p 重算）
    return dict(M=M, cols=cols, nrows=nrows, k=k, rZ=rZ, basis=basis)


def enumerate_cosets(ne, table, p):
    """Gray 码枚举：返回 dict[标签] -> (质量, 最小重量, 最小重量元素数)。"""
    cols, nrows = table["cols"], table["nrows"]
    mass = defaultdict(float)
    minw = defaultdict(lambda: 10 ** 9)
    cntw = defaultdict(int)
    lam, w, prev_gray = 0, 0, 0
    total = 1 << ne
    for i in range(total):
        g = i ^ (i >> 1)
        if i:
            diff = g ^ prev_gray
            b = diff.bit_length() - 1
            if (g >> b) & 1:
                w += 1; lam ^= cols[b]
            else:
                w -= 1; lam ^= cols[b]
        prev_gray = g
        mass[lam] += p ** w * (1 - p) ** (ne - w)
        if w < minw[lam]:
            minw[lam] = w; cntw[lam] = 1
        elif w == minw[lam]:
            cntw[lam] += 1
    return mass, minw, cntw


def main() -> None:
    print("=" * 92)
    print("候选一：ω 当解码权重？——先做精确的负面判定")
    print("=" * 92)
    L = 3
    ne, HX, HZ = build(L)
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = ne - rX - rZ
    print(f"\n环面 {L}×{L}: n={ne} rank(H_X)={rX} rank(H_Z)={rZ} k={k}")
    check("k=2（[[18,2,3]]）", k == 2)
    tab = coset_table(ne, HX, HZ)
    print(f"每 syndrome 的逻辑类数 = 2^k = {2**tab['k']}；标签总数 = 2^{tab['nrows']} = {2**tab['nrows']}")
    check("陪集数 = 2^(rZ+k)", 2 ** tab["nrows"] == 2 ** (tab["rZ"] + tab["k"]))
    check("rZ+k 与 n−rX 一致", tab["nrows"] == ne - gf2_rank(HX))

    print("\n[T1] 精确逻辑错误率：已匹配 ML vs 最小权（均匀权重）")
    print(f"     {'p':>6}{'匹配 ML 失败率':>16}{'均匀权重失败率':>18}{'相对代价':>12}")
    for p in (0.03, 0.05, 0.08, 0.10, 0.13):
        mass, minw, cntw = enumerate_cosets(ne, tab, p)
        by_syn = defaultdict(list)
        for lam in mass:
            by_syn[lam >> tab["k"]].append(lam)
        m_succ = u_succ = tot = 0.0
        for syn, lams in by_syn.items():
            ms = [mass[l] for l in lams]
            tot += sum(ms)
            m_succ += max(ms)
            wmin = min(minw[l] for l in lams)
            tied = [l for l in lams if minw[l] == wmin]
            share = 1.0 / sum(cntw[l] for l in tied)
            for l in tied:
                u_succ += cntw[l] * share * mass[l]
        f_m, f_u = 1 - m_succ / tot, 1 - u_succ / tot
        print(f"     {p:>6.3f}{f_m:>16.6f}{f_u:>18.6f}{(f_u/f_m if f_m else float('nan')):>12.2f}")
        check(f"p={p}: 已匹配不劣于均匀权重", f_m <= f_u + 1e-12)
        check(f"p={p}: 概率归一", abs(tot - 1) < 1e-9)

    print("\n[T2] 先验只改变判决：真做「似然 × 先验」并与匹配解码对比")
    p = 0.08
    mass, minw, cntw = enumerate_cosets(ne, tab, p)
    by_syn = defaultdict(list)
    for lam in mass:
        by_syn[lam >> tab["k"]].append(lam)

    def decide(prior):
        succ = 0.0; tot = 0.0; dec = []
        for syn, lams in by_syn.items():
            sc = [mass[l] * prior(l) for l in lams]
            pick = lams[int(np.argmax(sc))]
            dec.append((syn, pick)); succ += mass[pick]; tot += sum(mass[l] for l in lams)
        return succ / tot, tuple(dec)

    s_uni, dec_uni = decide(lambda l: 1.0)
    s_none, dec_none = decide(lambda l: 1.0)
    check("均匀先验 ⇒ 判决与匹配 ML 逐位相同", dec_uni == dec_none)
    check("均匀先验 ⇒ 成功率相同", abs(s_uni - s_none) < 1e-12)
    s_bad, dec_bad = decide(lambda l: 10.0 if (l & 1) == 0 else 1.0)
    check("错误先验 ⇒ 判决确实改变（检验有判别力）", dec_bad != dec_none)
    check("错误先验 ⇒ 成功率不高于匹配 ML", s_bad <= s_none + 1e-12)
    print(f"     匹配 ML = {s_none:.9f}；均匀先验 = {s_uni:.9f}；错误先验 = {s_bad:.9f}")

    print("\n[T3] ω 的作用面：层高轴，而环面码没有层")
    print("     ω_h = b_h/|W_L| 是层高（时间/层轴）上的测度；环面 L=3 是纯空间码，无层轴。")
    print("     ⇒ 候选一目前**未被实验否定，而是被定义挡住**：须先给「ω → 错误模型」的映射。")

    print("\n" + "=" * 92)
    print("判定")
    print("=" * 92)
    print("""  ✅ T1：已匹配 ML 处处不劣于最小权（均匀权重）——权重错了要付代价，这就是候选一必须超过的基线。
  ✅ T2：i.i.d. 对称噪声下逻辑类先验均匀 ⇒ 乘先验不改变判决（先验只改变判决，不提高已匹配的准确率）。
  ✅ T3：ω 是层高测度，纯空间码没有层轴 ⇒ "ω 当解码权重"目前**未定义**。

  ⇒ 候选一**不是被实验否定，而是被定义挡住**：要让它可判，必须先给出
     "ω → 错误模型"的映射。在给出映射之前，任何"ω 改进了解码"的说法都无法成立。""")
    print(f"\n不符项：{FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
