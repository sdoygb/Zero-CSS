#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tqec 电路上的简并画像（用 tqec 自己生成的电路 ＋ stim 的 DEM）

目的
  在 tqec 生成的电路上测量权重 1/2 层的简并画像：
    权重1 唯一率、权重2 类大小分布、fail(2) = 1 - <1/v>
  两种口径并列：
    口径A = 共享同一 syndrome 的机制归为一类
    口径B = 共享同一 syndrome 且同一逻辑类（真简并）
  数据来自 tqec 的电路生成 ＋ stim 的 detector error model，不依赖手写稳定子模型。

运行（需 tqec 环境）
  cd <tqec 工作副本> && uv run python tqec_degeneracy.py
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np
import stim
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.gallery.memory import memory
from tqec.utils.noise_model import NoiseModel


def tqec_memory_circuit(k: int = 1) -> stim.Circuit:
    """用 tqec 生成逻辑记忆电路（gallery 图），加噪声后返回 stim 电路。"""
    g = memory()
    compiled = compile_block_graph(g, FIXED_BULK_CONVENTION, "auto")
    tree = compiled.to_layer_tree()
    ideal = tree.generate_circuit(k, database_path=None)
    return NoiseModel.uniform_depolarizing(0.001).noisy_circuit(ideal)


def dem_to_matrix(circuit: stim.Circuit):
    """DEM -> (H, L, probs)：H[d,e] 机制 e 是否翻转探测器 d；L[o,e] 是否翻转可观察量 o。"""
    dem = circuit.detector_error_model(decompose_errors=False)
    mechs = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.add(t.val)
            elif t.is_logical_observable_id():
                obs.add(t.val)
        mechs.append((frozenset(dets), frozenset(obs), float(p)))
    n_det, n_obs = circuit.num_detectors, circuit.num_observables
    H = np.zeros((n_det, len(mechs)), dtype=np.uint8)
    L = np.zeros((n_obs, len(mechs)), dtype=np.uint8)
    for e, (dets, obs, _) in enumerate(mechs):
        for d in dets:
            H[d, e] = 1
        for o in obs:
            L[o, e] = 1
    return H, L, np.array([p for _, _, p in mechs])


def degeneracy_profile(H, L, probs, pool=None):
    """权重 1/2 层简并画像（两种口径）。"""
    n = len(probs)
    sel = list(range(n)) if (pool is None or pool >= n) else list(np.argsort(-probs)[:pool])
    syn = {e: tuple(np.nonzero(H[:, e])[0]) for e in sel}
    log = {e: tuple(np.nonzero(L[:, e])[0]) for e in sel}

    by_syn = defaultdict(list)
    for e in sel:
        by_syn[syn[e]].append(e)
    w1_unique = sum(1 for v in by_syn.values() if len(v) == 1) / len(sel)

    A, B = Counter(), Counter()
    for a, b in itertools.combinations_with_replacement(sel, 2):
        s = tuple(sorted(set(syn[a]) ^ set(syn[b])))
        A[s] += 1
        B[(s, tuple(sorted(set(log[a]) ^ set(log[b]))))] += 1
    vA = np.array(list(A.values()), float)
    vB = np.array(list(B.values()), float)
    failA = float(1 - np.mean(1 / vA))
    failB = float(1 - np.mean(1 / vB))
    return {
        "n_mech": n, "n_used": len(sel),
        "w1_unique": w1_unique,
        "A_classes": len(vA), "A_fail2": failA, "A_rec": 1 - failA,
        "A_mean": float(vA.mean()), "A_max": int(vA.max()),
        "A_hist": dict(sorted(Counter(vA.astype(int)).items())[:6]),
        "B_classes": len(vB), "B_fail2": failB, "B_rec": 1 - failB,
        "B_mean": float(vB.mean()), "B_max": int(vB.max()),
    }


def main() -> None:
    print("=" * 78)
    print("tqec 电路上的简并画像（tqec 生成电路 + stim DEM）")
    print("=" * 78)
    for k in (1, 2, 3):
        try:
            circ = tqec_memory_circuit(k)
        except Exception as exc:                       # noqa: BLE001
            print(f"\n=== k={k}: 生成失败（{type(exc).__name__}: {exc}）")
            continue
        H, L, probs = dem_to_matrix(circ)
        print(f"\n=== k={k}  qubits={circ.num_qubits} detectors={circ.num_detectors} "
              f"observables={circ.num_observables} 错误机制={len(probs)}")
        if len(probs) == 0:
            print("    无错误机制（噪声未生效）")
            continue
        r = degeneracy_profile(H, L, probs)
        print(f"    权重1 唯一率 = {r['w1_unique']*100:.1f}%  （{r['n_used']} 条机制）")
        print(f"    权重2 类直方图 = {r['A_hist']}")
        print(f"    口径A 同 syndrome          : 类数={r['A_classes']:6d} 平均={r['A_mean']:.3f} "
              f"最大={r['A_max']:5d}  fail(2)={r['A_fail2']:.4f}  恢复率={r['A_rec']*100:.1f}%")
        print(f"    口径B 同 syndrome+同逻辑类  : 类数={r['B_classes']:6d} 平均={r['B_mean']:.3f} "
              f"最大={r['B_max']:5d}  fail(2)={r['B_fail2']:.4f}  恢复率={r['B_rec']*100:.1f}%")


if __name__ == "__main__":
    main()
