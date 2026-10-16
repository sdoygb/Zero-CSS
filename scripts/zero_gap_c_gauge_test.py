#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口 (c) 的规范检验：不同胞腔化是否给出同一码？

判据（全部可判、退出码 0 = 全过）：
  G1 Euler：每个合法胞腔化都满足 |F| = E−V+2−2g（环面 g=1 ⇒ |F|=E−V）
  G2 每边恰属 2 个面 ⇒ rank(H_Z)=|F|−1
  G3 k = 2g（与取哪个胞腔化无关）
  G4 d_X、d_Z 在所有合法胞腔化之间**相同**（→ 面集是规范）
  G5 标准方格面集必在解中，且其值为 toric code [[2L², 2, L]]

去重：DFS 会由不同访问顺序重复到达同一解，报数前必须按面集去重。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, "scripts")
from zero_gap_c_closure import codespace_params, gf2_rank  # noqa: E402
from zero_gap_c_closure_dfs import (  # noqa: E402
    build_HX,
    cycles_of_weight,
    dfs_cellulations,
    torus,
)

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


def main() -> None:
    print("=" * 88)
    print("缺口 (c) 规范检验：不同胞腔化 → 同一码？")
    print("=" * 88)
    for L in (3, 4):
        nv, edges, faces = torus(L)
        ne = len(edges)
        HXn = np.array(build_HX(nv, edges), dtype=np.uint8)
        beta1 = ne - nv + 1
        g = 1
        c4 = cycles_of_weight(ne, HXn.tolist(), 4)
        sols, missing = dfs_cellulations(nv, ne, c4)
        assert missing is None, missing
        uniq = [sorted(u) for u in {frozenset(x) for x in sols}]
        std = frozenset(i for i, f in enumerate(c4) if set(f) in [set(x) for x in faces])
        print(f"\n环面 {L}×{L}: V={nv} E={ne} β₁={beta1}  4-圈={len(c4)}  去重后胞腔化={len(uniq)}")
        rows = []
        for j, s in enumerate(uniq):
            fs = [c4[i] for i in s]
            rhz, k, dX, dZ = codespace_params(fs, ne, HXn, beta1)
            rows.append((len(fs), rhz, k, dX, dZ))
            tag = "  ← 标准方格面" if frozenset(s) == std else ""
            print(f"     #{j+1}: |F|={len(fs):2d} rank={rhz:2d} k={k}  d_X={dX}  d_Z={dZ}{tag}")
        check(f"L={L}: 每个解 |F|=E−V={ne-nv}", all(r[0] == ne - nv for r in rows))
        check(f"L={L}: 每个解 rank=|F|−1", all(r[1] == r[0] - 1 for r in rows))
        check(f"L={L}: 每个解 k=2g=2", {r[2] for r in rows} == {2}, f"k ∈ {sorted({r[2] for r in rows})}")
        check(f"L={L}: d_X 与胞腔化无关", len({r[3] for r in rows}) == 1,
              f"d_X ∈ {sorted({r[3] for r in rows})}")
        check(f"L={L}: d_Z 与胞腔化无关", len({r[4] for r in rows}) == 1,
              f"d_Z ∈ {sorted({r[4] for r in rows})}")
        check(f"L={L}: 标准方格面在解中", std in {frozenset(s) for s in uniq})
        i_std = [i for i, s in enumerate(uniq) if frozenset(s) == std][0]
        check(f"L={L}: 标准解 = toric code [[{ne},2,{L}]]",
              (rows[i_std][2], rows[i_std][3], rows[i_std][4]) == (2, L, L),
              f"实测 {(rows[i_std][2], rows[i_std][3], rows[i_std][4])}")
    print("\n" + "=" * 88)
    print(f"不符项：{FAIL}   结论：面集选择对 (k,d) 是规范；真正的输入是亏格 g（k=2g）")
    print("=" * 88)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
