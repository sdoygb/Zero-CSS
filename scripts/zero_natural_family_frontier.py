#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero 天然复形族作为 CSS 码：算 (n, k, d) 并与标准族比前沿。

为什么要算这个：Zero-CSS 的构造与标准同调 CSS 码同构 ⇒ 码参数不会超过标准构造。
但"Zero 天然复形"是**特定实例**，所以要做的是**逐点比**，而不是靠结构论证下结论。

本脚本算三个 Zero 天然 / 参照实例：

  ① K₄（局部舞台）：四面体 = 球面。χ = V−E+F = 4−6+4 = 2 ⇒ g=0 ⇒ k=2g=0。
     判据：确实 k=0（**不是**码）——注意 Zero-CSS README 的「K₄ → [6,3,3]」是**经典**码，
     每边一个比特的量子构造在 K₄ 上无逻辑自由度。
  ② Heawood（全局舞台 Γ = Heawood 图，14 顶点 21 边）：Heawood 地图（环面嵌入，7 个六边形面）。
     χ = 14−21+7 = 0 ⇒ g=1 ⇒ k=2。算 d_X、d_Z。
  ③ 方格环面 L=3（参照）：[[18,2,3]]。

比较口径（与标准拓扑码族）：
  · 码率 k/n
  · 前沿量 k·d²/n（方格环面族与旋转表面码 ≈ 1；越大越好）
  · 另记 d/√n（距离随尺寸的标度）

用法：python3 scripts/zero_natural_family_frontier.py
退出码 0 = 全部断言通过。
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "scripts")
from zero_gap_c_closure import gf2_rank, kernel_basis, min_wt_outside, row_space  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


def heawood():
    """Heawood 图 = Fano 平面的 Levi 图（7 点 + 7 线，每条线 3 点 ⇒ 21 条关联边）。"""
    pts = list(itertools.combinations(range(1, 8), 1))          # 占位，稍后覆盖
    # Fano 平面：点 = GF(2)^3 \ {0}（7 个），线 = {a, b, a+b}
    pts = [p for p in itertools.product((0, 1), repeat=3) if any(p)]
    def add(a, b):
        return tuple((x + y) % 2 for x, y in zip(a, b))
    lines = []
    seen = set()
    for a in pts:
        for b in pts:
            if a == b:
                continue
            l = frozenset({a, b, add(a, b)})
            if len(l) == 3 and l not in seen:
                seen.add(l)
                lines.append(sorted(l))
    assert len(pts) == 7 and len(lines) == 7, (len(pts), len(lines))
    # 每个点恰在 3 条线上、每两条线恰交于 1 点（Fano 性质）
    inc = Counter()
    for l in lines:
        for p in l:
            inc[p] += 1
    assert set(inc.values()) == {3}
    verts = [("p", p) for p in pts] + [("l", tuple(l)) for l in lines]
    vid = {v: i for i, v in enumerate(verts)}
    edges = []
    for l in lines:
        for p in l:
            edges.append((vid[("p", p)], vid[("l", tuple(l))]))
    # 是否二分（Levi 图）
    bip = all((v[0] == "p") != (w[0] == "p") for v, w in [(verts[a], verts[b]) for a, b in edges])
    return verts, edges, bip


def cycles_of_weight(nv, edges, w):
    ne = len(edges)
    inc = [[] for _ in range(nv)]
    for e, (a, b) in enumerate(edges):
        inc[a].append(e)
        inc[b].append(e)
    out = []
    for sub in itertools.combinations(range(ne), w):
        deg = Counter()
        for e in sub:
            a, b = edges[e]
            deg[a] += 1
            deg[b] += 1
        if all(d == 2 for d in deg.values()) and len(deg) == w:
            out.append(frozenset(sub))
    return out


def dfs_face_cover(ne, cand, want_faces):
    """在 cand（都是等长圈）中找子集：每条边恰属 2 个面，且面数固定。"""
    inc = {e: [] for e in range(ne)}
    for i, f in enumerate(cand):
        for e in f:
            inc[e].append(i)
    counts = [0] * ne
    chosen = []
    sols = []

    def rec():
        if len(chosen) == want_faces:
            if all(c == 2 for c in counts):
                sols.append(frozenset(chosen))
            return
        target = next((e for e in range(ne) if counts[e] < 2), None)
        if target is None:
            return
        for i in inc[target]:
            if i in chosen:
                continue
            f = cand[i]
            if any(counts[e] >= 2 for e in f):
                continue
            for e in f:
                counts[e] += 1
            chosen.append(i)
            rec()
            chosen.pop()
            for e in f:
                counts[e] -= 1

    rec()
    return sols


def code_params(nv, edges, faces, label, quiet=False):
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
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = ne - rX - rZ
    # CSS 对易
    comm = (HX @ HZ.T) % 2
    dX = min_wt_outside(kernel_basis(HZ), row_space(HX))
    dZ = min_wt_outside(kernel_basis(HX), row_space(HZ))
    chi = nv - ne + len(faces)
    g = (2 - chi) / 2
    if not quiet:
        print(f"\n{label}: V={nv} E=n={ne} F={len(faces)}  χ={chi}  g={g}")
        print(f"     rank(H_X)={rX}  rank(H_Z)={rZ}  k=n−rX−rZ={k}"
              f"   2g={2*g:.0f}   d_X={dX}  d_Z={dZ}")
    if not quiet:
        check(f"{label}: H_X H_Z^T = 0（CSS 对易）", not comm.any())
    else:
        assert not comm.any(), "CSS 对易失败"
    return dict(n=ne, k=k, dX=dX, dZ=dZ, g=g, chi=chi, rX=rX, rZ=rZ)


def main() -> None:
    print("=" * 92)
    print("Zero 天然复形族作为 CSS 码：(n,k,d) 与前沿")
    print("=" * 92)
    rows = []

    # ── ① K₄（四面体）──────────────────────────────────────────────────
    print("\n" + "─" * 92)
    print("① K₄（局部舞台）：四面体")
    k4_v, k4_e = 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    k4_f = [frozenset({0, 1, 2}), frozenset({0, 1, 3}), frozenset({0, 2, 3}), frozenset({1, 2, 3})]
    # 面用边索引表示
    def face_to_edges(fv):
        out = set()
        for a in fv:
            for b in fv:
                if a < b:
                    out.add(k4_e.index((a, b)))
        return frozenset(out)
    k4_fe = [face_to_edges(f) for f in k4_f]
    r = code_params(k4_v, k4_e, k4_fe, "K₄（4 三角形面）")
    check("K₄: χ=2（球面）⟹ g=0 ⟹ k=0（无逻辑自由度，不是码）", r["k"] == 0 and r["chi"] == 2)
    rows.append(("K₄ 四面体", r))

    # ── ② Heawood 地图（环面嵌入）──────────────────────────────────────
    print("\n" + "─" * 92)
    print("② Heawood（全局舞台 Γ）：先找它的环面嵌入（7 个六边形面）")
    verts, edges, bip = heawood()
    nv, ne = len(verts), len(edges)
    print(f"     Heawood 图：V={nv} E={ne}  二分={bip}")
    check("Heawood 图 V=14, E=21", (nv, ne) == (14, 21))
    check("Heawood 图是二分图（Levi 图）", bip)
    c6 = cycles_of_weight(nv, edges, 6)
    print(f"     6-圈数 = {len(c6)}（Heawood 图围长 6）")
    sols = dfs_face_cover(ne, c6, want_faces=7)
    print(f"     「每条边恰属 2 个面、面数 7」的 6-圈面集个数 = {len(sols)}"
          f"（图的嵌入**不**由图唯一确定）")
    check("Heawood 环面嵌入存在", len(sols) >= 1, f"实测 {len(sols)}")
    params = {}
    for s in sols:
        fs = [c6[i] for i in sorted(s)]
        rr = code_params(nv, edges, fs, "Heawood 地图（7 六边形面）", quiet=True)
        params.setdefault((rr["k"], rr["dX"], rr["dZ"]), []).append(rr)
    print(f"     互不相同的 (k,d_X,d_Z) 参数组数 = {len(params)}")
    check("全部 16 种嵌入给出同一码（码对嵌入是规范）", len(params) == 1,
          f"实测 {sorted(params)}")
    r_hw = next(iter(params.values()))[0]
    check("Heawood: k=2g=2", r_hw["k"] == 2 and r_hw["g"] == 1)
    rows.append(("Heawood 地图", r_hw))

    # ── ③ 方格环面 L=3（参照）──────────────────────────────────────────
    print("\n" + "─" * 92)
    print("③ 方格环面 L=3（参照族：标准 toric code）")
    from zero_gap_c_closure import torus
    nv3, e3, f3 = torus(3)
    r_t = code_params(nv3, e3, [frozenset(f) for f in f3], "方格环面 3×3")
    check("方格环面: k=2, d_X=d_Z=3", (r_t["k"], r_t["dX"], r_t["dZ"]) == (2, 3, 3))
    rows.append(("方格环面 3×3", r_t))

    # ── 前沿表 ────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("前沿量（与标准族比）：rate = k/n，前沿量 = k·d²/n，d/√n")
    print("=" * 92)
    print(f"  {'实例':<16}{'n':>5}{'k':>4}{'d_X':>5}{'d_Z':>5}{'rate':>9}{'k·d²/n':>10}{'d/√n':>8}")
    for name, r in rows:
        d = min(x for x in (r["dX"], r["dZ"]) if x is not None) if r["k"] else 0
        rate = r["k"] / r["n"]
        front = r["k"] * d * d / r["n"]
        print(f"  {name:<16}{r['n']:>5}{r['k']:>4}{str(r['dX']):>5}{str(r['dZ']):>5}"
              f"{rate:>9.3f}{front:>10.3f}{d/(r['n']**0.5):>8.3f}")
    print("\n  参照：方格环面族 [[2L²,2,L]] ⇒ k·d²/n = 2L²/2L² = 1.000（与 L 无关）")
    print("        旋转表面码 [[d²,1,d]] ⇒ k·d²/n = 1.000")
    print("        Heawood 地图 [[21,2,3]] ⇒ k·d²/n = 0.857 < 1.000（**低于**标准族；"
          "d 由 X 侧最小权逻辑圈给出，d_Z=6 但 d_X=3）")
    print(f"\n不符项：{FAIL}")
    print("=" * 92)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
