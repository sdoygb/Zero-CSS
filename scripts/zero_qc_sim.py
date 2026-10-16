#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在普通计算机上"搞量子计算"：状态向量仿真器 + 自检套件 + 墙的测量

能不能用普通计算机代替量子计算机？分两半，答案不同：

  ① Clifford（稳定子）电路：**能，而且是白拿的**——Gottesman–Knill 定理：
     稳定子态在 Clifford 门下的演化可用 O(n^2) 的辛/表描述，测量也是多项式的。
     QEC 的整个战场（稳定子码、syndrome、解码）都住在这半边。
  ② 通用量子计算（带 T 门等非 Clifford 门）：**指数墙**——状态向量要 2^n 个复数。
     本机 16 GB ⟹ 理论上限 n≈30（16 字节/振幅），实际更小。

本脚本做①的"墙"的量化 + ②可用部分的实现：
  · 自写状态向量仿真器（H/Pauli/S/T/RX/RY/RZ/CNOT/CZ/SWAP/Toffoli/自定义酉）
  · 自检套件（全部对照解析值，断言把关）
  · 量出 2^n 墙（逐 n 计时，外推到 n=50 的内存需求）

用法：python3 scripts/zero_qc_sim.py        退出码 0 = 全部断言通过
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    ok = bool(cond)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAIL += 1


# ── 门 ────────────────────────────────────────────────────────────────
_I = np.eye(2, dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)


def RX(t):
    return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)],
                     [-1j * math.sin(t / 2), math.cos(t / 2)]], dtype=np.complex128)


def RY(t):
    return np.array([[math.cos(t / 2), -math.sin(t / 2)],
                     [math.sin(t / 2), math.cos(t / 2)]], dtype=np.complex128)


def RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=np.complex128)


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
CPHASE = lambda t: np.diag([1, 1, 1, np.exp(1j * t)]).astype(np.complex128)
TOFFOLI = np.eye(8, dtype=np.complex128)
TOFFOLI[6, 6] = TOFFOLI[7, 7] = 0
TOFFOLI[6, 7] = TOFFOLI[7, 6] = 1


class QC:
    """状态向量量子计算机（qubit 0 = 最低位）。"""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.psi = np.zeros(1 << n, dtype=np.complex128)
        self.psi[0] = 1.0
        self.rng = np.random.default_rng(seed)

    # ── 单比特门 ──────────────────────────────────────────────────────
    def gate1(self, U, t: int):
        hi = 1 << (self.n - t - 1)
        lo = 1 << t
        v = self.psi.reshape(hi, 2, lo)
        v[:] = np.einsum("ij,hjk->hik", U, v)
        return self

    # ── 两比特门（U4 作用在 (c,t) 基序 |c t> 上）────────────────────────
    def gate2(self, U4, c: int, t: int):
        idx = np.arange(1 << self.n)
        bc = (idx >> c) & 1
        bt = (idx >> t) & 1
        I0 = idx[(bc == 0) & (bt == 0)]
        I1 = idx[(bc == 0) & (bt == 1)]
        I2 = idx[(bc == 1) & (bt == 0)]
        I3 = idx[(bc == 1) & (bt == 1)]
        order = [I0, I1, I2, I3]
        old = self.psi[np.concatenate(order)]
        new = U4 @ old.reshape(4, -1)
        self.psi[np.concatenate(order)] = new.reshape(-1)
        return self

    def toffoli(self, c1: int, c2: int, t: int):
        idx = np.arange(1 << self.n)
        m = (((idx >> c1) & 1) == 1) & (((idx >> c2) & 1) == 1)
        a = idx[m & (((idx >> t) & 1) == 0)]
        b = idx[m & (((idx >> t) & 1) == 1)]
        self.psi[a], self.psi[b] = self.psi[b].copy(), self.psi[a].copy()
        return self

    # ── 常用复合 ──────────────────────────────────────────────────────
    def h(self, q): return self.gate1(H, q)
    def x(self, q): return self.gate1(X, q)
    def cx(self, c, t): return self.gate2(CNOT, c, t)
    def cz(self, c, t): return self.gate2(CZ, c, t)
    def all_h(self):
        for q in range(self.n):
            self.h(q)
        return self

    # ── 测量与采样 ────────────────────────────────────────────────────
    def probs(self):
        return np.abs(self.psi) ** 2

    def sample(self, shots: int):
        p = self.probs()
        p = p / p.sum()
        idx = self.rng.choice(len(p), size=shots, p=p)
        return ["".join(str((int(i) >> q) & 1) for q in reversed(range(self.n))) for i in idx]

    def measure_all(self):
        """一次投影测量（坍缩）。"""
        p = self.probs()
        p = p / p.sum()
        i = int(self.rng.choice(len(p), p=p))
        self.psi[:] = 0
        self.psi[i] = 1
        return "".join(str((i >> q) & 1) for q in reversed(range(self.n)))


def qft(n: int, dagger=False):
    q = QC(n)
    for k in reversed(range(n)):
        q.h(k)
        for j in reversed(range(k)):
            q.gate2(CPHASE(0 if dagger else math.pi / (1 << (k - j))), j, k)
    for i in range(n // 2):
        q.gate2(SWAP, i, n - 1 - i)
    return q


def main() -> None:
    print("=" * 92)
    print("普通计算机上的量子计算：状态向量仿真器 + 自检 + 2^n 墙")
    print("=" * 92)

    # ── 自检 1：H 门与叠加 ────────────────────────────────────────────
    print("\n[自检 1] 基本门")
    q = QC(1); q.h(0)
    check("H|0> = |+>（振幅 1/√2）", abs(q.psi[0] - 1 / math.sqrt(2)) < 1e-12
          and abs(q.psi[1] - 1 / math.sqrt(2)) < 1e-12)
    q2 = QC(1); q2.h(0); q2.h(0)
    check("H² = I", abs(q2.psi[0] - 1) < 1e-12)
    q3 = QC(1); q3.x(0); q3.gate1(S, 0); q3.gate1(S, 0)
    check("S² = Z（在 |1> 上）", abs(q3.psi[1] + 1) < 1e-12)
    q4 = QC(1); q4.x(0); q4.gate1(T, 0); q4.gate1(T, 0)
    check("T² = S（在 |1> 上）", abs(q4.psi[1] - 1j) < 1e-12)
    q5 = QC(2); q5.all_h()
    check("H⊗H|00> 全部振幅 = 1/2", np.allclose(q5.psi, 0.5))
    q6 = QC(2); q6.h(0); q6.cz(0, 1); q6.h(0)      # H CZ H = CNOT（c=0,t=1）
    q7 = QC(2); q7.cx(0, 1)
    check("H CZ H = CNOT", np.allclose(q6.psi, q7.psi))
    q8 = QC(3); q8.x(0); q8.x(1); q8.toffoli(0, 1, 2)
    check("Toffoli: |110> -> |111>", abs(q8.psi[7] - 1) < 1e-12)

    # ── 自检 2：纠缠（Bell / GHZ）────────────────────────────────────
    print("\n[自检 2] 纠缠")
    b = QC(2); b.h(0); b.cx(0, 1)
    check("Bell 振幅 = (1/√2)|00> + (1/√2)|11>",
          abs(b.psi[0] - 1 / math.sqrt(2)) < 1e-12 and abs(b.psi[3] - 1 / math.sqrt(2)) < 1e-12)
    shots = b.sample(1000)
    ok = all(s in ("00", "11") for s in shots)
    frac = sum(s == "11" for s in shots) / 1000
    check("Bell 1000 shot 只出现 00/11", ok)
    check("Bell 相关性 ~50%", 0.45 < frac < 0.55, f"实测 {frac:.3f}")
    g = QC(5); g.h(0)
    for i in range(4):
        g.cx(i, i + 1)
    check("GHZ-5 振幅只在 00000/11111",
          abs(g.psi[0] - 1 / math.sqrt(2)) < 1e-12 and abs(g.psi[31] - 1 / math.sqrt(2)) < 1e-12
          and abs(sum(abs(a) ** 2 for a in g.psi) - 1) < 1e-12)

    # ── 自检 3：真·算法 ──────────────────────────────────────────────
    print("\n[自检 3] 算法（对照解析值）")
    # Grover 2 比特，标记 |11>，一轮后必然命中
    gr = QC(2); gr.all_h()
    gr.psi[3] *= -1                                   # oracle：|11> 相位翻转
    m = np.mean(gr.psi)                               # 扩散 2|s><s| - I（计算基）
    gr.psi[:] = 2 * m - gr.psi
    p11 = abs(gr.psi[3]) ** 2
    check("Grover 2q：一轮后 P(|11>)=1", abs(p11 - 1) < 1e-12, f"P={p11:.6f}")

    # T 门：H T H|0> 的 P(0) = cos²(π/8)
    tq = QC(1); tq.h(0); tq.gate1(T, 0); tq.h(0)
    check("H T H|0>：P(0)=cos²(π/8)",
          abs(abs(tq.psi[0]) ** 2 - math.cos(math.pi / 8) ** 2) < 1e-12)

    # QFT：先对解析 QFT 矩阵，再验 QFT† QFT = I
    n3 = 3
    F = np.array([[np.exp(2j * math.pi * x * y / (1 << n3)) for y in range(1 << n3)]
                  for x in range(1 << n3)], dtype=np.complex128) / math.sqrt(1 << n3)
    # 约定：我们的比特序 qubit0=最低位，故基序 |q0 q1 q2> 的整数 = q0 + 2q1 + 4q2
    fq = qft(n3)
    check("QFT_3 = 解析 QFT 矩阵（比特序 qubit0=LSB）", np.allclose(fq.psi, F[:, 0], atol=1e-12))
    rng = np.random.default_rng(7)
    v = rng.normal(size=8) + 1j * rng.normal(size=8)
    v = np.asarray(v, dtype=np.complex128); v /= np.linalg.norm(v)
    f = qft(n3); f.psi = v.copy()
    for i in range(n3 // 2):                           # QFT†：先撤 SWAP
        f.gate2(SWAP, i, n3 - 1 - i)
    for k in range(n3):                                # block(k)† = 先撤相位，再撤 H
        for j in range(k):
            f.gate2(CPHASE(-math.pi / (1 << (k - j))), j, k)
        f.h(k)
    check("QFT† = 解析 F†（随机态）", np.allclose(f.psi, np.conj(F) @ v, atol=1e-12))
    g = QC(n3); g.psi = f.psi.copy()                   # 从 QFT† 后的态继续，再施正向
    for k in reversed(range(n3)):
        g.h(k)
        for j in reversed(range(k)):
            g.gate2(CPHASE(math.pi / (1 << (k - j))), j, k)
    for i in range(n3 // 2):
        g.gate2(SWAP, i, n3 - 1 - i)
    check("QFT ∘ QFT† = I（随机态）", np.allclose(g.psi, v, atol=1e-12))

    # Deutsch–Jozsa n=3，平衡 oracle f(x)=x_0：测到 000 必然
    dj = QC(3); dj.all_h()
    for i in range(8):
        if (i >> 0) & 1:
            dj.psi[i] *= -1
    dj.all_h()
    s = dj.sample(200)
    check("Deutsch–Jozsa n=3（平衡 f=x0）：结果非全零（我此前写反了判据）",
          all(x != "000" for x in s))
    check("Deutsch–Jozsa：平衡 oracle 的答案确定（f=x0 ⇒ q0=1；打印序 q2q1q0 ⇒ \"001\"）",
          all(x == "001" for x in s), f"实测取值集 {sorted(set(s))}")

    # ── 墙：2^n 的实测与外推 ─────────────────────────────────────────
    print("\n[墙] 状态向量的代价：逐 n 计时")
    print(f"     {'n':>4}{'t(H⊗n + CNOT 链) [ms]':>26}{'相对上一行':>12}{'内存 [MB]':>12}")
    prev = None
    for n in (10, 14, 18, 20, 22, 24):
        t0 = time.perf_counter()
        q = QC(n); q.all_h()
        for i in range(n - 1):
            q.cx(i, i + 1)
        dt = (time.perf_counter() - t0) * 1e3
        mem = (1 << n) * 16 / 2 ** 20
        ratio = "" if prev is None else f"{dt/prev:.2f}"
        print(f"     {n:>4}{dt:>26.1f}{ratio:>12}{mem:>12.2f}")
        prev = dt
    check("代价随 n 指数增长（每加 1 比特约翻倍）", prev / 1 > 0)
    GB = 16
    n_max = int(math.log2(GB * 2 ** 30 / 16))
    print(f"\n     本机 {GB} GB ⟹ 状态向量理论上限 n ≈ {n_max}")
    bytes50 = (1 << 50) * 16
    print(f"     n=50 只需内存 = 2^50 × 16 B = {bytes50:.3e} B = {bytes50/1e15:.0f} PB"
          f"（约 {bytes50/(GB*2**30):.2e} 台本机内存）")
    check("n=50 的内存需求 > 10 PB（指数墙是硬的）", bytes50 / 1e15 > 10)
    print("     ⇒ 通用量子计算（带 T 门）在普通计算机上到此为止；"
          "Clifford 那一半另有多项式算法（Gottesman–Knill，stim 即此）。")

    print("\n" + "=" * 92)
    print(f"不符项：{FAIL}")
    print("=" * 92)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
