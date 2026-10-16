# 暂存推导目录（_staging）

**用途**：这条"Zero 角度 → 量子纠错 → 量子计算"的推导线，**进行中的稿子放这里**；
定稿后再决定进 `docs/`／`scripts/` 还是留在暂存。

**纪律**（沿用母项目 Zero 的习惯）：
* 正文只写两样：**判定的结论**与**未决的问题**；不写过程叙述、不写"此前如何"。
* 结论只写「判定 ＋ 指针」（指向 `docs/` 的哪一节、`scripts/` 的哪个脚本、哪个数字）。
* 凡数字必须有脚本，脚本必须自带断言（退出码 0 = 全过）。
* **不许写恒真断言**（`check(..., True)` 之类）：每一条都要有判别力——
  即"如果它错了，检查会红"。对照组（如 T 门之于 Clifford 检验）是标准做法。

---

## 一、这条推导线做了什么（按时间序，皆为已提交）

| # | 主题 | 文档 | 脚本（退出码 0） | 判定 |
|--:|:--|:--|:--|:--|
| 1 | 缺口 (c)：面的选择 | `docs/TheoG_ext_closure.md` | `zero_gap_c_closure.py`、`..._dfs.py`、`..._gauge_test.py` | 加了"每边恰属 2 个面"后，**$k=2g$**；面集是**规范**（$L=4$ 有 3 种胞腔化，$(k,d)$ 全同） |
| 2 | Zero 天然复形族的前沿 | `docs/natural_family_frontier.md` | `zero_natural_family_frontier.py` | **不占优**：$K_4$ 无逻辑（$k=0$）；Heawood 地图 $[[21,2,3]]$，$kd^2/n=0.857<1.000$ |
| 3 | ω 当解码权重（候选一） | `docs/omega_decoder_verdict.md` | `zero_omega_decoder_test.py` | **被定义挡住**：$\omega_h$ 是层高（时间轴）测度，纯空间码没有层轴；附精确基线（已匹配 ML vs 最小权：代价 3–5%） |
| 4 | 普通计算机上做量子计算 | `docs/qc_on_classical.md` | `zero_qc_sim.py` | **Clifford 白拿，通用是指数墙**（$n=24$ 单层 24 s；$n=50$ 需 18 PB） |
| 5 | 用 Zero 世界看量子计算 | `docs/zero_native_qc.md` | `zero_native_qc.py` | 零和＋补偿移动 ⟹ 移动全是 Pauli 串 ⟹ 门集 ⊆ **Pauli ⊊ Clifford ⊊ 通用**：**给不出非 Clifford 门** |

## 二、由 1–5 归纳出的结构判断

1. **"能退化出 CSS" ≠ "能改进 CSS"**：构造与标准同调 CSS 码同构，且 Zero 对胞腔化施加不了额外约束
   （面集是规范、输入只剩亏格）⟹ 约束更少 ⟹ 挑不出更好的码。这一条是**结构性的**，不是没找对路。
2. **零和这一支的"门"全是 Pauli** ⟹ 它天然对应 **Clifford 那一半**（这正是经典计算机能白拿的那一半）
   ⟹ 它**不可能**给出通用性。
3. ⟹ 若要走到"通用"，非 Clifford 资源只能来自 Zero 的**另一支**：
   **非中心态 ⟹ 非平凡模流**（母项目 Z040／Z045 的机制）。这是标准理论里 **magic state** 的位置。

## 三、下一步（已定，待做）

**可判定的检验**：量化"模流对 Pauli 群的偏离"，与标准 magic 度量做**同步性检验**。

| 项 | 内容 |
|:--|:--|
| 对象 | $\sigma_t(P)=\rho^{it}P\rho^{-it}$ 离 Pauli 群多远（$\max_Q|\mathrm{tr}(Q^\dagger\sigma_t(P))|/d$） |
| 对照 | 标准 magic 度量：stabilizer entropy 或 $l_1$-norm |
| 起点 | $n=1$，态族 $\rho(p)=(1-p)\|\psi\rangle\langle\psi\|+p\,I/2$，$\|\psi\rangle\in\{\|0\rangle,\|T\rangle\}$ |
| **陷阱** | 偏离量对 $p$ **非单调**（$p=1$ 时 $\rho\propto I$、模流平凡、偏离 $=0$），而 $l_1$-magic 单调 ⟹ 二者**不会**简单单调同步；检验必须给出正确对应关系，不能只报"都大于零" |
| 输赢判据 | 同步 ⟹ 桥成立（Zero 的模流是 magic 资源的候选）；不同步 ⟹ 猜想需修正 |

## 四、环境事实（本机实测）

* 本机**没装** `stim`／`tqec` ⟹ 仓库里依赖它们的脚本（`tqec_degeneracy.py` 等）在**本机跑不了**；
  本文这一条线的脚本全部**只用 numpy/scipy**，可本机复跑。
* 本机 16 GB ⟹ 状态向量上限 $n\approx30$；**无 GNU `timeout`**（别写 `timeout …`）。
* 推送到 GitHub 用 SSH：`GIT_SSH_COMMAND='ssh -o ConnectTimeout=25 -o BatchMode=yes' git push`。
