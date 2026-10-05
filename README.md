# Zero-CSS · 纯 Zero 路线的量子纠错（CSS 码）

> 仓库名 **Zero-CSS**（原目录名 `zero-qec`）：Zero 路线的 **CSS 码**构造与判据。

**建立日期**：2026-10-04
**目的**：只用 **Zero（零和宇宙）** 的基础，构造并检验量子纠错结构。
**明确排除**：旧理论（几何论 / AG 完备码 / RM 矩代数 / 简并类闭式）——本目录**不引用**，
其脚本与文章仍在原处，不在此目录内。

> **母项目**：[**Zero**](https://github.com/sdoygb/Zero) —— 从单一公理 Z0（零不断乱动）出发的物理推导台账。
> 本仓库是它的**专题分支**：只用 Zero 的基础做量子纠错，**不引入旧理论**。

---

## 0 为什么另建目录

旧理论（几何论）与 Zero **有联系但基础不同**：旧理论的码族是 CSS(RM($`r`$,$`m`$))、简并类是 syndrome 类；
Zero 的原语是零和＋局域补偿移动＋循环次序。两者混放会**互相污染记号与判据**，故分开。

---

## 1 目录内容

| 文件 | 作用 | 依赖 |
|:--|:--|:--|
| `scripts/zero_native_code.py` | 零和 ⇒ 圈空间；$`[ E,\beta\_1,\text{girth} ]`$ 参数表；边按方向分组 ⇒ CSS 对易判据 | numpy |
| `scripts/zero_quantum_cycle_code.py` | 量子圈码（顶点星形 X/Z ＋ 面）；实测 $`n,k,d`$ | numpy |
| `scripts/zero_layer_qec_profile.py` | **四层 ＋ 读出面的纠错画像**（L0／L1／L1′／L2／$`\mathcal R`$） | numpy |
| `scripts/zero_layers_vs_code.py` | 分层 ↔ 码结构对应（稳定子／syndrome／MWD 各在哪层） | numpy |
| `scripts/zero_theorem_a_symplectic.py` | **定理 A**：顶点／面算符对易判据（GF(2) 辛向量，秒级） | numpy |
| `scripts/zero_theorem_a_code_table.py` | 定理 A 码的参数表（$`k=n-(V-1)-F`$） | numpy |
| `scripts/tqec_degeneracy.py` | 在 **tqec 生成的电路**上测权重 1/2 层简并画像 | tqec ＋ stim |
| `scripts/zero_theorem_a_lemmas_check.py` | **定理 A 引理 1–3 核验**（单格矩阵 ＋ 辛向量） | numpy |
| `scripts/zero_lemma3_unconditional.py` | **引理 3 无条件版**：闭合词 ⇒ 度数全偶 | numpy |
| `scripts/zero_k_formula_probe.py` | #1 探测：$`r\_X`$/$`r\_Z`$/$`k`$ 的关系（否证 $`k=2\beta\_1`$） | numpy |
| `scripts/zero_rz_law_probe.py` | $`r\_Z`$ 的规律：$`k=2E-(V-1)-r\_Z`$ 逐例成立 | numpy |
| `scripts/zero_lemma4_face_is_cycle.py` | 引理 4：面边界 ∈ 圈空间；$`r\_X=V-c`$；自环无关 | numpy |
| `docs/TheoA_commutation_proof.md` | **定理 A 的严格证明**（含更正记录） | — |
| `scripts/zero_gap_a_pauli_solve.py` | **定理 E**：从 L0-c 导出 Pauli 的 $`X,Z`$ 与反对易（求解版） | numpy |
| `scripts/zero_css_derivation_audit.py` | 归属审计：CSS 里哪些来自 Zero、哪些是借用 | numpy |
| `docs/TheoB_k_formula_proof.md` | **定理 B：$`k`$ 的闭式证明**（含 $`k=2\beta\_1`$ 的否证） | — |
| `scripts/zero_theorem_f_edge_algebra.py` | **定理 F**：边的二元算符空间（缺口 (b) 闭合） | numpy |
| `docs/TheoE_pauli_from_L0c.md` | **定理 E：从 L0-c 导出 Pauli 与反对易**（缺口 (a) 闭合；§3 已更正） | — |
| `scripts/zero_gap_c_decisive.py` | **定理 G**：面集不能从 Zero 原语唯一导出（决定性测试） | numpy |
| `docs/TheoF_edge_algebra.md` | **定理 F：边的二元算符空间**（"量子比特"借贷已去除） | — |
| `docs/TheoG_face_selection.md` | **定理 G：面的选择规则**（缺口 (c) 判定：不可导出 ＋ 必然关系） | — |
| `scripts/zero_which_code_family.py` | 哪个码族适合 Zero（平坦 vs 双曲 vs 高维 vs HGP） | numpy |
| `scripts/zero_improve_rate.py` | 提高码率的杠杆（加面/换复形） | numpy |
| `scripts/zero_rate_lever.py` | **减面 vs 距离**：(k,d) 前沿扫描 | numpy |
| `docs/rate_levers_verdict.md` | 码率杠杆判定（**§0 的"局域⟺低码率"已撤回**） | — |
| `scripts/zero_hgp_native.py` | **配方 B：HGP**（常码率构造） | numpy |
| `scripts/zero_hgp_dist2.py` | HGP 距离（枚举核空间，小图） | numpy |
| `scripts/zero_hgp_expander.py` | HGP + 高 girth 定度图 | numpy |
| `scripts/zero_hgp_distance_final.py` | **HGP 距离**：$`d=\min(girth\_1,girth\_2)`$ | numpy |
| `docs/hgp_recipe_B.md` | **配方 B 结论**：码率 0.23–0.67，距离 $`\sim\log n`$，三方对比 | — |

---

## 2 已确立的结果（全部可复现）

### 2.1 Zero 原生给出码

$$
\text{零和}\ \textstyle\sum w_i=0\ \Longrightarrow\ \text{圈空间}\ \Longrightarrow\ [[\,E,\ \beta_1=E-V+1,\ \text{girth}\,]]
$$

$`K\_4\to[ 6,3,3 ]`$；$`C\_n\to[ n,1,n ]`$；环面 $`3\times3\to`$ 见下。

### 2.2 Zero 原生给出**量子**码的前提

$$
\text{边按方向分组（水平／垂直）}\ \Longrightarrow\ H_XH_Z^\top=0\ \text{（实测非零元素}=0\text{）}
$$

### 2.3 定理 A（对易判据）

复形 $`(V,E,F)`$（顶点来自 Z0 的 $`(C,E)`$；**面来自 Z0③ 的闭合词**），
$`A\_v=\prod\_{e\ni v}X\_e`$、$`B\_f=\prod\_{e\in\partial f}Z\_e`$：

$$
\ A_v\ \text{与}\ B_f\ \text{对易}\iff |{\rm star}(v)\cap\partial f|\ \text{为偶}
$$

**已核验**：方格 3×3/4×4、环面 3×3/5×5 全部通过（同型恒对易；反对易 ⟺ 交集为奇；异型反对易对 $`=0`$）。

**已证**：引理 1–3 ＋ 定理 A（含引理 3 的**无条件版**：面取 Zero 的闭合词时，
$`|star(v)\cap\partial f|=\deg\_{\partial f}(v)\in\{0\}\cup2\mathbb N`$ 恒为偶
⇒ 对易**无条件成立**）。详见 [docs/TheoA_commutation_proof.md](docs/TheoA_commutation_proof.md)。

**已证（定理 B）**：$`k=2E-(V-c)-rank(K\_Z)`$；连通且面独立时 $`k=2E-(V-1)-F`$（11 例实测）。
$`k=2\beta\_1`$ **不普遍成立**（已否证）。详见 [docs/TheoB_k_formula_proof.md](docs/TheoB_k_formula_proof.md)。

**未证**：$`rank(K\_Z)`$ 的闭式（$`rel`$ 由什么决定）；$`d`$ 与 girth／最小割的关系；面（胞腔化）的选择如何影响 $`k`$。

### 2.4 分层把码的部件各就各位

| Zero 层 | 码的对应 | 实测（$`C\_8`$） |
|:--|:--|:--|
| **L0** | 稳定子（约束来源） | 零和 ＝ 校验约束 |
| **L1** | **syndrome／纠错信息** | 类数 36，fail(2) $`=0.000`$ |
| L1′ | 记录粗粒化 | 类数 5，fail(2) $`=0.850`$（**有损，非纠错层**） |
| **L2** | **最小权重解码** | 类数 29，fail(2) $`=0.030`$；轨道 ＝ Hamming 重量类 |
| $`\mathcal R`$ | 读出 | $`\equiv`$ L1 |

$$
\Longrightarrow\ \textbf{纠错住在 L1（历史层）};\ \text{L1}'\ \text{的旋转类是**记录**的代价，不是纠错能力}。
$$

### 2.5 码参数（**修正版**：物理比特 = 边，$`n=E`$）

| 复形 | $`V`$ | $`E=n`$ | $`F`$ | $`\beta\_1`$ | $`r\_X`$ | $`r\_Z`$ | $`k`$ | $`d`$ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 方格 1×1 … 5×5 | 4…36 | 4…60 | 1…25 | $`=F`$ | $`V-1`$ | $`F`$ | **0** | 无定义 |
| 环面 3×3 | 9 | 18 | 9 | 10 | 8 | 8 | **2** | **3** |
| 环面 4×4 | 16 | 32 | 16 | 17 | 15 | 15 | **2** | **4** |

$$
k=E-r_X-r_Z=\beta_1-\text{rank}(H_Z)\quad(\text{连通，已证});\qquad
\text{环面}=[[2L^2,2,L]]\ (\text{toric code})
$$

> **⚠ 作废**：早期版本用 $`n=2E`$（每条边两个比特）并给出环面 $`k=20`$、码率 0.556——**错误**。
> $`X`$ 型与 $`Z`$ 型稳定子作用在**同一批边比特**上，两组不独立。详见 [docs/TheoB_k_formula_proof.md](docs/TheoB_k_formula_proof.md) §0 的作废声明。

### 2.5b 距离 $`d\_X`$（定理 D，已证 + 已精确计算）

$$
d_X=\text{girth}_{\rm ess}=\min\{|c|:c\in Z_1\setminus\text{im}\delta_F\}
$$

| 复形 | $`d\_X`$ | girth | 说明 |
|:--|--:|--:|:--|
| 环面 3×3 | **3** | 3 | 首个非平凡圈：重量 3 |
| 环面 4×4 | **4** | 4 | 首个非平凡圈：重量 4 |
| 环面 5×5 | **5** | **4** | 重量 4 的 25 个圈**全部平凡**（它们就是 25 个面） |
| 方格族 | 无定义 | — | $`k=0`$ ⇒ 无逻辑算符 |

> **⚠ 提法更正**：$`d\_X\neq`$ girth。$`L\ge4`$ 时 girth $`=4`$（单个面）而 $`d\_X=L`$；
> $`L=5`$ 时两者数值**分离**（4 vs 5），直接测出。正确量是 **essential girth**。

**未证**：$`d\_Z=\min`$ cut 的严格证明；$`d\_X=L`$ 下界中"非可缩圈长 $`\ge L`$"的严格化；面集（胞腔化）的选择如何影响 $`k,d`$。

### 2.5c 归属：Zero ⇒ CSS 的借用清单（定理 E / F 后）

| 步骤 | 归属 |
|:--|:--|
| 零和 ⇒ 经典圈码 | ✅ Zero 原生（Z0③） |
| 面必须闭合 | ✅ Zero 原生（Z0③ 零和；非闭合则对易破裂） |
| 每边一个**二元算符空间**（翻转 $`F\_e`$ ＋ 读取 $`Z\_e`$） | ✅ **导出**（定理 F；不再是"量子比特"借贷） |
| $`F\_eZ\_e=-Z\_eF\_e`$（**实**反对易） | ✅ 原生（定理 F；不需 $`\times i`$） |
| 张量积 $`(\mathbb C^2)^{\otimes E}`$ | ✅ **导出**（F4：不同边对易 ＋ 维数 $`4^E`$） |
| 复化（复振幅／Born） | ✅ 用 `G62`(GNS) ＋ `G64`(双覆盖) |
| **面的选择规则** | ⚠️ **判定：不可从原语导出**（定理 G）；但 $`\textstyle\sum\_f\partial f=0`$ 必然 ⇒ $`k=\beta\_1-\lvert F \rvert+1`$ |

$$
\text{归属缺口：三处借用} \longrightarrow \textbf{仅剩一处}\ (\text{面的选择})
$$

### 2.6 与 tqec 的接口（同一套方法）

在 tqec 生成的电路上测简并画像：$`k=1`$ 权重1 唯一率 **100%**、权重2 恢复率 **81.2%**（真简并口径 82.9%）；
$`k=2`$ 为 **86.5%**。

---

## 3 已撤回 / 已废弃（教训记录）

| 项 | 处置 | 原因 |
|:--|:--|:--|
| "对易 ⟺ 欧拉（顶点度数全偶）" | **撤回** | 只对"顶点 X/Z"构造成立，而该构造在简单图上相邻顶点即反对易 ⇒ 非法 |
| "开边界会产生反对易对" | **撤回** | 实测 0；顶点星形与面边界交于 0 或 2，与边界无关 |
| 矩阵表示 Pauli 算符 | **废弃** | $`2^{2E}`$ 维：3×3 环面 $`4.7\times10^{21}`$ 元素，实测占 43% 内存、64 min CPU 未完成 |
| 辛向量表示 | **采用** | 同一任务 0.14 秒 |
| "旋转类可作纠错层" | **撤回** | 实测有损（100% 混 syndrome）；纠错在 L1 |

---

## 4 复现

```bash
cd scripts
for f in zero_native_code.py zero_quantum_cycle_code.py zero_layer_qec_profile.py \
         zero_layers_vs_code.py zero_theorem_a_symplectic.py zero_theorem_a_code_table.py; do
  python3 "$f"
done
# tqec 接口（需 tqec 环境）：
#   cd <tqec 工作副本> && uv run python <本目录>/scripts/tqec_degeneracy.py
```

全部脚本仅依赖 `numpy`（`tqec_degeneracy.py` 另需 tqec ＋ stim）。

---

## 5 下一步（候选）

| # | 事项 | 状态 |
|--:|:--|:--|
| 1 | 把 $`k=2E-(V-1)-F`$ 证成定理（秩-零化度 ＋ 欧拉数） | 未做 |
| 2 | $`d`$ 与 girth／最小割的精确关系 | 未做（只测了小例） |
| 3 | 定理 A 的严格证明落纸（引理 1–3） | 骨架已给，未落纸 |
| 4 | 相位群 $`\mu\_n`$ 与错误类型的有效关系 | **无有效结论**（前一版脚本构造有误，已废弃） |
| 5 | L2 的容量界 ⇒ 阈值 | 未做 |

---

## 6 许可

[**CC BY-NC 4.0**](LICENSE)（Attribution-NonCommercial 4.0 International）

可自由共享与改编（需署名）；**禁止商业使用**。

署名请注明：**Zero-CSS 项目**，<https://github.com/sdoygb/Zero-CSS>。
