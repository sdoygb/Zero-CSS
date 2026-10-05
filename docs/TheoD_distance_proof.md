# 定理 D：Zero 原生量子码的 X 型距离

**编号**：Zero-QEC-Thm-D
**日期**：2026-10-04
**范围**：只用 Zero（零和宇宙）的基础
**前置**：定理 A（对易判据）、定理 B v2（$k=E-r\_X-r\_Z$）
**核验脚本**：`scripts/zero_theorem_d_distance.py`

---

## 0 ⚠ 提法更正（最重要的一节）

**错误提法**（本文件初稿曾采用，且任务描述中亦如此）：

$$
d_X\overset{?}{=}\mathrm{girth}(G).
$$

**为什么错**：$L\ge4$ 时 $L\times L$ 环面方格图的 **girth $=4$**——单个面（4-圈）就是最短圈。
但这个 4-圈**就是面边界本身**，属于 $\mathrm{im}\delta\_F$，是**平凡**的，不构成逻辑算符。
而**实测** $d\_X=L$（$L=4$ 时 $d\_X=4$、$L=5$ 时 $d\_X=5$），与 girth$=4$ 不符。

**正确提法**：定义**本质围长（essential girth）**

$$
\mathrm{girth}_{\rm ess}(G,F)\ :=\ \min\{\,|c|\ :\ c\in Z_1,\ c\notin\mathrm{im}\delta_F\,\},
$$

则 $d\_X=\mathrm{girth}\_{\rm ess}$（即下面的 (D1)）。

$$
d_X=\mathrm{girth}_{\rm ess}\ \neq\ \mathrm{girth}\quad(L\ge4)
$$

---

## 1 设定

复形 $K=(V,E,F)$：$V,E$ 来自 Z0 的 $(C,E)$；面 = **Z0③ 的闭合词**。
物理比特 = 边，$n=E$；$H\_X$ = 顶点-边关联，$H\_Z$ = 面-边关联。

Z_1:=\ker H_X\subseteq\mathbb F_2^E (圈空间, \dim=\beta_1),\qquad
\delta_F:\mathbb F_2^F\to\mathbb F_2^E, \mathrm{im}\delta_F=\mathrm{rowspace}(H_Z).

**引理 4（已证）**：$\mathrm{im}\delta\_F\subseteq Z\_1$。

**定义 1.1（逻辑 X 算符）** $\mathcal L\_X:=Z\_1\setminus\mathrm{im}\delta\_F$；　$d\_X:=\min\{|c|:c\in\mathcal L\_X\}$。

**判据 1.2（可计算）** $c\in\mathrm{im}\delta\_F\iff$ 方程组 $\delta\_Fx=c$ 在 $\mathbb F\_2$ 上有解。

---

## 2 定理 D

### 命题 D1（距离的精确刻画）

$$
d_X=\mathrm{girth}_{\rm ess}=\min\{|c|:c\in Z_1\setminus\mathrm{im}\delta_F\}.
$$

*证明.* 由定义 1.1 直接重述；需验证"逻辑 X 算符 $=Z\_1\setminus\mathrm{im}\delta\_F$"。
X 型逻辑算符为"与所有 Z 型稳定子对易、且不在 $S\_X$ 中"的算符：
与所有 $Z\_f$ 对易 $\iff c\in\ker H\_Z$（在 X 基下）$=Z\_1$（引理 4 的对偶表述）；不在 $S\_X=\mathrm{im}\delta\_F$ 中即逻辑。
故 $\mathcal L\_X=Z\_1\setminus\mathrm{im}\delta\_F$。$\square$

### 命题 D2（平凡圈不改变距离）

若 $c\in Z\_1$ 且 $c\in\mathrm{im}\delta\_F$，则 $c$ 不对 $d\_X$ 贡献。
特别地，**所有面边界、以及面边界的任意对称差，都是平凡的**。

*证明.* $c\in\mathrm{im}\delta\_F=\mathrm{rowspace}(H\_Z)$ 意味着 $c$ 是若干 $Z\_f$ 的乘积所对应的向量，
即 $c$ 可由稳定子生成元生成 ⇒ 不改变逻辑类。$\square$

**推论 D2.1** 单个面是圈（$|\partial f|\ge3$）且平凡 ⇒ 若复形含 4-圈的面，则 $\mathrm{girth}\le4$ 而 $d\_X$ 与之无关。

### 命题 D3（上界）

$$
d_X\ \le\ \min\{\,|c|\ :\ c\in Z_1,\ c\notin\mathrm{im}\delta_F\,\}\ =\ \mathrm{girth}_{\rm ess}.
$$

平凡（即 D1 的定义式）；有用之处在于给出**算法**：按重量升序枚举子集，首个"是圈且无解"者即 $d\_X$。

### 命题 D4（环面）

对 $L\times L$ 环面（$L\ge3$）的方格胞腔化：

$$
d_X=L.
$$

*证明（骨架）.*
**(i) 上界 $d\_X\le L$**：取"直线绕一圈"的圈（水平方向，长 $L$）。
它的同调类非零（绕环面一周），故 $\notin\mathrm{im}\delta\_F=B\_1$。故 $d\_X\le L$。

**(ii) 下界 $d\_X\ge L$**：设 $c\in\mathcal L\_X$，则 $c$ 的同调类 $\neq0$，即 $c$ 非可缩。
把 $c$ 提升到万有覆盖 $\mathbb R^2$（方格格点）：$c$ 成为一条从 $u$ 到 $u+\lambda$ 的路径，
其中 $\lambda\in\mathbb Z^2\setminus\{0\}$ 为该同调类的代表。路径长度 $\ge\|\lambda\|\_1\ge L$。
（最后一步：$\lambda\neq0$ 的格向量其 $\ell^1$ 范数 $\ge1$，而在 $L\times L$ 环面上"绕一圈"对应 $|\lambda\_i|\ge1$，
投影回环面后路径长度 $\ge L$。此处"$\ge L$"需更细的论证——见 §4 诚实边界 #1。）
由 (i)(ii) 得 $d\_X=L$。$\square$

**(iii) 计算验证**：$L=3,4,5$ 全部由判据 1.2 **精确计算**得 $d\_X=3,4,5$（见 §3）。$\square$

---

## 3 实测核验（`zero_theorem_d_distance.py`）

### 3.1 环面族

| 复形 | $V$ | $E$ | $F$ | 重量 2 | 重量 3 | 重量 4 | 首个非平凡圈 | $d\_X$ | girth | $L$ |
|:--|--:|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| 环面 3×3 | 9 | 18 | 9 | 0 圈 | — | — | 重量 3，边 $(0,2,4)$ | **3** | 3 | 3 |
| 环面 4×4 | 16 | 32 | 16 | 0 圈 | 0 圈 | — | 重量 4，边 $(0,2,4,6)$ | **4** | **4** | 4 |
| 环面 5×5 | 25 | 50 | 25 | 0 圈 | 0 圈 | **25 圈，全平凡** | 重量 5，边 $(0,2,4,6,8)$ | **5** | **4** | 5 |

**注意最后两列**：$L=5$ 时 $\mathrm{girth}=4$（面）而 $d\_X=5$ ⇒ **$d\_X\neq\mathrm{girth}$ 被直接测出**。
（$L=4$ 时两者数值巧合相等，$L=5$ 时分离。）

**"25 圈全平凡"是关键**：5×5 环面的 25 个 4-圈恰是 25 个面，全在 $\mathrm{im}\delta\_F$ 内，
故 4 不给出逻辑；$d\_X$ 由重量 5 的首个非平凡圈给出。

### 3.2 方格族（$k=0$）

| 复形 | $E$ | $d\_X$ | 一致性 |
|:--|--:|:--|:--|
| 方格 2×2 | 12 | None（无逻辑算符） | ✅ 与 $k=0$ 一致 |
| 方格 3×3 | 24 | None（无逻辑算符） | ✅ 与 $k=0$ 一致 |

---

## 4 诚实边界

| # | 项 | 状态 |
|--:|:--|:--|
| 1 | D4 下界中"非可缩圈长度 $\ge L$" | **论证性，未严格化**。缺口：需要"环面的非可缩圈在万有覆盖中连接 $u$ 与 $u+\lambda$（$\lambda\neq0$）⇒ 投影后长度 $\ge L$"的严格证明。直觉上显然（绕一圈至少要 $L$ 步），但本文未写出严格版本 |
| 2 | D4 中 $\mathrm{im}\delta\_F=B\_1$（面边界生成全部边界） | 依赖维数计数 $\mathrm{rank}(H\_Z)=F-1=\dim B\_1$（环面族实测），未独立证明 |
| 3 | **提法更正** | 初稿与任务描述写 "$d\_X=\mathrm{girth}$"，**错**（$L\ge4$ 时 girth$=4\neq d\_X$）。正确量是 essential girth（§0） |
| 4 | $d\_Z$ | **未证**。构造性上界：给非平凡圈 $c$，其最小割 $S$ 给出 Z 型逻辑，$d\_Z\le\min\_c\min\mathrm{cut}(c)$ |
| 5 | 环面 5×5 的 $d\_Z$ 与总距离 $d=\min(d\_X,d\_Z)$ | **未计算**（$d\_X=5$ 已定） |
| 6 | 面集（胞腔化）的选择 | 输入量；改变 $\mathrm{im}\delta\_F$ 从而改变 $d\_X$ |

---

## 5 结论

**(i)（已证，命题 D1）**

$$
d_X=\mathrm{girth}_{\rm ess}=\min\{\,|c|\ :\ c\in Z_1\setminus\mathrm{im}\delta_F\,\}.
$$

**(ii)（提法更正）** $d\_X\neq\mathrm{girth}$：$L\ge4$ 时 $\mathrm{girth}=4$（单个面）而 $d\_X=L$；
$L=5$ 时两者数值直接分离（$4$ vs $5$）。

**(iii)（环面）** $L\times L$ 环面（$L\ge3$）：$d\_X=L$，已对 $L=3,4,5$ 精确计算。

**(iv)（平面）** $k=0$ ⇒ 无逻辑算符 ⇒ $d\_X$ **无定义**（不是 $0$ 也不是 $\infty$）。
