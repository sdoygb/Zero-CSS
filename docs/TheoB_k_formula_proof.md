# 定理 B（修正版）：Zero 原生量子码的参数闭式

**编号**：Zero-QEC-Thm-B（v2，取代 v1）
**日期**：2026-10-04
**范围**：只用 Zero（零和宇宙）的基础
**核验脚本**：`scripts/zero_css_structure_pin.py`、`scripts/zero_rank_and_distance.py`、`scripts/zero_lemma4_face_is_cycle.py`

> ## ⚠ v1 的作废声明
>
> v1 用 **$n=2E$**（每条边两个物理比特）并声称 $k=2E-\operatorname{rank}(K_X)-\operatorname{rank}(K_Z)$，给出环面 $k=20$、码率 0.556。
> **该式错误**：$X$ 型与 $Z$ 型稳定子**作用在同一批边比特上**，两组**不独立**，故 $r\neq r_X+r_Z$。
> 正确计数是 **$n=E$**（每条边一个物理比特），$k=E-r_X-r_Z$。环面的正确值是 $k=2$。
> v1 的 11 例"实测核验"全部是在错误的量上做的，**一并作废**。

---

## 0 记号与设定（CSS 形式）

复形 $K=(V,E,F)$：$V,E$ 来自 Z0 的 $(C,E)$；$F$ 为面集，每个面是 **Z0③ 的闭合词**。

**物理比特 = 边**：每条边 $e\in E$ 一个量子比特，$n:=|E|$，$\mathcal H=(\mathbb C^2)^{\otimes E}$。

H_X\in\mathbb F_2^{V\times E},　 H_X[v,e]=1\iff v\in e
\qquad（X 型稳定子：顶点星形）
H_Z\in\mathbb F_2^{F\times E},　 H_Z[f,e]=1\iff e\in\partial f
\qquad（Z 型稳定子：面边界）
$$S_X=\operatorname{rowspace}(H_X),\qquad S_Z=\operatorname{rowspace}(H_Z).$$

**对易条件**（定理 A）：$H_XH_Z^\top=0$（恒成立，因 $|\operatorname{star}(v)\cap\partial f|=\deg_{\partial f}(v)$ 为偶）。

---

## 1 引理 5（关联矩阵的秩）

**引理 5** 设底图有 $c$ 个连通分量，则
$$\operatorname{rank}(H_X)=V-c .$$

*证明.* 设 $x\in\mathbb F_2^V$ 且 $H_Xx=0$。第 $e=(a,b)$ 列为零给出 $x_a+x_b=0$，即 $x_a=x_b$；
故 $S=\{v:x_v=1\}$ 不含"恰一个端点在 $S$"的边，即 $\delta(S)=\varnothing$，$S$ 是连通分量之并。
反之取 $S$ 为连通分量之并，则每条边两端同属或同不属于 $S$，故 $H_Xx=0$。
故 $\ker H_X$ 的维数为 $c$，秩-零化度给 $\operatorname{rank}(H_X)=V-c$。$\square$

**推论 5.1** 连通（$c=1$）时 $\operatorname{rank}(H_X)=V-1$。
**自环**：$e=(v,v)$ 的列为 $2=0\in\mathbb F_2$，不影响秩（实测：三角 + 自环，$r_X=V-1$）。

---

## 2 定理 B：$k$ 的闭式

### 定理 B（v2）

$$k\ :=\ \dim_{\mathbb C}\mathcal C\ =\ E-\operatorname{rank}(H_X)-\operatorname{rank}(H_Z)
\ =\ \beta_1+(c-1)-\operatorname{rank}(H_Z),$$
其中 $\beta_1:=E-V+c$ 为圈空间维数。**连通时**
$$\boxed{\ k\ =\ \beta_1-\operatorname{rank}(H_Z)\ }$$

*证明.* CSS 码的码空间为 $\mathcal C=\{|\psi\rangle: X_v|\psi\rangle=|\psi\rangle,\ Z_f|\psi\rangle=|\psi\rangle\ \forall v,f\}$。

**第一步（X 侧的约束）** $Z$ 型约束 $H_Z|\psi\rangle=0$（在 $Z$ 基下）把态限制在 $\ker(H_Z)$ 上，$\dim\ker H_Z=E-\operatorname{rank}(H_Z)$。

**第二步（商掉 X 型稳定子）** $X$ 型约束再商去 $S_X=\operatorname{rowspace}(H_X)$。由于 $H_XH_Z^\top=0$，
$S_X\subseteq\ker(H_Z)$（X 侧稳定子不破坏 Z 约束），故
$$k=\dim\ker(H_Z)-\dim S_X=(E-\operatorname{rank}H_Z)-\operatorname{rank}H_X .$$

**第三步（对称性核验）** 同法从 X 侧出发得 $k=(E-\operatorname{rank}H_X)-\operatorname{rank}H_Z$，与上式相同（两者都是 $E-r_X-r_Z$），自洽。

**第四步（代入引理 5）** $k=E-(V-c)-\operatorname{rank}(H_Z)=\beta_1+(c-1)-\operatorname{rank}(H_Z)$。$\square$

### 推论 B.2（面独立时的特例）

记 $\mathrm{rel}:=|F|-\operatorname{rank}(H_Z)$（面边界之间的独立关系数）。则
$$k=\beta_1+(c-1)-|F|+\mathrm{rel}.$$
若面边界**线性无关**（$\mathrm{rel}=0$）且 $K$ 连通：$k=\beta_1-|F|$。

### 定理 C（面边界关系数的解释）

$\mathrm{rel}=\dim\ker\delta_F$，其中 $\delta_F:\mathbb F_2^F\to\mathbb F_2^E$ 为"面 $\mapsto$ 其边界边集之和"的线性映射。
同理 $k=\dim\operatorname{coker}\delta_F$（连通时）。两者与圈空间的合成分解
$$\mathbb F_2^E\ \supseteq\ Z_1=\ker\partial\ \ (\dim=\beta_1)$$
一致：$Z_1$ 中由面张成的部分是 $\operatorname{im}\delta_F$（$\dim=\operatorname{rank}H_Z$），余维数即 $k$。

*证明.* $\ker\delta_F$ 恰是"面边界之和为零"的线性关系组，维数 $=|F|-\operatorname{rank}\delta_F=|F|-\operatorname{rank}(H_Z)=\mathrm{rel}$。
$k=\beta_1-\operatorname{rank}(H_Z)=\dim Z_1/\operatorname{im}\delta_F=\dim\operatorname{coker}\delta_F$（连通）。$\square$

---

## 3 实测核验

### 3.1 码参数（正确计数 $n=E$）

| 复形 | $V$ | $E=n$ | $F$ | $\beta_1$ | $r_X$ | $r_Z$ | $k$ | $\mathrm{rel}$ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 方格 1×1 | 4 | 4 | 1 | 1 | 3 | 1 | **0** | 0 |
| 方格 2×2 | 9 | 12 | 4 | 4 | 8 | 4 | **0** | 0 |
| 方格 3×3 | 16 | 24 | 9 | 9 | 15 | 9 | **0** | 0 |
| 方格 4×4 | 25 | 40 | 16 | 16 | 24 | 16 | **0** | 0 |
| 方格 5×5 | 36 | 60 | 25 | 25 | 35 | 25 | **0** | 0 |
| 环面 3×3 | 9 | 18 | 9 | 10 | 8 | 8 | **2** | 1 |
| 环面 4×4 | 16 | 32 | 16 | 17 | 15 | 15 | **2** | 1 |
| 环面 5×5 | 25 | 50 | 25 | 26 | 24 | 24 | **2** | 1 |

**三式互校**（`dim ker H_Z − r_X`、`dim ker H_X − r_Z`、`E−r_X−r_Z`）在全部 8 例上**一致**。

\Longrightarrow 环面族 = [[2L^2, 2, L]] （即标准 toric code）;\qquad
平面族 k=0 （无逻辑比特）.

### 3.2 (ii) 距离 $d$

| 复形 | $d_X$ | $d_Z$ | $d$ | girth | $L$ |
|:--|--:|--:|--:|--:|--:|
| 环面 3×3 | 3 | 3 | **3** | 3 | 3 |
| 环面 4×4 | 4 | 4 | **4** | 4 | 4 |
| 环面 5×5 | 枚举超限 | 枚举超限 | — | 5 | 5 |

**结论（实测）**：环面族
d_X=(最短非平凡圈长)=girth=L,\qquad d_Z=(最小分离割)=L,\qquad d=L .
平面族：$k=0$ ⇒ 无逻辑算符 ⇒ $d$ **无定义**（不是 $0$ 也不是 $\infty$）。

**部分证明**：
- $d_X\le \text{girth}$：任何非平凡圈都是候选逻辑，取最短者即给上界。
- $d_X=\text{girth}$：等价于"每个最短圈都是非平凡（不可写成面之和）"——在环面上，最短圈 $L<\ldots$ 无法由面填满（面填满一个圈要求圈内面积 $=0$，而最短圈内部非空）。（**此处为论证草图，未严格化**。）
- $d_Z\le \min_{c\ \text{非平凡圈}}\ \min\text{cut}(c)$：给一个非平凡圈 $c$，取其最小割 $S$，则 $\sum_{v\in\partial S}v$ 是 Z 型逻辑（与面全交偶、与 $c$ 交奇 ⇒ 非平凡），重量 $=|S|$。

---

## 4 诚实边界

| # | 项 | 状态 |
|--:|:--|:--|
| 1 | v1 的全部结论（$n=2E$、$k=2E-r_X-r_Z$、环面 $k=20$、"11 例实测"） | **作废** |
| 2 | $k=E-r_X-r_Z$ 与 $k=\beta_1-\operatorname{rank}(H_Z)$ | **已证**（§2） |
| 3 | $\operatorname{rank}(H_X)=V-c$ | **已证**（引理 5） |
| 4 | $\mathrm{rel}=\dim\ker\delta_F$ 的解释 | **已证**（定理 C） |
| 5 | $d_X=\text{girth}$ 的**严格证明** | **未完成**（仅有上界与草图） |
| 6 | $d_Z=\min\text{cut}$ 的严格证明 | **未完成**（§3.2 给了构造性上界） |
| 7 | 环面 5×5 的距离 | 枚举超限（$\dim\ker=26$，$2^{26}$ 不可枚举）——需 BFS/整数规划 |
| 8 | **面集 $F$ 的选择（胞腔化）是输入** | 取哪些闭合作面会改变 $F,\operatorname{rank}(H_Z)$，从而改变 $k$ 与 $d$。这不是导出量 |
| 9 | 环面 2×2 | 其"面"退化为全部 4 条边（非正常胞腔化）；环面建议 $L\ge3$ |

---

## 5 结论

**(i)** Zero 的零和（顶点／边）＋ 闭合词（面）⇒ 合法 CSS 量子码（定理 A）。

**(ii)** 物理比特数 $n=E$（每条边一个）；逻辑量子比特数

k=E-r_X-r_Z=\beta_1-\operatorname{rank}(H_Z)　(连通，已证).

**(iii)** 环面胞腔化 $\Rightarrow[[2L^2,2,L]]$（toric code）；平面胞腔化 $\Rightarrow k=0$。
