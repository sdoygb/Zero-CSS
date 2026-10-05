# 定理 A：顶点／面稳定子的对易判据（严格证明）

**编号**：Zero-QEC-Thm-A
**日期**：2026-10-04
**范围**：只用 Zero（零和宇宙）的基础；不引用旧理论。
**核验脚本**：`scripts/zero_theorem_a_lemmas_check.py`（引理 1–3）、`scripts/zero_theorem_a_symplectic.py`（任意规模）

---

## 0 记号与约定

**定义 0.1（复形）**
$K=(V,E,F)$ 为一个有限胞腔复形：

- $V$ 为**顶点集**（有限集），来自 Z0 的 $(C,E)$ 配置空间；
- $E\subseteq\binom{V}{2}\times\mathbb N$ 为**边集**（允许重边：$(v,w)$ 可出现多次；先不含自环）；
- $F$ 为**面集**，每个面 $f\in F$ 是 $E$ 的一个有限子集 $\partial f\subseteq E$，称为 $f$ 的**边界**，其元素称为 $f$ 的**边界边**。

> **Zero 依据**：顶点与边来自 Z0 的 $(C,E)$；**面来自 Z0③ 的闭合词**——
> 一个满足零和 $\sum_i w_i=0$ 的闭合成边序列就是一个面（2-胞腔）。

**定义 0.2（星形）** 对 $v\in V$，$v$ 的**星形**为
$$\operatorname{star}(v)\ :=\ \{\,e\in E\ :\ v\in e\,\}\ \subseteq\ E .$$

**定义 0.3（重数）** 对 $v\neq w\in V$，记 $\mu(v,w)$ 为 $v,w$ 之间的边数（重边计重数）。

**定义 0.4（希尔伯特空间与算子）**
每条边 $e\in E$ 赋予**两个**量子比特，分别记为 $X_e$ 与 $Z_e$；总空间
$$\mathcal H=\bigotimes_{e\in E}\Big(\mathbb C^2_{X_e}\otimes\mathbb C^2_{Z_e}\Big),\qquad
\dim\mathcal H=2^{2|E|}.$$
定义**顶点算符**与**面算符**
$$A_v:=\prod_{e\in\operatorname{star}(v)}X_e,\qquad
B_f:=\prod_{e\in\partial f}Z_e .$$
（乘积按任意固定次序；下面证明它们良定义且两两对易。）

**约定 0.5（Pauli 代数）**
- 同一量子比特上：$XZ=-ZX$，$X^2=Z^2=I$；
- 不同量子比特上的算子**恒对易**（张量积因子）。

**引理 0.6（符号引理）**
设 $S,T\subseteq E$，则
$$\Big(\prod_{e\in S}X_e\Big)\Big(\prod_{e\in T}Z_e\Big)
=(-1)^{|S\cap T|}\ \Big(\prod_{e\in T}Z_e\Big)\Big(\prod_{e\in S}X_e\Big).$$

*证明.* 逐个张量因子比较。对 $e\in S\cap T$：该因子上左边是 $X_eZ_e=-Z_eX_e$，贡献一个 $-1$；对 $e\in S\setminus T$（$X_e$ 与 $I$）或 $e\in T\setminus S$（$I$ 与 $Z_e$）：该因子上两算子有一个是恒等，故对易，贡献 $+1$；对 $e\notin S\cup T$：两个恒等。$|S\cap T|$ 个 $-1$ 相乘即得。$\square$

---

## 1 三条引理

### 引理 1（同型对易）

对一切 $v,w\in V$ 与 $f,g\in F$：
$$[A_v,A_w]=0,\qquad [B_f,B_g]=0 .$$

*证明.* $A_v,A_w$ 都是 $\{X_e\}_{e\in E}$ 中算子的乘积，且每个 $X_e$ 只作用在第 $e$ 个边的 $X$ 比特上。
若 $e\neq e'$，则 $X_e,X_{e'}$ 作用在不同量子比特上，由约定 0.5 对易；
若 $e=e'$，则 $X_eX_e=X_eX_e$。故任意两个 $X$ 型算符对易，特别地 $A_vA_w=A_wA_v$。
同理（把 $X$ 换成 $Z$）得 $B_fB_g=B_gB_f$。$\square$

### 引理 2（星形交集 = 边数）

对一切 $v\neq w\in V$：
$$\big|\operatorname{star}(v)\cap\operatorname{star}(w)\big|=\mu(v,w).$$

*证明.* 由定义
\operatorname{star}(v)\cap\operatorname{star}(w)=\{\,e\in E : v\in e 且 w\in e\,\}.
一条边 $e$（作为 $\{v,w\}$ 型的二元组）同时含 $v$ 与 $w$，当且仅当 $e$ 的两个端点是 $v$ 与 $w$，即 $e$ 是 $v,w$ 之间的一条边。
故该集合恰为"$v,w$ 之间的边"之集，其元素个数按定义 0.3 即 $\mu(v,w)$。重边在此**逐条计数**，故等号两边同步计入。$\square$

**推论 2.1** 若 $K$ 是简单图（$\mu\le 1$），则 $|\operatorname{star}(v)\cap\operatorname{star}(w)|\in\{0,1\}$，其中取 $1$ 当且仅当 $vw\in E$。

> **注 2.2（必须记住的反例）** 推论 2.1 说明：若把 $X$ 与 $Z$ 型算符**都**放在顶点上（即用 $A_v$ 与 $\prod_{e\in\operatorname{star}(v)}Z_e$），
> 则相邻顶点共享恰 $1$ 条边，由引理 0.6 得**反对易**，该构造**不合法**。
> 这正是必须引入**面**的原因。

### 引理 3（星–面交界为偶：无条件版）

设 $f\in F$ 是**闭合词**（定义 0.1 的 Zero 面，即一条闭合边序列——这正是 Z0③ 零和所给出的对象）。
则对一切 $v\in V$：
$$\big|\operatorname{star}(v)\cap\partial f\big|\ =\ \deg_{\partial f}(v)\ \in\ \{0\}\cup 2\mathbb N .$$

特别地，若 $\partial f$ 是**简单圈**，则 $\deg_{\partial f}(v)\in\{0,2\}$。

*证明.* 分两步。

**第一步（交集 = 边界度数）**。
一条边 $e$ 属于 $\operatorname{star}(v)$ 当且仅当 $v\in e$；故
$$\operatorname{star}(v)\cap\partial f=\{\,e\in\partial f\ :\ v\in e\,\},$$
其元素个数逐条计数（重边重复计入）恰为 $v$ 在子（重）图 $\partial f$ 中的度数 $\deg_{\partial f}(v)$。$\square_{\text{step1}}$

**第二步（闭合 ⇒ 度数全偶）**。
设闭合词为边序列 $v_0\xrightarrow{e_1}v_1\xrightarrow{e_2}\cdots\xrightarrow{e_k}v_k$，且 $v_k=v_0$。
**闭合性**意味着该序列是一条闭合走步：每个顶点在序列中"进入"与"离开"的次数相等，
形式化地，对每个 $v$，
\#\{\,i : e_i 与 v 关联且 e_i 在序列中第奇数个出现\,\}
=\#\{\,i : 同类，第偶数个出现\,\},
因为序列是首尾相接的走步（每一次到达 $v$ 必伴随一次离开 $v$；起点 $v_0=v_k$ 亦被计入一次离开）。
故 $\deg_{\partial f}(v)$ 为偶。

**第三步（结合）**。由第一、二步，$|\operatorname{star}(v)\cap\partial f|=\deg_{\partial f}(v)\in\{0\}\cup2\mathbb N$。$\square$

> **必要性**：若序列**不闭合**（是路径），则其两个端点度数为 $1$（奇），引理 3 失效。
> 脚本 R2 核验了这一点（`zero_lemma3_unconditional.py`）：
> 路径 $(0{-}1{-}2)$ 的度数为 $\{0{:}1,1{:}2,2{:}1\}$ —— 不全偶。
> 故"**闭合**"（＝ Z0③ 的零和）是引理 3 的**必要结构**。

**推论 3.1（一般对易）** 由引理 0.6 与引理 3，$s:=|\operatorname{star}(v)\cap\partial f|$ 为偶数，故
$$A_vB_f=(-1)^{s}B_fA_v=B_fA_v .$$

**推论 3.2（$s$ 的界）** $\deg_{\partial f}(v)=0$ 或 $\ge2$；等号 $\deg=2$ 对所有 $v$ 成立当且仅当 $\partial f$ 为简单圈。
对一般的闭合词（例如"8 字"），可出现 $\deg=4$（脚本 R4 核验：顶点 $0$ 的 $\deg=4$）——
故引理 3 的**严格形式必为 $\{0\}\cup2\mathbb N$**，而 $\{0,2\}$ 是简单圈情形的特例。

---

## 2 定理 A

### 定理 A（对易判据）

设 $K=(V,E,F)$ 为定义 0.1 的复形，$A_v,B_f$ 如定义 0.4。则

**(i)** 同型算符两两对易：$[A_v,A_w]=[B_f,B_g]=0$；

**(ii)** 异型算符的对易符号为
$$A_vB_f=(-1)^{\,|\operatorname{star}(v)\cap\partial f|}\,B_fA_v ;$$

**(iii)** 从而 $A_vB_f=B_fA_v$ **恒成立**（对一切 $v\in V$、$f\in F$）。事实上由引理 3，$|\operatorname{star}(v)\cap\partial f|$ 恒为偶数，故符号 $(-1)^s=+1$。因此 $\{A_v\}\cup\{B_f\}$ 构成**两两对易**的算符族，生成一个良定义的稳定子群
$$\mathcal S=\Big\langle\,A_v\ (v\in V),\ B_f\ (f\in F)\,\Big\rangle\ \subseteq\ \mathcal P_{2|E|} .$$

*证明.* (i) 即引理 1。(ii) 即引理 0.6（取 $S=\operatorname{star}(v)$、$T=\partial f$）。
(iii) 由引理 3（无条件版），$|\operatorname{star}(v)\cap\partial f|\in\{0\}\cup2\mathbb N$ 恒为偶数，故符号为 $+1$，对易。

**良定义性**：每个生成元是自伴对合（$A_v^2=B_f^2=I$：每条边最多出现一次于星形／边界，$X^2=Z^2=I$），
且两两对易 ⇒ 生成一个阿贝尔群，其元素皆为对合 ⇒ $\mathcal S$ 同构于 $(\mathbb Z_2)^{r}$（$r$ 为生成元秩），
**无非平凡相位**（所有元素平方为 $I$、两两对易）。因此
\mathcal S 是合法的稳定子群,\qquad \mathcal C:=\{\,|\psi\rangle : g|\psi\rangle=|\psi\rangle \forall g\in\mathcal S\,\}
为 $\mathcal H$ 的非空子空间（稳定子码）。$\square$

### 推论 A.1（Zero 原生性）

$K$ 的三个部件都有 Zero 出处：

| 部件 | Zero 出处 |
|:--|:--|
| $V,E$ | Z0 的配置空间 $(C,E)$ |
| $F$ | Z0③ 的闭合词（零和 $\sum_i w_i=0$ ⇒ 闭合边序列） |
| $A_v,B_f$ | Z1 定理 1 的**局域补偿移动**：星形／边界都是**局域**支撑 |

\Longrightarrow **Zero 的零和原语原生给出一个合法的量子稳定子码**。

---

## 3 核验记录（脚本输出摘要）

**脚本**：`scripts/zero_theorem_a_lemmas_check.py`

| 检验 | 方法 | 结果 |
|:--|:--|:--|
| 引理 0.6（符号） | 单比特显式矩阵：$XZ=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，$ZX=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$，二者不等 | ✅（符号 $-1$） |
| 引理 0.6（共享 2 比特） | $X\otimes X$ 与 $Z\otimes Z$ | ✅ 对易（符号 $+1$） |
| 引理 0.6（共享 1 比特） | $X\otimes I$ 与 $Z\otimes I$ | ✅ 反对易（符号 $-1$） |
| 引理 1 | 单格复形（$V{=}4,E{=}4,F{=}1$，$2^8$ 维矩阵）：$A$-$A$、$B$-$B$ 全对易 | ✅ |
| 引理 2 | 5 个复形（含重边情形）逐对 $=$ | ✅ |
| 引理 3 | 5 个复形取值集合恰为 $[0,2]$ | ✅ |
| 引理 3（无条件版） | 闭合词度数全偶（三角／四边环／8 字／重边闭合）；非闭合路径**违反** ⇒ 闭合是必要条件 | ✅ |
| 引理 3（$s$ 的界） | "8 字"出现 $\deg=4$ ⇒ 严格形式为 $\{0\}\cup2\mathbb N$ | ✅ |
| 定理 A(ii) | 单格复形逐 $(v,f)$：$A_vB_f=(-1)^sB_fA_v$，$s=2$ 全部 | ✅ |
| 定理 A(iii) | 辛向量（任意规模）：方格 2×2/3×3/5×5、环面 3×3/5×5 异型反对易对 $=0$ | ✅ |

合计：引理 1–3（含无条件版）＋ 定理 A，全部通过。

---

## 4 诚实边界（尚未证明的部分）

| # | 项 | 状态 |
|--:|:--|:--|
| 1 | $k=2E-(V-1)-F$（码的逻辑量子比特数闭式） | **未证**；目前是实测（需 $\operatorname{rank}\{A_v\}=V-1$、$\operatorname{rank}\{B_f\}=|F|$ 的证明） |
| 3 | 距离 $d$ 与 girth／最小割的精确关系 | **未证**；仅小例实测 |
| 4 | 自环（$e=(v,v)$）的处理 | 本文件排除自环；若允许，$A_v$ 中 $X_e$ 出现一次，引理 2 的表述需调整 | 未处理 |
| 5 | 生成元的**独立性**（是否有冗余关系） | 未证；推论 A.1 只保证群良定义，不保证秩 |

---

## 5 与先前版本的差异（更正记录）

| 版本 | 主张 | 处置 |
|:--|:--|:--|
| v1（`zero_theorem_a_euler_commute.py`） | "对易 $\iff$ 图欧拉（顶点度数全偶）" | **撤回**：该构造把 $X,Z$ 都放顶点，由推论 2.1 在简单图上相邻顶点即反对易 ⇒ 构造本身非法 |
| v2（`zero_theorem_a_fixed.py`） | 推测"开边界会产生反对易对" | **撤回**：实测异型反对易对 $=0$；引理 3 给出原因 |
| v3（本文件） | 引理 1–3 ＋ 定理 A：判据是 $|\operatorname{star}(v)\cap\partial f|$ 为偶，**对一切闭合词恒成立**（无条件） | **本版** |
| 实现 | 矩阵表示（$2^{2E}$ 维） | **废弃**：3×3 环面 $4.7\times10^{21}$ 元素，实测占 43% 内存、64 min CPU 未完成 |
| 实现 | GF(2) 辛向量 | **采用**：同任务 0.14 秒 |
