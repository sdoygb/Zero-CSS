# -*- coding: utf-8 -*-
"""safe_macros.py —— GitHub 行内公式的**最小安全宏集**

背景：GitHub 的 MathJax 维护一个**未公开、且会变动**的宏白名单，
实测已确认被拒（github/markup#1688，2023 起至今未修）：
    \\operatorname  \\operatorname*  \\boxed  \\boldsymbol（时好时坏）
    \\llbracket     \\rrbracket（实测红字）

策略（A + B）：
  A. 行内公式只使用**最小安全宏集**——凡不在表内的宏，一律改写成
     不依赖该宏的等价写法。
  B. 行内公式一律用**反引号定界** `` $`...`$ ``（GitHub 官方文档推荐的
     "表达式与 markdown 语法冲突"时的写法），以规避 `_` `[[` `]]` `\\[`
     等字符被 Markdown 层吃掉。

安全集只收 MathJax **v2 base 包**（十余年未变、GitHub 官方文档示例中
出现过的）构造。任何"花哨"的宏都不进这个集。
"""

# ---- 安全：MathJax v2 base/ams 包中长期稳定的构造 ----
SAFE = set("""
frac dfrac tfrac sqrt sum prod int oint iint iiint lim
left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr
langle rangle lvert rvert vert Vert lVert rVert lceil rceil lfloor rfloor
partial nabla infty cdots ldots vdots ddots dots
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi
chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
mathbb mathcal mathbf mathfrak mathrm mathit mathsf mathtt
leq geq neq approx equiv sim simeq cong propto ll gg
subset supset subseteq supseteq in ni notin emptyset varnothing
cup cap setminus backslash times div pm mp cdot ast star circ bullet
oplus ominus otimes oslash odot
wedge vee neg land lor forall exists nexists
rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow
mapsto to gets uparrow downarrow updownarrow
quad qquad
hat bar vec dot ddot tilde widehat widetilde overline underline
stackrel overset underset
begin end
not
colon
square
angle
deg
prime
Re Im
aleph hbar ell wp
top bot
vdash dashv models
perp parallel
asymp
doteq
mid
mod bmod pmod
text textbf textit textrm textsf texttt
ker dim min max inf sup lim log ln exp sin cos tan det gcd
binom dbinom tbinom
le ge leq geq neq ne approx equiv sim simeq cong propto ll gg
iff implies impliedby Longrightarrow longrightarrow Longleftrightarrow
longleftrightarrow Longleftarrow longleftarrow
dagger ddagger bigotimes bigoplus bigodot bigcup bigcap bigsqcup
textstyle displaystyle scriptstyle scriptscriptstyle
rm bf it sf tt cal
square blacksquare triangle blacktriangle
hbar ell wp Re Im aleph
angle measuredangle
colon
sideset
overset underset stackrel
substack
pmod bmod mod
cases
aligned align array matrix pmatrix bmatrix vmatrix Vmatrix
""".split())

# ---- 已知不安全（实测被拒）----
BANNED = {
    "operatorname": "github/markup#1688，实测被拒",
    "operatorname*": "同上",
    "boxed": "bbox 扩展未加载",
    "llbracket": "实测红字",
    "rrbracket": "实测红字",
    "boldsymbol": "时好时坏",
    "bm": "非核心",
    "xrightarrow": "未验证，保守起见不用",
    "upharpoonright": "未验证，保守起见不用",
    "tag": "不支持",
    "label": "不支持",
    "eqref": "不支持",
}

# ---- 推荐改写表：不安全宏 -> 等价的安全写法 ----
REWRITE = [
    (r"\operatorname*", ""),          # 先处理带星的
    (r"\operatorname", ""),
    (r"\llbracket", "["),
    (r"\rrbracket", "]"),
    (r"\mathrm", ""),
    (r"\boxed", ""),
]


def is_safe(name: str) -> bool:
    return name in SAFE


def audit(text: str):
    """返回 (不安全的宏名 -> 出现次数)。"""
    import re
    bad = {}
    for m in re.finditer(r"\\([a-zA-Z]+)", text):
        n = m.group(1)
        if n not in SAFE:
            bad[n] = bad.get(n, 0) + 1
    return bad
