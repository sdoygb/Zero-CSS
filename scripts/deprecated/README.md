# deprecated —— 失败的尝试（保留溯源）

| 脚本 | 失败点 |
|:--|:--|
| `zero_hgp_distance.py` | 维度 bug（`L=2` 环图退化 + `in_rowspace` 变量覆盖）；已被 `../zero_hgp_dist2.py` 取代 |
| `zero_hyperbolic_test.py` | 格式串 bug；且只给抽象计数、不构造复形 |
| `zero_hyperbolic_explicit.py` | 随机粘合未做流形检验 ⇒ 对易列全 False（构造非法） |
| `zero_hgp_dist_formula.py` | 猜的公式 `d = min(girth1, girth2, 最小割)` **被否证**：$C_L$ 给 $d=2$，实测 $d=L$。正确式是 `d = min(girth1, girth2)`（见 `../zero_hgp_distance_final.py`） |

**双曲显式构造我连续失败 5 次**（记录见 `docs/rate_levers_verdict.md` §5 与 `docs/hgp_recipe_B.md`）。
正确做法应使用已发表的 $\{5,4\}$ 双曲码关联表，或 GAP 的 `SimplicialSurface`。
