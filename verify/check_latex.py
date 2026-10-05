#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_latex.py —— Zero-CSS 文档的 LaTeX/GitHub 渲染检查

检查四类**实际会导致 github.com 渲染失败或错乱**的问题：
  E1  $$ 与公式内容粘在同一行（多行显示块被切断）
  E2  表格行内的数学含裸 | （会切断表格单元格）
  E3  \\boxed （GitHub 的 MathJax 未加载 bbox 扩展）
  E4  制表符/退格符混入（应为反斜杠）

退出码 0 = 全部通过。
"""
import os
import re
import sys
import glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["README.md"] + sorted(
    os.path.relpath(p, HERE) for p in glob.glob(os.path.join(HERE, "docs", "*.md"))
)
# 只支持裸 | 定界符的替代
PIPES = {"|F|": r"\\lvert F\\rvert"}


def read(f):
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        return fh.read()


def e1_stuck_display(lines):
    """多行 $$ 块的边界：开块后、闭块前都必须是空行。

    用状态机区分开块/闭块（裸 $$ 成对出现）。
    """
    out = []
    in_block = False
    for i, l in enumerate(lines):
        st = l.strip()
        if st == "$$":
            if not in_block:
                # 开块：前面应为空行（文件头/表格外），后面应紧跟内容
                prev = lines[i - 1].strip() if i > 0 else ""
                if prev and not prev.startswith("|") and not prev.startswith("#"):
                    out.append((i + 1, "开块 $$ 前缺空行: " + prev[:50]))
                in_block = True
            else:
                # 闭块：前面（公式末行）必须非空，后面必须为空行
                prev = lines[i - 1].strip() if i > 0 else ""
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if not prev:
                    out.append((i + 1, "闭块 $$ 前有空行（块为空）"))
                if nxt:
                    out.append((i + 1, "闭块 $$ 后缺空行: " + nxt[:50]))
                in_block = False
    return out


def e2_table_pipe(lines):
    """表格行内数学含裸 | 。"""
    out = []
    for i, l in enumerate(lines, 1):
        if not l.lstrip().startswith("|"):
            continue
        if l.lstrip().startswith("|--") or l.lstrip().startswith("| --"):
            continue
        for seg in re.findall(r"\$\$(.+?)\$\$", l) + re.findall(r"(?<!\$)\$([^$]+)\$(?!\$)", l):
            if "|" in seg:
                out.append((i, seg[:70]))
    return out


def e3_boxed(lines):
    return [(i, l.strip()[:70]) for i, l in enumerate(lines, 1) if r"\boxed" in l]


def e4_control_chars(text):
    out = []
    for m in re.finditer(r"[\t\b]", text):
        ln = text[: m.start()].count("\n") + 1
        ctx = text[max(0, m.start() - 35) : m.start() + 35].replace("\n", " ")
        out.append((ln, "TAB" if m.group() == "\t" else "BACKSPACE", ctx))
    return out


def e5_table_columns(lines):
    """表格每行的 | 数必须一致（否则该行会被解析成额外列）。"""
    out = []
    tbl = []
    def flush():
        if len(tbl) > 1:
            cnts = set(c for _, c in tbl)
            if len(cnts) > 1:
                from collections import Counter
                mode = Counter(c for _, c in tbl).most_common(1)[0][0]
                for ln, c in tbl:
                    if c != mode:
                        out.append((ln, "| 数 %d（同表众数 %d）" % (c, mode)))
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith("|"):
            tbl.append((i, l.count("|")))
        else:
            flush()
            tbl = []
    flush()
    return out


def main():
    total = 0
    print("=" * 78)
    print("Zero-CSS · LaTeX / GitHub 渲染检查")
    print("=" * 78)
    for f in FILES:
        text = read(f)
        lines = text.split("\n")
        r1, r2, r3, r4, r5 = (
            e1_stuck_display(lines),
            e2_table_pipe(lines),
            e3_boxed(lines),
            e4_control_chars(text),
            e5_table_columns(lines),
        )
        n = len(r1) + len(r2) + len(r3) + len(r4) + len(r5)
        total += n
        status = "OK" if n == 0 else "%d 处问题" % n
        print("\n%-46s %s" % (f, status))
        for ln, txt in r1:
            print("   E1 L%-4d $$ 与公式同行: %s" % (ln, txt))
        for ln, txt in r2:
            print("   E2 L%-4d 表格内裸 |: %s" % (ln, txt))
        for ln, txt in r3:
            print("   E3 L%-4d \\boxed（GitHub 不渲染）: %s" % (ln, txt))
        for ln, kind, ctx in r4:
            print("   E4 L%-4d 控制字符(%s): ...%s..." % (ln, kind, ctx))
        for ln, txt in r5:
            print("   E5 L%-4d 表格列数不一致: %s" % (ln, txt))
    print("\n" + "=" * 78)
    print("合计问题: %d" % total)
    if total:
        print("不符 ✓ -> 需修复")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
