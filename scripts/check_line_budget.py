#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代码行数预算检查器 —— 对应 docs/CONVENTIONS.md §3.1 / D18。

用法:
    python3 scripts/check_line_budget.py                # 检查 src/ 全部 Rust 文件
    python3 scripts/check_line_budget.py --changed      # 只检查 git 已改动文件
    python3 scripts/check_line_budget.py --json         # 机器可读输出
    python3 scripts/check_line_budget.py --strict       # WARN 也算失败

判定口径（软上限 soft=300，硬上限 hard=600）:
    新建文件（基线中不存在）
        <= soft            OK
        soft..hard         WARN   提交信息里必须写明为什么不拆
        >  hard            ERROR  必须拆
    存量文件
        仍在 soft 以内     OK     预算内增长不算超限
        超 soft 且行数增加 ERROR 超限后只许降不许涨（冻结不增）
        超 soft 但行数减少 OK     降债，方向正确

豁免：文件头注释含 `line-budget: exempt <理由>` 即跳过，且必须在
      docs/line_budget_exemptions.md 登记——未登记的豁免直接判 ERROR。

退出码: 0 = 通过（可能有 WARN）；1 = 有 ERROR。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOFT = 300
DEFAULT_HARD = 600
DEFAULT_ROOTS = ("src",)

# 文件头豁免标记：任意注释行里出现 `line-budget: exempt` 即生效
EXEMPT_RE = re.compile(r"line-budget:\s*exempt", re.IGNORECASE)
# 生成代码不做预算约束
GENERATED_RE = re.compile(r"(^|/)(generated|[\w.]*_generated)\.rs$")

OK, INFO, WARN, ERROR, SKIP = "ok", "info", "warn", "error", "skip"
RANK = {OK: 0, INFO: 1, SKIP: 2, WARN: 3, ERROR: 4}


@dataclass
class Verdict:
    path: str
    lines: int
    base: int | None
    status: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "lines": self.lines,
            "base": self.base,
            "status": self.status,
            "reason": self.reason,
        }


def run_git(repo: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout


def repo_root(start: Path) -> Path:
    code, out = run_git(str(start), "rev-parse", "--show-toplevel")
    if code == 0 and out.strip():
        return Path(out.strip())
    return start


def count_lines(path: Path, effective: bool) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not effective:
        return len(lines)
    n = 0
    in_block = False
    for raw in lines:
        line = raw.strip()
        if in_block:
            if "*/" in line:
                in_block = False
                line = line.split("*/", 1)[1].strip()
            else:
                continue
        if not line:
            continue
        if line.startswith("//"):
            continue
        if line.startswith("/*"):
            if "*/" not in line:
                in_block = True
                continue
            line = line.split("*/", 1)[1].strip()
            if not line:
                continue
        if line.startswith("*"):
            continue
        n += 1
    return n


def has_exempt(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    return bool(EXEMPT_RE.search(head))


def registered_exempts(repo: Path, table: Path) -> set[str]:
    """解析豁免登记表（docs/line_budget_exemptions.md）的「文件」列。"""
    if not table.is_file():
        return set()
    paths: set[str] = set()
    for raw in table.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        if not first or first == "文件" or set(first) <= set("-: "):
            continue
        for token in re.split(r"[,、]", first):
            token = token.strip().strip("`").strip()
            if token and token != "—":
                paths.add(token)
    return paths


def base_lines(repo: Path, rel: str, baseline: str) -> int | None:
    """基线版本的行数；文件在基线中不存在则为 None（= 新文件）。"""
    code, out = run_git(repo, "show", f"{baseline}:{rel}")
    if code != 0:
        return None
    return len(out.splitlines())


def classify(
    rel: str,
    lines: int,
    base: int | None,
    soft: int,
    hard: int,
) -> Verdict:
    if base is None:
        if lines > hard:
            return Verdict(rel, lines, base, ERROR,
                           f"新建文件 {lines} 行 > 硬上限 {hard}，必须拆分")
        if lines > soft:
            return Verdict(rel, lines, base, WARN,
                           f"新建文件 {lines} 行 > 软上限 {soft}，提交信息需说明为何不拆")
        return Verdict(rel, lines, base, OK, "")

    if lines > soft:
        if lines > base:
            return Verdict(rel, lines, base, ERROR,
                           f"已超软上限 {soft} 且行数增长（{base} → {lines}），存量冻结不增")
        return Verdict(rel, lines, base, OK,
                       f"超软上限但已降债（{base} → {lines}）")
    return Verdict(rel, lines, base, OK, "")


def collect_files(repo: Path, roots: tuple[str, ...], excludes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        base = repo / root
        if base.is_file():
            files.append(base)
            continue
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            rel = path.relative_to(repo).as_posix()
            if any(part in ("target", ".git") for part in path.parts):
                continue
            if any(re.search(pat, rel) for pat in excludes):
                continue
            files.append(path)
    return files


def changed_files(repo: Path, baseline: str) -> set[str]:
    code, out = run_git(repo, "diff", "--name-only", baseline)
    rels: set[str] = set()
    if code == 0:
        rels |= {line.strip() for line in out.splitlines() if line.strip()}
    code, out = run_git(repo, "ls-files", "--others", "--exclude-standard")
    if code == 0:
        rels |= {line.strip() for line in out.splitlines() if line.strip()}
    return {r for r in rels if r.endswith(".rs")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="代码行数预算检查（CONVENTIONS.md §3.1）")
    ap.add_argument("--root", default=None, help="仓库根（默认自动探测）")
    ap.add_argument("--baseline", default="HEAD", help="基线 ref（默认 HEAD）")
    ap.add_argument("--soft", type=int, default=DEFAULT_SOFT, help="软上限行数（默认 300）")
    ap.add_argument("--hard", type=int, default=DEFAULT_HARD, help="硬上限行数（默认 600）")
    ap.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS), help="扫描目录（默认 src）")
    ap.add_argument("--exclude", nargs="*", default=[], help="排除路径正则")
    ap.add_argument("--changed", action="store_true", help="只检查相对基线有改动的文件")
    ap.add_argument("--count-effective", action="store_true",
                    help="按有效行计数（去空行与注释，近似）而非文件总行数")
    ap.add_argument("--strict", action="store_true", help="WARN 也算失败")
    ap.add_argument("--debt", action="store_true",
                    help="盘点模式：列出所有超软上限的存量文件（不判失败），用于看债务")
    ap.add_argument("--exempt-table", default="docs/line_budget_exemptions.md",
                    help="豁免登记表（相对仓库根），用于核对豁免是否登记")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    repo = Path(args.root).resolve() if args.root else repo_root(Path(__file__).resolve().parent)
    if not repo.is_dir():
        print(f"仓库根不存在: {repo}", file=sys.stderr)
        return 2

    files = collect_files(repo, tuple(args.roots), tuple(args.exclude))
    if args.changed:
        touched = changed_files(repo, args.baseline)
        files = [f for f in files if f.relative_to(repo).as_posix() in touched]

    table = Path(args.exempt_table)
    if not table.is_absolute():
        table = repo / table
    registered = registered_exempts(repo, table)

    verdicts: list[Verdict] = []
    for path in files:
        rel = path.relative_to(repo).as_posix()
        if GENERATED_RE.search(rel):
            verdicts.append(Verdict(rel, 0, None, SKIP, "生成代码"))
            continue
        lines = count_lines(path, args.count_effective)
        if has_exempt(path):
            if rel in registered:
                verdicts.append(Verdict(rel, lines, None, SKIP, "已豁免（已登记）"))
            else:
                verdicts.append(Verdict(rel, lines, None, ERROR,
                                        f"声明了豁免但未登记到 {table.name}"
                                        f"（未登记的豁免视为违规）"))
            continue
        base = base_lines(repo, rel, args.baseline)
        verdict = classify(rel, lines, base, args.soft, args.hard)
        if rel in registered:
            verdict.status = INFO if verdict.status == OK else verdict.status
            verdict.reason = (verdict.reason + "；" if verdict.reason else "") + \
                "登记表中存在该文件但未声明豁免标记（过期登记）"
        verdicts.append(verdict)

    errors = [v for v in verdicts if v.status == ERROR]
    warns = [v for v in verdicts if v.status == WARN]
    infos = [v for v in verdicts if v.status == INFO]
    failed = bool(errors) or (args.strict and bool(warns))

    if args.debt:
        debt = [v for v in verdicts if v.lines > args.soft and v.status != SKIP]
        debt.sort(key=lambda v: -v.lines)
        if args.json:
            print(json.dumps({
                "soft": args.soft,
                "hard": args.hard,
                "debt_files": len(debt),
                "debt_lines": sum(v.lines for v in debt),
                "files": [v.as_dict() for v in debt],
            }, ensure_ascii=False, indent=2))
            return 0
        print(f"行数债务盘点 · 软上限={args.soft} · 硬上限={args.hard}")
        print(f"超软上限文件 {len(debt)} 个，合计 {sum(v.lines for v in debt)} 行\n")
        for v in debt:
            flag = " [超硬上限]" if v.lines > args.hard else ""
            print(f"  {v.lines:>6} 行  {v.path}{flag}")
        return 0

    if args.json:
        print(json.dumps({
            "soft": args.soft,
            "hard": args.hard,
            "baseline": args.baseline,
            "checked": len(verdicts),
            "errors": len(errors),
            "warnings": len(warns),
            "infos": len(infos),
            "passed": not failed,
            "verdicts": [v.as_dict() for v in verdicts],
        }, ensure_ascii=False, indent=2))
        return 1 if failed else 0

    scope = "改动文件" if args.changed else "/".join(args.roots)
    print(f"行数预算检查 · 范围={scope} · 软上限={args.soft} · 硬上限={args.hard} · 基线={args.baseline}")
    print(f"检查 {len(verdicts)} 个文件：ERROR {len(errors)} · WARN {len(warns)} · INFO {len(infos)}")

    for status, items in ((ERROR, errors), (WARN, warns), (INFO, infos)):
        if not items:
            continue
        print()
        print(f"── {status.upper()} ──")
        for v in sorted(items, key=lambda x: -x.lines):
            base_txt = "新建" if v.base is None else f"基线 {v.base}"
            print(f"  {v.lines:>6} 行  ({base_txt})  {v.path}\n         {v.reason}")

    if failed:
        print("\n结果：不通过。拆分后重跑，或按 CONVENTIONS.md §3.1 登记豁免。")
        return 1
    print("\n结果：通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
