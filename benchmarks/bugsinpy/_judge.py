"""benchmarks/bugsinpy/_judge.py — BugsInPy 判分核心（F2P/P2P 双判据）。

从 QuixBugs run_benchmark.py 移植并泛化（那里按题读 meta.json 且固定 sys.executable；
这里 pytest 走每题独立 venv 的解释器、测试选择由调用方给），行为保持同构：
junitxml 逐用例解析 + 超时兜底 + 基线/判分轮对比。

与 QuixBugs 的关键差异（探针实测）：
- BugsInPy 的**纯基线采集会收进环境坏测试**（依赖漂移导致、buggy/fixed 都失败的用例，
  如 httpie 的 test_follow_show_redirects）。因此 F2P 集合应由调用方按
  「buggy 失败 ∧ fixed 通过」精确计算，不能直接拿全部基线失败当 F2P。
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(xml_path: Path) -> dict[str, str]:
    """解析 junitxml → {用例名: passed/failed/error/skipped}。缺失/解析失败返回 {}。"""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}
    cases: dict[str, str] = {}
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        if not name:
            continue
        if tc.find("failure") is not None:
            cases[name] = "failed"
        elif tc.find("error") is not None:
            cases[name] = "error"
        elif tc.find("skipped") is not None:
            cases[name] = "skipped"
        else:
            cases[name] = "passed"
    return cases


def run_pytest(cwd: Path, test_args: list[str] | str, python_bin: str,
               junit_name: str | None = None, timeout: float = 15) -> tuple[str, str, dict[str, str]]:
    """在 cwd 用 python_bin 跑 pytest，返回 (status, output, 逐用例结果)。

    - 不加 -x：双判据需要全量用例独立出结果。
    - junit_name 给定时写 <cwd>/<junit_name> 并解析逐用例结果；
      timeout/解析失败时逐用例结果为空 dict，调用方按基线缺失回落。
    - timeout 携带的部分输出总是 bytes（即使 text=True），需分别解码再拼。
    """
    cmd = [python_bin, "-m", "pytest"]
    cmd += [test_args] if isinstance(test_args, str) else list(test_args)
    cmd.append("--tb=short")
    if junit_name:
        cmd.append(f"--junitxml={cwd / junit_name}")
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        status = "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired as e:
        out, err = e.stdout or "", e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        output = out + err + f"\n[pytest 超时 {timeout}s]"
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        output = f"[pytest 异常] {e}"
        status = "error"
    cases = parse_junit(cwd / junit_name) if junit_name else {}
    return status, output, cases


def case_breakdown(base_cases: dict[str, str], final_cases: dict[str, str]) -> (
        tuple[list[str], list[str], list[str], list[str]] | None):
    """对比基线与判分轮逐用例结果，返回 (f2p, f2p_failed, p2p, p2p_broken)。

    任一轮逐用例结果缺失返回 None，调用方回落旧行为。
    final 中查不到的用例（被 agent 删/改名）按失败计。
    """
    if not base_cases or not final_cases:
        return None
    f2p = sorted(n for n, st in base_cases.items() if st in ("failed", "error"))
    p2p = sorted(n for n, st in base_cases.items() if st == "passed")
    f2p_failed = [n for n in f2p if final_cases.get(n) != "passed"]
    p2p_broken = [n for n in p2p if final_cases.get(n) != "passed"]
    return f2p, f2p_failed, p2p, p2p_broken


def fmt_cases(r: dict) -> str:
    """逐题输出行的 F2P/P2P 细分后缀。"""
    parts = []
    f2p, p2p = r.get("f2p"), r.get("p2p")
    if f2p:
        note = "" if not f2p["failed"] else f" 没修好: {','.join(f2p['failed'])}"
        parts.append(f"f2p={f2p['passed']}/{f2p['total']}{note}")
    if p2p:
        note = "" if not p2p["failed"] else f" 修坏回归: {','.join(p2p['failed'])}"
        parts.append(f"p2p={p2p['passed']}/{p2p['total']}{note}")
    return (" [" + "; ".join(parts) + "]") if parts else ""


def print_case_summary(results: list[dict]) -> None:
    """汇总 F2P/P2P 用例级数据（细分缺失的题不计入）。"""
    scored = [r for r in results if r.get("f2p")]
    if not scored:
        return
    f2p_pass = sum(r["f2p"]["passed"] for r in scored)
    f2p_total = sum(r["f2p"]["total"] for r in scored)
    p2p_pass = sum(r["p2p"]["passed"] for r in scored)
    p2p_total = sum(r["p2p"]["total"] for r in scored)
    print(f"F2P 用例修复: {f2p_pass}/{f2p_total}；P2P 用例保持: {p2p_pass}/{p2p_total}")


if __name__ == "__main__":
    # 干跑自检：无外部输入时直接确认模块可导入。
    print("judge module OK")
    sys.exit(0)
