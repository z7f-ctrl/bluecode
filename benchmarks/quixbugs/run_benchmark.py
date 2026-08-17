#!/usr/bin/env python3
"""QuixBugs benchmark runner（v0.7.3：F2P/P2P 双判据）。

流程（每题）：
1. 复制 tasks/<algo>/ 到临时 workdir；
2. **基线轮**：先对未修复的 buggy 版跑一次 pytest（junitxml 逐用例解析），
   采集 FAIL_TO_PASS 用例集（基线 failed/error）与 PASS_TO_PASS 用例集（基线 passed），
   输出落 `<run-dir>/<algo>.baseline.log`；
3. 以 --auto-approve 子进程启动 agent 修 buggy.py（日志落盘 <run-dir>/<algo>.agent.log）；
4. **判分轮**：跑 pytest 并逐用例对比基线——f2p 仍有失败 =「没修好」，
   p2p 出现失败 =「修坏回归」；日志落盘 <run-dir>/<algo>.pytest.log。

用法：
    python3 run_benchmark.py --dry-run          # 列出任务
    python3 run_benchmark.py --self-test        # 管线自验（正确实现应 f2p 全过 / buggy 版应有 f2p），不跑 agent
    python3 run_benchmark.py --algo gcd         # 单题冒烟
    python3 run_benchmark.py                    # 全量
    python3 run_benchmark.py --workers 4        # 4 题并行（注意 LLM 限流）

结果：results/run-<timestamp>.json（除 status 外加记 f2p/p2p 用例细分）；
完整输出在 results/run-<timestamp>/ 目录下逐题日志。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT_PY = ROOT.parents[1] / "agent.py"
AGENT_TIMEOUT = 180  # 单题 agent 上限（秒）
PYTEST_TIMEOUT = 15  # pytest 上限（死循环题）
# 管线自验抽样：两题参数化简单题 + 一题图题（tasks/ 自带 node.py 不依赖外部）；
# 选题要求 buggy 版快速 fail——bitcount 这类死循环题基线轮必超时拿不到逐用例结果
SELF_TEST_ALGOS = ["gcd", "mergesort", "shortest_paths"]


def list_tasks() -> list[str]:
    """返回 tasks/ 下所有任务名（有 meta.json 的目录）。"""
    return sorted(
        d.name for d in TASKS_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()
    )


def parse_junit(xml_path: Path) -> dict[str, str]:
    """解析 junitxml → {用例名: passed/failed/error/skipped}。文件缺失或解析失败返回 {}（调用方回落）。"""
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


def run_pytest(task_dir: Path, junit_name: str | None = None) -> tuple[str, str, dict[str, str]]:
    """在 task_dir 下跑 pytest，返回 (status, output, 逐用例结果)。

    不加 -x：双判据需要全量用例独立出结果（即便前面的用例失败）。
    junit_name 给定时写 <task_dir>/<junit_name> 并解析逐用例结果；
    timeout/解析失败时逐用例结果为空 dict，调用方按基线缺失回落。
    """
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    test_file = meta["test_file"]
    cmd = [sys.executable, "-m", "pytest", test_file, "--tb=short"]
    if junit_name:
        cmd.append(f"--junitxml={task_dir / junit_name}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT,
        )
        output = proc.stdout + proc.stderr
        status = "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired 携带的部分输出总是 bytes（即使 text=True），需分别解码再拼
        out, err = e.stdout or "", e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        output = out + err + f"\n[pytest 超时 {PYTEST_TIMEOUT}s]"
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        output = f"[pytest 异常] {e}"
        status = "error"
    cases = parse_junit(task_dir / junit_name) if junit_name else {}
    return status, output, cases


def setup_task_workdir(algo: str) -> Path:
    """复制任务到临时目录，返回 workdir 路径。"""
    src = TASKS_DIR / algo
    workdir = Path(tempfile.mkdtemp(prefix=f"quixbugs-{algo}-"))
    for f in src.iterdir():
        if f.suffix in (".py", ".json"):
            shutil.copy2(f, workdir / f.name)
    # 补一个兜底 conftest.py：knapsack 等测试引用 pytest.run_slow / use_correct，
    # 原仓库靠根 conftest 提供，缺了会 AttributeError 误判 fail
    conftest = workdir / "conftest.py"
    if not conftest.exists():
        conftest.write_text(
            "import pytest\n"
            "def pytest_configure(config):\n"
            "    pytest.use_correct = getattr(pytest, 'use_correct', False)\n"
            "    pytest.run_slow = getattr(pytest, 'run_slow', False)\n",
            encoding="utf-8",
        )
    return workdir


def _case_breakdown(
    base_cases: dict[str, str], final_cases: dict[str, str]
) -> tuple[list[str], list[str], list[str], list[str]] | None:
    """对比基线与判分轮逐用例结果，返回 (f2p, f2p_failed, p2p, p2p_broken)。

    任一轮逐用例结果缺失（timeout/解析失败）返回 None，调用方回落旧行为。
    final 中查不到的用例（被 agent 删/改名）按失败计。
    """
    if not base_cases or not final_cases:
        return None
    f2p = sorted(n for n, st in base_cases.items() if st in ("failed", "error"))
    p2p = sorted(n for n, st in base_cases.items() if st == "passed")
    f2p_failed = [n for n in f2p if final_cases.get(n) != "passed"]
    p2p_broken = [n for n in p2p if final_cases.get(n) != "passed"]
    return f2p, f2p_failed, p2p, p2p_broken


def run_single(algo: str, *, dry_run: bool = False, log_dir: Path | None = None) -> dict:
    """跑单个任务，返回结果 dict。"""
    workdir = setup_task_workdir(algo)
    request = (
        f"当前目录是 QuixBugs 任务 {algo} 的工作目录。"
        "请修复 buggy.py 中的 bug（文件里有且仅有一个 bug）。"
        "要求：用相对路径读写文件，不要 cd 到其他目录，不要修改测试文件；"
        "先用 read_file 读懂代码再改，优先 plan_patch 最小修改；"
        "不要自己跑 pytest（判分由外部执行）。"
    )
    if dry_run:
        return {"algo": algo, "workdir": str(workdir), "request": request}

    result = {
        "algo": algo,
        "workdir": str(workdir),
        "status": "pending",
        "pytest_status": None,
        "baseline_pytest_status": None,
        "f2p": None,
        "p2p": None,
        "agent_output": "",
        "duration": 0.0,
    }

    def _write_log(kind: str, text: str) -> None:
        if log_dir is None:
            return
        (log_dir / f"{algo}.{kind}.log").write_text(text, encoding="utf-8")
        result[f"{kind}_log"] = str(log_dir / f"{algo}.{kind}.log")

    start = time.time()

    # ── 基线轮：修复前对 buggy 版采集逐用例结果，得 F2P/P2P 用例集 ──
    base_status, base_output, base_cases = run_pytest(workdir, "baseline.xml")
    _write_log("baseline", base_output)
    result["baseline_pytest_status"] = base_status

    try:
        proc = subprocess.run(
            [sys.executable, str(AGENT_PY), request, "--auto-approve"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=AGENT_TIMEOUT,
        )
        full_output = proc.stdout + proc.stderr
        result["agent_output"] = full_output[-2000:]
        result["status"] = "agent-done" if proc.returncode == 0 else "agent-error"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        result["agent_output"] = out[-2000:]
        result["status"] = "agent-timeout"
        full_output = out + f"\n[agent 超时 {AGENT_TIMEOUT}s]"
    except Exception as e:  # noqa: BLE001
        result["agent_output"] = f"[runner 异常] {e}"
        result["status"] = "agent-error"
        full_output = result["agent_output"]
    _write_log("agent", full_output)

    result["duration"] = round(time.time() - start, 1)

    # ── 判分轮 ──
    pytest_status, pytest_output, final_cases = run_pytest(workdir, "final.xml")
    result["pytest_status"] = pytest_status
    _write_log("pytest", pytest_output)
    breakdown = _case_breakdown(base_cases, final_cases)
    if breakdown is not None:
        f2p, f2p_failed, p2p, p2p_broken = breakdown
        result["f2p"] = {
            "passed": len(f2p) - len(f2p_failed),
            "total": len(f2p),
            "failed": f2p_failed,
        }
        result["p2p"] = {
            "passed": len(p2p) - len(p2p_broken),
            "total": len(p2p),
            "failed": p2p_broken,
        }
    if pytest_status == "pass" and result["status"] == "agent-done":
        result["status"] = "pass"
    elif result["status"] == "agent-done":
        result["status"] = "fail"
    return result


def self_test() -> int:
    """管线自验：buggy 版基线应有 f2p 失败用例；换成 correct 实现后应 f2p 全过且无 p2p 回归。

    不启动 agent 子进程，纯判分管线检查（correct 源来自 /tmp/quixbugs-src）。
    返回进程退出码（0 全对 / 1 有不符）。
    """
    correct_dir = Path("/tmp/quixbugs-src/correct_python_programs")
    bad = 0
    for algo in SELF_TEST_ALGOS:
        workdir = setup_task_workdir(algo)
        try:
            base_status, _, base_cases = run_pytest(workdir, "baseline.xml")
            if not base_cases:
                print(f"{algo}: BAD — 基线轮无逐用例结果（{base_status}）")
                bad += 1
                continue
            f2p = sorted(n for n, s in base_cases.items() if s in ("failed", "error"))
            p2p = sorted(n for n, s in base_cases.items() if s == "passed")
            bug_ok = bool(f2p)
            print(f"{algo}: buggy 版基线 f2p={len(f2p)} p2p={len(p2p)} → {'OK' if bug_ok else 'BAD（buggy 版应有失败用例）'}")
            bad += 0 if bug_ok else 1

            src = correct_dir / f"{algo}.py"
            if not src.exists():
                print(f"{algo}: SKIP — correct 源缺失 {src}（确认 /tmp/quixbugs-src 已 clone）")
                continue
            shutil.copy2(src, workdir / "buggy.py")
            fin_status, fin_output, fin_cases = run_pytest(workdir, "final.xml")
            still_failed = [n for n in f2p if fin_cases.get(n) != "passed"]
            broken = [n for n in p2p if fin_cases.get(n) != "passed"]
            fix_ok = fin_status == "pass" and not still_failed and not broken
            print(
                f"{algo}: correct 版 pytest={fin_status} f2p 仍失败={still_failed or '无'}"
                f" p2p 回归={broken or '无'} → {'OK' if fix_ok else 'BAD'}"
            )
            if not fix_ok:
                print(fin_output[-1500:])
            bad += 0 if fix_ok else 1
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    print("SELF-TEST " + ("PASSED ✅" if bad == 0 else f"FAILED（{bad} 项不符）"))
    return 0 if bad == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="QuixBugs benchmark runner（F2P/P2P 双判据）")
    p.add_argument("--algo", help="只跑指定算法（如 gcd）")
    p.add_argument("--dry-run", action="store_true", help="只列出任务，不执行")
    p.add_argument("--self-test", action="store_true", help="管线自验：correct/buggy 基线对照，不跑 agent")
    p.add_argument("--output", help="结果 json 输出路径（默认 results/run-<timestamp>.json）")
    p.add_argument("--workers", type=int, default=1, help="并行任务数（默认 1 串行；注意 LLM 限流）")
    args = p.parse_args()

    if args.self_test:
        sys.exit(self_test())

    algos = [args.algo] if args.algo else list_tasks()
    if not algos:
        print("没有找到任务，先跑 prepare.py"); sys.exit(1)

    if args.dry_run:
        for a in algos:
            r = run_single(a, dry_run=True)
            print(f"- {r['algo']}\n  workdir: {r['workdir']}\n  request: {r['request']}")
        print(f"\n共 {len(algos)} 个任务")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"run-{ts}.json"
    log_dir = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    workers = max(1, args.workers)
    if workers == 1:
        for i, a in enumerate(algos, 1):
            r = run_single(a, log_dir=log_dir)
            print(f"[{i}/{len(algos)}] {r['algo']}: {r['status']} "
                  f"(pytest={r['pytest_status']}, {r['duration']}s){_fmt_cases(r)}", flush=True)
            results.append(r)
    else:
        print(f"并行 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_single, a, log_dir=log_dir): a for a in algos}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                print(f"[{done}/{len(algos)}] {r['algo']}: {r['status']} "
                      f"(pytest={r['pytest_status']}, {r['duration']}s){_fmt_cases(r)}", flush=True)
                results.append(r)
        # as_completed 顺序乱，按 algo 排序后写盘
        results.sort(key=lambda r: r["algo"])

    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n结果: {passed}/{len(results)} pass")
    _print_case_summary(results)

    out_path.write_text(
        json.dumps(
            {"timestamp": ts, "passed": passed, "total": len(results), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"写入 {out_path}")
    print(f"日志目录 {log_dir}")


def _fmt_cases(r: dict) -> str:
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


def _print_case_summary(results: list[dict]) -> None:
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
    main()
