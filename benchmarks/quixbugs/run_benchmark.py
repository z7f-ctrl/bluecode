#!/usr/bin/env python3
"""QuixBugs 基准 runner：让 bluecode agent 修 bug，用 pytest 判分。

流程：
1. 对每个任务，把 buggy.py 复制到独立工作目录
2. 起独立 thread 让 agent 读文件 → 修 bug → verifier 跑测试
3. 收集结果：pass（测试全过）/ fail（测试失败）/ timeout / error

用法：
  python benchmarks/quixbugs/run_benchmark.py                    # 跑全部 31 题
  python benchmarks/quixbugs/run_benchmark.py --algo bitcount    # 跑单题
  python benchmarks/quixbugs/run_benchmark.py --dry-run          # 只打印任务，不跑
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT_PY = ROOT.parents[1] / "agent.py"  # 项目根目录的 agent.py，避免硬编码绝对路径

# 超时配置（秒）
AGENT_TIMEOUT = 180  # agent 单题最大运行时间（120s 实测大量题顶格，fail-closed 后修订轮次增多）
PYTEST_TIMEOUT = 15   # pytest 单题最大运行时间（防死循环）


def list_tasks() -> list[str]:
    """列出所有可用任务名。"""
    return sorted(
        d.name for d in TASKS_DIR.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )


def run_pytest(task_dir: Path) -> tuple[str, str]:
    """在任务目录跑 pytest。返回 (status, output)。
    status: "pass" | "fail" | "timeout" | "error"
    """
    meta = json.loads((task_dir / "meta.json").read_text())
    test_file = meta["test_file"]
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-x", "--tb=short"],
            capture_output=True, text=True, timeout=PYTEST_TIMEOUT, cwd=task_dir,
        )
        output = r.stdout + r.stderr
        if r.returncode == 0:
            return "pass", output
        return "fail", output
    except subprocess.TimeoutExpired:
        return "timeout", f"pytest 超时（{PYTEST_TIMEOUT}s）"
    except Exception as exc:
        return "error", f"pytest 异常：{exc}"


def setup_task_workdir(algo: str) -> Path:
    """为任务创建独立工作目录，复制 buggy.py 和测试文件。"""
    src = TASKS_DIR / algo
    workdir = Path(tempfile.mkdtemp(prefix=f"quixbugs-{algo}-"))
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".py", ".json"):
            shutil.copy(f, workdir / f.name)
    # 兜底 conftest.py：QuixBugs 部分测试（knapsack）引用 pytest.run_slow /
    # use_correct，原仓库由根目录 conftest 的 pytest_configure 提供，prepare.py
    # 不会复制它，缺了直接 AttributeError——修对也判 fail。
    conftest = workdir / "conftest.py"
    if not conftest.exists():
        conftest.write_text(
            "import pytest\n\n\n"
            "def pytest_configure(config):\n"
            "    pytest.use_correct = False\n"
            "    pytest.run_slow = False\n",
            encoding="utf-8",
        )
    return workdir


def run_single(algo: str, *, dry_run: bool = False, log_dir: Path | None = None) -> dict:
    """跑一个任务。返回结果 dict。log_dir 非空时把完整 agent/pytest 输出落盘
    （json 里只存尾部摘要——之前只存截断文本，超时题的现场完全丢失）。"""
    workdir = setup_task_workdir(algo)
    result: dict = {
        "algo": algo,
        "workdir": str(workdir),
        "status": "pending",
        "pytest_status": None,
        "agent_output": "",
        "duration": 0,
    }
    if dry_run:
        result["status"] = "dry-run"
        return result

    def _write_log(kind: str, text: str) -> None:
        if log_dir is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{algo}.{kind}.log"
        path.write_text(text, encoding="utf-8")
        result[f"{kind}_log"] = str(path)

    start = time.time()
    try:
        # 用 subprocess 调 agent.py，工作目录设为任务目录
        # --auto-approve 让 guard 自动通过（benchmark 模式）
        # prompt 针对实测踩坑设计：幻觉 cd 路径、不读代码先跑 pytest 白耗轮次
        request = (
            f"当前工作目录就是任务目录，用相对路径操作文件即可，不要 cd 到任何其他路径。"
            f"buggy.py 是算法 {algo} 的带 bug 实现，test_{algo}.py 是对应测试。"
            f"请严格按顺序执行："
            f"1) 用 read_file 读 buggy.py 和 test_{algo}.py；"
            f"2) 定位 bug，用 plan_patch 对 buggy.py 做局部修复；"
            f"3) 不要自己跑 pytest——改动通过审批后系统会自动跑测试并把结果反馈给你。"
            f"不要修改测试文件。"
        )
        proc = subprocess.run(
            [
                sys.executable, str(AGENT_PY),
                request,
                "--auto-approve",
            ],
            capture_output=True, text=True, timeout=AGENT_TIMEOUT,
            cwd=workdir,
        )
        result["agent_output"] = proc.stdout[-2000:]  # 摘要保留尾部
        _write_log("agent", proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""))
        result["status"] = "agent-done"
    except subprocess.TimeoutExpired as exc:
        result["status"] = "agent-timeout"
        result["agent_output"] = f"agent 超时（{AGENT_TIMEOUT}s）"
        # TimeoutExpired 带部分输出——超时题的现场正是分析盲区，必须落盘
        partial = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        _write_log("agent", partial or "（超时前无任何输出）")
    except Exception as exc:
        result["status"] = "agent-error"
        result["agent_output"] = str(exc)

    result["duration"] = round(time.time() - start, 1)

    # 跑 pytest 判分
    pytest_status, pytest_output = run_pytest(workdir)
    result["pytest_status"] = pytest_status
    result["pytest_output"] = pytest_output[-1000:]
    _write_log("pytest", pytest_output)

    if pytest_status == "pass":
        result["status"] = "pass"
    elif result["status"] == "agent-done":
        result["status"] = "fail"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="QuixBugs benchmark runner")
    parser.add_argument("--algo", help="只跑指定算法")
    parser.add_argument("--dry-run", action="store_true", help="只列出任务不执行")
    parser.add_argument("--output", default=None, help="结果输出路径")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数（默认 1 串行）")
    args = parser.parse_args()

    tasks = [args.algo] if args.algo else list_tasks()
    if not tasks:
        print("无任务可跑")
        return

    print(f"QuixBugs benchmark：{len(tasks)} 个任务（workers={args.workers}）")
    # 完整输出落盘目录：与结果 json 同名（run-<ts>.json ↔ run-<ts>/）
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"run-{int(time.time())}.json"
    log_dir = RESULTS_DIR / out_path.stem
    results: list[dict] = []
    if args.workers <= 1 or args.dry_run:
        for i, algo in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {algo} ... ", end="", flush=True)
            r = run_single(algo, dry_run=args.dry_run, log_dir=log_dir)
            results.append(r)
            print(f"{r['status']} (pytest={r['pytest_status']}, {r['duration']}s)")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_single, algo, log_dir=log_dir): algo for algo in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done += 1
                print(f"[{done}/{len(tasks)}] {r['algo']} ... {r['status']} (pytest={r['pytest_status']}, {r['duration']}s)")
        # 按任务名排序，保持输出稳定
        results.sort(key=lambda r: r["algo"])

    # 汇总
    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n结果：{passed}/{len(results)} 通过 ({100*passed/len(results):.1f}%)")

    # 保存详细结果（out_path 在跑题前已定，与 log_dir 同名）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"详细结果已保存：{out_path}")
    if log_dir.exists():
        print(f"完整输出日志：{log_dir}/<algo>.agent.log / .pytest.log")


if __name__ == "__main__":
    main()
