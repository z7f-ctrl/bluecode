#!/usr/bin/env python3
"""BugsInPy benchmark runner（v0.8.1 pilot，F2P/P2P 双判据）。

流程（每题）：
1. clone 项目仓库 → checkout buggy commit；
2. 建立每题独立 venv（pyenv 提供旧版解释器）+ 安装放宽后的依赖（~/.blue/bip-envs 缓存）；
3. **基线轮**：在 buggy 版跑 test_file 采集逐用例结果；
4. **fixed 对照轮**：checkout fixed commit 跑同一 test_file——F2P = 「buggy 失败 ∧
   fixed 通过」的用例（探针实测：BugsInPy 元数据漂移 + 依赖放宽会造成与 bug 无关的
   环境坏测试在 buggy/fixed 都失败，纯基线采集会误收进 F2P，故必须用 fixed 收敛）；
5. checkout 回 buggy，以 --auto-approve 子进程启动 agent 修复（只读+改源码，不跑测试）；
6. **判分轮**：再跑 test_file——精确 F2P 全过且无 P2P 回归即 pass（整体 pytest 可能
   仍有环境坏测试失败，不影响判定）。

用法：
    python3 run_benchmark.py --dry-run           # 列出任务
    python3 run_benchmark.py --self-test         # 环境+判据自验（建 venv、buggy/fixed 对照），不跑 agent
    python3 run_benchmark.py --bug httpie/2      # 单题
    python3 run_benchmark.py                     # 全量（pilot 收录的题）
    python3 run_benchmark.py --workers 2

结果：results/run-<timestamp>.json（status + f2p/p2p 用例细分）；
完整输出在 results/run-<timestamp>/ 逐题日志。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _judge import run_pytest, fmt_cases, print_case_summary

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT_PY = ROOT.parents[1] / "agent.py"
AGENT_TIMEOUT = 300  # 单题 agent 上限（秒）：真实仓库比 QuixBugs 慢，放宽
PYTEST_TIMEOUT = 60  # pytest 上限（秒）：真实测试套件更慢
ENV_CACHE = Path.home() / ".blue" / "bip-envs"  # 每题 venv 缓存（依赖只装一次）
PYENV_ROOT = Path.home() / ".pyenv" / "versions"


def list_tasks() -> list[tuple[str, int]]:
    """返回 tasks/ 下所有任务（有 meta.json 的目录），形如 (proj, bug)。"""
    out = []
    for proj_dir in sorted(TASKS_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        for bug_dir in sorted(proj_dir.iterdir(), key=lambda p: int(p.name)):
            if (bug_dir / "meta.json").exists():
                out.append((proj_dir.name, int(bug_dir.name)))
    return out


def _read_meta(proj: str, bug: int) -> dict:
    return json.loads((TASKS_DIR / proj / str(bug) / "meta.json").read_text(encoding="utf-8"))


def resolve_python(python_version: str) -> str:
    """把 meta 里的 python_version（如 3.7.3）解析为 pyenv 已装解释器。

    优先同 major.minor；否则选 ≥ 目标的最小已装 minor；都没有回落当前解释器。
    （探针：3.7.3 → 装到的 3.8.16 可用。）
    """
    try:
        major, minor = (int(x) for x in python_version.split(".")[:2])
    except (ValueError, AttributeError):
        return sys.executable
    if not PYENV_ROOT.is_dir():
        return sys.executable
    installed = sorted(
        (d.name for d in PYENV_ROOT.iterdir()
         if d.is_dir() and (d / "bin" / "python").exists()), key=str)
    if not installed:
        return sys.executable
    same_minor = [v for v in installed if v.startswith(f"{major}.{minor}")]
    if same_minor:
        return str(PYENV_ROOT / sorted(same_minor)[-1] / "bin" / "python")
    newer = [v for v in installed
             if len(v.split(".")) >= 2 and int(v.split(".")[0]) == major
             and int(v.split(".")[1]) >= minor]
    if newer:
        return str(PYENV_ROOT / sorted(newer, key=lambda v: int(v.split(".")[1]))[0] / "bin" / "python")
    return sys.executable


def ensure_venv(proj: str, bug: int, meta: dict) -> Path:
    """建每题 venv 并装依赖（缓存在 ~/.blue/bip-envs/<proj>-<bug>）。"""
    venv_dir = ENV_CACHE / f"{proj}-{bug}"
    py_bin = venv_dir / "bin" / "python"
    if (venv_dir / ".ok").exists():
        return venv_dir
    base_python = resolve_python(meta.get("python_version", ""))
    if not venv_dir.exists():
        subprocess.run([base_python, "-m", "venv", str(venv_dir)], check=True,
                       capture_output=True, text=True)
    req = TASKS_DIR / proj / str(bug) / meta.get("requirements", "requirements.txt")
    if req.exists():
        proc = subprocess.run(
            [str(venv_dir / "bin" / "pip"), "install", "--disable-pip-version-check",
             "-r", str(req)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{proj}/{bug} 依赖安装失败：{proc.stdout[-800:] + proc.stderr[-800:]}")
    (venv_dir / ".ok").write_text("ok", encoding="utf-8")
    return venv_dir


def _git(repo_dir: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo_dir), *args], check=True,
                   capture_output=True, text=True)


def setup_repo(meta: dict) -> Path:
    """clone 项目到临时 workdir 并 checkout 到 buggy commit，返回 repo 目录。"""
    workdir = Path(tempfile.mkdtemp(prefix=f"bip-{meta['project']}-{meta['bug_id']}-"))
    repo = workdir / "repo"
    subprocess.run(["git", "clone", "--quiet", meta["repo_url"], str(repo)], check=True,
                   capture_output=True, text=True)
    _git(repo, "checkout", "--quiet", meta["buggy_commit"])
    return repo


def run_single(proj: str, bug: int, *, dry_run: bool = False,
               log_dir: Path | None = None, run_agent: bool = True) -> dict:
    meta = _read_meta(proj, bug)
    test_file = meta["test_file"]
    test_expr = meta.get("run_test", "")  # 失败测试表达式（agent prompt 用）

    request = (
        f"当前目录是 {meta['project']} 项目的一个 bug 版本（BugsInPy 任务 {meta['project']}/{meta['bug_id']}）。"
        f"有一个 bug 导致测试 {test_expr or test_file} 失败。"
        "请找出并修复这个 bug。要求：只改项目源码（不要改 tests/ 目录、不要改测试文件）；"
        "不要自己运行 pytest（判分由外部执行）；先 read_file 读懂代码再用 plan_patch 做最小修改。"
    )
    if dry_run:
        return {"proj": proj, "bug": bug, "request": request, "meta": meta}

    result = {
        "proj": proj, "bug": bug, "status": "pending",
        "pytest_status": None, "baseline_pytest_status": None,
        "f2p": None, "p2p": None, "agent_output": "", "duration": 0.0,
    }
    start = time.time()
    repo = setup_repo(meta)
    venv = ensure_venv(proj, bug, meta)
    py_bin = str(venv / "bin" / "python")

    def _write_log(kind: str, text: str) -> None:
        if log_dir is None:
            return
        (log_dir / f"{proj}-{bug}.{kind}.log").write_text(text, encoding="utf-8")
        result[f"{kind}_log"] = str(log_dir / f"{proj}-{bug}.{kind}.log")

    try:
        # ── 基线轮：buggy 跑 test_file ──
        base_status, base_output, base_cases = run_pytest(
            repo, test_file, py_bin, "baseline.xml", timeout=PYTEST_TIMEOUT)
        _write_log("baseline", base_output)
        result["baseline_pytest_status"] = base_status

        # ── fixed 对照轮：精确 F2P = buggy 失败 ∧ fixed 通过（排除环境坏测试）──
        _git(repo, "checkout", "--quiet", meta["fixed_commit"])
        _, fix_output, fixed_cases = run_pytest(
            repo, test_file, py_bin, "fixed.xml", timeout=PYTEST_TIMEOUT)
        _write_log("fixed", fix_output)
        _git(repo, "checkout", "--quiet", meta["buggy_commit"])

        # ── agent 修复 ──
        if run_agent:
            try:
                proc = subprocess.run(
                    [sys.executable, str(AGENT_PY), request, "--auto-approve"],
                    cwd=repo, capture_output=True, text=True, timeout=AGENT_TIMEOUT,
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
        else:
            result["status"] = "agent-skipped"
            full_output = "[--self-test 模式，不跑 agent]"
            _write_log("agent", full_output)

        result["duration"] = round(time.time() - start, 1)

        # ── 判分轮 ──
        pytest_status, pytest_output, final_cases = run_pytest(
            repo, test_file, py_bin, "final.xml", timeout=PYTEST_TIMEOUT)
        result["pytest_status"] = pytest_status
        _write_log("pytest", pytest_output)

        # 精确 F2P：buggy 失败 ∧ fixed 通过（final 查不到的用例按失败计）
        f2p = sorted(n for n, st in base_cases.items()
                     if st in ("failed", "error") and fixed_cases.get(n) == "passed")
        p2p = sorted(n for n, st in base_cases.items() if st == "passed")
        f2p_failed = [n for n in f2p if final_cases.get(n) != "passed"]
        p2p_broken = [n for n in p2p if final_cases.get(n) != "passed"]
        if f2p or p2p:
            result["f2p"] = {"passed": len(f2p) - len(f2p_failed), "total": len(f2p),
                             "failed": f2p_failed}
            result["p2p"] = {"passed": len(p2p) - len(p2p_broken), "total": len(p2p),
                             "failed": p2p_broken}

        # 判定：以精确 F2P/P2P 为准（整体 pytest 可能因环境坏测试非 pass）
        if result["status"] in ("agent-done", "agent-skipped"):
            if (f2p and not f2p_failed and not p2p_broken):
                result["status"] = "pass"
            elif result["status"] == "agent-skipped":
                result["status"] = "self-test-pass" if f2p and not f2p_failed else "self-test-bad"
            else:
                result["status"] = "fail"
    except Exception as e:  # noqa: BLE001
        result["status"] = "runner-error"
        result["agent_output"] = f"[runner 异常] {e}"
        _write_log("agent", f"[runner 异常] {e}")
    finally:
        shutil.rmtree(repo.parent, ignore_errors=True)  # 用完即弃（v0.8.1 pilot）
    return result


def self_test() -> int:
    """环境+判据自验：每题建 venv + buggy/fixed 对照，验证「有 F2P、fixed 全过、可判回归」。

    不启动 agent。是加题入库的门槛（探针实测元数据漂移，只信自验过的题）。
    """
    bad = 0
    for proj, bug in list_tasks():
        meta = _read_meta(proj, bug)
        print(f"== {proj}/{bug}: 建 venv + buggy/fixed 对照 ==", flush=True)
        try:
            repo = setup_repo(meta)
            venv = ensure_venv(proj, bug, meta)
            py_bin = str(venv / "bin" / "python")
            base_status, _, base_cases = run_pytest(
                repo, meta["test_file"], py_bin, "baseline.xml", timeout=PYTEST_TIMEOUT)
            _git(repo, "checkout", "--quiet", meta["fixed_commit"])
            fix_status, fix_output, fixed_cases = run_pytest(
                repo, meta["test_file"], py_bin, "fixed.xml", timeout=PYTEST_TIMEOUT)
            f2p = [n for n, st in base_cases.items()
                   if st in ("failed", "error") and fixed_cases.get(n) == "passed"]
            p2p = [n for n, st in base_cases.items() if st == "passed"]
            env_broken = [n for n, st in base_cases.items()
                          if st in ("failed", "error") and fixed_cases.get(n) != "passed"]
            # 门槛：有可修可验的 F2P + 可检回归的 P2P。环境坏测试（buggy/fixed 都失败）
            # 不算缺陷——精确 F2P 已把它们排除，这里仅报告。
            ok = bool(f2p) and bool(p2p)
            print(f"  buggy={base_status} fixed={fix_status} F2P={len(f2p)} "
                  f"P2P={len(p2p)} 环境坏测试={env_broken or '无'} → {'OK' if ok else 'BAD'}")
            if not ok:
                print(fix_output[-800:])
                bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"  BAD — {e}")
            bad += 1
    print("SELF-TEST " + ("PASSED ✅" if bad == 0 else f"FAILED（{bad} 项不符）"))
    return 0 if bad == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="BugsInPy benchmark runner（F2P/P2P 双判据）")
    p.add_argument("--bug", help="只跑指定题（如 httpie/2）")
    p.add_argument("--dry-run", action="store_true", help="只列出任务，不执行")
    p.add_argument("--self-test", action="store_true", help="环境+判据自验（不跑 agent）")
    p.add_argument("--output", help="结果 json 输出路径（默认 results/run-<timestamp>.json）")
    p.add_argument("--workers", type=int, default=1, help="并行任务数（默认 1 串行；注意 LLM 限流）")
    args = p.parse_args()

    if args.self_test:
        sys.exit(self_test())

    tasks = [tuple(args.bug.split("/"))] if args.bug else list_tasks()
    if not tasks:
        print("没有找到任务，先跑 prepare.py"); sys.exit(1)

    if args.dry_run:
        for proj, bug in tasks:
            r = run_single(proj, bug, dry_run=True)
            print(f"- {r['proj']}/{r['bug']}\n  request: {r['request']}")
        print(f"\n共 {len(tasks)} 个任务")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"run-{ts}.json"
    log_dir = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    workers = max(1, args.workers)
    if workers == 1:
        for i, (proj, bug) in enumerate(tasks, 1):
            r = run_single(proj, bug, log_dir=log_dir)
            print(f"[{i}/{len(tasks)}] {r['proj']}/{r['bug']}: {r['status']} "
                  f"(pytest={r['pytest_status']}, {r['duration']}s){fmt_cases(r)}", flush=True)
            results.append(r)
    else:
        print(f"并行 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_single, p, b, log_dir=log_dir): (p, b) for p, b in tasks}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                print(f"[{done}/{len(tasks)}] {r['proj']}/{r['bug']}: {r['status']} "
                      f"(pytest={r['pytest_status']}, {r['duration']}s){fmt_cases(r)}", flush=True)
                results.append(r)
        results.sort(key=lambda r: (r["proj"], r["bug"]))

    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n结果: {passed}/{len(results)} pass")
    print_case_summary(results)

    out_path.write_text(
        json.dumps(
            {"timestamp": ts, "passed": passed, "total": len(results), "results": results},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"写入 {out_path}")
    print(f"日志目录 {log_dir}")


if __name__ == "__main__":
    main()
