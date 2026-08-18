#!/usr/bin/env python3
"""把 BugsInPy 的已验证 bug 提取为独立任务目录（v0.8.1 pilot）。

任务目录结构：
  tasks/<proj>/<bug>/
    meta.json         # 任务元信息（repo/buggy/fixed commit、test_file、python 版本等）
    requirements.txt  # 依赖（已做「坏钉放宽」，见下）
    run_test.sh       # 失败测试表达式（agent prompt 与判分收敛用）

用法：python3 benchmarks/bugsinpy/prepare.py
（依赖 BugsInPy 元仓库已 clone 到 /tmp/bugsinpy-src）

⚠ 探针实测（2026-08）：BugsInPy 元仓库元数据存在漂移——bug.info 的 test_file 与
fail.txt / run_test.sh 互相错位；requirements 钉的是 2019-2020 年代版本，其中
cffi==1.14.0 / brotlipy==0.7.0 在 arm64 macOS 新 SDK 上编译失败（ffi_prep_closure
被移除）。因此：
1. 提取时对 requirements 做「坏钉放宽」：剔除 cffi== / brotlipy== 精确钉（让 pip 解析
   新版），其余保持原钉；放宽内容记入 meta.json 的 deps_relaxed 字段。
2. 入库的题必须通过 run_benchmark.py --self-test（buggy 有 F2P、fixed 全过、能检测
   回归）。CURATED 只放自验通过的题——加题先跑 self-test 过门。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
SRC = Path("/tmp/bugsinpy-src")  # BugsInPy 元仓库 clone 位置

# 已通过探针手工验证的题（buggy 有真实 F2P、fixed 全过、依赖可装）。
# 加新题：先手工/自验确认，再在此登记。
CURATED: list[tuple[str, int]] = [
    ("httpie", 2),  # --max-redirects 被忽略（client.py 缺一行透传），E2E 已跑通
]

# arm64 macOS 新 SDK / 新 Python 上编译失败的历史坏钉（剔除让 pip 解析新版）
BROKEN_PINS = ("cffi==", "brotlipy==")


def _read_bug_info(bug_dir: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in (bug_dir / "bug.info").read_text(encoding="utf-8").splitlines():
        key, _, val = line.partition("=")
        info[key.strip()] = val.strip().strip('"')
    return info


def _relax_requirements(src: Path, dst: Path) -> list[str]:
    """复制 requirements 并剔除坏钉，返回被剔除的钉（记录到 meta 供审计）。"""
    lines = src.read_text(encoding="utf-8").splitlines()
    dropped: list[str] = []
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(stripped.startswith(p) for p in BROKEN_PINS):
            dropped.append(stripped)
            continue
        kept.append(line)
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dropped


def extract_bug(proj: str, bug: int) -> dict:
    bug_dir = SRC / "projects" / proj / "bugs" / str(bug)
    if not bug_dir.is_dir():
        return {"proj": proj, "bug": bug, "status": "missing_bug_dir"}
    info = _read_bug_info(bug_dir)
    task_dir = TASKS / proj / str(bug)
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. 依赖（坏钉放宽）
    src_req = bug_dir / "requirements.txt"
    dropped = []
    if src_req.exists():
        dropped = _relax_requirements(src_req, task_dir / "requirements.txt")

    # 2. run_test.sh（失败测试表达式）
    run_sh = bug_dir / "run_test.sh"
    if run_sh.exists():
        shutil.copy(run_sh, task_dir / "run_test.sh")

    # 3. meta.json
    project_info: dict[str, str] = {}
    pinfo = SRC / "projects" / proj / "project.info"
    if pinfo.exists():
        for line in pinfo.read_text(encoding="utf-8").splitlines():
            key, _, val = line.partition("=")
            project_info[key.strip()] = val.strip().strip('"')
    meta = {
        "project": proj,
        "bug_id": bug,
        "repo_url": project_info.get("github_url", ""),
        "buggy_commit": info.get("buggy_commit_id", ""),
        "fixed_commit": info.get("fixed_commit_id", ""),
        "test_file": info.get("test_file", ""),
        "run_test": (run_sh.read_text().strip() if run_sh.exists() else ""),
        "python_version": info.get("python_version", ""),
        "requirements": "requirements.txt",
        "deps_relaxed": dropped,
    }
    (task_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"proj": proj, "bug": bug, "status": "ok", "meta": meta}


def main() -> None:
    if not SRC.exists():
        print(f"错误：BugsInPy 元仓库不在 {SRC}，请先 git clone https://github.com/soarsmu/BugsInPy.git")
        return
    results = [extract_bug(p, b) for p, b in CURATED]
    ok = [r for r in results if r["status"] == "ok"]
    print(f"提取完成：{len(ok)}/{len(CURATED)} 个任务就绪")
    for r in results:
        if r["status"] != "ok":
            print(f"  跳过 {r['proj']}/{r['bug']}: {r['status']}")
        else:
            meta = r["meta"]
            note = f"（坏钉放宽: {', '.join(meta['deps_relaxed'])}）" if meta["deps_relaxed"] else ""
            print(f"  {meta['project']}/{meta['bug_id']}: "
                  f"test={meta['test_file']} py={meta['python_version']} {note}")


if __name__ == "__main__":
    main()
