#!/usr/bin/env python3
"""把 QuixBugs 的 40 个算法题（31 个简单题 + 9 个图/链表题）提取为独立任务目录。

每个任务目录结构：
  tasks/<name>/
    buggy.py        # 带 bug 的实现（agent 要修的文件）
    test_<name>.py  # pytest 测试（验证修复）
    testdata.json   # 原始测试数据（供参考）
    meta.json       # 任务元信息（名称、难度标签、正确实现位置）

用法：python benchmarks/quixbugs/prepare.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC = Path("/tmp/quixbugs-src")
TASKS = ROOT / "tasks"

# 31 个有 json 测试数据的简单算法题（不含图算法）
SIMPLE_ALGOS = [
    "bitcount", "bucketsort", "find_first_in_sorted", "find_in_sorted",
    "flatten", "gcd", "get_factors", "hanoi", "is_valid_parenthesization",
    "kheapsort", "knapsack", "kth", "lcs_length", "levenshtein", "lis",
    "longest_common_subsequence", "max_sublist_sum", "mergesort",
    "next_palindrome", "next_permutation", "pascal", "possible_change",
    "powerset", "quicksort", "rpn_eval", "shunting_yard", "sieve",
    "sqrt", "subsequences", "to_base", "wrap",
]

# 9 个依赖 node.py（Node 类）的图/链表算法题；测试在文件内直接构造图，无 json 数据
GRAPH_ALGOS = [
    "breadth_first_search", "depth_first_search", "detect_cycle",
    "minimum_spanning_tree", "reverse_linked_list", "shortest_path_length",
    "shortest_path_lengths", "shortest_paths", "topological_ordering",
]


def extract_task(algo: str) -> dict:
    """提取一个任务到独立目录。返回 meta。"""
    task_dir = TASKS / algo
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. buggy 实现
    buggy_src = SRC / "python_programs" / f"{algo}.py"
    if not buggy_src.exists():
        return {"name": algo, "status": "missing_buggy"}
    shutil.copy(buggy_src, task_dir / "buggy.py")
    # 图/链表题额外依赖公共 Node 类定义（测试文件里 from node import Node）
    if algo in GRAPH_ALGOS:
        shutil.copy(SRC / "python_programs" / "node.py", task_dir / "node.py")

    # 2. 测试文件：改写 import 路径，让测试直接 import buggy
    test_src = SRC / "python_testcases" / f"test_{algo}.py"
    if not test_src.exists():
        return {"name": algo, "status": "missing_test"}
    test_content = test_src.read_text()
    # 通用处理：去掉 if/else import 块，统一改为 from buggy import X
    lines = test_content.split("\n")
    out_lines: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if "if pytest.use_correct:" in stripped:
            skip = True
            continue
        if skip:
            # 跳过 correct import 和 else: 行
            if stripped.startswith("from correct_python_programs") or stripped == "else:":
                continue
            # 遇到非 import 行结束 skip
            if not stripped.startswith("from python_programs"):
                skip = False
                out_lines.append(line)
            else:
                # 把 buggy import 提到顶层
                out_lines.append(f"from buggy import {algo}")
            continue
        if not skip and f"from python_programs.{algo} import {algo}" in line:
            out_lines.append(f"from buggy import {algo}")
            continue
        out_lines.append(line)
    test_content = "\n".join(out_lines)
    # load_testdata 路径适配：测试数据放在同目录 testdata.json
    test_content = test_content.replace(
        f'load_json_testcases({algo}.__name__)',
        f'load_json_testcases("{algo}")',
    )
    (task_dir / f"test_{algo}.py").write_text(test_content)

    # 3. 测试数据（按算法名命名，与 load_testdata 的查找约定一致）
    json_src = SRC / "json_testcases" / f"{algo}.json"
    if json_src.exists():
        shutil.copy(json_src, task_dir / f"{algo}.json")

    # 4. load_testdata 辅助模块（路径改为同目录 testdata.json）
    load_src = SRC / "python_testcases" / "load_testdata.py"
    load_content = load_src.read_text()
    load_content = load_content.replace(
        'quixbugs_root = Path(__file__).parent / ".."',
        'quixbugs_root = Path(__file__).parent',
    )
    load_content = load_content.replace(
        'f"json_testcases/{algorithm}.json"',
        'f"{algorithm}.json"',
    )
    (task_dir / "load_testdata.py").write_text(load_content)

    # 5. meta
    correct_src = SRC / "correct_python_programs" / f"{algo}.py"
    meta = {
        "name": algo,
        "status": "ok",
        "has_correct": correct_src.exists(),
        "test_file": f"test_{algo}.py",
        "buggy_file": "buggy.py",
    }
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> None:
    if not SRC.exists():
        print(f"错误：QuixBugs 源码不在 {SRC}，请先 git clone")
        return
    algos = SIMPLE_ALGOS + GRAPH_ALGOS
    results = [extract_task(a) for a in algos]
    ok = [r for r in results if r["status"] == "ok"]
    print(f"提取完成：{len(ok)}/{len(algos)} 个任务就绪")
    for r in results:
        if r["status"] != "ok":
            print(f"  跳过 {r['name']}: {r['status']}")


if __name__ == "__main__":
    main()
