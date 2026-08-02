# QuixBugs Benchmark

把 [QuixBugs](https://github.com/jkoppel/QuixBugs) 的 40 个算法修 bug 题接入 bluecode，作为真实任务的回归测试 + 能力评测。

## 用法

```bash
# 1. 准备题库（需要先 clone QuixBugs 到 /tmp/quixbugs-src）
git clone --depth 1 https://github.com/jkoppel/QuixBugs.git /tmp/quixbugs-src
python3 benchmarks/quixbugs/prepare.py          # 生成 40 个任务目录到 tasks/

# 2. 跑基准（需要 .env 配好模型）
python3 benchmarks/quixbugs/run_benchmark.py                    # 全部 40 题
python3 benchmarks/quixbugs/run_benchmark.py --algo bitcount    # 单题
python3 benchmarks/quixbugs/run_benchmark.py --workers 4        # 并行跑题（默认 1 串行）
python3 benchmarks/quixbugs/run_benchmark.py --dry-run          # 只列出任务
```

## 判分方式

- 每题独立 workdir，agent 以 `--auto-approve` 模式修 `buggy.py`（单题 180s 超时）
- 修完后跑 `pytest test_<algo>.py`，带 15s 超时（防死循环 bug）
- 结果：`pass`（全过）/ `fail`（测试失败）/ `agent-timeout` / `agent-error`
- 输出 resolve rate，详细结果存 `results/run-<timestamp>.json`
- workdir 自动补 `conftest.py`（knapsack 等测试引用 `pytest.run_slow`/`use_correct`，原仓库靠根 conftest 提供，缺失会 AttributeError 误判 fail）

## 任务结构

`prepare.py` 生成的每个任务目录：

```
tasks/<algo>/
  buggy.py         # 带 bug 的实现（agent 要修的文件）
  test_<algo>.py   # pytest 测试（import 路径已改写为 from buggy import X）
  <algo>.json      # 参数化测试数据（图算法题无此文件，测试内直接构造）
  node.py          # 仅图/链表题：公共 Node 类定义
  load_testdata.py # 数据加载辅助
  meta.json        # 任务元信息
```

题库 = 31 个有 json 测试数据的简单算法题 + 9 个依赖 `node.py` 的图/链表题（BFS/DFS/detect_cycle/MST/reverse_linked_list/最短路×3/拓扑排序）。

## 注意

- `tasks/` 和 `results/` 不入库（.gitignore 已排除），由 prepare.py 重新生成
- benchmark 需要真实模型 API（.env 里的 OPENAI_API_KEY 等），不是离线测试
- 并行 workers 过高可能触发模型 API 限流、加剧结果波动；数字要做稳可串行或 `--workers 2`
