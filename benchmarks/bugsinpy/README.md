# BugsInPy benchmark（v0.8.1 pilot）

[bugsInPy](https://github.com/soarsmu/BugsInPy) 真实仓库 bug 基准。pilot 收录经过
自验（`--self-test`）的题，每题独立 venv、F2P/P2P 双判据判分。

## 用法

```bash
# 1. 准备 BugsInPy 元仓库（只需元数据，题目代码按 commit 从各项目仓库实时 clone）
git clone --depth 1 https://github.com/soarsmu/BugsInPy.git /tmp/bugsinpy-src

# 2. 提取任务（tasks/ 由 CURATED 清单生成）
python3 benchmarks/bugsinpy/prepare.py

# 3. 环境+判据自验（建 venv + buggy/fixed 对照，不跑 agent）——加题的入库门槛
python3 benchmarks/bugsinpy/run_benchmark.py --self-test

# 4. 跑真实修复
python3 benchmarks/bugsinpy/run_benchmark.py --bug httpie/2   # 单题
python3 benchmarks/bugsinpy/run_benchmark.py                   # 全量
python3 benchmarks/bugsinpy/run_benchmark.py --workers 2       # 并行（注意 LLM 限流）
```

结果与 QuixBugs 同构：`results/run-<ts>.json`（status + f2p/p2p 用例细分），
完整输出落 `results/run-<ts>/` 逐题日志。

## 与 QuixBugs 的差异（探针实测，2026-08）

1. **每题独立 venv**：依赖是各 bug 时代钉死的旧版本（2019–2020），用 `pyenv` 提供
   旧解释器（meta 的 `python_version` → 最近的已装 pyenv 版本），venv 缓存在
   `~/.blue/bip-envs/<proj>-<bug>`（只装一次）。
2. **坏钉放宽**：`prepare.py` 提取时剔除 `cffi==` / `brotlipy==` 精确钉——这两个包在
   arm64 macOS 新 SDK 上编译失败（`ffi_prep_closure` 被移除），删钉让 pip 解析新版。
   放宽内容记入 meta 的 `deps_relaxed`。
3. **精确 F2P**：BugsInPy 元数据漂移 + 依赖放宽会产生「buggy/fixed 都失败」的
   环境坏测试（如 httpie 的 `test_follow_show_redirects`）。因此 F2P 不用纯基线失败集，
   而是 `buggy 失败 ∧ fixed 通过`（用官方 fixed_commit 对照收敛）；环境坏测试仅报告、
   不影响判定。整体 pytest 非 pass 不代表失败——以精确 F2P/P2P 为准。
4. **元数据漂移**：shallow clone 的 `bug.info` / `run_test.sh` / fail.txt 互相错位，
   入库题一律以 `--self-test` 为准（buggy 有 F2P、fixed 全过、可检回归）。

## 收录情况

| 题 | 项目 | bug | 修复点 | 状态 |
|---|---|---|---|---|
| httpie/2 | httpie | 2 | `client.py` 缺 `requests_session.max_redirects = args.max_redirects` 透传 | ✅ 自验通过，E2E 跑通（探针 agent 一次修对 → pass；实跑一次模型波动修错 → 判分器正确判 fail） |

扩量：把新题加进 `prepare.py` 的 `CURATED` 清单 → 重跑 `prepare.py` → 跑 `--self-test`，
通过即入库。依赖重/有编译问题的项目（pandas/matplotlib 等）暂不收录。
