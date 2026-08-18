# TODO（Bluecode 小蓝 待办与计划）

> 已交付版本的变更记录见 [CHANGELOG.md](CHANGELOG.md)。本文档收集 roadmap 未实现项与已记录的欠账。

## backlog（暂缓）

| 项 | 说明 | 依赖 / 备注 |
|---|---|---|
| token 级流式输出 + Ctrl-C 打断 | `astream_events` 逐 token 流式；当前是节点级播报 | Web 协议已预留 `delta` 事件（design-web.md §4.2/§14），落地即升级 |
| prompts 中英双语化 | `prompts.py` 全部系统提示词双语化 | — |
| BugsInPy 扩量 | 从 pilot（httpie/2 自验+E2E）扩题 | CURATED 加题 + `--self-test` 过门 |
| benchmark 结果查看器 | 读 `results/run-*.json` 做结果展示 | 依赖 v0.9 数据 |
| Web 审批键盘快捷键 | `Y` 全批 / `N` 全拒 / `1-9` 切勾选——CLI 肌肉记忆的网页版（design-web §6.4） | 当前仅 ⌘/Ctrl+Enter 提交 |
| Web 命令面板 | `/` 唤起命令面板（design-web §6.4/§15） | — |

## 已关闭（明确不做）

- **LangGraph Studio**：v0.8 图小地图 + 事件流满足约 80% 需求（design-web.md §14），不再单独集成。
