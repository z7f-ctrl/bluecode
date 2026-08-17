"""小蓝 Blue —— 自检 + 初始化（blue doctor / blue init，v0.7 阶段二）。

从 agent.py 拆出（#7 模块拆分）。doctor 六项自检在启动时拦下「配错端点/模型名
不存在」的实测坑；init 交互写全局 ~/.blue/.env（key 不回显），写完接 doctor。
渲染工具 _c/_C 来自 cli（模块级 import，加载顺序 agent→cli→doctor 保证无环）；
_make_plain_model / _fetch_model_ids / ENV_GLOBAL_PATH 经函数内 `import agent` 取用
——validate_graph.py 通过 patch("agent.X") 注入假实现，必须走 agent 命名空间解析。
ENV_GLOBAL_PATH 来自 session（目录常量集中处）。
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from langchain_core.messages import HumanMessage

from cli import _C, _c


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 11)  # tomllib（.blue.toml 权限分级）需要 3.11+
    return ok, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else "（需要 ≥3.11）")


def _check_deps() -> list[tuple[bool, str]]:
    out = []
    for mod, required in (("langgraph", True), ("langchain_openai", True),
                          ("langchain_core", True), ("dotenv", True),
                          ("rich", False), ("grandalf", False)):
        try:
            __import__(mod)
            out.append((True, f"{mod} 已安装"))
        except ImportError:
            out.append((not required,
                        f"{mod} 缺失" + ("" if required else "（可选，缺失自动降级，不影响主流程）")))
    return out


def _check_config() -> tuple[bool, str]:
    """配置自检：env 三键为主，多模型注册表（~/.blue/models.toml）可补
    base_url/model；api_key 需 env 或激活条目自带其一。"""
    import agent  # 测试契约：validate_graph patch("agent.X")，必须走 agent 命名空间
    registry = agent.load_models()
    active_cfg = registry.get(agent.current_model_name()) or {}
    missing = []
    if not (os.environ.get("OPENAI_API_KEY", "").strip() or active_cfg.get("api_key")):
        missing.append("OPENAI_API_KEY")
    if not (os.environ.get("OPENAI_BASE_URL", "").strip() or active_cfg.get("base_url")):
        missing.append("OPENAI_BASE_URL")
    if not (os.environ.get("MODEL_NAME", "").strip() or registry):
        missing.append("MODEL_NAME")
    if missing:
        return False, f"配置缺失：{', '.join(missing)}（跑 blue init，或检查 .env / ~/.blue/.env）"
    src = "env + models.toml" if registry else "三项环境变量"
    return True, f"配置齐全（{src}）"


def _check_blue_dir() -> tuple[bool, str]:
    from session import BLUE_DIR
    try:
        os.makedirs(BLUE_DIR, exist_ok=True)
        probe = os.path.join(BLUE_DIR, ".write-test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True, f"{BLUE_DIR} 可写"
    except OSError as exc:
        return False, f"{BLUE_DIR} 不可写：{exc}"


def _fetch_model_ids() -> list[str]:
    """GET {base}/models 返回可用模型 id 列表；异常原样抛出，由调用方整形为诊断文本。
    端点/key 按激活模型解析（models.toml 条目优先，缺省回落 env）。"""
    from models import model_kwargs
    kw = model_kwargs()
    base = str(kw.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")).strip().rstrip("/")
    req = urllib.request.Request(
        base + "/models",
        headers={"Authorization": "Bearer " + str(kw.get("api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return [m.get("id", "") for m in data.get("data", [])]


def _check_api_and_model() -> tuple[bool, str]:
    """API 可达 + key 有效 + 激活模型在可用列表（v1h typo / glm5.1 不存在的实测坑）。
    激活模型经 models 注册表解析（/model 切换或 [active] 配置），无注册表回落 env。"""
    import agent  # 测试契约：validate_graph patch("agent._fetch_model_ids")，必须走 agent 命名空间
    model = agent.current_model_name().strip()
    try:
        # 经 agent 取用：validate_graph.py 等通过 patch("agent._fetch_model_ids")
        # 注入假响应，必须走 agent 命名空间解析（见 #7 拆分说明）。
        ids = agent._fetch_model_ids()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, f"API 可达但认证失败（HTTP {exc.code}）——OPENAI_API_KEY 无效"
        return False, (f"模型列表请求失败（HTTP {exc.code}）——检查 OPENAI_BASE_URL 路径"
                       f"（<base>/models 应可达；末尾多个字母之类 typo 会 404）")
    except Exception as exc:
        return False, f"API 不可达（{type(exc).__name__}: {exc}）——检查 OPENAI_BASE_URL / 网络"
    if model in ids:
        return True, f"API 可达，模型 {model} 在线（共 {len(ids)} 个可用）"
    similar = [i for i in ids if model.split("-")[0].lower() in i.lower()]
    hint = "、".join(similar[:5]) or "、".join(ids[:5])
    return False, f"模型 {model} 不在可用列表，相近：{hint}"


def _check_tool_calling() -> tuple[bool, str]:
    """真实调一次最小 tool calling 请求（几个 token）：模型不支持工具调用则 agent 无法工作。"""
    import agent  # 测试契约：validate_graph patch("agent._make_plain_model")，必须走 agent 命名空间
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def _ping(x: str) -> str:
        """自检探针。"""
        return x

    try:
        resp = agent._make_plain_model().bind_tools([_ping]).invoke(
            [HumanMessage(content='调用 _ping 工具，参数 x="ok"。')])
    except Exception as exc:
        return False, f"tool calling 探测失败（{type(exc).__name__}: {exc}）"
    if getattr(resp, "tool_calls", None):
        return True, "模型正确返回 tool_calls"
    return False, "模型未返回 tool_calls（该模型可能不支持工具调用，agent 无法工作）"


def cmd_doctor() -> int:
    """自检：环境/依赖/配置/数据目录/API 与模型/tool calling。返回进程退出码（0=全过）。"""
    c = _c
    checks: list[tuple[str, bool, str]] = [("Python 版本", *_check_python())]
    checks += [("依赖", ok, msg) for ok, msg in _check_deps()]
    checks.append(("配置", *_check_config()))
    checks.append(("数据目录", *_check_blue_dir()))
    config_ok = checks[-2][1]
    if config_ok:  # 配置齐全才测 API，避免无 key 时的误导性报错
        checks.append(("API 与模型", *_check_api_and_model()))
        if checks[-1][1]:
            checks.append(("tool calling", *_check_tool_calling()))
    failed = 0
    print("[蓝] 🩺 自检：")
    for name, ok, msg in checks:
        mark = c("✓", _C.GREEN) if ok else c("✗", _C.RED)
        print(f"  {mark} {name}：{msg}")
        failed += 0 if ok else 1
    if failed:
        print(c(f"[蓝] {failed} 项未过，先修复再用。", _C.RED))
        return 1
    print(c("[蓝] 全部通过，可以干活。", _C.GREEN))
    return 0


def _write_env_file(path: str, values: dict) -> None:
    """写 .env（权限 600；已存在则先备份 <path>.bak）。空值键跳过。"""
    if os.path.exists(path):
        shutil.copy(path, path + ".bak")
    lines = [f"{k}={v}" for k, v in values.items() if v]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# bluecode 配置（blue init 生成）\n" + "\n".join(lines) + "\n")
    os.chmod(path, 0o600)


def cmd_init() -> int:
    """交互式初始化：写全局 ~/.blue/.env（项目级 .env 可覆盖同名键），写完跑 doctor 验证。"""
    import agent  # ENV_GLOBAL_PATH 经 agent 取用：validate_graph 可 patch("agent.ENV_GLOBAL_PATH")
    c = _c
    print("[蓝] ⚙ 初始化配置（写入全局 ~/.blue/.env；项目根目录的 .env 可覆盖同名键；"
          "显式环境变量优先级最高）")
    env_path = agent.ENV_GLOBAL_PATH
    if os.path.exists(env_path):
        print(f"[蓝] 已存在 {env_path}，继续将覆盖（原文件备份为 .bak）。")
        if input("继续？[y/N] > ").strip().lower() != "y":
            print("[蓝] 已取消。")
            return 1
    cur = {k: os.environ.get(k, "").strip()
           for k in ("OPENAI_BASE_URL", "MODEL_NAME", "OPENAI_API_KEY", "TAVILY_API_KEY")}
    base = input(f"OPENAI_BASE_URL [{cur['OPENAI_BASE_URL'] or 'https://api.openai.com/v1'}] > ").strip() \
        or cur["OPENAI_BASE_URL"] or "https://api.openai.com/v1"
    model = input(f"MODEL_NAME [{cur['MODEL_NAME'] or 'gpt-4o-mini'}] > ").strip() \
        or cur["MODEL_NAME"] or "gpt-4o-mini"
    key_hint = "，回车保留已配置" if cur["OPENAI_API_KEY"] else ""
    key = getpass.getpass(f"OPENAI_API_KEY（不回显{key_hint}）> ").strip() or cur["OPENAI_API_KEY"]
    if not key:
        print(c("[蓝] ✗ OPENAI_API_KEY 必填。", _C.RED))
        return 1
    tavily = getpass.getpass("TAVILY_API_KEY（可选，web_search 用，回车跳过）> ").strip() \
        or cur["TAVILY_API_KEY"]
    _write_env_file(env_path, {
        "OPENAI_BASE_URL": base, "MODEL_NAME": model,
        "OPENAI_API_KEY": key, "TAVILY_API_KEY": tavily,
    })
    print(f"[蓝] 已写入 {env_path}（权限 600）。开始连通性自检…")
    agent.load_dotenv(agent.ENV_GLOBAL_PATH, override=True)  # 让紧随的 doctor 读到新值
    return agent.cmd_doctor()  # 经 agent 取用：validate_graph 可 patch("agent.cmd_doctor")
