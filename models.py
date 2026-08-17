"""小蓝 Blue —— 多模型注册表 + 激活模型管理（v0.8）。

`~/.blue/models.toml` 定义多个可用模型（name → model/base_url/api_key/
context_window/note），`/model` 命令在会话中切换激活模型。纯配置逻辑，
不依赖 agent（避免循环导入）；agent 经 `_base_kwargs()` / `set_active_model()`
等包装接入（测试 patch "agent.X" 契约不变）。

激活优先级（高 → 低）：
  1. 运行时 override（/model 切换，进程内内存态，重启回落）
  2. 环境变量 MODEL_NAME（命中注册表时用注册表配置；未命中回落裸 env 旧行为）
  3. models.toml [active].name
  4. 注册表第一个模型
  5. 缺省 "gpt-4o-mini"（纯 env 旧行为）

安全：api_key 可选存注册表（权限 600 由用户自持），缺省回落环境变量
OPENAI_API_KEY；base_url 缺省回落 OPENAI_BASE_URL。TOML 解析失败回落空
注册表（fail-closed，行为等同未配置多模型）。
"""

from __future__ import annotations

import os
import tomllib

MODELS_PATH = os.path.join(os.path.expanduser("~/.blue"), "models.toml")
DEFAULT_CONTEXT_WINDOW = 128_000  # 注册表未配 context_window 时的缺省窗口

# 运行时激活 override（/model 切换写入；进程内有效，不落盘）
_ACTIVE_OVERRIDE: str | None = None

_config_warned: set[str] = set()  # 同一配置问题只警告一次，防每轮刷屏


def load_models() -> dict[str, dict]:
    """读注册表返回 {name: {model, base_url, api_key, context_window, note}}。
    文件缺失 → {}；TOML 语法错误 → 警告一次并回落 {}（fail-closed）。"""
    try:
        with open(MODELS_PATH, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 — 配置解析失败绝不阻断主流程
        if MODELS_PATH not in _config_warned:
            _config_warned.add(MODELS_PATH)
            print(f"[蓝] ⚠ 模型配置 {MODELS_PATH} 解析失败（{exc}），多模型注册表回落为空")
        return {}
    raw = data.get("models")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        entry: dict = {"model": "", "base_url": "", "api_key": "",
                       "context_window": 0, "note": ""}
        entry.update({k: str(cfg[k]) for k in ("model", "base_url", "api_key", "note") if cfg.get(k)})
        cw = cfg.get("context_window")
        try:
            entry["context_window"] = int(cw) if cw else 0
        except (TypeError, ValueError):
            entry["context_window"] = 0
        out[str(name)] = entry
    return out


def _active_section_name() -> str:
    """读 models.toml [active].name（默认激活的模型名，可被 env/override 覆盖）。"""
    try:
        with open(MODELS_PATH, "rb") as f:
            data = tomllib.load(f)
    except Exception:  # noqa: BLE001 — 缺失/损坏均回落
        return ""
    active = data.get("active")
    if isinstance(active, dict) and active.get("name"):
        return str(active["name"]).strip()
    return ""


def active_model_name() -> str:
    """当前激活模型名（按激活优先级解析，见模块 docstring）。"""
    if _ACTIVE_OVERRIDE:
        return _ACTIVE_OVERRIDE
    env_name = os.environ.get("MODEL_NAME", "").strip()
    models = load_models()
    if env_name:
        return env_name  # env 命中注册表用注册表配置（model_kwargs 处理），未命中回落裸 env
    cfg_name = _active_section_name()
    if cfg_name in models:
        return cfg_name
    if models:
        return next(iter(models))
    return "gpt-4o-mini"


def set_active_model(name: str) -> tuple[bool, str]:
    """设置运行时激活模型。name 必须在注册表中（或等于 env MODEL_NAME）。
    返回 (是否成功, 提示文本)。切换后调用方（agent）须清模型缓存。"""
    global _ACTIVE_OVERRIDE
    name = (name or "").strip()
    models = load_models()
    env_name = os.environ.get("MODEL_NAME", "").strip()
    if name and (name in models or name == env_name):
        _ACTIVE_OVERRIDE = name
        return True, name
    if not models:
        return False, f"未配置多模型注册表（{MODELS_PATH} 不存在），无法切换。"
    return False, f"模型 {name!r} 不在注册表，可用：{', '.join(models)}"


def clear_active_override() -> None:
    """清空运行时 override（回落 env/注册表默认）。"""
    global _ACTIVE_OVERRIDE
    _ACTIVE_OVERRIDE = None


def list_models() -> list[dict]:
    """有序模型列表（供 /model 展示）：注册表条目 + env 当前模型（未注册时补在末尾）。"""
    models = load_models()
    out = [
        {
            "name": n,
            "model": cfg["model"] or n,
            "base_url": cfg["base_url"],
            "context_window": cfg["context_window"] or DEFAULT_CONTEXT_WINDOW,
            "note": cfg["note"],
        }
        for n, cfg in models.items()
    ]
    env_name = os.environ.get("MODEL_NAME", "").strip()
    if env_name and env_name not in models:
        out.append({
            "name": env_name,
            "model": env_name,
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "context_window": DEFAULT_CONTEXT_WINDOW,
            "note": "环境变量 MODEL_NAME（未注册）",
        })
    return out


def model_kwargs(name: str | None = None) -> dict:
    """构造 ChatOpenAI 的 kwargs。注册表命中 → 用注册表配置（缺省回落 env）；
    未注册 → 纯 env 旧行为。api_key/base_url 缺省均回落环境变量。"""
    name = name or active_model_name()
    cfg = load_models().get(name)
    kwargs: dict = {"model": os.environ.get("MODEL_NAME", "gpt-4o-mini")}
    if cfg:
        kwargs["model"] = cfg["model"] or name
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        elif os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        if cfg.get("api_key"):
            kwargs["api_key"] = cfg["api_key"]
        elif os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        if os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    return kwargs


def context_window(name: str | None = None) -> int:
    """激活（或指定）模型的上下文窗口大小；未配或缺省回落 DEFAULT_CONTEXT_WINDOW。"""
    name = name or active_model_name()
    cfg = load_models().get(name)
    if cfg and cfg.get("context_window"):
        return cfg["context_window"]
    return DEFAULT_CONTEXT_WINDOW
