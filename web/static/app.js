/* 小蓝 Blue Web 控制台 —— 前端单页（v0.8 M0–M2）
 * 零构建原生 ES module：无框架 / 无构建步骤 / 无 CDN / 无字体外链，离线可用。
 * 安全红线：模型输出与文件内容一律以 textContent 渲染，绝不进入 innerHTML。
 * 协议对齐 design-web.md §4：REST + 六种 SSE 事件
 * （round_start / node / approval_required / round_end / error / info）。
 */
"use strict";

// ── 常量 ──
const LS_TOKEN_KEY = "blue_web_token";
const EVENT_TYPES = ["round_start", "node", "approval_required", "round_end", "error", "info"];
const NODE_LABELS = {
  planner: "计划", agent: "执行", worker: "并行 worker", guard: "执行结果",
  verifier: "自动验证", reviewer: "评审", report: "交付报告",
};
const ACTION_INFO = {
  plan_write_file: { tag: "write", icon: "📝", label: "write" },
  plan_patch: { tag: "patch", icon: "🔧", label: "patch" },
  plan_run_command: { tag: "command", icon: "⚠️", label: "command" },
  plan_run_python: { tag: "python", icon: "🐍", label: "python" },
};
// 与 tools.py ACTION_CATEGORY 一致（快照重建给裸 change 兜底分类用）
const ACTION_CATEGORY = {
  plan_write_file: "write", plan_patch: "write",
  plan_run_command: "command", plan_run_python: "python",
};
const RISKY_CATEGORIES = new Set(["command", "python"]);
const MAX_RENDER_LINES = 4000; // 大内容渲染行数上限，防卡顿

// ── 全局状态 ──
const state = {
  token: "",
  sessions: [],
  current: null,        // 当前 thread_id
  es: null,             // EventSource
  running: false,       // 有轮次在执行（含等审批期间，图未结束）
  awaiting: false,      // 等待审批
  approvalId: null,
  round: null,
  seenApprovals: new Set(), // 已渲染的 approval_id，防历史回放与快照重建重复出卡
  sessionSeq: 0,        // 会话切换年代，旧 SSE 回调据此作废
  // v0.8.2 观测面板（M3/M4）
  lastNode: null,       // 图小地图最前沿节点
  parallelTasks: 0,     // planner 拆出的并行子任务数（worker chip）
  roundUsage: [],       // 每轮用量 {round, prompt, completion, calls, context}（SSE round_end）
  sessionTotal: null,   // 会话累计 {prompt, completion}（快照提供，重启后仍准）
  contextWindow: 0,     // 激活模型上下文窗口（health）
  models: [],           // 模型注册表（/api/models）
  activeModel: null,
};
let tokenPrompted = false; // 401 时每页只 prompt 一次

// ── 小工具 ──
const $ = (sel) => document.querySelector(sel);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function pre(className, text) {
  const p = el("pre", className);
  p.textContent = text == null ? "" : String(text);
  return p;
}
const num0 = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
function txt(v) {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v); } catch { return String(v); }
}
const safeJson = (v) => { try { return JSON.stringify(v, null, 2); } catch { return String(v); } };
function fmtBytes(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v < 0) return "";
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${(v / 1024 / 1024).toFixed(1)} MB`;
}
function fmtTime(iso) {
  const t = Date.parse(iso || "");
  if (!Number.isFinite(t)) return String(iso || "");
  const diff = Date.now() - t;
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`;
  const d = new Date(t);
  const pad = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ── 观测面板（v0.8.2 M3/M4）：图小地图 + 用量 + 模型 + 环境 ──
const TOPO_NODES = [
  { id: "planner", y: 18 },
  { id: "mid", y: 74 },
  { id: "guard", y: 130 },
  { id: "verifier", y: 186 },
  { id: "reviewer", y: 242 },
  { id: "report", y: 298 },
];
const MM_W = 190, MM_H = 344, MM_X = 30, MM_WID = 130, MM_HGT = 30;

function svgEl(tag, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs ?? {})) n.setAttribute(k, v);
  return n;
}
function buildMinimap() {
  const wrap = $("#minimap");
  wrap.replaceChildren();
  const svg = svgEl("svg", { viewBox: `0 0 ${MM_W} ${MM_H}`, class: "mm" });
  const cx = MM_X + MM_WID / 2;
  for (let i = 0; i < TOPO_NODES.length - 1; i++) {
    const a = TOPO_NODES[i], b = TOPO_NODES[i + 1];
    svg.append(svgEl("line", { x1: cx, y1: a.y + MM_HGT, x2: cx, y2: b.y, class: "mm-edge" }));
  }
  // revise 回边：reviewer 左外侧虚线回到 agent（激活时闪一下）
  const rev = svgEl("path", {
    d: `M ${MM_X} ${TOPO_NODES[4].y + MM_HGT / 2} H 8 V ${TOPO_NODES[1].y + MM_HGT / 2} H ${MM_X}`,
    class: "mm-edge revise",
  });
  rev.dataset.edge = "revise";
  svg.append(rev);
  for (const nd of TOPO_NODES) {
    const g = svgEl("g", { class: "mm-node todo", "data-node": nd.id });
    g.append(svgEl("rect", { x: MM_X, y: nd.y, width: MM_WID, height: MM_HGT, rx: 6 }));
    const t = svgEl("text", { x: cx, y: nd.y + 19, class: "mm-label" });
    t.textContent = nd.id === "mid" ? "agent" : nd.id;
    g.append(t);
    svg.append(g);
  }
  wrap.append(svg);
}
function minimapSet(id, cls) {
  document.querySelectorAll(`#minimap .mm-node[data-node="${id}"]`)
    .forEach((n) => { n.classList.remove("todo", "active", "done"); if (cls) n.classList.add(cls); });
}
function minimapLabel(id, text) {
  const n = document.querySelector(`#minimap .mm-node[data-node="${id}"]`);
  if (n) n.querySelector("text").textContent = text;
}
function resetMinimap() {
  for (const nd of TOPO_NODES) minimapSet(nd.id, "todo");
  minimapLabel("mid", "agent");
  state.parallelTasks = 0;
  state.lastNode = null;
  const rev = document.querySelector(`#minimap path[data-edge="revise"]`);
  if (rev) rev.classList.remove("flash");
}
function flashRevise() {
  const rev = document.querySelector(`#minimap path[data-edge="revise"]`);
  if (!rev) return;
  rev.classList.remove("flash");
  void rev.getBoundingClientRect(); // 重触发 CSS 动画
  rev.classList.add("flash");
}
function markNodeDone(nodeName) {
  if (!nodeName) return;
  if (state.lastNode) minimapSet(state.lastNode, "done");
  minimapSet(nodeName, "active");
  state.lastNode = nodeName;
}
function renderTokenPanel() {
  const body = $("#token-body");
  const rounds = state.roundUsage.slice(-8);
  if (!rounds.length && !state.sessionTotal) {
    body.textContent = "尚未有轮次";
    body.className = "obs-dim";
    return;
  }
  body.className = "";
  const lines = [];
  for (const u of rounds) {
    lines.push(`第 ${u.round} 轮：${num0(u.prompt)} + ${num0(u.completion)} = `
      + `${num0(u.prompt) + num0(u.completion)}（${num0(u.calls)} 次）`);
  }
  if (rounds.length && state.roundUsage.length > rounds.length) {
    lines.push(`…（共 ${state.roundUsage.length} 轮）`);
  }
  let sIn = 0, sOut = 0, peak = 0;
  for (const u of state.roundUsage) {
    sIn += num0(u.prompt); sOut += num0(u.completion);
    peak = Math.max(peak, num0(u.context));
  }
  if (state.sessionTotal) {
    sIn = Math.max(sIn, num0(state.sessionTotal.prompt));
    sOut = Math.max(sOut, num0(state.sessionTotal.completion));
  }
  if (sIn || sOut) lines.push(`会话累计：${sIn} + ${sOut} = ${sIn + sOut}`);
  const win = state.contextWindow || 0;
  if (peak && win) lines.push(`上下文峰值 ${peak} / ${win}（${(peak / win * 100).toFixed(1)}%）`);
  body.textContent = lines.join("\n");
}
function renderEnvPanel(h) {
  const body = $("#env-body");
  if (!h) { body.textContent = "加载中…"; body.className = "obs-dim"; return; }
  body.className = "";
  const lines = [
    `模型 ${h.model}`, `窗口 ${h.context_window ?? "?"}`,
    `key ${h.key_configured ? "已配置" : "未配置"}`, `auth ${h.auth}`,
  ];
  const p = h.permissions;
  if (p) lines.push(`权限 write=${p.write} command=${p.command} python=${p.python}`);
  if (h.base_url) lines.push(`base_url ${h.base_url}`);
  body.textContent = lines.join("\n");
  if (h.context_window) state.contextWindow = h.context_window;
}
async function loadModels() {
  const list = $("#model-list");
  try {
    const r = await api("/api/models");
    state.models = r.models ?? [];
    state.activeModel = r.active;
    list.className = "";
    list.replaceChildren();
    for (const m of state.models) {
      const name = m?.name ?? m?.model ?? "";
      if (!name) continue;
      const row = el("div", "model-row" + (name === state.activeModel ? " active" : ""));
      row.append(el("span", "", name + (name === state.activeModel ? " ✓" : "")));
      if (m?.note) row.append(el("span", "obs-dim", m.note));
      if (name !== state.activeModel) {
        row.onclick = async () => {
          const msg = await setModel(name);
          $("#model-note").textContent = msg;
          loadHealth(); loadModels();
        };
      }
      list.append(row);
    }
    if (!state.models.length) list.textContent = "（未配置模型注册表）";
  } catch (e) {
    list.textContent = `模型列表加载失败：${e.message}`;
  }
}
async function setModel(name) {
  try {
    const r = await api("/api/models", { method: "POST", body: { name } });
    return `✅ ${r.message}（下一轮生效）`;
  } catch (e) {
    return `❌ ${e.message}`;
  }
}
async function openAudit() {
  const modal = $("#audit-modal");
  const body = $("#audit-body");
  modal.classList.remove("hidden");
  body.textContent = "加载中…";
  body.className = "modal-body obs-dim";
  try {
    const r = await api("/api/audit?limit=80");
    const entries = r.entries ?? [];
    body.className = "modal-body";
    if (!entries.length) { body.textContent = "（暂无审计记录）"; return; }
    body.replaceChildren();
    for (const e of entries) {
      const row = el("div", "audit-row");
      const idx = Array.isArray(e.indices) ? ` 批 ${e.indices.map((i) => i + 1).join(",")}` : "";
      const note = e.note ? `「${e.note}」` : "";
      row.append(el("div", "audit-line",
        `${e.ts ?? ""}  ${e.action ?? ""}${idx}${note}  source=${e.source ?? ""}  `
        + `thread=${String(e.thread ?? "").slice(0, 12)}`));
      if (Array.isArray(e.changes) && e.changes.length) {
        const brief = e.changes.map((c) => {
          const tgt = c?.path ?? c?.command ?? "";
          return `${c?.action ?? ""} ${String(tgt).slice(0, 60)}`;
        }).slice(0, 5).join("；");
        row.append(el("div", "audit-changes obs-dim", brief));
      }
      body.append(row);
    }
  } catch (e) {
    body.className = "modal-body";
    body.textContent = `❌ 审计加载失败：${e.message}`;
  }
}

// ── 流区域操作 ──
const nearBottom = () => {
  const s = $("#stream");
  return s.scrollHeight - s.scrollTop - s.clientHeight < 140;
};
function appendCard(node) {
  const s = $("#stream");
  const stick = nearBottom();
  s.append(node);
  if (stick) s.scrollTop = s.scrollHeight;
}
const clearStream = () => { $("#stream").replaceChildren(); state.seenApprovals.clear(); };
const addInfoLine = (text) => appendCard(el("div", "info-line", text));
function divider(text) {
  const d = el("div", "divider");
  d.append(el("span", "line"), el("span", "text", text), el("span", "line"));
  appendCard(d);
}
function requestBubble(round, text) {
  const b = el("div", "req");
  b.append(el("div", "req-meta", `你 · 第 ${round ?? "?"} 轮`), pre("req-text", text));
  appendCard(b);
}

// ── token / fetch 封装 ──
class ApiError extends Error {
  constructor(status, message, code) { super(message); this.status = status; this.code = code; }
}
const loadToken = () => { try { return localStorage.getItem(LS_TOKEN_KEY) || ""; } catch { return ""; } };
function saveToken(t) {
  state.token = t;
  try { localStorage.setItem(LS_TOKEN_KEY, t); } catch { /* 隐私模式忽略 */ }
}
function promptTokenOnce() {
  if (tokenPrompted) return !!state.token;
  tokenPrompted = true;
  const t = window.prompt(
    "服务端要求访问 token（blue web 启动时打印在控制台，或 BLUE_WEB_TOKEN）：", "");
  if (t && t.trim()) {
    saveToken(t.trim());
    // 换 token 后重建事件流（EventSource 无法带 Authorization 头，走 ?token= 查询参数）
    if (state.current) connectEvents(state.current);
    return true;
  }
  return !!state.token;
}
function buildOpts(method, body) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const opts = { method, headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  return opts;
}
async function api(path, { method = "GET", body } = {}) {
  let res = await fetch(path, buildOpts(method, body));
  if (res.status === 401 && promptTokenOnce()) res = await fetch(path, buildOpts(method, body));
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }
  if (!res.ok) {
    // FastAPI 的 detail 可能是字符串或 {code, message} 结构，宽松提取
    let msg = `HTTP ${res.status}`;
    let code;
    if (data && typeof data === "object") {
      const d = data.detail ?? data.error ?? data.message;
      if (typeof d === "string") msg = d;
      else if (d && typeof d === "object") { msg = d.message ?? d.error ?? safeJson(d); code = d.code; }
      code = code ?? data.code;
    } else if (typeof data === "string" && data) msg = data;
    throw new ApiError(res.status, msg, code);
  }
  return data;
}

// ── 会话侧栏 ──
async function loadHealth() {
  const badge = $("#health-badge");
  try {
    const h = await api("/api/health");
    const model = h?.model ?? h?.model_name ?? h?.active_model ?? "未知模型";
    const keyOk = h?.key_configured ?? h?.key_set ?? h?.has_key;
    badge.replaceChildren(
      el("span", "dot " + (keyOk === true ? "ok" : keyOk === false ? "bad" : "unk")),
      el("span", "model", String(model)));
    const host = h?.base_url_host ?? h?.base_url ?? h?.base_domain ?? "";
    const p = h?.permissions;
    const permTxt = p
      ? `权限 write=${p.write ?? "?"} command=${p.command ?? "?"} python=${p.python ?? "?"}` : "";
    const keyTxt = keyOk === true ? "key 已配置" : keyOk === false ? "key 未配置" : "key 状态未知";
    const verEl = $("#ver");
    if (verEl && h?.version) verEl.textContent = `v${h.version}`;
    renderEnvPanel(h); // v0.8.2：环境面板（模型/窗口/key/权限）
    badge.title = [model, host, keyTxt, permTxt].filter(Boolean).join("｜");
  } catch (e) {
    badge.replaceChildren(el("span", "dot bad"), el("span", "model", "服务不可达"));
    badge.title = `health 检查失败：${e.message}`;
  }
}
async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = Array.isArray(data) ? data : (data?.sessions ?? data?.items ?? []);
    renderSessions();
  } catch (e) {
    addInfoLine(`❌ 会话列表加载失败：${e.message}`);
  }
}
function renderSessions() {
  const box = $("#session-list");
  box.replaceChildren();
  if (!state.sessions.length) { box.append(el("div", "dim empty", "暂无历史会话")); return; }
  for (const s of state.sessions) {
    const tid = s.thread_id ?? s.id ?? "";
    const item = el("button", "sess" + (tid === state.current ? " active" : ""));
    item.append(el("div", "sess-id", tid || "（无 id）"));
    const meta = [
      `${s.rounds ?? s.round ?? 0} 轮`,
      fmtTime(s.last_active ?? s.updated_at ?? s.created_at),
    ].filter(Boolean).join(" · ");
    item.append(el("div", "sess-meta dim", meta));
    item.onclick = () => selectSession(tid);
    box.append(item);
  }
}
async function createSession() {
  const data = await api("/api/sessions", { method: "POST", body: {} });
  const tid = data?.thread_id ?? data?.session?.thread_id ?? data?.id;
  if (!tid) throw new Error("新建会话响应缺少 thread_id");
  await loadSessions();
  await selectSession(tid);
  return tid;
}
async function selectSession(tid) {
  if (!tid || tid === state.current) return;
  state.sessionSeq++;
  closeEvents();
  state.current = tid;
  state.round = null;
  state.running = false;
  state.awaiting = false;
  state.approvalId = null;
  state.roundUsage = [];   // 每轮用量按会话隔离
  state.sessionTotal = null;
  resetMinimap();
  renderTokenPanel();
  clearStream();
  renderSessions();
  refreshUi();
  addInfoLine(`已切换到 ${tid}`);
  connectEvents(tid);
  await loadSnapshot(tid);
}

// ── SSE 事件流 ──
function closeEvents() {
  if (state.es) { state.es.close(); state.es = null; }
}
function setSseDot(cls, title) {
  const dot = $("#sse-dot");
  dot.className = "dot " + cls;
  if (title) dot.title = title;
}
function connectEvents(tid) {
  closeEvents();
  const myTid = tid;
  const mySeq = state.sessionSeq;
  const tokenQp = state.token ? `?token=${encodeURIComponent(state.token)}` : "";
  const es = new EventSource(`/api/sessions/${encodeURIComponent(tid)}/events${tokenQp}`);
  state.es = es;
  const alive = () => state.current === myTid && state.sessionSeq === mySeq;
  for (const type of EVENT_TYPES) {
    es.addEventListener(type, (ev) => {
      if (!ev.data || !alive()) return; // 同名网络错误事件没有 data，跳过
      let data;
      try { data = JSON.parse(ev.data); } catch { data = ev.data; }
      handleEvent(type, data);
    });
  }
  // 兼容服务端只发默认 message（事件名放在 JSON 的 event/type 字段里）
  es.onmessage = (ev) => {
    if (!ev.data || !alive()) return;
    try {
      const m = JSON.parse(ev.data);
      const t = m?.event ?? m?.type;
      if (t) handleEvent(t, m.data ?? m);
    } catch { /* 非 JSON 忽略 */ }
  };
  es.onopen = () => { if (alive()) { setSseDot("ok", "SSE 已连接"); refreshStatus(); } };
  es.onerror = (ev) => {
    if (!alive() || ev.data) return; // ev.data 存在说明是服务端业务 error 事件，非断线
    // EventSource 原生自动重连 + Last-Event-ID，服务端从环形缓冲重放
    if (es.readyState === EventSource.CLOSED) setSseDot("bad", "SSE 连接已关闭");
    else setSseDot("warn", "SSE 断线，自动重连中…");
    refreshStatus();
  };
}

// ── SSE 事件分发 ──
function handleEvent(type, data) {
  switch (type) {
    case "round_start":
      state.round = data?.round ?? state.round;
      state.running = true;
      state.awaiting = false;
      state.approvalId = null;
      resetMinimap(); // 新一轮：执行图清零重画
      divider(`第 ${state.round ?? "?"} 轮`);
      if (data?.request) requestBubble(state.round, data.request);
      break;
    case "node":
      nodeCard(data?.node, data?.data);
      // 观测：节点完成推进小地图；planner 并行任务 → worker chip；reviewer revise → 回边闪
      const n = data?.node;
      if (n) {
        markNodeDone(n);
        const nd = data?.data;
        if (n === "planner" && Array.isArray(nd?.parallel_tasks) && nd.parallel_tasks.length >= 2) {
          state.parallelTasks = nd.parallel_tasks.length;
          minimapLabel("mid", `worker ×${state.parallelTasks}`);
        }
        if (n === "reviewer" && (nd?.verdict === "revise" || nd?.verdict?.verdict === "revise")) {
          flashRevise();
        }
      }
      break;
    case "approval_required":
      approvalCard(data);
      minimapSet("guard", "active"); // guard 待批闪烁（CSS pulse）
      state.lastNode = "guard";
      break;
    case "round_end":
      usageLine(data);
      if (data?.usage) {
        const u = data.usage;
        state.roundUsage.push({
          round: data.round ?? state.round ?? state.roundUsage.length + 1,
          prompt: num0(u.prompt), completion: num0(u.completion),
          calls: num0(u.calls), context: num0(u.context),
        });
      }
      renderTokenPanel();
      minimapSet("report", "done");
      state.running = false;
      state.awaiting = false;
      state.approvalId = null;
      loadSessions(); // 轮次/活跃时间已变化，顺手刷新侧栏
      break;
    case "error":
      errorCard(data);
      state.running = false;
      state.awaiting = false;
      state.approvalId = null;
      break;
    case "info":
      addInfoLine(data?.message ?? txt(data));
      break;
    default:
      addInfoLine(`[${type}] ${txt(data)}`);
      break;
  }
  refreshUi();
}

// ── 过程节点卡 ──
function stepText(s) {
  if (typeof s === "string") return s;
  if (s && typeof s === "object") return s.step ?? s.title ?? s.task ?? txt(s);
  return String(s);
}
function nodeDataText(nodeName, data) {
  if (data == null) return "（无输出）";
  if (typeof data === "string") return data;
  if (Array.isArray(data)) return safeJson(data);
  const chunks = [];
  const plan = data.plan ?? data.steps;
  if (Array.isArray(plan) && plan.length) {
    chunks.push("【计划】\n" + plan.map((s, i) => `  ${i + 1}. ${stepText(s)}`).join("\n"));
  }
  const pt = data.parallel_tasks;
  if (Array.isArray(pt) && pt.length >= 2) chunks.push(`【并行子任务】× ${pt.length}`);
  if (Array.isArray(data.worker_notes) && data.worker_notes.length) {
    chunks.push("【并行子任务】\n" + data.worker_notes.map(String).join("\n"));
  }
  if (typeof data.feedback === "string" && data.feedback) chunks.push(data.feedback);
  if (typeof data.report === "string" && data.report) chunks.push(data.report);
  if (typeof data.last_report === "string" && data.last_report) chunks.push(data.last_report);
  if (Array.isArray(data.changed_files) && data.changed_files.length) {
    chunks.push("【改动文件】" + data.changed_files.join("; "));
  }
  if (typeof data.verdict === "string" && data.verdict && !data.feedback) {
    chunks.push(`verdict: ${data.verdict}`);
  }
  if (!chunks.length) {
    const json = safeJson(data);
    return json.length > 6000 ? json.slice(0, 6000) + "\n…（已截断）" : json;
  }
  return chunks.join("\n\n");
}
// verifier 输出按行着色：✓ 绿 / ✗ 红（对齐 CLI _print_node 的着色语义）
function coloredLinesPre(text) {
  const wrap = el("div", "code-lines");
  for (const ln of String(text).split("\n").slice(0, MAX_RENDER_LINES)) {
    let cls = "code-row";
    if (ln.startsWith("✓")) cls += " line-ok";
    else if (ln.startsWith("✗")) cls += " line-bad";
    const row = el("div", cls);
    row.append(el("span", "lc", ln));
    wrap.append(row);
  }
  return wrap;
}
function nodeCard(nodeName, data) {
  const det = el("details", "card node");
  if (nodeName === "report") det.open = true; // 交付报告默认展开
  const sum = el("summary");
  sum.append(
    el("span", "node-label", NODE_LABELS[nodeName] || nodeName || "node"),
    el("span", "node-name dim", nodeName || ""));
  det.append(sum);
  const text = nodeDataText(nodeName, data);
  det.append(nodeName === "verifier" ? coloredLinesPre(text) : pre("node-pre", text));
  appendCard(det);
}

// ── 用量行 / 错误卡 ──
function usageLine(data) {
  const u = data?.usage ?? {};
  const total = num0(u.prompt) + num0(u.completion);
  const calls = u.calls ?? u.n_calls ?? 0;
  let sess = data?.session_total;
  if (sess && typeof sess === "object") sess = num0(sess.prompt) + num0(sess.completion);
  let verdict = data?.verdict ?? "";
  if (verdict && typeof verdict === "object") verdict = verdict.verdict ?? txt(verdict);
  let text = `📊 第 ${data?.round ?? state.round ?? "?"} 轮用量：`
    + `输入 ${num0(u.prompt)} + 输出 ${num0(u.completion)} = ${total}（${calls} 次调用）`;
  if (sess != null) text += `｜会话累计 ${sess}`;
  if (u.context) text += `｜上下文峰值 ${u.context}`;
  if (verdict) text += `｜verdict ${verdict}`;
  appendCard(el("div", "usage", text));
}
function errorCard(data) {
  const c = el("div", "card err");
  c.append(el("div", "err-title", `❌ 出错${data?.phase ? `（${data.phase}）` : ""}`));
  c.append(pre("err-msg", data?.message ?? "未知错误"));
  if (data?.hint) c.append(el("div", "err-hint", `提示：${data.hint}`));
  if (data?.recoverable) {
    const b = el("button", "btn", "🔁 断点续跑");
    b.onclick = () => doRetry();
    c.append(b);
  }
  appendCard(c);
}

// ── 审批卡（核心交互） ──
const categoryOf = (action) => ACTION_CATEGORY[action] ?? "write";
// 快照里拿到的可能是卡片格式（带 preview），也可能是 guard 裸 change（content/old/new/code）。
// 裸 change 在前端按 CLI 同款预览规则补齐（patch 3 行 / write 5 行 / python 10 行 / 命令全文）。
function normalizeChange(ch, i) {
  const c = ch && typeof ch === "object" ? { ...ch } : { action: String(ch) };
  c.index = Number.isInteger(c.index) ? c.index : i;
  c.action = c.action ?? "unknown";
  c.category = c.category ?? categoryOf(c.action);
  c.permission = c.permission ?? "ask";
  if (c.preview == null) c.preview = previewFromRaw(c);
  if (c.bytes_hint == null) c.bytes_hint = rawSizeHint(c);
  return c;
}
function previewFromRaw(c) {
  const linesOf = (t) => String(t ?? "").split("\n");
  switch (c.action) {
    case "plan_run_command":
      return { rule: "command_full" };
    case "plan_patch": {
      const o = linesOf(c.old), n = linesOf(c.new);
      return { rule: "patch_3lines", old_head: o.slice(0, 3), new_head: n.slice(0, 3),
               total_lines: Math.max(o.length, n.length) };
    }
    case "plan_write_file": {
      const l = linesOf(c.content);
      return { rule: "write_5lines", lines: l.slice(0, 5), total_lines: l.length };
    }
    case "plan_run_python": {
      const l = linesOf(c.code);
      return { rule: "python_10lines", lines: l.slice(0, 10), total_lines: l.length };
    }
    default:
      return { rule: "generic" };
  }
}
function rawSizeHint(c) {
  const parts = [c.content, c.old, c.new, c.code, c.command].filter((x) => typeof x === "string");
  return parts.length ? parts.reduce((s, x) => s + x.length, 0) : undefined;
}
// 预览渲染：兼容 preview 对象的多种字段约定（old_head/new_head、lines、text、head）
function previewPre(preview) {
  if (preview == null) return null;
  if (typeof preview !== "object") return pre("preview", preview);
  if (preview.rule === "command_full") return null; // 命令全文已在 target 行展示
  let text = "";
  if (Array.isArray(preview.old_head) || Array.isArray(preview.new_head)) {
    text = "--- old\n" + (preview.old_head || []).join("\n")
      + "\n+++ new\n" + (preview.new_head || []).join("\n");
  } else if (Array.isArray(preview.lines)) text = preview.lines.join("\n");
  else if (typeof preview.text === "string") text = preview.text;
  else if (typeof preview.head === "string") text = preview.head;
  else text = safeJson(preview);
  const shown = Array.isArray(preview.lines) ? preview.lines.length : 0;
  if (preview.total_lines && shown && preview.total_lines > shown) {
    text += `\n  …（共 ${preview.total_lines} 行，点「展开全文」查看）`;
  }
  return pre("preview", text);
}
function changeRow(ch, boxes) {
  const risky = RISKY_CATEGORIES.has(ch.category);
  const row = el("div", "appr-row" + (risky ? " risky" : ""));
  const top = el("div", "appr-row-top");
  const cb = el("input");
  cb.type = "checkbox";
  cb.checked = true;
  cb.dataset.index = String(ch.index);
  boxes.push(cb);
  const info = ACTION_INFO[ch.action] ?? { tag: "other", icon: "•", label: ch.action };
  top.append(
    cb,
    el("span", "idx", `${ch.index + 1}.`),
    el("span", `badge a-${info.tag}`, `${info.icon} ${info.label}`),
    el("span", `badge perm-${ch.permission}`, `${ch.category}·${ch.permission}`),
    el("span", "target", ch.path ?? ch.command ?? ""));
  if (ch.bytes_hint != null) top.append(el("span", "dim", fmtBytes(ch.bytes_hint)));
  row.append(top);
  const pv = previewPre(ch.preview);
  if (pv) row.append(pv);
  const btn = el("button", "btn tiny", "展开全文");
  btn.onclick = () => expandChange(ch, btn, row);
  row.append(btn);
  return row;
}
async function expandChange(ch, btn, row) {
  const tid = state.current;
  if (!tid) return;
  btn.disabled = true;
  btn.textContent = "加载中…";
  try {
    const payload = await api(`/api/sessions/${encodeURIComponent(tid)}/changes/${ch.index}`);
    const holder = el("div", "appr-detail");
    renderDetailInto(holder, payload, ch);
    row.append(holder);
    btn.remove();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "展开全文";
    addInfoLine(`❌ 详情加载失败：${e.message}`);
  }
}
// changes/{i} 返回形状宽松适配：unified_diff/diff 文本、hunks 结构化、
// content/code 全文（带行号）、或裸 change（old/new 前后对照兜底）
function renderDetailInto(holder, payload, ch) {
  const d = (payload && typeof payload === "object" && payload.data != null) ? payload.data : payload;
  if (typeof d === "string") { holder.append(pre("preview", d)); return; }
  const diffText = d?.unified_diff ?? d?.diff ?? null;
  if (typeof diffText === "string") { holder.append(diffBlock(diffText.split("\n"))); return; }
  if (Array.isArray(d?.hunks)) {
    const lines = [];
    for (const h of d.hunks) {
      if (h?.header) lines.push(String(h.header));
      for (const l of h?.lines ?? []) lines.push(l);
    }
    holder.append(diffBlock(lines));
    return;
  }
  const body = d?.content ?? d?.code ?? d?.text ?? null;
  if (typeof body === "string") {
    if (d?.path) holder.append(el("div", "detail-title", String(d.path)));
    holder.append(codeBlockWithLines(body));
    return;
  }
  if (d?.old != null || d?.new != null) {
    // 服务端未给 hunks 的兜底：old 整块红 / new 整块绿（不对齐，仅示意）
    const lines = [`--- ${d.path ?? "old"}`];
    for (const l of String(d.old ?? "").split("\n")) lines.push({ kind: "del", text: l });
    lines.push(`+++ ${d.path ?? "new"}`);
    for (const l of String(d.new ?? "").split("\n")) lines.push({ kind: "add", text: l });
    holder.append(diffBlock(lines));
    return;
  }
  holder.append(pre("preview", safeJson(d ?? payload ?? ch)));
}
// diff 行按行首 +/- 染红绿（行可以是字符串、{kind,text} 或 {type,text}）
function diffBlock(lines) {
  const wrap = el("div", "code-lines diff");
  for (const ln of lines.slice(0, MAX_RENDER_LINES)) {
    const s = typeof ln === "string" ? ln : (ln?.text ?? ln?.line ?? String(ln));
    const kind = typeof ln === "object" && ln ? (ln.kind ?? ln.type) : null;
    let cls = "code-row";
    if (s.startsWith("--- ") || s.startsWith("+++ ")) cls += " filehdr";
    else if (kind === "add" || (!kind && s.startsWith("+"))) cls += " add";
    else if (kind === "del" || (!kind && s.startsWith("-"))) cls += " del";
    else if (s.startsWith("@@")) cls += " hunk";
    const row = el("div", cls);
    row.append(el("span", "lc", s));
    wrap.append(row);
  }
  if (lines.length > MAX_RENDER_LINES) {
    wrap.append(el("div", "code-row hunk",
      `…（共 ${lines.length} 行，仅渲染前 ${MAX_RENDER_LINES} 行）`));
  }
  return wrap;
}
// write/python 全文：带行号代码块
function codeBlockWithLines(text) {
  const wrap = el("div", "code-lines");
  const lines = String(text).split("\n");
  lines.slice(0, MAX_RENDER_LINES).forEach((ln, i) => {
    const row = el("div", "code-row");
    row.append(el("span", "ln", String(i + 1)), el("span", "lc", ln));
    wrap.append(row);
  });
  if (lines.length > MAX_RENDER_LINES) {
    wrap.append(el("div", "code-row hunk",
      `…（共 ${lines.length} 行，仅渲染前 ${MAX_RENDER_LINES} 行）`));
  }
  return wrap;
}
function approvalCard(data) {
  const aid = data?.approval_id ?? data?.id ?? "";
  if (aid && state.seenApprovals.has(aid)) return; // 历史回放与快照重建去重
  if (aid) state.seenApprovals.add(aid);
  const round = data?.round ?? state.round;
  const changes = (Array.isArray(data?.changes) ? data.changes : [])
    .map((ch, i) => normalizeChange(ch, i));
  state.awaiting = true;
  state.approvalId = aid || null;
  refreshUi();
  const card = el("div", "card approval");
  const head = el("div", "appr-head");
  head.append(
    el("span", "appr-title", `⏸ 审批请求（${changes.length} 项）`),
    el("span", "dim", `第 ${round ?? "?"} 轮`));
  card.append(head);
  const list = el("div", "appr-list");
  const boxes = [];
  for (const ch of changes) list.append(changeRow(ch, boxes));
  card.append(list);
  // 决策区（对齐 CLI：y 全批 / n 全拒 / m 意见 / 序号选批）
  const dz = el("div", "appr-actions");
  const note = el("input", "note");
  note.type = "text";
  note.placeholder = "附言 / 修改意见（退回修改必填）";
  const bAll = el("button", "btn primary", "全部批准");
  const bSel = el("button", "btn", "批准勾选");
  const bRej = el("button", "btn danger", "全部拒绝");
  const bMod = el("button", "btn", "退回修改");
  dz.append(note, bAll, bSel, bRej, bMod);
  card.append(dz);
  const controls = [note, bAll, bSel, bRej, bMod, ...boxes];
  const refreshSelLabel = () => {
    const n = boxes.filter((c) => c.checked).length;
    bSel.textContent = `批准勾选（${n} 项）`;
    bSel.disabled = n === 0;
  };
  boxes.forEach((c) => c.addEventListener("change", refreshSelLabel));
  refreshSelLabel();
  const setControlsEnabled = (on) => {
    for (const c of controls) c.disabled = !on;
    refreshSelLabel();
  };
  const submit = async (kind) => {
    const tid = state.current;
    if (!tid) return;
    const noteTxt = note.value.trim();
    // 决策回传即 guard 的 resume 协议原样（design-web.md §4.2）
    const body = { approval_id: aid, action: kind === "approve-selected" ? "approve" : kind };
    let approvedCount = changes.length;
    if (kind === "approve-selected") {
      const idx = changes.filter((c, i) => boxes[i].checked).map((c) => c.index);
      if (!idx.length) return;
      body.indices = idx;
      approvedCount = idx.length;
    } else if (kind === "reject") {
      body.note = noteTxt || "用户拒绝";
    } else if (kind === "modify") {
      if (!noteTxt) {
        note.classList.add("need");
        note.focus();
        addInfoLine("退回修改需要填写修改意见");
        return;
      }
      body.note = noteTxt;
    }
    setControlsEnabled(false);
    try {
      await api(`/api/sessions/${encodeURIComponent(tid)}/approvals`, { method: "POST", body });
      freezeCard(card, kind, approvedCount, changes.length);
      state.awaiting = false;
      state.approvalId = null;
    } catch (e) {
      setControlsEnabled(true);
      addInfoLine(`❌ 审批提交失败：${e.message}`);
    }
    refreshUi();
  };
  bAll.onclick = () => submit("approve");
  bSel.onclick = () => submit("approve-selected");
  bRej.onclick = () => submit("reject");
  bMod.onclick = () => submit("modify");
  appendCard(card);
}
function freezeCard(card, kind, count, total) {
  card.classList.add("resolved");
  card.querySelectorAll("button, input").forEach((x) => { x.disabled = true; });
  const msg = kind === "approve" ? `✅ 已全部批准（${total} 项）`
    : kind === "approve-selected" ? `✅ 已批准 ${count} 项 / 共 ${total} 项（其余跳过）`
    : kind === "reject" ? "⛔ 已全部拒绝"
    : "✏️ 已退回修改";
  card.append(el("div", "appr-result", msg));
}

// ── 会话快照重建（M2） ──
async function loadSnapshot(tid) {
  let snap;
  try {
    snap = await api(`/api/sessions/${encodeURIComponent(tid)}`);
  } catch (e) {
    addInfoLine(`❌ 会话快照加载失败：${e.message}`);
    return;
  }
  const data = (snap && typeof snap === "object" && (snap.data ?? snap.snapshot)) ?? snap ?? {};
  // 历史回放（若后端提供 history/events 数组，元素形如 {type, data}）
  const history = data.history ?? data.events;
  if (Array.isArray(history) && history.length) {
    for (const item of history) {
      if (!item || typeof item !== "object") continue;
      const t = item.type ?? item.event ?? item.kind;
      if (t) handleEvent(t, item.data ?? item.payload ?? item);
    }
  } else {
    renderSnapshotFallback(data);
  }
  // 挂起中的审批 → 重建审批卡（服务重启后也能继续批，M2 关键场景）
  const pending = extractPendingApproval(data);
  if (pending) {
    approvalCard(pending);
    state.running = true;
    state.awaiting = true;
    state.approvalId = pending.approval_id || null;
    addInfoLine("检测到挂起中的审批，可直接在此决策");
  } else {
    // 无挂起审批：历史回放里若有已决议的 approval_required，awaiting 至此复位
    state.awaiting = false;
    state.approvalId = null;
    const st = String(data.status ?? data.graph_status ?? "").toLowerCase();
    if (st === "running" || st === "awaiting_approval" || st === "busy") {
      state.running = true;
    } else {
      state.running = false;
      const nextArr = Array.isArray(data.next) ? data.next : [];
      if (nextArr.length) addInfoLine("存在未完成的执行（停在非审批节点），可点「断点续跑」");
    }
  }
  // v0.8.2：会话累计用量喂 token 面板（重启后 per-round 丢失，但累计仍在）
  const tu = data.token_usage ?? data.usage;
  if (tu && typeof tu === "object") {
    state.sessionTotal = { prompt: num0(tu.prompt), completion: num0(tu.completion) };
  }
  renderTokenPanel();
  refreshUi();
}
// 无 history 时的兜底视图：轮次分隔 + 当前需求 + 计划 + 累计用量 + 最近报告
function renderSnapshotFallback(data) {
  const rounds = data.rounds ?? data.round ?? data.session?.rounds;
  if (rounds) divider(`会话恢复 · 共 ${rounds} 轮`);
  if (data.request) requestBubble(data.round ?? rounds ?? null, data.request);
  const plan = data.plan ?? data.steps;
  if (Array.isArray(plan) && plan.length) {
    nodeCard("planner", { plan, parallel_tasks: data.parallel_tasks });
  }
  const usage = data.token_usage ?? data.usage;
  if (usage && typeof usage === "object") {
    appendCard(el("div", "usage",
      `会话累计 token：输入 ${num0(usage.prompt)} + 输出 ${num0(usage.completion)} = `
      + `${num0(usage.prompt) + num0(usage.completion)}`));
  }
  const rep = data.last_report ?? data.report;
  if (typeof rep === "string" && rep.trim()) nodeCard("report", { report: rep });
}
// 从快照提取挂起审批：优先卡片格式（pending_approval，含 approval_id 与预览），
// 其次 LangGraph 原生 interrupts（裸 change，前端补预览）
function extractPendingApproval(data) {
  for (const key of ["pending_approval", "pendingApproval", "approval"]) {
    const c = data?.[key];
    if (c && Array.isArray(c.changes) && c.changes.length) return materializeCard(c, data);
  }
  const ints = data?.interrupts ?? data?.pending?.interrupts ?? data?.state?.interrupts;
  if (Array.isArray(ints) && ints.length) {
    const it = ints.find((x) => x && (x.value ?? x)?.changes) ?? ints[0];
    const v = it?.value ?? it;
    if (v && Array.isArray(v.changes) && v.changes.length) {
      return materializeCard(
        { ...v, approval_id: v.approval_id ?? it?.id ?? data?.approval_id }, data);
    }
  }
  return null;
}
function materializeCard(c, data) {
  return {
    approval_id: c.approval_id ?? c.id ?? "",
    round: c.round ?? data.round ?? data.rounds ?? null,
    changes: c.changes.map((ch, i) => normalizeChange(ch, i)),
  };
}

// ── 用户动作 ──
async function sendMessage() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text || state.running || state.awaiting) return;
  try {
    if (!state.current) await createSession();
    await api(`/api/sessions/${encodeURIComponent(state.current)}/messages`,
      { method: "POST", body: { text } });
    input.value = "";
    autoGrow(input);
    state.running = true; // round_start 事件随后到达，幂等
  } catch (e) {
    if (e.status === 409) addInfoLine("上一轮还在跑，请稍候（服务端已拒绝本轮）");
    else addInfoLine(`❌ 发送失败：${e.message}`);
  }
  refreshUi();
}
function resultText(r) {
  if (r == null) return "完成";
  if (typeof r === "string") return r;
  return r.message ?? r.detail ?? r.summary ?? safeJson(r);
}
async function doUndo() {
  const tid = state.current;
  if (!tid) return;
  if (!window.confirm("回退最近一轮的文件改动？（仅 write/patch 可撤，命令副作用不可撤）")) return;
  try {
    const r = await api(`/api/sessions/${encodeURIComponent(tid)}/undo`,
      { method: "POST", body: {} });
    addInfoLine(`↩ 撤销完成：${resultText(r)}`);
    loadSessions();
  } catch (e) {
    addInfoLine(`❌ 撤销失败：${e.message}`);
  }
}
async function doRetry() {
  const tid = state.current;
  if (!tid) return;
  try {
    await api(`/api/sessions/${encodeURIComponent(tid)}/retry`, { method: "POST", body: {} });
    addInfoLine("🔁 已从断点继续执行");
    state.running = true;
  } catch (e) {
    if (e.status === 409) addInfoLine("没有可续跑的挂起执行");
    else addInfoLine(`❌ 断点续跑失败：${e.message}`);
  }
  refreshUi();
}

// ── UI 状态刷新 ──
function refreshComposer() {
  const busy = state.running || state.awaiting;
  const input = $("#input");
  const send = $("#btn-send");
  input.disabled = busy;
  send.disabled = busy || !input.value.trim();
}
function refreshButtons() {
  const idle = !!state.current && !state.running && !state.awaiting;
  $("#btn-undo").disabled = !idle;
  $("#btn-retry").disabled = !idle;
}
function refreshStatus() {
  const s = $("#status-line");
  if (state.awaiting) s.textContent = "⏸ 等待审批";
  else if (state.running) s.textContent = "▶ 执行中…";
  else s.textContent = state.es ? "已连接 · 空闲" : "空闲";
}
const refreshUi = () => { refreshComposer(); refreshButtons(); refreshStatus(); };
const autoGrow = (ta) => {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 170) + "px";
};
const renderEmptyHint = () => {
  clearStream();
  addInfoLine("左侧选择会话，或点「＋ 新会话」开始；直接在下方输入需求会自动创建会话。");
};

// ── 启动 ──
function wireUi() {
  $("#btn-new").onclick = async () => {
    try {
      await createSession();
      $("#input").focus();
    } catch (e) {
      addInfoLine(`❌ 新建会话失败：${e.message}`);
    }
  };
  $("#btn-undo").onclick = () => doUndo();
  $("#btn-retry").onclick = () => doRetry();
  $("#btn-send").onclick = () => sendMessage();
  // v0.8.2 观测面板：审计尾部 / 观测折叠 / 模型切换
  $("#btn-audit").onclick = () => openAudit();
  $("#audit-close").onclick = () => $("#audit-modal").classList.add("hidden");
  $("#audit-modal").addEventListener("click", (e) => {
    if (e.target.id === "audit-modal") $("#audit-modal").classList.add("hidden");
  });
  const obs = $("#obs");
  $("#obs-toggle").onclick = () => {
    obs.classList.toggle("collapsed");
    $("#obs-toggle").textContent = obs.classList.contains("collapsed") ? "◂" : "▸";
  };
  const input = $("#input");
  input.addEventListener("input", () => { autoGrow(input); refreshComposer(); });
  input.addEventListener("keydown", (e) => {
    if (e.isComposing) return; // 中文输入法组词中不触发发送
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      sendMessage();
    }
  });
}
async function boot() {
  state.token = loadToken();
  wireUi();
  refreshUi();
  renderEmptyHint();
  buildMinimap();
  resetMinimap();
  await loadHealth();
  await loadModels();
  await loadSessions();
  if (state.sessions.length) {
    const first = state.sessions[0];
    const tid = first?.thread_id ?? first?.id;
    if (tid) await selectSession(tid);
  }
  setInterval(loadHealth, 30_000);
}
boot();
