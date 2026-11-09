#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简历 AI 助手 · ReAct Agent 服务
--------------------------------
纯 Python 标准库实现，零第三方依赖。单端口同时提供：
  - 静态文件（简历站点）
  - POST /api/chat  （RAG + ReAct 问答）

ReAct 循环：Thought → Action(tool) → Observation → (可选第二轮) → Answer
工具：search_kb / get_by_type / get_contact / get_profile

生成模式：
  - 配置 LLM_API_KEY  → 调 OpenAI 兼容接口，用检索到的原文作答（RAG 防幻觉）
  - 未配置            → 检索模板模式，答案仍全部来自简历原文，绝不编造
"""

import json
import math
import os
import re
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.environ.get("STATIC_DIR", HERE)
KB_PATH = os.path.join(HERE, "kb.json")

# ---------- 载入知识库 ----------
with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)
CHUNKS = KB["chunks"]
OWNER = KB.get("meta", {}).get("owner", "本人")

# ---------- 分词：中文 bigram + 英文/数字词 ----------
_CJK = r"\u4e00-\u9fff"


def tokenize(text: str):
    t = (text or "").lower()
    toks = []
    for w in re.findall(r"[a-z0-9][a-z0-9\.%\+#/_-]*", t):
        toks.append(w)
    for seg in re.findall(r"[" + _CJK + r"]+", t):
        if len(seg) == 1:
            toks.append(seg)
        for i in range(len(seg) - 1):
            toks.append(seg[i:i + 2])
    return toks


# ---------- 构建 TF-IDF 索引 ----------
DOC_TOKENS = [tokenize(c["text"] + " " + " ".join(c.get("tags", [])) + " " + c.get("title", "")) for c in CHUNKS]
DOC_TF = []
DF = {}
for toks in DOC_TOKENS:
    tf = {}
    for tk in toks:
        tf[tk] = tf.get(tk, 0) + 1
    DOC_TF.append(tf)
    for tk in set(toks):
        DF[tk] = DF.get(tk, 0) + 1
N = len(CHUNKS)
IDF = {tk: math.log((N + 1) / (df + 0.5)) for tk, df in DF.items()}


def vec(tf):
    v = {}
    for tk, c in tf.items():
        v[tk] = (1 + math.log(c)) * IDF.get(tk, 0.0)
    return v


DOC_VEC = [vec(tf) for tf in DOC_TF]


def cosine(v1, v2):
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    num = sum(v1[k] * v2[k] for k in common)
    d1 = math.sqrt(sum(x * x for x in v1.values()))
    d2 = math.sqrt(sum(x * x for x in v2.values()))
    return num / (d1 * d2) if d1 and d2 else 0.0


# ---------- 工具集（Agent 可调用） ----------
_NUM_RE = re.compile(r"\d")
# 问「成果/数据/效果」时，优先命中含数字的片段
_NUM_HINT_KW = ("成果", "数据", "效果", "增长", "提升", "降低", "优化", "roi", "ctr", "cpe", "消耗", "多少", "成绩", "亮点", "亮眼")


def tool_search_kb(query, topk=3, type_filter=None):
    qv = vec({tk: 1 for tk in tokenize(query)})
    want_num = any(k in (query or "").lower() for k in _NUM_HINT_KW)
    scored = []
    for i, c in enumerate(CHUNKS):
        if type_filter and c.get("type") != type_filter:
            continue
        s = cosine(qv, DOC_VEC[i])
        # 标题/标签命中加权
        title_blob = (c.get("title", "") + " " + " ".join(c.get("tags", []))).lower()
        for tk in set(tokenize(query)):
            if len(tk) >= 2 and tk in title_blob:
                s += 0.06
        # 问成果类问题时，给含数字的片段加权
        if want_num and _NUM_RE.search(c["text"]):
            s += 0.10
        scored.append((s, c))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    top = scored[0][0]
    # 相对阈值：只保留达到最高分 35% 的，避免噪音片段混入
    cut = max(top * 0.35, 0.03)
    kept = [(s, c) for s, c in scored if s >= cut][:topk]
    return [(round(s, 4), c) for s, c in kept]


def tool_get_by_type(type_name):
    return [(1.0, c) for c in CHUNKS if c.get("type") == type_name]


def tool_get_contact():
    return [(1.0, c) for c in CHUNKS if c.get("type") == "contact"]


# 「AI 相关」跨类型聚合：按标签命中，避免语义检索把无关片段捞进来
_AI_KEYS = ("ai", "agent", "llm", "rag", "mcp", "prompt", "智能体", "人工智能", "自动化", "工具调用")


def tool_get_ai():
    out = []
    for c in CHUNKS:
        tags = [t.lower() for t in c.get("tags", [])]
        blob = " ".join(tags) + " " + c.get("title", "").lower()
        if any(k in blob for k in _AI_KEYS):
            out.append((1.0, c))
    return out


# 「数据成果」召回：只要带强数字信号（百分比 / 倍数 / 效果指标）的片段
_METRIC_RE = re.compile(r"(\d+(\.\d+)?\s*%|\d+\s*倍|ROI|CTR|CPE|CPC|CPTI)")


def tool_get_metrics():
    return [(1.0, c) for c in CHUNKS if _METRIC_RE.search(c["text"]) or _METRIC_RE.search(c.get("title", ""))]


TOOLS = {
    "search_kb": tool_search_kb,
    "get_by_type": tool_get_by_type,
    "get_contact": tool_get_contact,
}

# ---------- 意图识别（Agent 的决策依据） ----------
INTENT_RULES = [
    ("contact", ("联系", "邮箱", "电话", "怎么找", "微信", "如何联系"), tool_get_contact, "get_contact"),
    ("work", ("工作经历", "就职", "任职", "履历", "在哪家公司", "几家公司", "做过什么工作", "职业经历", "待过"),
     lambda: tool_get_by_type("work"), "get_by_type"),
    ("project", ("项目", "作品", "做过的东西", "看板", "助手", "demo", "开源"),
     lambda: tool_get_by_type("project"), "get_by_type"),
    ("skill", ("技能", "会什么", "能力", "擅长", "工具", "技术栈", "会不会"),
     lambda: tool_get_by_type("skill"), "get_by_type"),
    ("education", ("学历", "学校", "专业", "毕业", "教育"), lambda: tool_get_by_type("education"), "get_by_type"),
    ("self", ("评价", "性格", "怎么样", "优点", "自我评价", "为人"), lambda: tool_get_by_type("self"), "get_by_type"),
    ("ai", ("ai", "人工智能", "大模型", "llm", "agent", "智能体", "rag", "mcp", "机器人"),
     tool_get_ai, "get_ai"),
    ("metric", ("成果", "数据表现", "效果如何", "增长", "roi", "ctr", "cpe", "消耗", "成绩", "亮点", "亮眼", "多少"),
     tool_get_metrics, "get_metrics"),
]


def detect_intent(q):
    ql = (q or "").lower()
    for name, kws, fn, tool in INTENT_RULES:
        if any(k in ql for k in kws):
            return name, fn, tool
    return "general", None, "search_kb"


# ---------- 提示问题（检索为空时给的建议） ----------
SUGGESTIONS = [
    "她做过哪些 AI 相关的事？",
    "有哪些亮眼的数据成果？",
    "工作经历是怎样的？",
    "介绍一下项目经历",
    "她会哪些技能？",
    "怎么联系她？",
]


def build_answer_template(q, hits, intent):
    """无 LLM 时的降级生成：答案全部由检索到的原文拼装，不编造"""
    if not hits:
        return ("这个问题在我的简历知识库里没检索到明确内容。"
                "可以换个问法，比如：" + "、".join(SUGGESTIONS[:3])), []

    lead = {
        "contact": "联系方式如下：",
        "project": "ta 自主搭建过的项目主要有这些：",
        "skill": "能力标签覆盖这些方向：",
        "education": "教育背景：",
        "self": "自我评价：",
    }.get(intent, "根据简历内容：")

    lines = []
    srcs = []
    for score, c in hits:
        srcs.append({"id": c["id"], "title": c.get("title", ""), "score": score})
        head = c.get("title", "")
        if c.get("org"):
            head += f"（{c['org']}"
            if c.get("period"):
                head += f" · {c['period']})"
            else:
                head += ")"
        elif c.get("period"):
            head += f"（{c['period']}）"
        lines.append(f"• {head}：{c['text']}")

    answer = lead + "\n" + "\n".join(lines)
    return answer, srcs


def call_llm(question, hits):
    """有 LLM_API_KEY 时：用检索结果作为上下文生成回答（RAG）"""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")

    context = "\n\n".join(
        f"[{i + 1}] {c.get('title', '')}\n{c['text']}" for i, (s, c) in enumerate(hits)
    )
    system = (
        f"你是{OWNER}简历页上的 AI 助手，代表 ta 回答访客（可能是 HR 或面试官）的提问。\n"
        "严格规则：\n"
        "1. 只能使用【简历原文】里的信息回答，禁止编造任何简历中没有的内容；\n"
        "2. 原文没有的，直接说明「简历里没有提到」，不要猜测；\n"
        "3. 用中文回答，简洁自然，2-4 句话，可用项目符号；\n"
        "4. 涉及数字必须原样引用原文数字。\n"
        "5. 称呼用第三人称「她/ta」或「我」均可，保持口语化自然。"
    )
    user = f"【简历原文】\n{context}\n\n【问题】{question}"

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": 600,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[LLM 调用失败，降级为检索模式]", e)
        return None


# ---------- ReAct 主循环 ----------
# 语义检索达到该分数即视为「问的是具体实体」，优先于分类路由
SEM_THRESHOLD = 0.45


def run_agent(question):
    trace = []
    step = 0

    # Step 1 · Thought：先语义检索一次，判断是不是在问具体实体
    step += 1
    intent, fn, tool_name = detect_intent(question)
    sem = tool_search_kb(question, topk=3)
    sem_top = sem[0][0] if sem else 0.0
    use_sem = sem_top >= SEM_THRESHOLD
    trace.append({
        "step": step, "type": "thought",
        "text": (f"分析提问意图 → 归类为「{intent}」。同时做一次语义检索，最高相关度 {sem_top:.2f}。"
                 + ("高于阈值 %.2f，判定为具体实体提问，按语义命中作答。" % SEM_THRESHOLD if use_sem
                    else "未达阈值 %.2f，改用分类路由召回该类目下的完整内容。" % SEM_THRESHOLD))
    })

    # Step 2 · Action
    step += 1
    if use_sem:
        hits = sem
        tool_name = "search_kb"
        args = {"tool": "search_kb", "query": question, "topk": 3}
    elif tool_name == "get_contact":
        hits = tool_get_contact()
        args = {"tool": "get_contact"}
    elif tool_name == "get_ai":
        hits = tool_get_ai()
        args = {"tool": "get_ai"}
    elif tool_name == "get_metrics":
        hits = tool_get_metrics()
        args = {"tool": "get_metrics"}
    elif tool_name == "get_by_type" and fn:
        hits = fn()
        args = {"tool": "get_by_type", "type": intent}
    else:
        hits = tool_search_kb(question, topk=3)
        args = {"tool": "search_kb", "query": question, "topk": 3}

    trace.append({"step": step, "type": "action", "tool": tool_name, "args": args})

    # Step 3 · Observation
    step += 1
    if hits:
        obs = "命中 " + str(len(hits)) + " 条：" + "、".join(h[1].get("title", h[1]["id"]) for h in hits)
    else:
        obs = "未命中相关内容，需要提示访客换个问法。"
    trace.append({"step": step, "type": "observation", "text": obs,
                  "hits": [{"id": h[1]["id"], "title": h[1].get("title", ""), "score": h[0]} for h in hits]})

    # Step 3.5 · 若首轮没命中，放宽条件再检索一轮（ReAct 的迭代）
    if not hits:
        step += 1
        trace.append({"step": step, "type": "thought", "text": "首轮未命中，去掉类型过滤再检索一次。"})
        hits = tool_search_kb(question, topk=3)
        step += 1
        trace.append({"step": step, "type": "observation",
                      "text": ("二次检索命中 " + str(len(hits)) + " 条") if hits else "二次检索仍无结果",
                      "hits": [{"id": h[1]["id"], "title": h[1].get("title", ""), "score": h[0]} for h in hits]})

    # Step 4 · Answer
    step += 1
    mode = "rag-template"
    answer, srcs = build_answer_template(question, hits, intent)
    if hits:
        llm_ans = call_llm(question, hits)
        if llm_ans:
            answer = llm_ans
            mode = "llm"
            srcs = [{"id": h[1]["id"], "title": h[1].get("title", ""), "score": h[0]} for h in hits]

    trace.append({"step": step, "type": "answer", "text": answer, "mode": mode})
    return {"answer": answer, "sources": srcs, "trace": trace, "mode": mode}


# ---------- HTTP 服务 ----------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, rel):
        safe = os.path.normpath(rel).lstrip("/")
        path = os.path.join(STATIC_DIR, safe)
        if not path.startswith(os.path.abspath(STATIC_DIR)):
            return self._send(403, "forbidden", "text/plain; charset=utf-8")
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if not os.path.exists(path):
            return self._send(404, "not found", "text/plain; charset=utf-8")
        ext = os.path.splitext(path)[1].lower()
        ctype = {
            ".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".ico": "image/x-icon", ".woff2": "font/woff2",
        }.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        rel = unquote(urlparse(self.path).path)
        if rel in ("/", "/api/health"):
            if rel == "/api/health":
                return self._send(200, json.dumps({"ok": True, "mode": "llm" if os.environ.get("LLM_API_KEY") else "rag-template"}))
            rel = "/index.html"
        self._send_file(rel)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            return self._send(400, json.dumps({"error": "bad request"}))
        q = (body.get("question") or "").strip()
        if not q:
            return self._send(400, json.dumps({"error": "empty question"}))
        try:
            result = run_agent(q)
            result["question"] = q
            return self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"[resume-agent] 服务启动 http://0.0.0.0:{port}  静态目录={STATIC_DIR}  "
          f"LLM={'已配置' if os.environ.get('LLM_API_KEY') else '未配置(检索模式)'}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
