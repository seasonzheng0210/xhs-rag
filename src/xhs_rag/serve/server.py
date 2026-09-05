"""M6 Web UI —— 手机可访问的收藏夹检索界面。

特性:
  - 标准库 http.server 零依赖
  - 启动时预热模型(embedding + rerank),检索响应不再有冷启动
  - 0.0.0.0 监听,局域网手机可直接访问
  - /        搜索页面(移动端优先)
  - /api/search?q=xxx   语义检索 JSON
  - /api/stats          数据统计 JSON
用法: python -m xhs_rag.cli serve
"""
from __future__ import annotations

import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from loguru import logger

from ..core.config import Config
from ..index.retriever import Retriever
from ..store.db import DB

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>收藏夹 RAG</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1f2328;--muted:#6e7781;--accent:#ff2442;
  --border:#e5e7eb;--radius:14px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg);color:var(--fg);padding:16px;max-width:720px;margin:0 auto}
header{display:flex;align-items:baseline;gap:10px;margin-bottom:14px}
h1{font-size:20px;font-weight:700}
header .sub{font-size:12px;color:var(--muted)}
.searchbox{display:flex;gap:8px;margin-bottom:14px}
#q{flex:1;padding:12px 14px;font-size:16px;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--card);outline:none}
#q:focus{border-color:var(--accent)}
#btn{padding:12px 20px;font-size:15px;font-weight:600;border:none;
  background:var(--accent);color:#fff;border-radius:var(--radius);cursor:pointer}
#btn:disabled{opacity:.5}
.hint{font-size:12px;color:var(--muted);margin-bottom:14px}
#status{font-size:13px;color:var(--muted);margin-bottom:10px;min-height:18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px 16px;margin-bottom:10px}
.card .title{font-size:15px;font-weight:600;color:var(--fg);
  text-decoration:none;display:block;margin-bottom:6px}
.card .meta{font-size:12px;color:var(--muted);margin-bottom:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;
  margin-right:6px}
.badge.score{background:#fff1f2;color:var(--accent)}
.badge.kind{background:#f0f0ee;color:var(--muted)}
.card .text{font-size:13px;line-height:1.6;color:#3a3f45;
  display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.card.hl{border-color:var(--accent);box-shadow:0 0 0 3px #ffe4e8}
/* ── M8 AI 回答 ── */
.answer{background:linear-gradient(180deg,#fff9fa,#fff);border:1px solid #ffd7dd;
  border-radius:var(--radius);padding:14px 16px;margin-bottom:14px}
.answer .hd{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700;
  color:var(--accent);margin-bottom:8px}
.answer .hd .tag{font-size:11px;font-weight:500;color:var(--muted);
  background:#fff;border:1px solid var(--border);border-radius:99px;padding:1px 8px}
.answer .body{font-size:14px;line-height:1.75;color:#2b2f35;white-space:pre-wrap;
  word-break:break-word}
.answer .body:empty::after{content:'思考中…';color:var(--muted);font-size:13px}
sup.cite{display:inline-block;min-width:15px;padding:0 3px;margin:0 1px;
  font-size:11px;line-height:1.5;text-align:center;color:var(--accent);
  background:#fff1f2;border-radius:4px;cursor:pointer;vertical-align:super;
  text-decoration:none}
sup.cite:hover{background:var(--accent);color:#fff}
.notice{font-size:12px;color:#8a6d3b;background:#fff8e6;border:1px solid #ffe6a8;
  border-radius:10px;padding:8px 12px;margin-bottom:12px}
/* ── 调试模式: 错误卡片 ── */
.errbox{background:#fff5f5;border:1px solid #ffcdd2;border-radius:10px;
  padding:10px 12px;margin-bottom:12px;font-size:13px;color:#b71c1c}
.errbox .msg{font-weight:600;margin-bottom:6px}
.errbox .ops{display:flex;gap:8px;flex-wrap:wrap}
.errbox button{font-size:12px;padding:5px 12px;border-radius:6px;border:1px solid #f0b4b4;
  background:#fff;color:#b71c1c;cursor:pointer}
.errbox button:hover{background:#ffe9e9}
.errbox .trace{margin-top:8px;display:none}
.errbox .trace pre{background:#2d2d2d;color:#e8e8e8;font-size:11px;line-height:1.5;
  padding:10px;border-radius:6px;overflow-x:auto;max-height:260px;overflow-y:auto;
  white-space:pre;margin:0}
.errbox .copied{color:#2e7d32;font-size:12px;margin-top:6px;display:none}
.footer{margin-top:18px;font-size:12px;color:var(--muted);text-align:center}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #ffd7dd;
  border-top-color:var(--accent);border-radius:50%;animation:sp .7s linear infinite;
  vertical-align:-2px;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header><h1>📚 收藏夹 RAG</h1><span class="sub" id="stat"></span></header>
<div class="searchbox">
  <input id="q" placeholder="搜收藏夹里的内容…" autocomplete="off">
  <button id="btn" onclick="go()">搜索</button>
</div>
<div class="hint">试试：怎么护理宝宝私处 / 衣物清洗 / 月子喂养</div>
<div id="status"></div>
<div id="answer-box"></div>
<div id="results"></div>
<div class="footer" id="foot"></div>
<script>
let loading=false;
const $=s=>document.querySelector(s);
async function go(){
  const q=$('#q').value.trim();
  if(!q||loading)return;
  loading=true;$('#btn').disabled=true;
  $('#status').innerHTML='<span class="spin"></span>检索中…';
  $('#results').innerHTML='';$('#answer-box').innerHTML='';
  const t0=Date.now();
  try{
    // 一次性流式接口：先推检索结果，再逐字推 LLM 回答
    // 多轮: sid 存 localStorage,同会话追问服务端自动结合历史改写检索词
    if(!localStorage.xhsSid)localStorage.xhsSid=crypto.randomUUID();
    const resp=await fetch('/api/answer?q='+encodeURIComponent(q)
      +'&sid='+encodeURIComponent(localStorage.xhsSid));
    const reader=resp.body.getReader(),dec=new TextDecoder();
    let buf='';
    for(;;){
      const {done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');buf=lines.pop();
      for(const ln of lines){
        if(!ln.startsWith('data: '))continue;
        let d;try{d=JSON.parse(ln.slice(6))}catch(e){continue}
        if(d.type==='meta'){
          $('#status').textContent='检索到 '+d.results.length+' 条，正在生成回答…';
          if(d.rewritten)$('#status').innerHTML=
            '<span class="spin"></span>按「'+d.rewritten+'」检索到 '+
            d.results.length+' 条，正在生成回答…';
          render(d.results);
        }else if(d.type==='delta'){
          let b=$('#answer-box .body');
          if(!b){$('#answer-box').innerHTML=
            '<div class="answer"><div class="hd">🤖 AI 回答<span class="tag" id="amodel"></span></div><div class="body"></div></div>';
            b=$('#answer-box .body');
            if(d.model)$('#amodel').textContent=d.model;
          }
          b.appendChild(document.createTextNode(d.text));
        }else if(d.type==='notice'){
          const n=document.createElement('div');n.className='notice';
          n.textContent=d.message;
          $('#answer-box').appendChild(n);
        }else if(d.type==='error'){  // 调试模式: 服务端抛异常, 带完整 traceback
          showError(d.message,d.trace,d.q);
        }else if(d.type==='done'){
          // 生成完毕，把正文里的 [n] 统一渲染成可点击角标
          const b=$('#answer-box .body');
          if(b)b.innerHTML=withCites(b.textContent);
          const secs=((Date.now()-t0)/1000).toFixed(1);
          $('#status').textContent='共耗时 '+secs+' 秒（检索 '+d.search_secs
            +' 秒 + 生成 '+d.llm_secs+' 秒）';
        }
      }
    }
    if(!$('#answer-box .body'))$('#status').textContent='没有相关内容，换个关键词试试';
  }catch(e){showError('请求失败：'+e,null)}
  finally{loading=false;$('#btn').disabled=false}
}
// ── 调试模式: 错误卡片(查看 traceback / 复制报告 / 跳转 WorkBuddy) ──
let lastReport='';
function showError(msg,trace,q){
  $('#status').textContent='';$('#answer-box').innerHTML='';$('#results').innerHTML='';
  const lines=['======== xhs-rag 错误报告 ========',
    '时间: '+new Date().toLocaleString('zh-CN',{hour12:false}),
    '问题: '+(q||$('#q').value||'(空)'),
    '错误: '+msg];
  if(trace&&trace.length)lines.push('Traceback:',...trace);
  lines.push('====================================',
    '本报告已自动保存到 data/debug/last_error.txt。',
    '打开 WorkBuddy 说「修复上次的错误」即可，无需复制粘贴。');
  lastReport=lines.join('\\n');
  const box=document.createElement('div');box.className='errbox';
  box.innerHTML=
    '<div class="msg">⚠️ '+esc(msg)+'</div>'+
    '<div class="ops">'+
      '<button onclick="toggleTrace(this)">查看错误详情</button>'+
      '<button onclick="copyReport(this)">复制错误报告</button>'+
      '<button onclick="goWorkbuddy(this)">去 WorkBuddy 修复</button>'+
    '</div>'+
    '<div class="trace"><pre>'+esc(trace?trace.join('\\n'):'(无 traceback，请查看 data/logs/xhs-rag.log)')+'</pre></div>'+
    '<div class="copied"></div>';
  $('#answer-box').appendChild(box);
}
function toggleTrace(btn){
  const t=btn.closest('.errbox').querySelector('.trace');
  const show=t.style.display!=='block';t.style.display=show?'block':'none';
  btn.textContent=show?'收起错误详情':'查看错误详情';
}
async function copyReport(btn){
  try{await navigator.clipboard.writeText(lastReport)}
  catch(e){const ta=document.createElement('textarea');ta.value=lastReport;
    document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove()}
  const box=btn.closest('.errbox');const c=box.querySelector('.copied');
  c.textContent='✅ 错误报告已复制';c.style.display='block';
}
function goWorkbuddy(btn){
  copyReport(btn).then(()=>{
    const box=btn.closest('.errbox');const c=box.querySelector('.copied');
    c.textContent='✅ 已复制！到 WorkBuddy 说「修复上次的错误」，或直接粘贴此报告';
    c.style.display='block';
    if(box.scrollIntoView)box.scrollIntoView({behavior:'smooth',block:'nearest'});
  });
}
function render(items){
  const box=$('#results');box.innerHTML='';
  if(!items.length){box.innerHTML='<div class="card">没有相关内容，换个关键词试试</div>';return}
  items.forEach((it,i)=>{
    const d=document.createElement('div');d.className='card';d.id='r'+(i+1);
    const kind=it.note_type==='video'?'视频':'图文';
    d.innerHTML=
      '<a class="title" href="'+it.url+'" target="_blank">'+esc(it.title)+'</a>'+
      '<div class="meta"><span class="badge score">['+(i+1)+'] '+it.score+'</span>'+
      '<span class="badge kind">'+kind+'</span>'+
      (it.section?'<span class="badge kind">'+esc(it.section)+'</span>':'')+'</div>'+
      '<div class="text">'+esc(it.text)+'</div>';
    box.appendChild(d);
  });
}
// 引用角标 [n] → 可点击上标，点击滚动到对应卡片
function withCites(txt){
  return esc(txt).replace(/\\[(\\d+)\\]/g,(m,n)=>'<sup class="cite" data-n="'+n+'">'+n+'</sup>');
}
document.addEventListener('click',e=>{
  const s=e.target.closest('sup.cite');if(!s)return;
  const card=document.getElementById('r'+s.dataset.n);
  if(!card)return;
  document.querySelectorAll('.card.hl').forEach(c=>c.classList.remove('hl'));
  card.classList.add('hl');
  card.scrollIntoView({behavior:'smooth',block:'center'});
});
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
$('#q').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
(async()=>{try{const r=await fetch('/api/stats');const d=await r.json();
  $('#stat').textContent='共 '+d.notes+' 篇 · '+d.chunks+' chunks';
  $('#foot').textContent='笔记 '+d.notes+' · 图片 OCR '+d.images+' · 视频 '+d.videos+' · 转写 '+d.asr_chars+' 字';
}catch(e){}})();
</script>
</body>
</html>
"""


def lan_ip() -> str:
    """获取本机局域网 IP(优先 192.168 段)。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    # SSE 需要长连接 + chunked，HTTP/1.0 会每次关连接
    protocol_version = "HTTP/1.1"
    retriever: Retriever = None
    db: DB = None
    started_at: float = time.time()
    db_path: str = ""
    answerer = None  # qa.Answerer，未配置则 None
    debug: bool = False  # 调试模式: 错误响应携带完整 traceback, 前端可查看/复制/跳 WorkBuddy
    # 多轮对话会话存储: sid -> [{role, content}] (扁平 messages, 最新在后)
    # 内存态, 重启即清空; 每会话最多保留 6 轮, 30 分钟无活动整段过期
    sessions: dict = {}
    SESSION_TTL = 1800
    SESSION_MAX_ROUNDS = 6

    def _db_conn(self):
        """每请求新建 sqlite 连接(sqlite 连接不能跨线程)。"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_message(self, fmt, *args):  # 静默访问日志
        logger.debug(fmt % args)

    # ── SSE ────────────────────────────────────────────────
    def _sse_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")  # 关掉 nginx 缓冲
        # ★ 标准库不会自动分块：不声明 chunked 的话客户端会一直等 body
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj: dict):
        """推一条 SSE 事件（手动 chunked 编码）。
        客户端中途断开会抛 ConnectionResetError / BrokenPipeError，由调用方吞掉。"""
        data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        """写终止 chunk，告诉客户端流结束。"""
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def _enrich(self, results: list[dict]):
        """补 url / note_type（独立 sqlite 连接，避免跨线程）。"""
        if not results:
            return results
        conn = self._db_conn()
        try:
            for r in results:
                note = conn.execute(
                    "SELECT url, note_type FROM notes WHERE note_id=?",
                    (r["note_id"],)).fetchone()
                r["url"] = note["url"] if note else ""
                r["note_type"] = note["note_type"] if note else "note"
        finally:
            conn.close()
        return results

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── 调试模式: 错误报告 ────────────────────────────────
    debug_dir: str = ""  # serve() 里赋值为 data/debug，报错自动落盘

    def _dump_error(self, text: str):
        """错误报告落盘到 data/debug/last_error.txt（固定文件名，WorkBuddy 可直接读取）。

        用户侧闭环：Web 页面报错 → 自动存盘 → 打开 WorkBuddy 说
        「修复上次的错误」→ 直接读此文件定位修复，无需复制粘贴。"""
        if not self.debug or not self.debug_dir:
            return
        try:
            p = Path(self.debug_dir) / "last_error.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            logger.warning(f"错误报告已落盘: {p}")
        except Exception:
            pass  # 落盘失败不影响主流程

    def _err_body(self, where: str, exc: Exception) -> dict:
        """构造错误响应体。调试模式携带完整 traceback, 生产模式只给消息。"""
        import traceback as tb

        body = {"error": f"{where}: {exc}"}
        if self.debug:
            body["trace"] = tb.format_exc().splitlines()
            body["debug"] = True
            self._dump_error(self._report_text("", exc, self.path))
        return body

    def _report_text(self, q: str, exc: Exception, path: str = "") -> str:
        """生成可直接粘贴给 WorkBuddy 的错误报告文本。"""
        import datetime
        import traceback as tb

        lines = [
            "======== xhs-rag 错误报告 ========",
            f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"接口: {path or self.path}",
            f"问题: {q}",
            f"错误: {type(exc).__name__}: {exc}",
            "Traceback:",
        ]
        lines += tb.format_exc().splitlines()
        lines.append("====================================")
        lines.append("本报告已自动保存到 data/debug/last_error.txt。")
        lines.append("打开 WorkBuddy 说「修复上次的错误」即可，无需复制粘贴。")
        return "\n".join(lines)

    def _html(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self._html(PAGE.encode("utf-8"))
        elif url.path == "/api/search":
            q = (parse_qs(url.query).get("q") or [""])[0].strip()
            if not q:
                self._json({"error": "缺少 q 参数"}, 400)
                return
            t0 = time.time()
            try:
                results = self._enrich(self.retriever.search(q))
                self._json({"q": q, "secs": round(time.time() - t0, 1),
                            "results": results})
            except Exception as e:
                logger.exception("搜索失败")
                self._json(self._err_body("搜索失败", e), 500)
        elif url.path == "/api/answer":
            self._handle_answer(url)
        elif url.path == "/api/stats":
            self._json(self._stats())
        elif url.path == "/api/health":
            # 健康检查: Docker healthcheck / 监控探活。不触发模型推理, 秒回。
            try:
                lancedb = __import__("lancedb")
                tbl = lancedb.connect(str(
                    self.retriever.lance_dir)).open_table(self.retriever.table_name)
                chunks = tbl.count_rows()
            except Exception:
                chunks = -1
            self._json({
                "status": "ok",
                "chunks": chunks,
                "llm": self.answerer.provider if getattr(self, "answerer", None) else None,
                "debug": bool(getattr(self, "debug", False)),
                "uptime_secs": round(time.time() - self.started_at),
            })
        else:
            # HTTP/1.1 下必须给 Content-Length，否则客户端会一直等 body
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _handle_answer(self, url):
        """检索 + LLM 问答，SSE 流式：
        先推 meta(检索结果) → 逐字推 delta(回答) → done(耗时)。
        LLM 不可用时只推 notice，检索结果照常返回。
        多轮: 带 sid 时取会话历史; 追问式 query 先经 LLM 改写成独立检索词。"""
        q = (parse_qs(url.query).get("q") or [""])[0].strip()
        sid = (parse_qs(url.query).get("sid") or [""])[0].strip()
        if not q:
            self._json({"error": "缺少 q 参数"}, 400)
            return

        # ── 多轮: 取会话历史(过期整段清理), 判断是否需要改写检索词 ──
        history = self.sessions.get(sid, []) if sid else []
        if history and time.time() - history[0].get("_ts", 0) > self.SESSION_TTL:
            self.sessions.pop(sid, None)
            history = []
        history = [h for h in history if h.get("role") in ("user", "assistant")]

        rewritten = ""
        if history and self.answerer is not None:
            try:
                if self.answerer.needs_rewrite(q, history):
                    rewritten = self.answerer.rewrite_query(q, history)
            except Exception as e:
                logger.warning(f"改写环节异常,按原 query 检索: {e}")

        search_q = rewritten or q  # 检索用改写词; 生成仍用原始 q(历史已注入)
        self._sse_head()
        try:
            t0 = time.time()
            results = self._enrich(self.retriever.search(search_q))
            search_secs = round(time.time() - t0, 1)
            meta = {"type": "meta", "q": q, "secs": search_secs,
                    "results": results}
            if rewritten and rewritten != q:
                meta["rewritten"] = rewritten
            self._sse(meta)
            if not results:
                self._sse({"type": "done", "search_secs": search_secs,
                           "llm_secs": 0})
                return

            answerer = self.answerer
            if answerer is None:
                self._sse({"type": "notice",
                           "message": "未启用 LLM 问答（llm.enabled 为 false）"})
                self._sse({"type": "done", "search_secs": search_secs,
                           "llm_secs": 0})
                return
            ok, why = answerer.available()
            if not ok:
                self._sse({"type": "notice",
                           "message": f"跳过 AI 回答：{why}"})
                self._sse({"type": "done", "search_secs": search_secs,
                           "llm_secs": 0})
                return

            t1 = time.time()
            first = True
            answer_buf: list[str] = []
            for piece in answerer.stream(q, results, history or None):
                answer_buf.append(piece)
                evt = {"type": "delta", "text": piece}
                if first:  # 首块带上模型名，用于 UI 角标
                    evt["model"] = answerer.model
                    first = False
                self._sse(evt)
            # ── 多轮: 回答完成, 写回会话(最多保留 N 轮) ──
            if sid:
                sess = self.sessions.setdefault(sid, [])
                sess.append({"role": "user", "content": q, "_ts": time.time()})
                sess.append({"role": "assistant",
                             "content": "".join(answer_buf)})
                # 截断到最近 N 轮(user+assistant 成对)
                keep = self.SESSION_MAX_ROUNDS * 2
                if len(sess) > keep:
                    del sess[:-keep]
            self._sse({"type": "done", "search_secs": search_secs,
                       "llm_secs": round(time.time() - t1, 1)})
        except Exception as e:  # 含 LLMUnavailable 与客户端断开
            logger.warning(f"问答失败: {e}")
            try:
                if self.debug:
                    self._dump_error(self._report_text(q, e, self.path))
                    self._sse({"type": "error", "message": f"AI 回答失败：{e}",
                               "trace": __import__("traceback").format_exc().splitlines(),
                               "q": q, "path": self.path})
                else:
                    self._sse({"type": "notice", "message": f"AI 回答失败：{e}"})
                self._sse({"type": "done", "search_secs": 0, "llm_secs": 0})
            except Exception:
                pass  # 客户端已断开，忽略
        finally:
            self._sse_end()  # 所有出口都补终止 chunk

    def _note_type(self, nid: str) -> str:
        if not self.db_path:
            return "note"
        conn = self._db_conn()
        try:
            note = conn.execute(
                "SELECT note_type FROM notes WHERE note_id=?", (nid,)).fetchone()
            return note["note_type"] if note else "note"
        finally:
            conn.close()

    def _stats(self) -> dict:
        if not self.db_path:
            return {"notes": 0, "images": 0, "videos": 0, "asr_chars": 0, "chunks": 0}
        import lancedb

        conn = self._db_conn()
        try:
            notes = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
            images = conn.execute(
                "SELECT COUNT(*) c FROM images WHERE ocr_done=1").fetchone()["c"]
            vids = conn.execute("SELECT COUNT(*) c FROM videos").fetchone()["c"]
            asr_chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(asr_text)),0) c FROM videos").fetchone()["c"]
        finally:
            conn.close()
        chunks = 0
        try:
            tbl = lancedb.connect(str(
                self.retriever.lance_dir)).open_table(self.retriever.table_name)
            chunks = tbl.count_rows()
        except Exception:
            pass
        return {"notes": notes, "images": images, "videos": vids,
                "asr_chars": asr_chars, "chunks": chunks}


def _build_answerer(cfg: Config):
    """构造 LLM 问答器。任何异常都不该拖垮 Web 服务 —— 降级为纯检索。"""
    try:
        from ..qa.answer import Answerer

        ans = Answerer(cfg)
        ok, why = ans.available()
        if ok:
            logger.info(f"LLM 问答已启用: {ans.provider} / {ans.model}"
                        f"（thinking={'on' if ans.thinking else 'off'}）")
        else:
            logger.warning(f"LLM 问答不可用，只提供检索结果：{why}")
        return ans
    except Exception as e:
        logger.warning(f"LLM 模块加载失败，只提供检索结果：{e}")
        return None


def serve(cfg: Config) -> int:
    """启动 Web 服务(模型预热 + 0.0.0.0 监听)。"""
    from ..store.db import DB

    db = DB(cfg.path("paths.db"))

    logger.info("预热模型(embedding + rerank,首次约 5 分钟)...")
    retriever = Retriever(cfg, db)
    t0 = time.time()
    retriever.warmup()
    logger.info(f"模型预热完成,耗时 {time.time()-t0:.0f}s")

    Handler.retriever = retriever
    Handler.db = db
    Handler.db_path = str(cfg.path("paths.db"))
    Handler.answerer = _build_answerer(cfg)
    Handler.debug = bool(cfg.get("serve.debug", False))  # 调试模式(错误详情+跳转修复)
    Handler.debug_dir = str(cfg.path("paths.data_dir") / "debug")  # 错误报告落盘目录

    host = cfg.get("serve.host", "0.0.0.0")
    port = int(cfg.get("serve.port", 8765))
    httpd = ThreadingHTTPServer((host, port), Handler)
    ip = lan_ip()
    if Handler.debug:
        logger.info("调试模式已开启: 接口报错时前端可查看 traceback / 复制错误报告 / 跳转修复")
    logger.success(
        f"Web UI 已启动: 本机 http://127.0.0.1:{port}  手机 http://{ip}:{port}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web UI 已停止")
    return 0
