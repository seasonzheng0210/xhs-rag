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
<div id="results"></div>
<div class="footer" id="foot"></div>
<script>
let loading=false;
const $=s=>document.querySelector(s);
async function go(){
  const q=$('#q').value.trim();
  if(!q||loading)return;
  loading=true;$('#btn').disabled=true;
  $('#status').innerHTML='<span class="spin"></span>搜索中…';
  $('#results').innerHTML='';
  const t0=Date.now();
  try{
    const r=await fetch('/api/search?q='+encodeURIComponent(q));
    const d=await r.json();
    if(d.error){$('#status').textContent='出错了：'+d.error;return}
    const secs=((Date.now()-t0)/1000).toFixed(1);
    $('#status').textContent='找到 '+d.results.length+' 条，耗时 '+secs+' 秒';
    render(d.results);
  }catch(e){$('#status').textContent='请求失败：'+e}
  finally{loading=false;$('#btn').disabled=false}
}
function render(items){
  const box=$('#results');box.innerHTML='';
  if(!items.length){box.innerHTML='<div class="card">没有相关内容，换个关键词试试</div>';return}
  for(const it of items){
    const d=document.createElement('div');d.className='card';
    const kind=it.note_type==='video'?'视频':'图文';
    d.innerHTML=
      '<a class="title" href="'+it.url+'" target="_blank">'+esc(it.title)+'</a>'+
      '<div class="meta"><span class="badge score">'+it.score+'</span>'+
      '<span class="badge kind">'+kind+'</span>'+
      (it.section?'<span class="badge kind">'+esc(it.section)+'</span>':'')+'</div>'+
      '<div class="text">'+esc(it.text)+'</div>';
    box.appendChild(d);
  }
}
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
    retriever: Retriever = None
    db: DB = None
    db_path: str = ""

    def _db_conn(self):
        """每请求新建 sqlite 连接(sqlite 连接不能跨线程)。"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_message(self, fmt, *args):  # 静默访问日志
        logger.debug(fmt % args)

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                results = self.retriever.search(q)
                secs = round(time.time() - t0, 1)
                # 独立连接补全 url / 类型(避免 sqlite 跨线程)
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
                self._json({"q": q, "secs": secs, "results": results})
            except Exception as e:
                logger.exception("搜索失败")
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/stats":
            self._json(self._stats())
        else:
            self.send_response(404)
            self.end_headers()

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


def serve(cfg: Config) -> int:
    """启动 Web 服务(模型预热 + 0.0.0.0 监听)。"""
    from ..store.db import DB

    db = DB(cfg.path("paths.db"))

    logger.info("预热模型(embedding + rerank,首次约 5 分钟)...")
    retriever = Retriever(cfg, db)
    t0 = time.time()
    retriever.embedder.encode(["预热"])
    retriever._ensure_reranker()
    logger.info(f"模型预热完成,耗时 {time.time()-t0:.0f}s")

    Handler.retriever = retriever
    Handler.db = db
    Handler.db_path = str(cfg.path("paths.db"))

    host = cfg.get("serve.host", "0.0.0.0")
    port = int(cfg.get("serve.port", 8765))
    httpd = ThreadingHTTPServer((host, port), Handler)
    ip = lan_ip()
    logger.success(
        f"Web UI 已启动: 本机 http://127.0.0.1:{port}  手机 http://{ip}:{port}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web UI 已停止")
    return 0
