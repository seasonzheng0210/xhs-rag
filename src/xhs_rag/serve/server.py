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

import base64
import json
import socket
import threading
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
/* ── P2-3 Web 端传图检索 ── */
.imgbtn{flex:0 0 auto;display:flex;align-items:center;justify-content:center;
  width:46px;font-size:19px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--card);cursor:pointer;user-select:none}
.imgbtn:hover{border-color:var(--accent)}
.imgbtn.busy{opacity:.5;pointer-events:none}
#imgfile{display:none}
.preview{position:relative;display:inline-block;margin-bottom:10px}
.preview img{max-height:110px;max-width:100%;border-radius:10px;
  border:1px solid var(--border);display:block}
.preview .x{position:absolute;top:-8px;right:-8px;width:22px;height:22px;
  border-radius:50%;background:#1f2328;color:#fff;border:none;cursor:pointer;
  font-size:12px;line-height:1;padding:0}
/* ── M10 Agent 深度思考开关 ── */
.deeprow{display:flex;align-items:center;gap:8px;font-size:12px;
  color:var(--muted);margin:-6px 0 12px}
.deeprow label{display:flex;align-items:center;gap:6px;cursor:pointer}
.deeprow input{width:15px;height:15px;accent-color:var(--accent);cursor:pointer}
/* Agent 中间步骤进度 */
.steps{background:#f8f8f6;border:1px solid var(--border);border-radius:10px;
  padding:8px 12px;margin-bottom:12px;font-size:12px;color:#555}
.steps .st{padding:2px 0;line-height:1.6}
.steps .st b{color:var(--accent);font-weight:600}
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
/* ── P1-1 告警横幅 ── */
#alertbar:empty{display:none}
.alert{background:#fef2f2;border:1px solid #fca5a5;color:#7f1d1d;
  padding:10px 12px;border-radius:10px;margin:10px 0;font-size:13px}
.alert .hd{font-weight:600;margin-bottom:6px;display:flex;
  justify-content:space-between;align-items:center;gap:8px}
.alert .hd button{background:#b91c1c;color:#fff;border:0;border-radius:6px;
  padding:3px 9px;font-size:12px;cursor:pointer;white-space:nowrap}
.alert .item{padding:4px 0;border-top:1px dashed #fca5a5}
.alert .item .ts{color:#9a3412;font-size:11px;margin-right:6px}
/* ── P1-2 同步健康面板 ── */
.syncpanel{margin:16px 0 4px;font-size:13px}
.syncpanel .hd{display:flex;justify-content:space-between;align-items:center;
  cursor:pointer;padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;
  border-radius:10px;font-weight:600}
.syncpanel .hd .meta{font-weight:400;color:#64748b;font-size:12px}
.syncpanel .body{border:1px solid #e2e8f0;border-top:0;border-radius:0 0 10px 10px;
  padding:8px 10px;display:none;background:#fff}
.syncpanel.open .body{display:block}
.syncpanel table{width:100%;border-collapse:collapse;font-size:12px}
.syncpanel th{text-align:left;color:#64748b;font-weight:500;padding:4px 6px;
  border-bottom:1px solid #e2e8f0;white-space:nowrap}
.syncpanel td{padding:5px 6px;border-bottom:1px solid #f1f5f9;
  font-variant-numeric:tabular-nums}
.syncpanel tr.fail td{background:#fef2f2}
.syncpanel tr.run td{background:#fffbeb}
.syncpanel .ok{color:#15803d}.syncpanel .bad{color:#b91c1c}
.syncpanel .warn{color:#b45309}
</style>
</head>
<body>
<header><h1>📚 收藏夹 RAG</h1><span class="sub" id="stat"></span></header>
<div id="alertbar"></div>
<div class="searchbox">
  <input id="q" placeholder="搜收藏夹里的内容…" autocomplete="off">
  <label class="imgbtn" for="imgfile" id="imgbtn" title="上传截图：识别图中文字后检索">📷</label>
  <input type="file" id="imgfile" accept="image/*" onchange="pickImage(this)">
  <button id="btn" onclick="go()">搜索</button>
</div>
<div id="preview"></div>
<div class="deeprow">
  <label><input type="checkbox" id="deep"> 深度思考</label>
  <span>Agent 多步检索，更准但慢 3-5 倍</span>
</div>
<div class="hint">试试：怎么护理宝宝私处 / 衣物清洗 / 月子喂养</div>
<div id="status"></div>
<div id="answer-box"></div>
<div id="results"></div>
<div class="syncpanel" id="syncpanel">
  <div class="hd" onclick="this.parentNode.classList.toggle('open');loadSync()">
    <span>🔄 同步健康</span><span class="meta" id="syncmeta">点击展开</span>
  </div>
  <div class="body" id="syncbody"></div>
</div>
<div class="footer" id="foot"></div>
<script>
let loading=false;
const $=s=>document.querySelector(s);
/* ── P2-3 传图检索：前端先压到 ≤1600px JPEG 再传 ──
   手机原图常 5MB+，base64 后约 6.7MB，既慢又可能触发体积限制；
   压到 1600px/0.85 后通常 <600KB，OCR 精度基本无损。 */
function shrink(file,maxSide,quality){
  return new Promise((res,rej)=>{
    const img=new Image(),url=URL.createObjectURL(file);
    img.onload=()=>{
      const w=img.naturalWidth,h=img.naturalHeight;
      const s=Math.min(1,maxSide/Math.max(w,h));
      const c=document.createElement('canvas');
      c.width=Math.round(w*s);c.height=Math.round(h*s);
      c.getContext('2d').drawImage(img,0,0,c.width,c.height);
      URL.revokeObjectURL(url);
      res(c.toDataURL('image/jpeg',quality));
    };
    img.onerror=()=>{URL.revokeObjectURL(url);rej(new Error('图片读取失败'))};
    img.src=url;
  });
}
async function pickImage(inp){
  const f=inp.files&&inp.files[0];if(!f)return;
  const btn=$('#imgbtn');btn.classList.add('busy');
  $('#status').innerHTML='<span class="spin"></span>识别图片文字…';
  try{
    const dataUrl=await shrink(f,1600,0.85);
    const r=await fetch('/api/ocr',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:dataUrl})});
    const d=await r.json();
    if(d.error){showError('图片识别失败：'+d.error,d.trace,null);return}
    if(!d.text){$('#status').textContent='图片里没识别出文字，换个更清晰的截图';return}
    $('#q').value=(($('#q').value.trim()+' '+d.text).trim()).slice(0,300);
    $('#preview').innerHTML='<div class="preview"><img src="'+dataUrl+'">'+
      '<button class="x" onclick="clearImage()">×</button></div>';
    $('#status').textContent='已识别（'+d.engine+' · '+d.secs+'s），可直接搜索';
  }catch(e){showError('图片上传失败：'+e,null,null)}
  finally{btn.classList.remove('busy');inp.value=''}
}
function clearImage(){$('#preview').innerHTML='';$('#status').textContent=''}
async function go(){
  const q=$('#q').value.trim();
  if(!q||loading)return;
  const deep=$('#deep').checked;
  loading=true;$('#btn').disabled=true;
  $('#status').innerHTML='<span class="spin"></span>'+(deep?'Agent 多步检索中…':'检索中…');
  $('#results').innerHTML='';$('#answer-box').innerHTML='';
  const t0=Date.now();
  try{
    // 一次性流式接口：先推检索结果，再逐字推 LLM 回答
    // 深度思考: 走 /api/agent(LangGraph 多步循环)，事件协议与 /api/answer
    //           兼容(meta/delta/notice/done/error)，额外有 step 事件报中间步骤
    // 多轮: sid 存 localStorage,同会话追问服务端自动结合历史改写检索词
    if(!localStorage.xhsSid)localStorage.xhsSid=crypto.randomUUID();
    const resp=await fetch((deep?'/api/agent':'/api/answer')+'?q='+encodeURIComponent(q)
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
        if(d.type==='step'){
          // Agent 模式专有：逐节点报进度，避免用户对着白屏等 30-60 秒
          let s=$('#answer-box .steps');
          if(!s){$('#answer-box').innerHTML='<div class="steps"></div>';
            s=$('#answer-box .steps')}
          const line=document.createElement('div');line.className='st';
          line.innerHTML=d.detail||('第 '+d.steps+' 步');
          s.appendChild(line);
          $('#status').textContent='Agent 执行中（第 '+d.steps+' 步 / 上限 '
            +(d.max_steps||8)+'）…';
        }else if(d.type==='meta'){
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
          if(d.steps!=null){  // Agent 模式: 报步数与工具调用次数
            $('#status').textContent='共耗时 '+secs+' 秒（Agent '+d.steps+' 步 · '
              +((d.tools||[]).length)+' 次工具调用）';
          }else{
            $('#status').textContent='共耗时 '+secs+' 秒（检索 '+d.search_secs
              +' 秒 + 生成 '+d.llm_secs+' 秒）';
          }
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
/* ── P1-1 告警横幅：未读告警置顶显示，可一键标记已读 ── */
async function loadAlerts(){
  try{
    const r=await fetch('/api/alerts');const d=await r.json();
    const bar=$('#alertbar');
    if(!d.unacked){bar.innerHTML='';return}
    let h='<div class="alert"><div class="hd"><span>⚠️ 有 '+d.unacked+' 条未读告警</span>'+
      '<button onclick="ackAlerts()">全部标记已读</button></div>';
    d.items.forEach(a=>{h+='<div class="item"><span class="ts">'+esc(a.ts)+'</span>'+
      '<b>'+esc(a.title)+'</b> —— '+esc(a.message)+'</div>'});
    bar.innerHTML=h+'</div>';
  }catch(e){}
}
async function ackAlerts(){
  try{await fetch('/api/alerts?ack=1');loadAlerts()}catch(e){}
}
/* ── P1-2 同步健康面板：读 sync_runs 最近 N 轮 ── */
let syncLoaded=false;
async function loadSync(){
  if(syncLoaded)return;syncLoaded=true;
  const body=$('#syncbody');
  try{
    const r=await fetch('/api/sync');const d=await r.json();
    $('#syncmeta').textContent=d.runs.length?('最近 '+d.runs.length+' 轮 · 失败 '+d.failed)+' 轮':'暂无记录';
    if(!d.runs.length){body.innerHTML='<div style="color:#64748b">还没有同步记录</div>';return}
    let h='<table><tr><th>开始时间</th><th>触发</th><th>状态</th><th>耗时</th><th>listed</th><th>indexed</th><th>说明</th></tr>';
    d.runs.forEach(r=>{
      const cls=r.status==='failed'?'fail':(r.status==='running'?'run':'');
      const st=r.status==='success'?'<span class="ok">成功</span>':
        (r.status==='running'?'<span class="warn">运行中</span>':'<span class="bad">失败</span>');
      h+='<tr class="'+cls+'"><td>'+esc(r.started||'')+'</td><td>'+esc(r.trigger||'')+'</td>'+
        '<td>'+st+'</td><td>'+(r.duration_s==null?'-':r.duration_s+'s')+'</td>'+
        '<td>'+(r.listed||0)+'</td><td>'+(r.indexed||0)+'</td>'+
        '<td>'+esc(r.error_msg||'')+'</td></tr>';
    });
    body.innerHTML=h+'</table>';
  }catch(e){body.innerHTML='<div style="color:#b91c1c">同步记录读取失败</div>'}
}
(async()=>{try{const r=await fetch('/api/stats');const d=await r.json();
  $('#stat').textContent='共 '+d.notes+' 篇 · '+d.chunks+' chunks';
  $('#foot').textContent='笔记 '+d.notes+' · 图片 OCR '+d.images+' · 视频 '+d.videos+' · 转写 '+d.asr_chars+' 字';
}catch(e){}})();
loadAlerts();
/* 有失败轮次时自动展开同步健康面板，省得用户自己找 */
(async()=>{try{
  const r=await fetch('/api/sync');const d=await r.json();
  if(d.failed>0||d.running>0){document.getElementById('syncpanel').classList.add('open');loadSync()}
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
    cfg: object = None  # P1-1/P1-2: 告警与同步健康需要读配置
    answerer = None  # qa.Answerer，未配置则 None
    debug: bool = False  # 调试模式: 错误响应携带完整 traceback, 前端可查看/复制/跳 WorkBuddy
    # 多轮对话会话存储: sid -> [{role, content}] (扁平 messages, 最新在后)
    # 内存态, 重启即清空; 每会话最多保留 6 轮, 30 分钟无活动整段过期
    sessions: dict = {}
    SESSION_TTL = 1800
    SESSION_MAX_ROUNDS = 6
    # Web 传图体积上限。前端已把手机原图压到 ≤1600px/JPEG(~600KB)，
    # 8MB 是给"绕过前端压缩的直传"留的余量；超过直接拒，不读进内存。
    MAX_UPLOAD_BYTES = 8 * 1024 * 1024
    ocr_ok: bool = False  # serve() 启动时主线程预热 OCR 的结果

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

    def do_POST(self):
        """POST 端点：目前只有 /api/ocr（图片上传）。

        /api/search 与 /api/answer 保持 GET（query 短，走 URL 更省事）；
        图片体积大必须走 body，故单独开一个 POST 端点。
        """
        url = urlparse(self.path)
        if url.path == "/api/ocr":
            self._handle_ocr(url)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _handle_ocr(self, url):
        """P2-3 查询端多模态(Web 侧)：把上传的截图 OCR 成检索词。

        与 CLI 的 `--image` 复用同一套 OcrEngine，不需要任何新依赖。

        ⚠️ OcrEngine 的模型必须在 serve 启动时于**主线程**预热（见 serve()）：
        Windows 上在工作线程里首次加载推理引擎会死等挂起，而 HTTP handler
        恰好跑在 ThreadingHTTPServer 的 worker 线程里。这里只做推理，不加载。
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise ValueError("空请求体")
            if n > self.MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"图片过大({n / 1048576:.1f}MB)，上限 "
                    f"{self.MAX_UPLOAD_BYTES // 1048576}MB")
            obj = json.loads(self.rfile.read(n).decode("utf-8"))
            data_url = str(obj.get("image") or "").strip()
            if not data_url.startswith("data:image/"):
                raise ValueError("image 需为 data:image/* 形式的数据 URL")
            t0 = time.time()
            # 直接复用查询端统一实现：它已经处理了 data URL 解码、临时文件
            # 落盘与清理、多行压平、300 字截断 —— 这里不再重复一遍。
            # 引擎是进程级共享实例，所以每张图不会重付 RapidOCR 加载开销。
            from ..process.ocr import ocr_image_to_query

            info = ocr_image_to_query(self.cfg, data_url)
            query = (info or {}).get("text") or ""
            if not query:
                logger.warning("上传图片没识别出可用文字")
            b64 = data_url.split(",", 1)[1] if "," in data_url else ""
            self._json({
                "text": query,
                "engine": (info or {}).get("engine", ""),
                "confidence": round(float((info or {}).get("confidence") or 0), 2),
                "bytes": len(b64) * 3 // 4,   # base64 → 原始字节数（估算）
                "secs": round(time.time() - t0, 1),
            })
        except Exception as e:
            logger.exception("图片 OCR 失败")
            self._json(self._err_body("图片识别失败", e), 400)

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
        elif url.path == "/api/agent":
            self._handle_agent(url)
        elif url.path == "/api/stats":
            self._json(self._stats())
        elif url.path == "/api/memory":
            # M11: 长期记忆状态(对话流水/摘要/画像), 只读不加工
            try:
                from ..memory import counts, digests, profiles

                self._json({
                    "counts": counts(self.db_path),
                    "digests": digests(self.db_path, 5),
                    "profiles": profiles(self.db_path, 20),
                })
            except Exception as e:
                self._json({"error": f"memory: {e}"}, 500)
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
        elif url.path == "/api/sync":
            # P1-2: 同步健康（读 sync_runs 最近 N 轮）。不触发模型推理。
            try:
                self._json(self._sync_health())
            except Exception as e:
                self._json({"error": f"sync: {e}"}, 500)
        elif url.path == "/api/alerts":
            # P1-1: 告警查询；带 ?ack=1 则全部标记已读
            try:
                from .. import notify as _n
                if parse_qs(url.query).get("ack"):
                    self._json({"acked": _n.ack_all(self.cfg)})
                    return
                self._json({"unacked": _n.unacked_count(self.cfg),
                            "items": _n.recent(self.cfg, 10)})
            except Exception as e:
                self._json({"error": f"alerts: {e}"}, 500)
        else:
            # HTTP/1.1 下必须给 Content-Length，否则客户端会一直等 body
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _handle_agent(self, url):
        """M10 Agent 多步问答（SSE）。

        与 /api/answer（单跳：检索 → 生成）的区别：走 LangGraph 决策循环，
        4 个只读工具、最多 8 步、步数耗尽有 finalize 兜底。

        事件协议与 /api/answer 保持兼容（meta/delta/notice/done/error），
        额外增加 step 事件报告中间步骤 —— 否则用户要对着白屏等 30-60 秒。
        记忆落盘由 RAGAgent 内部按 session 处理，这里不重复写。
        """
        q = (parse_qs(url.query).get("q") or [""])[0].strip()
        sid = (parse_qs(url.query).get("sid") or [""])[0].strip()
        if not q:
            self._json({"error": "缺少 q 参数"}, 400)
            return

        self._sse_head()
        try:
            from ..agent import RAGAgent

            agent = RAGAgent(self.cfg, verbose=False,
                             session=f"web:{sid}" if sid else "web")
            hits: list[dict] = []
            out: dict = {}
            for kind, payload in agent.stream(q):
                if kind == "step":
                    self._sse({"type": "step", "max_steps": agent.max_steps,
                               **payload})
                    if payload.get("hits"):
                        hits = payload["hits"]
                elif kind == "final":
                    out = payload

            # 引用卡片：用 Agent 实际检索到的命中补全 url / note_type
            if hits:
                self._sse({"type": "meta", "q": q, "secs": out.get("secs", 0),
                           "results": self._enrich(hits)})

            answer = out.get("answer") or ""
            if answer:
                self._sse({"type": "delta", "text": answer,
                           "model": getattr(self.answerer, "model", "")})
                # 多轮: 写回会话（与 /api/answer 共用同一套 session 存储）
                if sid:
                    sess = self.sessions.setdefault(sid, [])
                    sess.append({"role": "user", "content": q,
                                 "_ts": time.time()})
                    sess.append({"role": "assistant", "content": answer})
                    keep = self.SESSION_MAX_ROUNDS * 2
                    if len(sess) > keep:
                        del sess[:-keep]
            else:
                self._sse({"type": "notice",
                           "message": "Agent 没有产出回答（多半是收藏里没有相关内容）"})
            self._sse({"type": "done", "steps": out.get("steps", 0),
                       "tools": out.get("tool_calls", []),
                       "search_secs": 0, "llm_secs": out.get("secs", 0)})
        except Exception as e:  # 含客户端断开
            logger.warning(f"Agent 问答失败: {e}")
            try:
                if self.debug:
                    import traceback as _tb

                    self._dump_error(self._report_text(q, e, self.path))
                    self._sse({"type": "error", "message": f"Agent 失败：{e}",
                               "trace": _tb.format_exc().splitlines(),
                               "q": q, "path": self.path})
                else:
                    self._sse({"type": "notice", "message": f"Agent 失败：{e}"})
                self._sse({"type": "done", "steps": 0, "tools": [],
                           "search_secs": 0, "llm_secs": 0})
            except Exception:
                pass

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
        # ── M11: 用户提问原文落盘(零 LLM 成本, 离线 digest 消费) ──
        from ..memory import log_dialog
        log_dialog(self.db_path, "user", q, sid)

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
            # CRAG-lite: 低置信时用 LLM 改写重检一次(LLM 不可用则退纯检索)
            rewrite_fn = None
            if self.answerer is not None:
                rewrite_fn = lambda qq: self.answerer.rewrite_query(qq)  # noqa: E731
            results = self._enrich(self.retriever.search(search_q,
                                                         rewrite_fn=rewrite_fn))
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
            # ── M11: 注入长期记忆背景注记(摘要+画像, 空则不影响) ──
            memory_note = ""
            if self.db_path:
                try:
                    from ..memory import recent_context
                    memory_note = recent_context(self.db_path)
                except Exception as e:
                    logger.warning(f"记忆背景注记读取失败(忽略): {e}")
            for piece in answerer.stream(q, results, history or None,
                                         memory_note):
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
            # ── M11: 回答原文落盘(digest 消费), 失败不影响主流程 ──
            log_dialog(self.db_path, "assistant", "".join(answer_buf), sid)
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

    def _sync_health(self, limit: int = 10) -> dict:
        """P1-2: 读 sync_runs 最近 N 轮（纯 DB 读，无模型推理，秒回）。"""
        from datetime import datetime

        runs: list[dict] = []
        if self.db_path:
            conn = None
            try:
                # ⚠️ 必须走 _db_conn(): 本模块没有模块级 import sqlite3
                # (历史原因, 各处按需局部导入)。2026-09-16 冒烟抓到过
                # 这里直接写 sqlite3.connect 导致 NameError 被 except 吞掉,
                # 接口静默返回空列表 —— 面板永远显示"无记录"。
                conn = self._db_conn()
                cur = conn.execute(
                    "SELECT run_id, started_at, finished_at, trigger, status, "
                    "listed, indexed, error_msg FROM sync_runs "
                    "ORDER BY started_at DESC LIMIT ?", (limit,))
                for r in cur.fetchall():
                    d = dict(r)
                    st, fi = d.get("started_at"), d.get("finished_at")
                    status = d.get("status") or ""
                    # 僵尸运行态归一：正常一次同步是分钟级，超 6h 还挂 running
                    # 的必然是被杀/崩溃遗留（DB 侧下次 start_run 会正式收尸，
                    # 这里先保证面板不说谎）。2026-09-16 实测库里积了 3 条，
                    # 其中一条 392 小时，面板当时谎报"2 轮在运行中"。
                    if (status == "running" and st
                            and (time.time() * 1000 - st) > 6 * 3600 * 1000):
                        status = "aborted"
                    runs.append({
                        "run_id": (d.get("run_id") or "")[:8],
                        "started": (datetime.fromtimestamp(st / 1000)
                                    .strftime("%m-%d %H:%M") if st else ""),
                        "trigger": d.get("trigger") or "",
                        "status": status,
                        "duration_s": (round((fi - st) / 1000, 1)
                                       if (st and fi) else None),
                        "listed": d.get("listed") or 0,
                        "indexed": d.get("indexed") or 0,
                        "error_msg": (d.get("error_msg") or "")[:80],
                    })
            except Exception as e:
                logger.warning(f"同步健康读取失败: {e}")
            finally:
                if conn is not None:
                    conn.close()
        return {
            "runs": runs,
            "failed": sum(1 for r in runs if r["status"] == "failed"),
            "running": sum(1 for r in runs if r["status"] == "running"),
        }

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


def _warmup_ocr(cfg) -> bool:
    """主线程预热 OCR 引擎（供 Web 传图检索）。失败只降级，不拖垮服务。

    为什么必须在这里做：Windows 上在工作线程里首次加载推理引擎会死等挂起
    （同 mcp_server 顶部说明），而 HTTP handler 跑在 ThreadingHTTPServer 的
    worker 线程里 —— 若等首次传图才加载，那个请求会直接卡死。
    """
    try:
        from ..process.ocr import shared_ocr_engine

        t0 = time.time()
        # 必须走共享实例：否则这里预热的对象会被丢掉，
        # 而每次传图 ocr_image_to_query 又新建一个 → 预热白做、每张图重付加载。
        ok = shared_ocr_engine(cfg).warmup()
        logger.info(f"OCR 引擎预热{'完成' if ok else '失败(降级为云端 API)'}"
                    f",耗时 {time.time() - t0:.0f}s")
        return ok
    except Exception as e:
        logger.warning(f"OCR 预热异常，Web 传图将不可用：{e}")
        return False


def _startup_digest(cfg) -> None:
    """后台线程: 启动时消化一次未处理的对话记忆(digest)。

    失败/无 LLM 都只记日志, 不影响 serve 主流程。
    """
    try:
        from ..memory import run_digest

        out = run_digest(str(cfg.path("paths.db")), Handler.answerer,
                         verbose=False)
        if out.get("processed_blocks"):
            logger.info(f"启动记忆消化完成: {out}")
    except Exception as e:
        logger.warning(f"启动记忆消化失败(不影响服务): {e}")


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
    Handler.cfg = cfg  # P1-1/P1-2
    Handler.answerer = _build_answerer(cfg)
    Handler.debug = bool(cfg.get("serve.debug", False))  # 调试模式(错误详情+跳转修复)
    Handler.debug_dir = str(cfg.path("paths.data_dir") / "debug")  # 错误报告落盘目录

    # ── P2-3: Web 传图检索的 OCR 引擎必须在主线程预热（见 _warmup_ocr）──
    Handler.ocr_ok = _warmup_ocr(cfg)

    # ── M11: 启动后后台消化一次对话记忆(不阻塞 UI 启动/首问) ──
    if Handler.answerer is not None:
        threading.Thread(target=_startup_digest, args=(cfg,), daemon=True).start()

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
