"""快速验证新笔记可搜索（独立运行）。"""
import sys, json, urllib.request, urllib.parse

q = "好单"
url = f"http://127.0.0.1:8765/api/search?q={urllib.parse.quote(q)}"
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.loads(r.read())
print(f"q={d['q']} secs={d['secs']} 命中{len(d['results'])}条")
for r in d['results'][:3]:
    print(f"  [{r['score']:.2f}] {r['title'][:40]}")