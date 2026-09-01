"""端到端验证调试模式：让 retriever 抛异常，确认 /api/search 响应携带 traceback。

用法: PYTHONPATH=src python scripts/_debug_mode_test.py
不依赖真实模型/数据库，用假 retriever 构造 500 场景。
"""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request

from xhs_rag.serve.server import Handler


class _BoomRetriever:
    """一调用就抛异常的假 retriever。"""

    def search(self, q):
        raise RuntimeError("模拟检索崩溃: embedding 模型未加载")


def _serve(port: int, debug: bool):
    Handler.retriever = _BoomRetriever()
    Handler.db = None
    Handler.db_path = ""
    Handler.answerer = None
    Handler.debug = debug
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test(port: int, debug: bool, label: str):
    httpd = _serve(port, debug)
    # ★ URL 必须百分号编码，urllib 不会自动编码中文
    from urllib.parse import quote

    q = quote("测试")
    req = Request(f"http://127.0.0.1:{port}/api/search?q={q}", method="GET")
    try:
        resp = urlopen(req, timeout=10)
        code = resp.status
        body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", 0)
        body = json.loads(e.read().decode("utf-8")) if hasattr(e, "read") else {}
    httpd.shutdown()

    print(f"\n=== {label} ===")
    print(f"HTTP {code}")
    print(f"字段: {sorted(body.keys())}")
    if debug:
        assert code == 500, "debug 模式应返回 500"
        assert "trace" in body, "debug 模式应携带 trace"
        assert "Traceback" in body["trace"][0], "trace 应为真实堆栈"
        assert body["debug"] is True
        print(f"error: {body['error'][:40]}")
        print(f"trace 行数: {len(body['trace'])}, 首行: {body['trace'][0]}")
        print("✅ debug=True  → 错误+完整 traceback 随响应下发")
    else:
        assert code == 500
        assert "trace" not in body, "生产模式不应暴露 trace"
        assert "debug" not in body
        print(f"error: {body['error'][:40]}")
        print("✅ debug=False → 只有错误消息，不暴露堆栈")


if __name__ == "__main__":
    test(8871, True, "调试模式(debug=true)")
    test(8872, False, "生产模式(debug=false)")
    print("\n全部通过 ✅")
