"""面试演示截图脚本 —— 用 Playwright 对本地查询服务截图。

产出（docs/ 下）：
  demo-01-首页.png          Web 界面首页（含统计信息）
  demo-02-白灼秋葵回答.png   真实查询：检索 + LLM 回答 + 原帖引用
  demo-03-沙茶酱缓存命中.png 第二个查询（命中 LRU 缓存）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
OUT = Path(__file__).resolve().parent.parent / "docs"


def wait_answer(page, timeout_s: float = 120.0) -> None:
    """轮询等待 #answer-box 出现回答，同时打印 #status 的进度文本。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = page.eval_on_selector("#status", "el => el.textContent")
        if status:
            print(f"  [status] {status.strip()}")
        try:
            body = page.eval_on_selector("#answer-box .body", "el => el.textContent")
            if body and len(body.strip()) > 10:
                print(f"  [answer] {body.strip()[:80]}...")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"等待回答超时 {timeout_s}s")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            print(f"chromium 启动失败: {e}", file=sys.stderr)
            return 2
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(BASE, wait_until="networkidle", timeout=30_000)
        time.sleep(1)

        stat = page.eval_on_selector("#stat", "el => el.textContent")
        print(f"[首页统计] {stat}")
        page.screenshot(path=str(OUT / "demo-01-首页.png"), full_page=False)

        # 查询 1：白灼秋葵（首次检索，展示真实耗时）
        page.fill("#q", "白灼秋葵的做法和要点")
        t0 = time.time()
        page.click("#btn")
        wait_answer(page)
        dt = time.time() - t0
        print(f"[查询1] 白灼秋葵 端到端 {dt:.1f}s")
        page.screenshot(path=str(OUT / "demo-02-白灼秋葵回答.png"), full_page=False)

        # 查询 2：沙茶酱（期望命中 LRU 缓存，检索≈0s）
        page.fill("#q", "沙茶酱怎么吃")
        t0 = time.time()
        page.click("#btn")
        wait_answer(page)
        dt = time.time() - t0
        print(f"[查询2] 沙茶酱 端到端 {dt:.1f}s")
        page.screenshot(path=str(OUT / "demo-03-沙茶酱缓存命中.png"), full_page=False)

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
