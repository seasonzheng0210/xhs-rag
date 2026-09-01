"""诊断 user_id 的取法。

M1 抓收藏列表要拼 /user/profile/{user_id} 的 URL，缺了它就得卡住。
一次会话里把三条候选路径全试一遍，看哪条能拿到，别反复试错浪费访问次数
（每次访问都在累加风控）。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger  # noqa: E402

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.core.logging import setup_logging  # noqa: E402
from xhs_rag.crawler.browser import BrowserSession  # noqa: E402

UID_RE = re.compile(r"/user/profile/([0-9a-fA-F]{24})")


def main() -> int:
    cfg = load_config()
    setup_logging("INFO", None)

    with BrowserSession(cfg, headless=bool(cfg.get("auth.headless", False)),
                        save_state=False) as ctx:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        # ── 1. /user/profile/me 是否 302 到真实 ID ──────────
        print("\n[1] /user/profile/me")
        try:
            page.goto("https://www.xiaohongshu.com/user/profile/me",
                      wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(2000)
            print(f"    最终 URL : {page.url}")
            m = UID_RE.search(page.url)
            print(f"    URL 提取 : {m.group(1) if m else '未命中'}")
        except Exception as e:
            print(f"    异常: {e}")

        # ── 2. 首页 HTML 里出现的所有 user_id ───────────────
        print("\n[2] 首页 HTML 里的 /user/profile/{id}")
        try:
            page.goto("https://www.xiaohongshu.com/explore",
                      wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(2500)
            html = page.content()
            ids = UID_RE.findall(html)
            print(f"    命中 {len(ids)} 处，去重后 {len(set(ids))} 个")
            for uid, n in Counter(ids).most_common(8):
                print(f"      {uid}  出现 {n} 次")
            # 侧边栏「我的主页」链接通常带自己的 ID
            for pat in (r'class="[^"]*side-bar[^"]*"[\s\S]{0,2000}?/user/profile/([0-9a-fA-F]{24})',
                        r'我的主页[\s\S]{0,500}?/user/profile/([0-9a-fA-F]{24})'):
                m = re.search(pat, html)
                if m:
                    print(f"    side-bar/我的主页 命中: {m.group(1)}")
                    break
        except Exception as e:
            print(f"    异常: {e}")

        # ── 3. __INITIAL_STATE__ / localStorage ─────────────
        print("\n[3] __INITIAL_STATE__ / localStorage")
        try:
            info = page.evaluate("""() => {
              const out = {stateKeys: null, userId: null, lsKeys: [], lsUser: null};
              try {
                const s = window.__INITIAL_STATE__;
                if (s) {
                  out.stateKeys = Object.keys(s);
                  const u = s.user && (s.user.userInfo || s.user.user || s.user);
                  if (u && u.id) out.userId = String(u.id);
                  if (s.user && s.user.userIdFromMct) out.userId = String(s.user.userIdFromMct);
                }
              } catch (e) { out.err = String(e); }
              try {
                for (let i = 0; i < localStorage.length; i++) out.lsKeys.push(localStorage.key(i));
                const raw = localStorage.getItem('USER_INFO') || localStorage.getItem('user');
                if (raw) out.lsUser = raw.slice(0, 200);
              } catch (e) {}
              return out;
            }""")
            print(f"    stateKeys: {info.get('stateKeys')}")
            print(f"    userId   : {info.get('userId')}")
            print(f"    lsKeys   : {info.get('lsKeys')}")
            print(f"    lsUser   : {info.get('lsUser')}")
        except Exception as e:
            print(f"    异常: {e}")

    logger.info("诊断结束。把拿到的 user_id 填到 config.yaml 的 auth.user_id 即可")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
