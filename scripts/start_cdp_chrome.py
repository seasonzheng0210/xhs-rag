"""CDP 模式一键启动脚本（B 计划日常运维入口）。

作用：启动带调试端口的 Chrome，供 xhs-rag 以 CDP 模式连接（真实指纹 + 真实登录态）。

★ 为什么必须这个脚本（2026-08-31 实测踩坑）：
  1. Chrome 151+ 安全策略：默认 profile 拒绝开 --remote-debugging-port，
     必须指定独立的 --user-data-dir。
  2. 直接把 user-data-dir 指向默认目录会撞同一限制，所以要复制一份 profile。
  3. 复制时机：用户默认 Chrome 里的登录态（web_session）失效时，重跑本脚本
     （--refresh-profile）会重新复制并保留最新 cookie。

用法：
  python scripts/start_cdp_chrome.py              # 启动（profile 不存在则自动复制）
  python scripts/start_cdp_chrome.py --refresh    # 强制重新复制 profile 后启动
  python scripts/start_cdp_chrome.py --check      # 只检查端口，不启动

注意：启动前需确保用户日常 Chrome 已完全退出（进程残留会让复制缺 Cookies 文件，
     见 2026-08-31 日志）。本脚本发现 Chrome 在跑会先尝试优雅关闭。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]
# 用户默认 profile（Chrome 安装的登录态所在）
DEFAULT_USER_DATA = Path.home() / r"AppData\Local\Google\Chrome\User Data"
# 项目内独立 profile（CDP 调试用）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CDP_PROFILE = PROJECT_ROOT / "data" / "chrome_cdp_profile"
CDP_PORT = 9222
ENDPOINT = f"http://127.0.0.1:{CDP_PORT}/json/version"

# 复制时排除的缓存目录（只有登录态相关文件才需要）
EXCLUDE_DIRS = [
    "Cache", "Code Cache", "GPUCache", "GrShaderCache", "DawnCache",
    "ShaderCache", "Crashpad", "Service Worker", "Dictionaries", "Component Updater",
]


def find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    return None


def chrome_running() -> int:
    """统计 chrome 进程数（0 = 未运行）。"""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return out.count("chrome.exe")
    except Exception:
        return -1


def kill_chrome() -> None:
    """优雅关闭 + 兜底强杀。复制 profile 必须等 Chrome 完全退出。"""
    subprocess.run(["taskkill", "/IM", "chrome.exe"], capture_output=True, text=True)
    time.sleep(3)
    if chrome_running() > 0:
        subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"],
                       capture_output=True, text=True)
        time.sleep(2)


def copy_profile() -> None:
    """把用户默认 profile 复制到项目目录（robocopy 镜像，排除缓存）。"""
    if not DEFAULT_USER_DATA.exists():
        sys.exit(f"[FAIL] 找不到默认 profile: {DEFAULT_USER_DATA}")
    if CDP_PROFILE.exists() and CDP_PROFILE.is_dir():
        shutil.rmtree(CDP_PROFILE, ignore_errors=True)

    ex = " ".join(f'/XD "{d}"' for d in EXCLUDE_DIRS)
    CDP_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f'robocopy "{DEFAULT_USER_DATA}" "{CDP_PROFILE}" /MIR /COPY:DAT '
           f"/R:1 /W:1 /NFL /NDL /NJH /NP {ex}")
    print(f"[..] 复制 profile → {CDP_PROFILE}")
    rc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if rc.returncode > 7:
        sys.exit(f"[FAIL] robocopy 失败: rc={rc.returncode}")
    cookies = CDP_PROFILE / "Default" / "Network" / "Cookies"
    if not cookies.exists():
        print("[WARN] Cookies 文件缺失！可能 Chrome 复制时仍在运行，请完全退出后 --refresh 重试")
    else:
        print(f"[OK ] profile 就绪（{cookies.stat().st_size} bytes Cookies）")


def port_open() -> bool:
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="CDP 模式 Chrome 启动器")
    ap.add_argument("--refresh", action="store_true", help="强制重新复制 profile")
    ap.add_argument("--check", action="store_true", help="只检查端口状态")
    args = ap.parse_args()

    if args.check:
        print("端口已开（CDP 可用）" if port_open() else "端口未开（Chrome 未以调试模式运行）")
        return 0 if port_open() else 1

    chrome = find_chrome()
    if not chrome:
        sys.exit("[FAIL] 未找到 Chrome")

    if args.refresh:
        if chrome_running() > 0:
            print("[..] Chrome 正在运行，先关闭…")
            kill_chrome()
        copy_profile()
    elif not CDP_PROFILE.exists():
        print("[..] profile 不存在，先复制（需先关 Chrome）…")
        if chrome_running() > 0:
            print("[..] Chrome 正在运行，先关闭…")
            kill_chrome()
        copy_profile()

    if port_open():
        print(f"[OK ] 9222 已在监听，直接复用（{ENDPOINT}）")
        return 0

    print(f"[..] 启动调试 Chrome（{chrome.name}）…")
    subprocess.Popen([
        str(chrome),
        f"--remote-debugging-port={CDP_PORT}",
        f'--user-data-dir="{CDP_PROFILE}"',
        "--no-first-run", "--no-default-browser-check",
    ])

    for _ in range(12):
        time.sleep(1)
        if port_open():
            print("[OK ] CDP 就绪：Chrome 已带调试端口运行")
            print(f"      profile: {CDP_PROFILE}")
            print("      之后跑 xhs-rag 命令时记得清代理环境变量（见 README/日志）")
            return 0
    print("[FAIL] 10 秒内未监听 9222，看 Chrome 窗口是否有报错")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
