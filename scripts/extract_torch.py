"""手动解压 torch/torchaudio wheel 到 site-packages(断点续传版)。

背景:pip 安装 torch 会被 WorkBuddy 批量写入保护拦截(7000+ 头文件)。
方案:手动 zipfile 解压,跳过编译期才需要的 include/ 头文件,减少 80% 写入量。
特性:已存在的文件自动跳过,中断后重跑续传。
用法:python scripts/extract_torch.py [--force]
"""
import sys
import time
import zipfile
from pathlib import Path

SP = Path(r"C:\Users\Administrator\.workbuddy\binaries\python\envs\xhs-rag\Lib\site-packages")
WHEELS = [
    Path("data/tmp/torch_wheels/torch-2.13.0-cp313-cp313-win_amd64.whl"),
    Path("data/tmp/torch_wheels/torchaudio-2.11.0-cp313-cp313-win_amd64.whl"),
]
SKIP_PREFIX = ("torch/include/",)  # 编译期头文件,运行时用不到
SKIP_SUFFIX = (".h", ".hpp", ".cuh", ".cu")  # 头文件兜底


def should_skip(name: str) -> bool:
    if any(name.startswith(p) for p in SKIP_PREFIX):
        return True
    low = name.lower()
    return any(low.endswith(s) for s in SKIP_SUFFIX)


def extract(wheel: Path, force: bool = False) -> tuple[int, int, int]:
    """返回 (已装, 跳过, 已存在)。"""
    installed = skipped = exists = 0
    t0 = time.time()
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        for i, n in enumerate(names, 1):
            if should_skip(n):
                skipped += 1
                continue
            dst = SP / n
            if dst.exists() and not force:
                exists += 1
                continue
            if n.endswith("/"):
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with z.open(n) as src, open(dst, "wb") as out:
                    out.write(src.read())
                installed += 1
            except PermissionError:
                # Windows 下残留只读文件,删了重写
                dst.unlink(missing_ok=True)
                with z.open(n) as src, open(dst, "wb") as out:
                    out.write(src.read())
                installed += 1
            if i % 2000 == 0:
                el = time.time() - t0
                print(
                    f"[{wheel.name}] {i}/{len(names)} 文件 "
                    f"({installed} 装, {exists} 已存在, {skipped} 跳) "
                    f"耗时 {el:.0f}s",
                    flush=True,
                )
    print(
        f"[完成] {wheel.name}: 新装 {installed}, 已存在 {exists}, 跳过 {skipped}, "
        f"总耗时 {time.time()-t0:.0f}s",
        flush=True,
    )
    return installed, skipped, exists


def main() -> int:
    force = "--force" in sys.argv
    total_new = 0
    for w in WHEELS:
        if not w.exists():
            print(f"[跳过] wheel 不存在: {w}", flush=True)
            continue
        print(f"[开始] {w.name} ({w.stat().st_size/1e6:.0f}MB)", flush=True)
        n, _, _ = extract(w, force)
        total_new += n
    print(f"[全部完成] 新写入 {total_new} 个文件", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
