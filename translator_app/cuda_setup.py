"""让 CTranslate2 (faster-whisper) 在 Windows 上找到 pip 安装的 CUDA 12 / cuDNN 9 DLL。

ctranslate2 的 Windows 版本需要 cublas64_12.dll 以及 cuDNN 9 的各个子库，
这些库通过 `nvidia-cublas-cu12`、`nvidia-cudnn-cu12` 安装在 site-packages/nvidia/*/bin 下，
但默认不在 DLL 搜索路径里。必须在 import faster_whisper / ctranslate2 之前调用。
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> list[str]:
    if sys.platform != "win32":
        return []

    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:  # pragma: no cover - 某些嵌入式环境没有 getsitepackages
        pass
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))

    added: list[str] = []
    for root in dict.fromkeys(roots):
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
                d = str(bin_dir.resolve())
                if d not in added:
                    os.add_dll_directory(d)
                    added.append(d)

    if added:
        # cuDNN 会用 LoadLibrary 按 PATH 动态加载子库，所以 PATH 也要加上
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
    return added
