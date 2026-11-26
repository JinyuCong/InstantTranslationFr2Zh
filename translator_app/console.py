"""控制台输出（彩色、流式）与会议记录文件。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

try:
    import colorama
    from colorama import Fore, Style
except ImportError:  # pragma: no cover
    colorama = None


class _NoColor:
    def __getattr__(self, name: str) -> str:
        return ""


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if colorama is not None:
        colorama.just_fix_windows_console()


class Printer:
    """译文流式输出期间，其它线程的提示信息先缓存，等这一句输出完再打印，避免串行和互相阻塞。"""

    INDENT = " " * 9

    def __init__(self, color: bool = True, show_french: bool = True):
        self.show_french = show_french
        use = color and colorama is not None
        self.F = Fore if use else _NoColor()
        self.S = Style if use else _NoColor()
        self._lock = threading.Lock()
        self._owner: int | None = None   # 正在流式输出译文的线程
        self._deferred: list[str] = []

    @staticmethod
    def _write(s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    def _msg(self, s: str) -> None:
        with self._lock:
            if self._owner is not None and self._owner != threading.get_ident():
                self._deferred.append(s)
            else:
                self._write(s)

    # ---------------------------------------------------------------- 普通消息
    def info(self, msg: str) -> None:
        self._msg(f"{self.F.CYAN}• {msg}{self.S.RESET_ALL}\n")

    def ok(self, msg: str) -> None:
        self._msg(f"{self.F.GREEN}✔ {msg}{self.S.RESET_ALL}\n")

    def warn(self, msg: str) -> None:
        self._msg(f"{self.F.YELLOW}! {msg}{self.S.RESET_ALL}\n")

    def error(self, msg: str) -> None:
        self._msg(f"{self.F.RED}✘ {msg}{self.S.RESET_ALL}\n")

    def debug(self, msg: str) -> None:
        self._msg(f"{self.S.DIM}  · {msg}{self.S.RESET_ALL}\n")

    # ---------------------------------------------------------------- 译文流
    def begin(self, ts: str, label: str, french: str) -> None:
        with self._lock:
            self._owner = threading.get_ident()
            head = f"{self.S.DIM}{ts}{self.S.RESET_ALL} {self.F.MAGENTA}[{label}]{self.S.RESET_ALL}"
            if self.show_french:
                self._write(f"\n{head} {self.S.DIM}{french}{self.S.RESET_ALL}\n"
                            f"{self.INDENT}{self.F.GREEN}{self.S.BRIGHT}")
            else:
                self._write(f"{head} {self.F.GREEN}{self.S.BRIGHT}")

    def piece(self, text: str) -> None:
        self._write(text.replace("\n", "\n" + self.INDENT))

    def correction(self, text: str) -> None:
        self._msg(f"{self.INDENT}{self.F.YELLOW}↳ 更正：{self.S.RESET_ALL}"
                  f"{self.F.GREEN}{self.S.BRIGHT}{text}{self.S.RESET_ALL}\n")

    def end(self, note: str = "") -> None:
        with self._lock:
            tail = f" {self.S.DIM}{note}" if note else ""
            self._write(f"{self.S.RESET_ALL}{tail}{self.S.RESET_ALL}\n")
            self._owner = None
            for s in self._deferred:
                self._write(s)
            self._deferred.clear()


class Transcript:
    """把双语结果追加写入 transcripts/日期_时间.md。"""

    def __init__(self, directory: Path, header: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime("%Y-%m-%d_%H-%M-%S.md")
        self._f = open(self.path, "a", encoding="utf-8")
        self._f.write(f"# 实时翻译记录 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{header}\n\n")
        self._f.flush()
        self.count = 0

    def add(self, ts: str, label: str, french: str, chinese: str) -> None:
        self._f.write(f"**{ts} [{label}]**  \n{french}  \n> {chinese}\n\n")
        self._f.flush()
        self.count += 1

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
