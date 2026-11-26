"""法语 → 中文 实时翻译（本地模型）

用法示例：
  python main.py                       # 现场：麦克风
  python main.py --source loopback     # 线上会议：翻译电脑正在播放的声音
  python main.py --source both         # 两路同时（请戴耳机，避免麦克风收到扬声器的声音）
  python main.py --file 录音.mp3        # 翻译音频/视频文件
  python main.py --list-devices        # 列出可用的麦克风和环回设备
  python main.py --check               # 自检：加载模型并翻译几句测试文本
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 必须在 import faster_whisper / ctranslate2 之前
from translator_app.cuda_setup import add_nvidia_dll_dirs  # noqa: E402

add_nvidia_dll_dirs()

from translator_app.config import load_config  # noqa: E402
from translator_app.console import Printer, setup_console  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="法语 → 中文 实时语音翻译（本地部署）")
    ap.add_argument("--config", default=str(BASE_DIR / "config.yaml"), help="配置文件（默认 config.yaml）")
    ap.add_argument("--source", choices=["mic", "loopback", "both", "file"],
                    help="mic=现场麦克风，loopback=线上会议/电脑播放的声音，both=两者，file=音频文件")
    ap.add_argument("--file", help="要翻译的音频/视频文件（自动使用 --source file）")
    ap.add_argument("--fast", action="store_true", help="文件模式下不按实时速度，尽快处理")
    ap.add_argument("--mic-device", help="麦克风序号或名称片段")
    ap.add_argument("--loopback-device", help="环回（输出）设备序号或名称片段")
    ap.add_argument("--llm", help="翻译模型，例如 qwen2.5:7b、translategemma:4b")
    ap.add_argument("--asr-model", help="Whisper 模型，例如 large-v3-turbo、large-v3")
    ap.add_argument("--hide-french", action="store_true", help="只显示中文")
    ap.add_argument("--no-merge", action="store_true", help="不等待半句话合并，延迟更低")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--debug", action="store_true", help="显示识别耗时、首字延迟、被过滤的幻觉文本")
    ap.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    ap.add_argument("--check", action="store_true", help="自检：加载模型、翻译测试句后退出")
    return ap.parse_args(argv)


def apply_args(cfg, args) -> None:
    if args.file:
        cfg.audio.source = "file"
        cfg.audio.file = args.file
    if args.source:
        cfg.audio.source = args.source
    if args.fast:
        cfg.audio.file_realtime = False
    if args.mic_device is not None:
        cfg.audio.mic_device = args.mic_device
    if args.loopback_device is not None:
        cfg.audio.loopback_device = args.loopback_device
    if args.llm:
        cfg.translation.model = args.llm
    if args.asr_model:
        cfg.asr.model = args.asr_model
    if args.hide_french:
        cfg.output.show_french = False
    if args.no_merge:
        cfg.translation.merge_incomplete = False
    if args.no_color:
        cfg.output.color = False
    if args.debug:
        cfg.output.debug = True


CHECK_SENTENCES = [
    "Bonjour à tous, merci d'être venus aussi nombreux ce matin.",
    "Le chiffre d'affaires a augmenté de 12,5 % au troisième trimestre, soit 3,2 millions d'euros.",
    "Je ne pense pas qu'on puisse livrer le projet avant la fin du mois de mars.",
]


def run_check(cfg, printer: Printer) -> int:
    import subprocess

    from translator_app.pipeline import App

    app = App(cfg, BASE_DIR, printer)
    translator, _asr = app.prepare()
    for s in CHECK_SENTENCES:
        printer.begin("测试", "检查", s)
        t0 = time.perf_counter()
        for piece in translator.translate_stream(s):
            printer.piece(piece)
        printer.end(f"（{time.perf_counter() - t0:.2f}s）")
        for c in translator.number_issues:
            printer.warn(f"数字核对：原文 {c.original} = {c.expected}，译文中未找到")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        printer.info(f"当前显存占用：{out}")
    except Exception:
        pass
    translator.unload()
    printer.ok("自检完成")
    return 0


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    cfg, unknown = load_config(args.config)
    apply_args(cfg, args)
    printer = Printer(color=cfg.output.color, show_french=cfg.output.show_french)
    for key in unknown:
        printer.warn(f"配置项未识别，已忽略：{key}")

    if args.list_devices:
        from translator_app.audio import list_devices, terminate_pyaudio
        print(list_devices())
        terminate_pyaudio()
        return 0

    from translator_app.translate import TranslationError

    try:
        if args.check:
            return run_check(cfg, printer)
        from translator_app.pipeline import App
        return App(cfg, BASE_DIR, printer).run()
    except TranslationError as e:
        printer.error(str(e))
        return 2
    except KeyboardInterrupt:
        printer.warn("已取消")
        return 130


if __name__ == "__main__":
    sys.exit(main())
