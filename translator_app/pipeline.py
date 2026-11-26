"""主流程：音源 → VAD 切句 → Whisper 识别 → 合并半句 → LLM 流式翻译 → 控制台 / 记录文件。

线程：
  每个音源一个采集线程（读音频 + VAD 切句）
  一个识别线程（GPU，faster-whisper）
  一个翻译线程（Ollama，流式打印）
识别和翻译流水线并行：翻译第 N 句时已经在识别第 N+1 句。
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .asr import Transcriber
from .audio import FileSource, open_loopback, open_microphone, terminate_pyaudio
from .config import Config, resolve_path
from .console import Printer, Transcript
from .translate import TranslationError, Translator, load_glossary
from .vad import Segmenter, SpeechSegment

_STOP = object()
_ELLIPSIS_END = re.compile(r"(?:\.\.\.|…)\s*$")
_ELLIPSIS_START = re.compile(r"^\s*(?:\.\.\.|…)\s*")
_TERMINAL = re.compile(r"[.!?。！？](?:\s*[»\"”')\]])*\s*$")


def is_incomplete(text: str, forced: bool = False) -> bool:
    """没有句末标点、以省略号/逗号结尾，或者被强制切开的片段，认为句子还没说完。"""
    if forced or _ELLIPSIS_END.search(text):
        return True
    return not _TERMINAL.search(text)


def join_fragments(a: str, b: str) -> str:
    a = _ELLIPSIS_END.sub("", a).rstrip()
    b = _ELLIPSIS_START.sub("", b).strip()
    # Whisper 常把半句话末尾补成句号，下一段又以小写开头：去掉这个多余的句号
    if a.endswith(".") and b[:1].islower():
        a = a[:-1]
    return f"{a} {b}".strip()


@dataclass
class Utterance:
    text: str
    source: str
    start_time: float
    forced: bool = False
    first_ready: float = field(default_factory=time.monotonic)  # 第一段识别完成的时间（限制最长等待）

    def merged(self, other: "Utterance") -> "Utterance":
        return Utterance(join_fragments(self.text, other.text), self.source, self.start_time,
                         other.forced, self.first_ready)


class Activity:
    """记录每个音源是否正在说话、是否还有片段在识别中，用于决定半句话要不要继续等。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._speaking: dict[str, bool] = defaultdict(bool)
        self._inflight: dict[str, int] = defaultdict(int)

    def set_speaking(self, source: str, value: bool) -> None:
        with self._lock:
            self._speaking[source] = value

    def add_inflight(self, source: str, delta: int) -> None:
        with self._lock:
            self._inflight[source] += delta

    def busy(self, source: str) -> bool:
        with self._lock:
            return self._speaking[source] or self._inflight[source] > 0


class App:
    def __init__(self, cfg: Config, base_dir: Path, printer: Printer):
        self.cfg = cfg
        self.base = base_dir
        self.p = printer
        self.stop_event = threading.Event()
        self.asr_q: queue.Queue = queue.Queue()
        self.trans_q: queue.Queue = queue.Queue()
        self.activity = Activity()
        self.is_file = cfg.audio.source == "file"
        self._backlog_warned = 0.0

    # ------------------------------------------------------------------ 准备
    def _build_sources(self):
        a = self.cfg.audio
        if a.source == "file":
            if not a.file:
                raise SystemExit("文件模式需要 --file 路径")
            src = FileSource(a.file, realtime=a.file_realtime)
            self.p.info(f"音源：文件 {a.file}（{src.duration:.0f} 秒，{'实时速度' if a.file_realtime else '全速'}）")
            return [src]
        sources = []
        if a.source in ("mic", "both"):
            src = open_microphone(a.mic_device)
            self.p.info(f"音源 [{src.label}]：麦克风 {src.device['name']}")
            sources.append(src)
        if a.source in ("loopback", "both"):
            src = open_loopback(a.loopback_device)
            self.p.info(f"音源 [{src.label}]：电脑播放的声音（环回）{src.device['name']}")
            sources.append(src)
        if not sources:
            raise SystemExit(f"未知音源：{a.source}（可选 mic / loopback / both / file）")
        return sources

    def _format_ts(self, t: float) -> str:
        if self.is_file:
            t = int(t)
            return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
        return time.strftime("%H:%M:%S", time.localtime(t))

    def prepare(self) -> tuple[Translator, Transcriber]:
        cfg = self.cfg
        glossary = load_glossary(resolve_path(cfg.glossary_file, self.base))
        if glossary:
            self.p.info(f"术语表：{len(glossary)} 条")

        translator = Translator(cfg.translation, glossary)
        self.p.info(f"连接翻译模型 {cfg.translation.model}（提示词模式：{translator.style}）…")
        desc = translator.check()
        self.p.ok(f"翻译服务就绪：{desc}")

        self.p.info(f"加载语音识别模型 Whisper {cfg.asr.model}（{cfg.asr.compute_type}，首次运行会自动下载）…")
        t = time.perf_counter()
        asr = Transcriber(cfg.asr, [fr for fr, _ in glossary],
                          model_root=str(resolve_path(cfg.asr.download_dir, self.base)))
        asr.warmup()
        self.p.ok(f"语音识别就绪（{time.perf_counter() - t:.1f}s）")

        # 先加载 Whisper 再加载 LLM：Ollama 会根据剩余显存决定放多少层到 GPU，避免 Whisper 爆显存
        self.p.info("加载翻译模型到显存…")
        t = time.perf_counter()
        translator.warmup()
        share = translator.gpu_share()
        where = "" if share is None else f"，{share:.0%} 在 GPU 上"
        self.p.ok(f"翻译模型就绪（{time.perf_counter() - t:.1f}s{where}）")
        if share is not None and share < 0.99:
            self.p.warn("翻译模型有一部分在 CPU 上运行，速度会变慢。可关闭其他占用显存的程序（浏览器硬件加速、游戏等），"
                        "或换用更小的模型（--llm translategemma:4b）")
        return translator, asr

    # ------------------------------------------------------------------ 线程
    def _on_segment(self, seg: SpeechSegment) -> None:
        self.activity.add_inflight(seg.source, 1)
        self.asr_q.put(seg)
        if self.is_file and not self.cfg.audio.file_realtime:
            return  # 全速处理文件时排队是正常的
        backlog = self.asr_q.qsize()
        if backlog >= 4 and time.monotonic() - self._backlog_warned > 30:
            self._backlog_warned = time.monotonic()
            self.p.warn(f"识别队列积压 {backlog} 段，GPU 可能跟不上（可换更小的模型或 compute_type）")

    def _capture(self, src) -> None:
        clock = (lambda: src.position_s) if isinstance(src, FileSource) else time.time
        seg = Segmenter(self.cfg.vad, src.label, self._on_segment,
                        on_speech_state=lambda v: self.activity.set_speaking(src.label, v), clock=clock)
        try:
            while not self.stop_event.is_set():
                chunk = src.read(0.1)
                if chunk is None:
                    if src.finished:
                        break
                    seg.check_idle()
                    continue
                seg.feed(chunk)
            seg.flush()
        except Exception as e:
            self.p.error(f"[{src.label}] 音频采集出错：{e!r}")
        finally:
            src.stop()
            self.activity.set_speaking(src.label, False)

    def _recognize(self, asr: Transcriber) -> None:
        debug = self.cfg.output.debug
        while True:
            seg = self.asr_q.get()
            if seg is _STOP:
                self.trans_q.put(_STOP)
                return
            try:
                res = asr.transcribe(seg.audio)
                if debug:
                    extra = f"，丢弃幻觉：{res.dropped}" if res.dropped else ""
                    self.p.debug(f"[{seg.source}] 语音 {seg.duration:.1f}s{'（强制切分）' if seg.forced else ''}"
                                 f" → 识别 {res.seconds:.2f}s：{res.text!r}{extra}")
                if res.text:
                    self.trans_q.put(Utterance(res.text, seg.source, seg.start_time, seg.forced))
            except Exception as e:
                self.p.error(f"语音识别出错：{e!r}")
            finally:
                self.activity.add_inflight(seg.source, -1)

    def _translate_loop(self, translator: Translator, transcript: Transcript) -> None:
        tc = self.cfg.translation
        pending: Utterance | None = None
        deadline = 0.0
        while True:
            timeout = None if pending is None else max(0.0, deadline - time.monotonic())
            try:
                item = self.trans_q.get(timeout=timeout)
            except queue.Empty:
                # 说话人还在继续说这句话：再等等（有上限）
                now = time.monotonic()
                if self.activity.busy(pending.source) and now - pending.first_ready < tc.merge_max_hold_s:
                    deadline = now + 0.2
                    continue
                self._emit(pending, translator, transcript)
                pending = None
                continue

            if item is _STOP:
                if pending is not None:
                    self._emit(pending, translator, transcript)
                return

            if pending is not None:
                if item.source == pending.source and len(pending.text) + len(item.text) <= tc.merge_max_chars:
                    item = pending.merged(item)
                else:
                    self._emit(pending, translator, transcript)
                pending = None

            if (tc.merge_incomplete and not self.stop_event.is_set()
                    and is_incomplete(item.text, item.forced) and len(item.text) < tc.merge_max_chars):
                pending = item
                deadline = time.monotonic() + tc.merge_wait_s
            else:
                self._emit(item, translator, transcript)

    def _emit(self, utt: Utterance, translator: Translator, transcript: Transcript) -> None:
        ts = self._format_ts(utt.start_time)
        parts: list[str] = []
        err = None
        t0 = time.perf_counter()
        first = None
        self.p.begin(ts, utt.source, utt.text)
        try:
            for piece in translator.translate_stream(utt.text):
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(piece)
                self.p.piece(piece)
        except TranslationError as e:
            err = str(e)
        except Exception as e:  # 不让单句错误中断整个会议
            err = repr(e)
        finally:
            note = ""
            if self.cfg.output.debug and first is not None:
                note = f"（首字 {first:.2f}s / 共 {time.perf_counter() - t0:.2f}s）"
            self.p.end(note)
        chinese = "".join(parts).strip() or f"[翻译失败] {err or ''}"
        if err:
            self.p.error(f"翻译失败：{err}")
            transcript.add(ts, utt.source, utt.text, chinese)
            return
        if translator.leaks and translator.style == "chat":
            # 译文夹杂外语单词：自动重译一次并给出更正
            if self.cfg.output.debug:
                self.p.debug(f"译文夹杂外语：{translator.leaks}，重译")
            try:
                fixed = translator.retranslate(chinese)
                if fixed != chinese:
                    chinese = fixed
                    self.p.correction(fixed)
            except Exception as e:
                self.p.debug(f"重译失败：{e!r}")
        if translator.number_issues:
            fix = "；".join(f"{c.original} = {c.expected}" for c in translator.number_issues)
            self.p.warn(f"数字核对：译文可能有误，原文 {fix}")
            chinese += f"  ⚠ 数字核对：{fix}"
        transcript.add(ts, utt.source, utt.text, chinese)

    # ------------------------------------------------------------------ 运行
    def run(self) -> int:
        sources = self._build_sources()
        translator, asr = self.prepare()
        transcript = Transcript(resolve_path(self.cfg.output.transcript_dir, self.base),
                                f"音源：{self.cfg.audio.source}　识别：Whisper {self.cfg.asr.model}　"
                                f"翻译：{self.cfg.translation.model}")

        # 音频流必须在主线程打开：WASAPI（环回）需要在初始化 PortAudio/COM 的线程里打开，否则报 -9999
        try:
            for s in sources:
                s.start()
        except Exception as e:
            self.p.error(f"打开音频设备失败：{e!r}（用 --list-devices 查看设备，或在系统声音设置里检查设备是否被占用）")
            for s in sources:
                s.stop()
            transcript.close()
            transcript.path.unlink(missing_ok=True)
            translator.unload()
            terminate_pyaudio()
            return 1

        captures = [threading.Thread(target=self._capture, args=(s,), daemon=True, name=f"capture-{s.label}")
                    for s in sources]
        recognizer = threading.Thread(target=self._recognize, args=(asr,), daemon=True, name="asr")
        translator_t = threading.Thread(target=self._translate_loop, args=(translator, transcript),
                                        daemon=True, name="translate")
        for t in (*captures, recognizer, translator_t):
            t.start()

        self.p.ok("开始实时翻译（按 Ctrl+C 结束）")
        interrupted = False
        try:
            while any(t.is_alive() for t in captures):
                time.sleep(0.2)
        except KeyboardInterrupt:
            interrupted = True
            self.p.warn("正在停止：处理完已识别的内容后退出（再按一次 Ctrl+C 立即退出）")
            self.stop_event.set()
        try:
            for t in captures:
                t.join(timeout=3)
            self.asr_q.put(_STOP)
            while recognizer.is_alive() or translator_t.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.p.warn("立即退出")
        finally:
            transcript.close()
            if transcript.count:
                self.p.info(f"本次记录已保存：{transcript.path}")
            else:
                try:
                    transcript.path.unlink()
                except OSError:
                    pass
            translator.unload()  # 释放 Ollama 占用的显存
            terminate_pyaudio()
        return 130 if interrupted else 0
