"""流式语音活动检测（Silero VAD, ONNX）+ 按停顿切句。

直接复用 faster-whisper 自带的 Silero VAD ONNX 模型，不需要额外下载或 PyTorch。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from .audio import TARGET_SR
from .config import VadConfig

FRAME = 512          # Silero 在 16 kHz 下的帧长（32 ms）
CONTEXT = 64
FRAME_S = FRAME / TARGET_SR


def _find_bundled_model() -> Path:
    import faster_whisper

    assets = Path(faster_whisper.__file__).parent / "assets"
    models = sorted(assets.glob("silero_vad*.onnx"))
    if not models:
        raise FileNotFoundError(f"在 {assets} 中没有找到 Silero VAD 模型")
    return models[-1]


class SileroVAD:
    """逐帧（512 采样点）输出语音概率，内部保留 LSTM 状态。兼容两种 ONNX 接口：
    faster-whisper 版 (input, h, c) 和官方版 (input, state, sr)。"""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else _find_bundled_model()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)
        names = {i.name for i in self._sess.get_inputs()}
        if {"h", "c"} <= names:
            self._mode = "hc"
        elif {"state", "sr"} <= names:
            self._mode = "state"
        else:
            raise RuntimeError(f"不支持的 Silero VAD 模型接口：{sorted(names)}")
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.reshape(1, FRAME)], axis=1).astype(np.float32, copy=False)
        self._context = x[:, -CONTEXT:]
        if self._mode == "hc":
            out, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
        else:
            out, self._state = self._sess.run(
                None, {"input": x, "state": self._state, "sr": np.array(TARGET_SR, dtype=np.int64)}
            )
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class SpeechSegment:
    audio: np.ndarray
    source: str
    start_time: float              # 片段开始时间：实时音源为 time.time()，文件为文件内秒数
    forced: bool = False           # 因过长被强制切分（句子大概率没说完）
    created: float = field(default_factory=time.monotonic)

    @property
    def duration(self) -> float:
        return len(self.audio) / TARGET_SR


class Segmenter:
    """状态机：静音 → 检测到语音 → 持续语音 → 停顿足够长 → 输出一个片段。"""

    def __init__(self, cfg: VadConfig, source: str, on_segment: Callable[[SpeechSegment], None],
                 on_speech_state: Callable[[bool], None] | None = None,
                 vad: SileroVAD | None = None,
                 clock: Callable[[], float] = time.time):
        self.vad = vad or SileroVAD()
        self.clock = clock
        self.source = source
        self.on_segment = on_segment
        self.on_speech_state = on_speech_state or (lambda active: None)
        self.thr = cfg.threshold
        self.neg_thr = max(0.05, cfg.threshold - 0.15)
        f = lambda ms: max(1, int(round(ms / 1000 / FRAME_S)))  # noqa: E731
        self.min_silence = f(cfg.min_silence_ms)
        self.short_silence = f(cfg.short_silence_ms)
        self.pad = f(cfg.speech_pad_ms)
        self.min_speech = f(cfg.min_speech_ms)
        self.soft_max = f(cfg.soft_max_s * 1000)
        self.hard_max = f(cfg.max_segment_s * 1000)
        self.idle_flush_s = cfg.min_silence_ms / 1000

        self._rest = np.zeros(0, dtype=np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=self.pad)
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._triggered = False
        self._silence = 0
        self._speech = 0
        self._start_time = 0.0
        self._last_audio = time.monotonic()

    @property
    def in_speech(self) -> bool:
        return self._triggered

    # ------------------------------------------------------------------ 输入
    def feed(self, audio: np.ndarray) -> None:
        self._last_audio = time.monotonic()
        buf = np.concatenate([self._rest, audio]) if len(self._rest) else audio
        n = len(buf) // FRAME
        for i in range(n):
            self._process(buf[i * FRAME:(i + 1) * FRAME])
        self._rest = buf[n * FRAME:].copy()

    def check_idle(self) -> None:
        """WASAPI 环回在没有声音播放时不会送来任何数据，需要按时间主动结束当前句子。"""
        if self._triggered and time.monotonic() - self._last_audio >= self.idle_flush_s:
            self._finish(trailing_silence=0)

    def flush(self) -> None:
        if self._triggered:
            if len(self._rest):
                self._frames.append(np.pad(self._rest, (0, FRAME - len(self._rest))))
                self._rest = np.zeros(0, dtype=np.float32)
            self._finish(trailing_silence=0)

    # ------------------------------------------------------------------ 状态机
    def _process(self, frame: np.ndarray) -> None:
        p = self.vad(frame)
        if not self._triggered:
            if p >= self.thr:
                self._triggered = True
                self._frames = list(self._pre) + [frame]
                self._probs = [0.0] * len(self._pre) + [p]
                self._pre.clear()
                self._silence = 0
                self._speech = 1
                self._start_time = self.clock() - len(self._frames) * FRAME_S
                self.on_speech_state(True)
            else:
                self._pre.append(frame)
            return

        self._frames.append(frame)
        self._probs.append(p)
        if p >= self.thr:
            self._silence = 0
            self._speech += 1
        elif p < self.neg_thr:
            self._silence += 1

        need = self.min_silence if len(self._frames) < self.soft_max else self.short_silence
        if self._silence >= need:
            self._finish(trailing_silence=self._silence)
        elif len(self._frames) >= self.hard_max:
            self._force_split()

    def _emit(self, frames: list[np.ndarray], speech_frames: int, start_time: float, forced: bool) -> None:
        if speech_frames >= self.min_speech and frames:
            self.on_segment(SpeechSegment(np.concatenate(frames), self.source, start_time, forced))

    def _finish(self, trailing_silence: int) -> None:
        keep = len(self._frames) - max(0, trailing_silence - self.pad)
        self._emit(self._frames[:keep], self._speech, self._start_time, forced=False)
        tail = self._frames[keep:]
        self._frames, self._probs = [], []
        self._triggered = False
        self._silence = self._speech = 0
        self._pre.clear()
        self._pre.extend(tail[-self.pad:] if self.pad else [])
        self.on_speech_state(False)

    def _force_split(self) -> None:
        """句子过长：在后 40% 中语音概率最低（最像停顿）的位置切开，余下部分继续累积。"""
        n = len(self._frames)
        lo, hi = int(n * 0.6), n - self.pad
        probs = np.asarray(self._probs[lo:hi], dtype=np.float32)
        if len(probs) >= 5:
            probs = np.convolve(probs, np.ones(5, dtype=np.float32) / 5, mode="same")
        cut = lo + int(np.argmin(probs)) if len(probs) else n
        head, tail = self._frames[:cut], self._frames[cut:]
        head_speech = sum(1 for p in self._probs[:cut] if p >= self.thr)
        self._emit(head, head_speech, self._start_time, forced=True)
        self._frames = tail
        self._probs = self._probs[cut:]
        self._speech = sum(1 for p in self._probs if p >= self.thr)
        self._silence = 0
        self._start_time = self.clock() - len(self._frames) * FRAME_S
