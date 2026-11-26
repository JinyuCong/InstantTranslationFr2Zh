"""音频输入：麦克风（现场）、WASAPI 环回（线上会议 / 电脑播放的声音）、音频文件。

所有音源统一输出 16 kHz 单声道 float32，供 VAD 与 Whisper 使用。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Protocol

import numpy as np
import soxr

TARGET_SR = 16000


class AudioSource(Protocol):
    label: str

    def start(self) -> None: ...
    def read(self, timeout: float) -> np.ndarray | None: ...
    def stop(self) -> None: ...
    @property
    def finished(self) -> bool: ...


# --------------------------------------------------------------------------- PyAudio
_pa_lock = threading.Lock()
_pa_instance = None


def _pyaudio():
    """共享一个 PyAudio 实例（PyAudioWPatch，支持 WASAPI loopback）。"""
    global _pa_instance
    import pyaudiowpatch as pyaudio

    with _pa_lock:
        if _pa_instance is None:
            _pa_instance = pyaudio.PyAudio()
        return pyaudio, _pa_instance


def terminate_pyaudio() -> None:
    global _pa_instance
    with _pa_lock:
        if _pa_instance is not None:
            try:
                _pa_instance.terminate()
            finally:
                _pa_instance = None


def _match(info: dict, selector: str | int) -> bool:
    if isinstance(selector, int) or (isinstance(selector, str) and selector.isdigit()):
        return info["index"] == int(selector)
    return str(selector).lower() in info["name"].lower()


class _StreamSource:
    """把 PyAudio 回调中的原始数据放进队列，在读取线程里做混音 + 重采样。"""

    def __init__(self, label: str, device: dict, channels: int, fallback_channels: int | None = None):
        self.label = label
        self.device = device
        self._channels = channels
        self._fallback_channels = fallback_channels
        self._rate = int(device["defaultSampleRate"])
        self._q: queue.Queue[bytes] = queue.Queue()
        self._stream = None
        self._resampler = (
            soxr.ResampleStream(self._rate, TARGET_SR, 1, dtype="float32", quality="HQ")
            if self._rate != TARGET_SR else None
        )

    @property
    def finished(self) -> bool:
        return False

    def _callback(self, in_data, frame_count, time_info, status):
        pyaudio, _ = _pyaudio()
        self._q.put(in_data)
        return (None, pyaudio.paContinue)

    def _open(self):
        pyaudio, pa = _pyaudio()
        return pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=self.device["index"],
            frames_per_buffer=max(256, self._rate // 50),  # 20 ms
            stream_callback=self._callback,
        )

    def start(self) -> None:
        try:
            self._stream = self._open()
        except OSError:
            # 某些设备（WASAPI 共享模式）只接受原生声道数
            if not self._fallback_channels or self._fallback_channels == self._channels:
                raise
            self._channels = self._fallback_channels
            self._stream = self._open()
        self._stream.start_stream()

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            data = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        # 一次把积压的数据都取出来，减少调用次数
        chunks = [data]
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        x = np.frombuffer(b"".join(chunks), dtype=np.float32)
        if self._channels > 1:
            x = x.reshape(-1, self._channels).mean(axis=1)
        if self._resampler is not None:
            x = self._resampler.resample_chunk(x)
        return np.ascontiguousarray(x, dtype=np.float32)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._resampler = None


def open_microphone(selector: str | int | None = None, label: str = "现场") -> _StreamSource:
    pyaudio, pa = _pyaudio()
    if selector is None or selector == "":
        info = pa.get_default_input_device_info()
    else:
        candidates = [
            pa.get_device_info_by_index(i) for i in range(pa.get_device_count())
        ]
        candidates = [
            d for d in candidates
            if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False) and _match(d, selector)
        ]
        if not candidates:
            raise RuntimeError(f"找不到麦克风设备：{selector!r}（用 --list-devices 查看）")
        # 同一设备会在多个 Host API 下出现，优先系统默认 API（MME，可任意声道/采样率）
        default_api = pa.get_default_host_api_info()["index"]
        candidates.sort(key=lambda d: d["hostApi"] != default_api)
        info = candidates[0]
    return _StreamSource(label, info, 1, fallback_channels=int(info["maxInputChannels"]))


def open_loopback(selector: str | int | None = None, label: str = "线上") -> _StreamSource:
    """WASAPI 环回：录制扬声器/耳机正在播放的声音（Teams、Zoom、腾讯会议、浏览器……）。"""
    pyaudio, pa = _pyaudio()
    loopbacks = list(pa.get_loopback_device_info_generator())
    if not loopbacks:
        raise RuntimeError("没有找到 WASAPI 环回设备")
    if selector is None or selector == "":
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            info = speakers
        else:
            match = [d for d in loopbacks if speakers["name"] in d["name"]]
            if not match:
                raise RuntimeError(f"找不到默认输出设备 {speakers['name']} 的环回设备")
            info = match[0]
    else:
        match = [d for d in loopbacks if _match(d, selector)]
        if not match:
            raise RuntimeError(f"找不到环回设备：{selector!r}（用 --list-devices 查看）")
        info = match[0]
    return _StreamSource(label, info, int(info["maxInputChannels"]))


def list_devices() -> str:
    pyaudio, pa = _pyaudio()
    lines = []
    try:
        default_in = pa.get_default_input_device_info()["index"]
    except Exception:
        default_in = -1
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = wasapi["defaultOutputDevice"]
    lines.append("== 麦克风（--source mic，--mic-device 序号或名称）==")
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False):
            api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            mark = " *默认" if i == default_in else ""
            lines.append(f"  [{i:>2}] {d['name']}  ({api}, {int(d['defaultSampleRate'])} Hz){mark}")
    lines.append("")
    lines.append("== 环回 / 线上会议声音（--source loopback，--loopback-device 序号或名称）==")
    default_name = pa.get_device_info_by_index(default_out)["name"]
    for d in pa.get_loopback_device_info_generator():
        mark = " *默认扬声器" if default_name in d["name"] else ""
        lines.append(f"  [{d['index']:>2}] {d['name']}  ({int(d['defaultSampleRate'])} Hz){mark}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 文件
def decode_file(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """用 PyAV 把任意音频/视频文件解码为单声道 float32。
    （faster-whisper 1.2.1 自带的 decode_audio 与 PyAV ≥17 不兼容，所以自己实现）"""
    import av

    chunks: list[np.ndarray] = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"文件里没有音轨：{path}")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class FileSource:
    """把音频/视频文件当作实时音源（用于测试和离线转写）。"""

    def __init__(self, path: str, realtime: bool = True, label: str = "文件", chunk_s: float = 0.1):
        self.label = label
        self._audio = decode_file(path)
        self._realtime = realtime
        self._chunk = int(TARGET_SR * chunk_s)
        self._pos = 0
        self._t0 = 0.0
        self._finished = False

    @property
    def duration(self) -> float:
        return len(self._audio) / TARGET_SR

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def position_s(self) -> float:
        return self._pos / TARGET_SR

    def start(self) -> None:
        self._t0 = time.monotonic()

    def read(self, timeout: float) -> np.ndarray | None:
        if self._pos >= len(self._audio):
            self._finished = True
            return None
        if self._realtime:
            due = self._t0 + self._pos / TARGET_SR
            wait = due - time.monotonic()
            if wait > timeout:
                time.sleep(timeout)
                return None
            if wait > 0:
                time.sleep(wait)
        chunk = self._audio[self._pos:self._pos + self._chunk]
        self._pos += len(chunk)
        return chunk

    def stop(self) -> None:
        self._pos = len(self._audio)
