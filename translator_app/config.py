"""配置：config.yaml + 命令行覆盖。"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    source: str = "mic"                 # mic | loopback | both | file
    mic_device: str | int | None = None  # 设备序号或名称片段；None = 系统默认麦克风
    loopback_device: str | int | None = None  # 输出设备序号或名称片段；None = 系统默认扬声器
    file: str | None = None              # source=file 时的音频/视频文件
    file_realtime: bool = True           # 文件模式是否按真实时间速度播放


@dataclass
class VadConfig:
    threshold: float = 0.5        # Silero VAD 语音概率阈值
    min_silence_ms: int = 600     # 停顿多久认为一句话结束
    short_silence_ms: int = 250   # 句子已经很长时，用更短的停顿切分
    speech_pad_ms: int = 250      # 片段前后保留的余量，避免吞掉首尾音节
    min_speech_ms: int = 300      # 短于这个长度的声音（咳嗽、点击）丢弃
    soft_max_s: float = 12.0      # 超过这个长度开始使用 short_silence_ms
    max_segment_s: float = 25.0   # 硬上限：在最低语音概率处强制切分（Whisper 窗口为 30 s）


@dataclass
class AsrConfig:
    model: str = "large-v3-turbo"     # large-v3-turbo（快）| large-v3（最准，显存更多）| 本地路径
    device: str = "cuda"
    compute_type: str = "int8_float16"  # 4060 上 float16 / int8_float16 都可以
    language: str = "fr"
    beam_size: int = 5
    use_context_prompt: bool = True   # 用上文作为 Whisper 提示，改善专有名词与标点
    no_speech_threshold: float = 0.6
    download_dir: str = "models/whisper"
    hf_endpoint: str = ""             # 国内网络可设为 https://hf-mirror.com


@dataclass
class TranslationConfig:
    backend: str = "ollama"           # ollama | openai（任何 OpenAI 兼容服务：llama.cpp server、LM Studio、vLLM…）
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    model: str = "qwen2.5:7b"
    prompt_style: str = "auto"        # auto | chat | translategemma
    temperature: float = 0.1
    context_pairs: int = 4            # 带上前几句原文/译文，保证上下文连贯、术语一致
    num_ctx: int = 4096
    keep_alive: str = "60m"           # 让 Ollama 在会议间歇也不卸载模型
    timeout_s: float = 60.0
    merge_incomplete: bool = True     # 句子没说完（无句末标点）时等后半句一起翻译
    merge_wait_s: float = 0.8         # 未完结句等待下一段的基础时间
    merge_max_hold_s: float = 6.0     # 说话人一直在说时最多再等多久
    merge_max_chars: int = 400


@dataclass
class OutputConfig:
    show_french: bool = True
    color: bool = True
    transcript_dir: str = "transcripts"
    debug: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    glossary_file: str = "glossary.txt"


def _apply(obj: Any, data: dict, prefix: str = "") -> list[str]:
    unknown: list[str] = []
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in (data or {}).items():
        if key not in fields:
            unknown.append(prefix + key)
            continue
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if isinstance(value, dict):
                unknown += _apply(current, value, prefix + key + ".")
        else:
            setattr(obj, key, value)
    return unknown


def load_config(path: str | Path | None) -> tuple[Config, list[str]]:
    """返回 (配置, 未识别的键列表)。文件不存在时使用默认值。"""
    cfg = Config()
    unknown: list[str] = []
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        unknown = _apply(cfg, data)
    return cfg, unknown


def resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path)
