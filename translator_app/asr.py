"""法语语音识别：faster-whisper（CTranslate2，GPU）。"""
from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import AsrConfig

# Whisper 在静音/噪声/音乐上常见的法语“幻觉”字幕（训练数据里的字幕署名等）
_HALLUCINATIONS = [re.compile(p, re.IGNORECASE) for p in (
    r"amara\.org",
    r"sous[- ]?titr\w*\s+(?:r[ée]alis[ée]e?s?|par|fait|fr\b|st\b|st'|soci[ée]t[ée]|de la communaut)",
    r"^\s*sous[- ]?titrage\b",
    r"radio[- ]canada\s*$",
    r"merci d'avoir regard[ée]",
    r"abonnez[- ]vous",
    r"n'oubliez pas de vous abonner",
    r"(?:like|pouce bleu).{0,30}(?:abonne|cha[îi]ne)",
    r"transcription (?:r[ée]alis[ée]e )?par",
)]
_FILLER = re.compile(
    r"^(?:\s|[,.!?…\-]|euh|heu|hum|hmm+|mm+|mh+|ben|bah|bon|ah|oh|eh|hein|voil[àa])*$", re.IGNORECASE
)
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def is_hallucination(text: str) -> bool:
    return any(p.search(text) for p in _HALLUCINATIONS)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not _HAS_LETTER.search(text) or _FILLER.match(text):
        return ""
    return text


@dataclass
class AsrResult:
    text: str
    seconds: float
    dropped: list[str]


class Transcriber:
    def __init__(self, cfg: AsrConfig, glossary_terms: list[str] | None = None, model_root: str | None = None):
        if cfg.hf_endpoint:
            os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model, device=cfg.device, compute_type=cfg.compute_type,
            download_root=model_root or cfg.download_dir,
        )
        terms = [t for t in (glossary_terms or []) if t]
        self._glossary_prompt = ("Vocabulaire : " + ", ".join(terms) + ".") if terms else ""
        self._history: deque[str] = deque(maxlen=6)

    def warmup(self) -> None:
        rng = np.random.default_rng(0)
        self.model.transcribe((rng.standard_normal(16000) * 0.001).astype(np.float32),
                              language=self.cfg.language, beam_size=1, without_timestamps=True)

    def _prompt(self) -> str | None:
        parts = []
        if self._glossary_prompt:
            parts.append(self._glossary_prompt)
        if self.cfg.use_context_prompt and self._history:
            ctx = " ".join(self._history)
            parts.append(ctx[-400:])
        return " ".join(parts) or None

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            task="transcribe",
            beam_size=self.cfg.beam_size,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=self.cfg.no_speech_threshold,
            condition_on_previous_text=False,
            initial_prompt=self._prompt(),
            without_timestamps=True,
            vad_filter=False,
        )
        kept, dropped = [], []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if (is_hallucination(text)
                    or seg.compression_ratio > 2.6
                    or (seg.no_speech_prob > 0.5 and seg.avg_logprob < -0.7)):
                dropped.append(text)
                continue
            kept.append(text)
        text = clean_text(" ".join(kept))
        if text:
            self._history.append(text)
        return AsrResult(text, time.perf_counter() - t0, dropped)
