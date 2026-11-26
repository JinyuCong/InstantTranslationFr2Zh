"""数字保护：把法语里容易译错的数字先确定性地换算好，再交给大模型，并在翻译后核对。

小模型最常见的严重错误是数量级换算：3,2 millions → 应为 320万，却译成 32万。
这里在送入大模型前，把
  - 「3,2 millions」「1,5 milliard」这类 → 320万 / 15亿
  - 「3 200 000」这类大整数 → 320万
  - 法语小数逗号「12,5」 → 12.5
先换算好（控制台显示的法语原文不变），翻译后再检查这些数字是否原样出现在译文里。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_SEP = "[   .]"   # 法语千位分隔：空格 / 不换行空格 / 窄空格 / 点
_INT = rf"\d{{1,3}}(?:{_SEP}\d{{3}})+|\d+"
_DEC = rf"(?:{_INT})(?:[,.]\d+)?"

_MAGNITUDE = re.compile(
    rf"(?<![\d,.])({_DEC})\s*(mille milliards|milliards?|millions?|billions?)\b(?:\s+d(?:e\s+|['’]\s*))?",
    re.IGNORECASE,
)
# 0 开头的长数字多半是电话号码/编号，不换算
_BIG_INT = re.compile(rf"(?<![\d,.])(?!0)(\d{{1,3}}(?:{_SEP}\d{{3}}){{2,}}|\d{{6,}})(?![\d,])")
_DEC_COMMA = re.compile(r"(?<![\d,])(\d+),(\d+)(?![\d,])")

_FACTORS = {
    "million": Decimal(10) ** 6,
    "milliard": Decimal(10) ** 9,
    "mille milliard": Decimal(10) ** 12,
    "billion": Decimal(10) ** 12,   # 法语 billion = 10^12（万亿）
}


def _to_decimal(s: str) -> Decimal | None:
    s = s.strip()
    if "," in s:                                   # 法语小数逗号：3,2
        int_part, frac = s.rsplit(",", 1)
    elif re.fullmatch(r"\d+\.(?:\d{1,2}|\d{4,})", s):  # 3.2 / 3.25（不是 3.200 这种千位分隔）
        int_part, frac = s.split(".")
    else:
        int_part, frac = s, ""
    try:
        return Decimal(re.sub(_SEP, "", int_part) + ("." + frac if frac else ""))
    except InvalidOperation:
        return None


def _fmt(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def to_chinese_amount(value: Decimal) -> str:
    """3200000 → 320万，1500000000 → 15亿，2.5e12 → 2.5万亿"""
    if value >= Decimal(10) ** 12:
        return _fmt(value / Decimal(10) ** 12) + "万亿"
    if value >= Decimal(10) ** 8:
        return _fmt(value / Decimal(10) ** 8) + "亿"
    if value >= Decimal(10) ** 4:
        return _fmt(value / Decimal(10) ** 4) + "万"
    return _fmt(value)


@dataclass
class NumberCheck:
    original: str   # 法语原文中的写法
    expected: str   # 译文中应出现的写法


def protect_numbers(text: str) -> tuple[str, list[NumberCheck]]:
    checks: list[NumberCheck] = []

    def magnitude(m: re.Match) -> str:
        num, unit = m.group(1), m.group(2).lower()
        base = _to_decimal(num)
        key = "mille milliard" if unit.startswith("mille") else unit.rstrip("s")
        if base is None or key not in _FACTORS:
            return m.group(0)
        zh = to_chinese_amount(base * _FACTORS[key])
        checks.append(NumberCheck(f"{num} {m.group(2)}", zh))
        return zh + " "

    def big_int(m: re.Match) -> str:
        value = _to_decimal(m.group(1))
        if value is None or value < 100000:
            return m.group(0)
        zh = to_chinese_amount(value)
        checks.append(NumberCheck(m.group(1), zh))
        return zh

    def dec_comma(m: re.Match) -> str:
        dot = f"{m.group(1)}.{m.group(2)}"
        checks.append(NumberCheck(m.group(0), dot))
        return dot

    text = _MAGNITUDE.sub(magnitude, text)
    text = _BIG_INT.sub(big_int, text)
    text = _DEC_COMMA.sub(dec_comma, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), checks


def verify_numbers(translation: str, checks: list[NumberCheck]) -> list[NumberCheck]:
    """返回译文中没有找到的数字。"""
    norm = re.sub(r"[\s,，]", "", translation)
    missing = []
    for c in checks:
        if c.expected.replace(",", "") not in norm:
            missing.append(c)
    return missing
