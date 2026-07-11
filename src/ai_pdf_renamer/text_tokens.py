"""Token, filename text, and case conversion helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .rename_ops import FILENAME_RESERVED_WIN

MAX_NORMALIZED_KEYWORDS = 7


def chunk_text(text: str, *, chunk_size: int = 8000, overlap: int = 1000) -> list[str]:
    """Split text into overlapping chunks of chunk_size characters with overlap for LLM context windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    if not (text and text.strip()):
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def tokens_similar(token_a: str, token_b: str) -> bool:
    """Check if two tokens are similar."""
    a = token_a.lower()
    b = token_b.lower()
    if a == b:
        return True
    return bool((a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 2)


def _stripped_tokens(tokens: Iterable[str]) -> list[str]:
    return [token for token in (t.strip() for t in tokens) if token]


def _token_should_remain(token: str, remove_tokens: list[str]) -> bool:
    return not any(tokens_similar(token, remove_token) for remove_token in remove_tokens)


def subtract_tokens(main_tokens: Iterable[str], remove_tokens: Iterable[str]) -> list[str]:
    """Remove tokens from main_tokens that are similar to any token in remove_tokens."""
    remove = _stripped_tokens(remove_tokens)
    result: list[str] = []
    for token in _stripped_tokens(main_tokens):
        if _token_should_remain(token, remove):
            result.append(token)
    return result


def _normalized_words(tokens: Iterable[str]) -> list[str]:
    return [word for word in (clean_token(token) for token in tokens) if word and word != "na"]


def _camel_words(words: Iterable[str]) -> list[str]:
    split_words: list[str] = []
    for word in words:
        split_words.extend(part for part in word.split("_") if part)
    return split_words


def _join_camel(words: list[str]) -> str:
    first = words[0].lower()
    rest = "".join(word.capitalize() for word in words[1:])
    return first + rest


def _join_case(words: list[str], desired_case: str) -> str:
    if desired_case == "camelCase":
        return _join_camel(_camel_words(words))
    separator = "_" if desired_case == "snakeCase" else "-"
    return separator.join(word.lower() for word in words)


def _validated_case(desired_case: str) -> str:
    if desired_case in _VALID_CASES:
        return desired_case
    raise ValueError(f"Unknown desired_case: {desired_case!r}. Use one of: {sorted(_VALID_CASES)}")


def convert_case(tokens: Iterable[str], desired_case: str) -> str:
    """Convert a token list to a single string in camelCase, snakeCase, or kebabCase."""
    desired_case = _validated_case(desired_case)
    words = _normalized_words(tokens)
    if desired_case == "camelCase":
        words = _camel_words(words)
    if not words:
        return ""
    return _join_case(words, desired_case)


def normalize_keywords(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Clean, deduplicate, and filter placeholder tokens from keywords."""
    if raw is None:
        return []
    tokens = (
        [str(x).strip() for x in raw] if isinstance(raw, list | tuple) else [t.strip() for t in str(raw).split(",")]
    )

    filtered: list[str] = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        low = t.lower()
        if low in {
            "...",
            "…",
            "na",
            "n/a",
            "xxx",
            "w1",
            "w2",
            "tbd",
            "tba",
            "etc",
            "etc.",
            "tbd.",
            "tba.",
        }:
            continue
        filtered.append(t)
    return filtered[:MAX_NORMALIZED_KEYWORDS]


_FILENAME_RESERVED_WIN = frozenset(name.lower() for name in FILENAME_RESERVED_WIN)


def clean_token(text: str) -> str:
    """Normalize a token for filenames."""
    text = text.strip().rstrip(".")
    if not text:
        return "na"
    text = (
        text.replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("ẞ", "SS")
    )
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.lower()
    if text in _FILENAME_RESERVED_WIN:
        text = text + "_"
    return text


_VALID_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
VALID_CASE_CHOICES: tuple[str, ...] = tuple(sorted(_VALID_CASES))


@dataclass(frozen=True)
class Stopwords:
    words: set[str]

    def filter_tokens(self, tokens: Iterable[str]) -> list[str]:
        out: list[str] = []
        for token in tokens:
            t = token.strip()
            if not t:
                continue
            if t.lower() in self.words:
                continue
            out.append(t)
        return out


def split_to_tokens(text: str | None) -> list[str]:
    """Split text on whitespace, commas, underscores, and hyphens into non-empty lowercase tokens."""
    if text is None or not isinstance(text, str):
        return []
    return [t for t in re.split(r"[\s,_-]+", text) if t]
