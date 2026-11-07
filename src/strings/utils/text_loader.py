"""
Loading large text files without surprises.

    load_text        whole file (optionally capped at max_bytes, cut on
                     a character boundary), with normalization options
    iter_chunks      fixed-size chunks with an overlap, so a streaming
                     matcher never misses a pattern that straddles two
                     chunks (overlap ≥ pattern length − 1)
    search_stream    run any matcher over a file chunk by chunk, with
                     global positions — constant memory for huge logs
    normalize        lower-casing / whitespace collapsing / ASCII folding
    write_text       save generated datasets
"""
import io
import os
import re
import unicodedata
from typing import Callable, Iterator, List, Optional, Tuple


def normalize(text: str, lower: bool = False, collapse_whitespace: bool = False,
              ascii_only: bool = False) -> str:
    if ascii_only:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    if lower:
        text = text.lower()
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def load_text(path: str, max_bytes: Optional[int] = None, encoding: str = "utf-8",
              errors: str = "replace", **normalize_options) -> str:
    """Read a text file (the first max_bytes bytes if given)."""
    with open(path, "rb") as fh:
        data = fh.read() if max_bytes is None else fh.read(max_bytes)
    text = data.decode(encoding, errors="ignore" if max_bytes else errors)
    return normalize(text, **normalize_options) if normalize_options else text


def write_text(text: str, path: str, encoding: str = "utf-8") -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding=encoding) as fh:
        return fh.write(text)


def iter_chunks(path: str, chunk_size: int = 1 << 20, overlap: int = 0,
                encoding: str = "utf-8") -> Iterator[Tuple[int, str]]:
    """
    Yield (offset, chunk) where offset is the character position of the
    chunk's first character; consecutive chunks share `overlap`
    characters.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    with io.open(path, encoding=encoding, errors="replace") as fh:
        offset, carry = 0, ""
        while True:
            block = fh.read(chunk_size - len(carry))
            if not block:
                return
            chunk = carry + block
            yield offset, chunk
            carry = chunk[len(chunk) - overlap:] if overlap else ""
            offset += len(chunk) - len(carry)


def search_stream(path: str, pattern: str,
                  matcher: Callable[[str, str], List[int]],
                  chunk_size: int = 1 << 20) -> List[int]:
    """
    Global positions of `pattern` in a file of any size, searching
    chunk by chunk. Chunks overlap by len(pattern) − 1 characters, and
    matches lying entirely inside an overlap are reported once.
    """
    m = len(pattern)
    if m == 0:
        raise ValueError("empty pattern")
    overlap = m - 1
    found, seen_until = [], -1
    for offset, chunk in iter_chunks(path, max(chunk_size, 2 * m), overlap):
        for p in matcher(chunk, pattern):
            g = offset + p
            if g > seen_until:
                found.append(g)
                seen_until = g
    return found


def file_size(path: str) -> int:
    return os.path.getsize(path)
