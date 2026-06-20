from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal


@dataclass(frozen=True)
class ChunkMeta:
    category: str | None
    year: int | None
    path: str | None
    lang: str | None
    repo: str | None
    source: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    type: Literal["paper", "code"]
    title: str
    text: str
    meta: ChunkMeta


def chunk_id_of(doc_id: str, idx: int, text: str) -> str:
    """Return the shared deterministic chunk id.

    The text is intentionally part of the hash so changed re-ingests get a new
    identity, while identical content re-ingests collide correctly.
    """
    return hashlib.sha1(f"{doc_id}|{idx}|{text}".encode()).hexdigest()[:16]
