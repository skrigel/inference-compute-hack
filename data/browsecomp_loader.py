"""Load BrowseComp-Plus corpus and convert to Chunk schema."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from data.schema import Chunk, ChunkMeta, chunk_id_of

# browsecomp-plus web docs run from ~300 chars to ~920k chars (~230k tokens) — far
# past the model context. Rather than truncate, split each doc into passages that
# fit the window and score each independently (also better recall: the predicate is
# matched per passage, not per whole doc). We chunk by CHARACTERS, not words: these
# are multilingual web docs (incl. space-free scripts like Arabic/CJK), so a word
# count does NOT bound the token count. ~6000 chars stays well under the window
# across scripts. Cap passages/doc so one giant doc can't dominate the corpus /
# scoring cost (raise BROWSECOMP_MAX_CHUNKS_PER_DOC for fuller coverage).
CHUNK_CHARS = int(os.environ.get("BROWSECOMP_CHUNK_CHARS", "6000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("BROWSECOMP_CHUNK_OVERLAP_CHARS", "600"))
MAX_CHUNKS_PER_DOC = int(os.environ.get("BROWSECOMP_MAX_CHUNKS_PER_DOC", "20"))


def _char_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size character windows. Robust to docs
    with few/no whitespace boundaries (multilingual web text), unlike word-based
    chunking which can emit a single multi-thousand-token 'word' run."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
    return chunks


def _extract_title(text: str, docid: str) -> str:
    """Extract title from YAML frontmatter if present, else use docid."""
    match = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text)
    if match:
        return match.group(1).strip()
    return docid


@lru_cache(maxsize=1)
def _load_raw_corpus() -> list[dict]:
    """Load raw corpus from HuggingFace (cached)."""
    from datasets import load_dataset  # deferred: heavy optional dependency

    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    return list(ds)


def load_browsecomp_corpus(limit: int | None = None) -> list[Chunk]:
    """Load BrowseComp-Plus corpus as Chunk objects.

    Args:
        limit: Max number of chunks to return. None for all (~100k).

    Returns:
        List of Chunk objects.
    """
    raw = _load_raw_corpus()

    chunks: list[Chunk] = []
    for item in raw:
        if limit is not None and len(chunks) >= limit:
            break  # `limit` caps CHUNKS (the results shown in the UI), not source docs
        docid = str(item["docid"])
        text = item["text"]
        url = item["url"]
        title = _extract_title(text, docid)

        passages = _char_chunks(text, CHUNK_CHARS, CHUNK_OVERLAP_CHARS)[:MAX_CHUNKS_PER_DOC]
        n = len(passages)
        for idx, passage in enumerate(passages):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_of(docid, idx, passage),
                    doc_id=docid,
                    type="paper",  # web documents, closest match
                    title=title if n == 1 else f"{title} (part {idx + 1}/{n})",
                    text=passage,
                    meta=ChunkMeta(
                        category=None,
                        year=None,
                        path=None,
                        lang=None,
                        repo=None,
                        source=url,
                    ),
                )
            )

    if limit is not None:
        chunks = chunks[:limit]  # trim the doc that pushed us over to land exactly on limit
    return chunks


def browsecomp_docs(limit: int | None = None) -> list[tuple[str, str]]:
    """Return (doc_id, text) tuples for use in _scaled_docs pattern.

    Args:
        limit: Max number of docs to return.

    Returns:
        List of (doc_id, text) tuples.
    """
    chunks = load_browsecomp_corpus(limit=limit)
    return [(chunk.doc_id, chunk.text) for chunk in chunks]
