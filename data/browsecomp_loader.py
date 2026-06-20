"""Load BrowseComp-Plus corpus and convert to Chunk schema."""

from __future__ import annotations

import re
from functools import lru_cache

from data.schema import Chunk, ChunkMeta, chunk_id_of


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
    if limit is not None:
        raw = raw[:limit]

    chunks = []
    for item in raw:
        docid = str(item["docid"])
        text = item["text"]
        url = item["url"]

        chunk = Chunk(
            chunk_id=chunk_id_of(docid, 0, text),
            doc_id=docid,
            type="paper",  # web documents, closest match
            title=_extract_title(text, docid),
            text=text,
            meta=ChunkMeta(
                category=None,
                year=None,
                path=None,
                lang=None,
                repo=None,
                source=url,
            ),
        )
        chunks.append(chunk)

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
