from __future__ import annotations

import os


CHUNK_OVERLAP_RATIO = float(os.environ.get("CHUNK_OVERLAP_RATIO", "0.0"))


def chunk_with_overlap(
    text: str,
    chunk_size: int = 512,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> list[str]:
    """Split text into chunks with optional overlap.

    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap_ratio: Fraction of overlap between consecutive chunks (0.0-0.5)

    Returns:
        List of text chunks with specified overlap
    """
    overlap_ratio = max(0.0, min(0.5, overlap_ratio))
    words = text.split()

    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    overlap_words = int(chunk_size * overlap_ratio)
    step_size = max(1, chunk_size - overlap_words)

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step_size

    return chunks
