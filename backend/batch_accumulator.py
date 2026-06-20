from __future__ import annotations

import os
import time

from inference.scorer import ScoreRequest


BATCH_ACCUMULATE_MS = int(os.environ.get("BATCH_ACCUMULATE_MS", "0"))


class BatchAccumulator:
    """Accumulates requests until batch is full or time window expires.

    When BATCH_ACCUMULATE_MS=0 (default), requests dispatch immediately.
    Otherwise, requests accumulate for up to max_wait_ms before dispatch.
    """

    def __init__(
        self,
        max_wait_ms: int = BATCH_ACCUMULATE_MS,
        max_batch_size: int = 64,
    ) -> None:
        self._max_wait_ms = max(0, max_wait_ms)
        self._max_batch_size = max(1, max_batch_size)
        self._pending: list[ScoreRequest] = []
        self._first_added_at: float | None = None

    def add(self, request: ScoreRequest) -> list[ScoreRequest]:
        """Add a request. Returns batch to dispatch if full or disabled."""
        if self._max_wait_ms == 0:
            return [request]

        if not self._pending:
            self._first_added_at = time.perf_counter()

        self._pending.append(request)

        if len(self._pending) >= self._max_batch_size:
            return self.flush()

        return []

    def should_flush(self) -> bool:
        """Check if time window has expired."""
        if not self._pending or self._first_added_at is None:
            return False
        elapsed_ms = (time.perf_counter() - self._first_added_at) * 1000.0
        return elapsed_ms >= self._max_wait_ms

    def flush(self) -> list[ScoreRequest]:
        """Return and clear all pending requests."""
        batch = self._pending
        self._pending = []
        self._first_added_at = None
        return batch

    def pending(self) -> list[ScoreRequest]:
        """Return current pending requests without clearing."""
        return list(self._pending)
