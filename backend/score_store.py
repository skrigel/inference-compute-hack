"""Persistent score cache backed by SQLite — fetch instead of re-scan.

The in-memory ``ScoreCache`` (backend/cache.py) only lives for one process; every
restart / new corpus load re-scores from scratch. This persists scored results
keyed by ``(collection, chunk_id, predicate, model)`` so a repeated query fetches
stored scores instead of re-running inference on the GPU.

Organised per ``collection`` (e.g. ``browsecomp``, ``demo``, and other
heterogeneous datasets) so corpora stay separate. ``chunk_id`` already hashes the
chunk text (see data.schema.chunk_id_of), so re-ingesting changed content misses
correctly; ``model`` is in the key so switching scorer backends never serves stale
scores. This is a deliberate "store" layer for the demo — the opposite of the
project's recompute-over-store thesis — kept behind a clean fetch interface.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

from inference.scorer import ScoreResult

DEFAULT_DB = str(Path(__file__).resolve().parents[1] / "data" / "score_cache.db")
# SQLite caps host parameters per statement (~999); chunk the IN-list well under it.
_IN_BATCH = 400


class ScoreStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.environ.get("SCORE_CACHE_DB", DEFAULT_DB)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                collection TEXT NOT NULL,
                chunk_id   TEXT NOT NULL,
                predicate  TEXT NOT NULL,
                model_id   TEXT NOT NULL,
                score REAL NOT NULL,
                p_yes REAL NOT NULL,
                p_no  REAL NOT NULL,
                tier  INTEGER NOT NULL DEFAULT 1,
                ts    REAL NOT NULL,
                PRIMARY KEY (collection, chunk_id, predicate, model_id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scores_lookup ON scores (collection, predicate, model_id)"
        )
        self._conn.commit()

    def get_scores(
        self, collection: str, predicate: str, model_id: str, chunk_ids
    ) -> dict[str, ScoreResult]:
        """Fetch stored scores for the given chunk ids. Missing ids are simply absent."""
        ids = list(chunk_ids)
        if not ids:
            return {}
        out: dict[str, ScoreResult] = {}
        with self._lock:
            for start in range(0, len(ids), _IN_BATCH):
                window = ids[start : start + _IN_BATCH]
                placeholders = ",".join("?" * len(window))
                rows = self._conn.execute(
                    "SELECT chunk_id, score, p_yes, p_no, tier FROM scores "
                    "WHERE collection=? AND predicate=? AND model_id=? "
                    f"AND chunk_id IN ({placeholders})",
                    [collection, predicate, model_id, *window],
                ).fetchall()
                for cid, score, p_yes, p_no, tier in rows:
                    out[cid] = ScoreResult(
                        chunk_id=cid, score=score, p_yes=p_yes, p_no=p_no,
                        tier=tier, from_cache=True,
                    )
        return out

    def put_scores(
        self, collection: str, predicate: str, model_id: str, results
    ) -> None:
        rows = [
            (collection, r.chunk_id, predicate, model_id, r.score, r.p_yes, r.p_no, r.tier, time.time())
            for r in results
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO scores "
                "(collection, chunk_id, predicate, model_id, score, p_yes, p_no, tier, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
            by_collection = dict(
                self._conn.execute(
                    "SELECT collection, COUNT(*) FROM scores GROUP BY collection"
                ).fetchall()
            )
        return {"n_scores": total, "by_collection": by_collection}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
