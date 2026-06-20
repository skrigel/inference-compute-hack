from __future__ import annotations

from dataclasses import dataclass

from backend.schemas import RefineOp


@dataclass(frozen=True)
class ClauseRecord:
    clause_id: str
    op: RefineOp | str
    text: str
    parent_clause_id: str | None
    removable: bool = True
    target_chunk_id: str | None = None


def label_for(op: RefineOp) -> str:
    return {
        RefineOp.require: "Require",
        RefineOp.exclude: "Exclude",
        RefineOp.include: "Include",
        RefineOp.refocus: "Refocus",
        RefineOp.brush: "Range",
    }[op]

