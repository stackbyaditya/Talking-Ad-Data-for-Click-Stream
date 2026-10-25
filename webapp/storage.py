"""Session storage. One JSON file per session so concurrent gunicorn workers never interleave
writes. The same key scheme (sessions/<id>.json) maps 1:1 onto an S3 bucket later."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class SessionStore:
    def __init__(self, root: str | Path):
        self.dir = Path(root) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: Dict[str, Any]) -> str:
        sid = str(record["id"])
        if not _ID_RE.match(sid):
            raise ValueError("bad session id")
        tmp = self.dir / f".{sid}.tmp"
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(self.dir / f"{sid}.json")  # atomic on the same filesystem
        return sid

    def count(self) -> int:
        return sum(1 for _ in self.dir.glob("*.json"))

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.dir.glob("*.json"):
            try:
                label = json.loads(p.read_text(encoding="utf-8")).get("label", "?")
            except (OSError, ValueError):
                continue
            counts[label] = counts.get(label, 0) + 1
        return counts

    def all(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out
