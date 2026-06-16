import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

MEMORY_DIR = Path("data/cache")


@dataclass
class Turn:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            query=d["query"],
            answer=d["answer"],
            sources=d.get("sources", []),
            confidence=d.get("confidence", 0.0),
            mode=d.get("mode", "general"),
            timestamp=d.get("timestamp", ""),
        )


class ConversationMemory:
    def __init__(self, session_id: str = "default", max_turns: int = 10):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        self._load()

    def _path(self) -> Path:
        safe_id = re.sub(r"[^\w\-]", "_", self.session_id)
        safe_id = safe_id[:200]
        return MEMORY_DIR / f"memory_{safe_id}.json"

    def _load(self):
        p = self._path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.turns = [Turn.from_dict(t) for t in data[-self.max_turns:]]
            except Exception:
                self.turns = []

    def save(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        data = [t.to_dict() for t in self.turns[-self.max_turns:]]
        self._path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_turn(self, query: str, answer: str, sources: List[dict], confidence: float, mode: str = "general"):
        turn = Turn(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            mode=mode,
            timestamp=datetime.now().isoformat(),
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        self.save()

    def get_history(self, max_chars: int = 2000) -> str:
        parts = []
        for t in self.turns[-5:]:
            parts.append(f"Q: {t.query}")
            parts.append(f"A: {t.answer[:300]}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def clear(self):
        self.turns = []
        p = self._path()
        if p.exists():
            p.unlink()

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0
