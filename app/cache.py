"""
Small in-process TTL+LRU cache, used for two things:
  1. query embedding cache (skip the CPU-bound encode() call entirely on a
     repeat query - very common in a demo-booth setting where lots of
     people type near-identical canonical questions, or a judge re-runs the
     same query after tweaking something)
  2. full /query answer cache, keyed on (query, jurisdiction, category) -
     skips retrieval AND the Groq call entirely on an exact repeat.

Deliberately NOT Redis or any external cache: this is a single-process
hackathon demo, not a distributed deployment, and adding an external cache
dependency here would be exactly the kind of complexity-you-can't-defend
this project's own README already warns against. If this backend ever runs
as multiple processes/machines, replace this with a real shared cache -
don't assume cache hits are shared across workers, they aren't.

Thread-safety note: reads/writes happen inside asyncio coroutines without an
`await` in between, so there's no interleaving risk within one process
despite no explicit lock. If you later add a lock here "to be safe", you're
solving a problem this cache doesn't actually have in its current usage.
"""

import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        self._store[key] = (expires_at, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }
