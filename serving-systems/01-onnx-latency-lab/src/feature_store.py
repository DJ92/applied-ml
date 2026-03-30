from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockFeatureStore:
    records: dict[str, dict[str, Any]]
    lookup_latency_ms: float = 2.0
    timeout_ms: float = 25.0
    cache_enabled: bool = True
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> dict[str, Any]:
        if self.cache_enabled and key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        if self.lookup_latency_ms > self.timeout_ms:
            raise TimeoutError(f"Feature lookup for {key} exceeded timeout {self.timeout_ms} ms")
        time.sleep(self.lookup_latency_ms / 1000.0)
        value = self.records[key]
        if self.cache_enabled:
            self.cache[key] = value
        return value

    def stats(self) -> dict[str, int]:
        return {"cache_hits": self.hits, "cache_misses": self.misses}
