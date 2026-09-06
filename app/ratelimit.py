"""
Minimal in-process rate limiter - fixed-window counter per client IP, no
external dependency (no Redis, no slowapi). This is a single-process
hackathon demo; a real distributed rate limiter would need shared state
across processes/machines, which this deliberately doesn't attempt.

Purpose here is specifically "better fallback under load", not security:
this protects the demo from one client (or one runaway frontend retry loop)
accidentally hammering the single Groq API key and DB pool hard enough to
degrade the experience for everyone else standing at the booth.
"""

import time
from collections import defaultdict


class FixedWindowRateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit_per_minute = limit_per_minute
        self._windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    def allow(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds). Known, accepted edge case:
        a fixed window can allow up to 2x the limit right at a window
        boundary. A sliding-window/token-bucket would close that gap at the
        cost of more state per key - not worth it for a demo-scale limiter."""
        now = int(time.time())
        window_start, count = self._windows[key]
        current_window = now - (now % 60)

        if window_start != current_window:
            self._windows[key] = (current_window, 1)
            return True, 0

        if count < self.limit_per_minute:
            self._windows[key] = (current_window, count + 1)
            return True, 0

        retry_after = current_window + 60 - now
        return False, max(retry_after, 1)

    def reset(self) -> None:
        self._windows.clear()
