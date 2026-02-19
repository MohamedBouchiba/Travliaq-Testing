"""Detect agent stuck loops via sliding window pattern matching."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoopDetection:
    """Result of a loop detection check."""

    detected: bool
    pattern_type: Optional[str] = None  # "repeat" | "alternating"
    pattern: Optional[str] = None
    window: List[str] = field(default_factory=list)


class LoopDetector:
    """Sliding-window loop detector for agent action sequences.

    Detects:
    - AAA pattern: same action repeated `repeat_threshold`+ times consecutively
    - ABAB pattern: two actions alternating `alternating_cycles`+ full cycles
    """

    def __init__(
        self,
        window_size: int = 8,
        repeat_threshold: int = 3,
        alternating_cycles: int = 2,
    ):
        self._window: List[str] = []
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold
        self._alternating_cycles = alternating_cycles
        self._detected_count = 0
        self._suppressed_pattern: Optional[str] = None

    @property
    def detected_count(self) -> int:
        return self._detected_count

    def push(self, action_name: str) -> LoopDetection:
        """Add an action and check for loops.

        Suppresses consecutive re-detections of the same ongoing pattern
        (e.g., "click[5] x3" fires once; x4, x5, x6 are suppressed until
        the pattern breaks).
        """
        self._window.append(action_name)
        if len(self._window) > self._window_size:
            self._window = self._window[-self._window_size :]

        result = self._check()
        if result.detected:
            # Extract stable key: "click[5]" from "click[5] x3"
            base = result.pattern.rsplit(" x", 1)[0]
            # Normalize alternating: "a <-> b" and "b <-> a" are the same loop
            if result.pattern_type == "alternating" and " <-> " in base:
                parts = base.split(" <-> ")
                base = " <-> ".join(sorted(parts))
            suppression_key = f"{result.pattern_type}:{base}"
            if suppression_key == self._suppressed_pattern:
                # Same ongoing pattern — suppress
                return LoopDetection(detected=False, window=list(self._window))
            self._suppressed_pattern = suppression_key
            self._detected_count += 1
            return result
        else:
            # Pattern broken — reset suppression
            self._suppressed_pattern = None
            return result

    def _check(self) -> LoopDetection:
        w = self._window

        # Check AAA: last N actions are identical
        if len(w) >= self._repeat_threshold:
            tail = w[-self._repeat_threshold :]
            if len(set(tail)) == 1:
                count = 0
                for a in reversed(w):
                    if a == tail[0]:
                        count += 1
                    else:
                        break
                return LoopDetection(
                    detected=True,
                    pattern_type="repeat",
                    pattern=f"{tail[0]} x{count}",
                    window=list(w),
                )

        # Check ABAB: alternating pair for N cycles
        min_len = self._alternating_cycles * 2
        if len(w) >= min_len:
            tail = w[-min_len:]
            a, b = tail[0], tail[1]
            if a != b and all(
                tail[i] == (a if i % 2 == 0 else b) for i in range(min_len)
            ):
                return LoopDetection(
                    detected=True,
                    pattern_type="alternating",
                    pattern=f"{a} <-> {b} x{self._alternating_cycles}",
                    window=list(w),
                )

        return LoopDetection(detected=False, window=list(w))
