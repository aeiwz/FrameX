"""Pytest configuration helpers for FrameX test output."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _extract_skip_reason(report: Any) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2]).strip()
        return reason or "no reason provided"
    if hasattr(longrepr, "reprcrash") and hasattr(longrepr, "message"):
        message = str(getattr(longrepr, "message", "")).strip()
        return message or "no reason provided"
    return str(longrepr).strip() if longrepr else "no reason provided"


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return

    counts = Counter(_extract_skip_reason(report) for report in skipped)
    terminalreporter.section("Skip Reasons")
    for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        terminalreporter.write_line(f"{count}x {reason}")
