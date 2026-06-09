"""UTC timestamp policy for xpkg payloads.

This is the single home for the ISO-8601 UTC timestamp format used across
descriptors, provenance records, and the durable store. Other modules should
call :func:`now_utc_iso` instead of formatting ``datetime.now`` by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime


def now_utc_iso(*, drop_microseconds: bool = False) -> str:
    """Return an ISO-8601 UTC timestamp with a trailing ``Z``.

    Args:
        drop_microseconds: When True, truncate to whole seconds. Used for
            human-facing ``imported_at`` and descriptor timestamps. The durable
            store keeps full precision so close-together commits stay ordered by
            their ``updated_at`` tiebreak.
    """
    now = datetime.now(tz=UTC)
    if drop_microseconds:
        now = now.replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


__all__ = ["now_utc_iso"]
