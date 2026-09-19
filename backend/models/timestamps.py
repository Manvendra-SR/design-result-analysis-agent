"""
backend/models/timestamps.py
==============================
One annotated ``datetime`` type, used by every model that exposes a timestamp
to the API.

The problem it solves
---------------------
Timestamps are written as naive UTC (``datetime.utcnow``) into
``TIMESTAMP`` columns, which carry no offset, so Pydantic serialised them as
``"2026-09-08T11:26:09.913555"`` - a wall-clock reading with nothing saying
which clock. JavaScript's ``new Date()`` parses an ISO date-time *without* an
offset as **local** time, so the browser took UTC digits to be IST digits and
every time in the UI read 5h30m early.

The fix belongs here, at the serialisation boundary, not in the UI: the value
really is UTC, so it is labelled UTC (``...+00:00``) on the way out and every
client - the React app, curl, pgAdmin's user alike - can convert it into
whatever zone it displays in. The frontend then just calls
``toLocaleString()`` and gets the viewer's own zone (IST on an Indian
machine), with no hardcoded offset anywhere in the codebase.

Storage is deliberately unchanged: naive UTC in ``TIMESTAMP`` columns. Only
what crosses the API is tagged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import PlainSerializer


def _as_utc_iso(value: datetime) -> str:
    """Serialise a datetime as an explicitly-UTC ISO 8601 string.

    A naive value is *assumed* to be UTC - which it is, everything in this
    project is written with ``datetime.utcnow``. An aware value is converted,
    so a timestamp that arrived from somewhere else is still correct.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


#: ``datetime`` that always serialises with an explicit UTC offset.
UTCDateTime = Annotated[datetime, PlainSerializer(_as_utc_iso, return_type=str)]
