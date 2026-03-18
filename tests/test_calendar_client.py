"""Tests for src.calendar_client module."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.calendar_client import (
    CalendarClient,
    CalendarEvent,
    CalendarFetchResult,
    estimate_recording_window,
)
from src.transcriber import Segment

_TZ = ZoneInfo("Asia/Tokyo")

_CFG = SimpleNamespace(
    enabled=True,
    credentials_path="credentials.json",
    calendar_id="primary",
    timezone="Asia/Tokyo",
    match_tolerance_minutes=30,
    api_timeout_sec=10,
    max_retries=2,
)


def _make_raw_event(
    *,
    summary: str = "Weekly Standup",
    start_dt: str = "2026-03-17T10:00:00+09:00",
    end_dt: str = "2026-03-17T11:00:00+09:00",
    event_id: str = "evt_001",
    attendees: list[dict] | None = None,
    description: str = "Discuss sprint progress",
    use_date: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Build a raw Google Calendar API event dict."""
    raw: dict = {
        "summary": summary,
        "id": event_id,
        "description": description,
        "organizer": {"email": "team@example.com"},
    }
    if use_date:
        raw["start"] = {"date": start_date or "2026-03-17"}
        raw["end"] = {"date": end_date or "2026-03-18"}
    else:
        raw["start"] = {"dateTime": start_dt}
        raw["end"] = {"dateTime": end_dt}

    if attendees is not None:
        raw["attendees"] = attendees
    return raw


# ---------------------------------------------------------------------------
# estimate_recording_window
# ---------------------------------------------------------------------------

class TestEstimateRecordingWindow:

    def test_estimate_recording_window_basic(self):
        """Segments with known duration produce correct window."""
        segments = [
            Segment(start=0.0, end=300.0, text="hello", speaker="Alice"),
            Segment(start=300.0, end=600.0, text="world", speaker="Bob"),
            Segment(start=600.0, end=1800.0, text="bye", speaker="Alice"),
        ]
        pipeline_start = datetime(2026, 3, 17, 11, 0, 0)
        start, end = estimate_recording_window(segments, pipeline_start, "Asia/Tokyo")

        assert end.tzinfo is not None
        assert start.tzinfo is not None
        # Max segment end is 1800s = 30 minutes
        expected_start = end - timedelta(seconds=1800)
        assert start == expected_start
        # End should be pipeline_start in the target timezone
        assert end == datetime(2026, 3, 17, 11, 0, 0, tzinfo=_TZ)

    def test_estimate_recording_window_empty(self):
        """Empty segments fall back to 1-hour window."""
        pipeline_start = datetime(2026, 3, 17, 11, 0, 0)
        start, end = estimate_recording_window([], pipeline_start, "Asia/Tokyo")

        assert end == datetime(2026, 3, 17, 11, 0, 0, tzinfo=_TZ)
        assert start == end - timedelta(hours=1)


# ---------------------------------------------------------------------------
# CalendarClient._parse_event
# ---------------------------------------------------------------------------

class TestParseEvent:

    def test_parse_event_basic(self):
        """Standard event with dateTime fields is parsed correctly."""
        raw = _make_raw_event(
            attendees=[
                {"displayName": "Alice", "email": "alice@example.com"},
                {"displayName": "Bob", "email": "bob@example.com"},
            ],
        )
        event = CalendarClient._parse_event(raw)

        assert event is not None
        assert event.title == "Weekly Standup"
        assert event.description == "Discuss sprint progress"
        assert event.event_id == "evt_001"
        assert event.attendees == ["Alice", "Bob"]
        assert event.start == datetime.fromisoformat("2026-03-17T10:00:00+09:00")
        assert event.end == datetime.fromisoformat("2026-03-17T11:00:00+09:00")
        assert event.calendar_id == "team@example.com"

    def test_parse_event_all_day(self):
        """All-day event with date (not dateTime) is parsed."""
        raw = _make_raw_event(use_date=True, start_date="2026-03-17", end_date="2026-03-18")
        event = CalendarClient._parse_event(raw)

        assert event is not None
        assert event.start.tzinfo == ZoneInfo("UTC")
        assert event.end.tzinfo == ZoneInfo("UTC")
        assert event.start.day == 17
        assert event.end.day == 18

    def test_parse_event_no_attendees(self):
        """Event without attendees list produces empty list."""
        raw = _make_raw_event()
        # No 'attendees' key at all
        event = CalendarClient._parse_event(raw)

        assert event is not None
        assert event.attendees == []

    def test_parse_event_no_description(self):
        """Event with empty description is parsed with empty string."""
        raw = _make_raw_event(description="")
        event = CalendarClient._parse_event(raw)

        assert event is not None
        assert event.description == ""

    def test_parse_event_invalid(self):
        """Malformed event returns None."""
        # Missing both start.dateTime and start.date
        raw = {"summary": "Bad Event", "start": {}, "end": {}, "id": "bad"}
        event = CalendarClient._parse_event(raw)
        assert event is None

    def test_parse_event_completely_empty(self):
        """Completely empty dict returns None."""
        event = CalendarClient._parse_event({})
        assert event is None

    def test_parse_event_attendee_email_fallback(self):
        """Attendee without displayName falls back to email."""
        raw = _make_raw_event(
            attendees=[
                {"email": "noname@example.com"},
                {"displayName": "Alice", "email": "alice@example.com"},
            ],
        )
        event = CalendarClient._parse_event(raw)

        assert event is not None
        assert event.attendees == ["noname@example.com", "Alice"]


# ---------------------------------------------------------------------------
# CalendarClient._compute_overlap
# ---------------------------------------------------------------------------

class TestComputeOverlap:

    def test_compute_overlap_full(self):
        """Event fully contains the recording."""
        event_start = datetime(2026, 3, 17, 9, 0, tzinfo=_TZ)
        event_end = datetime(2026, 3, 17, 12, 0, tzinfo=_TZ)
        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)

        overlap = CalendarClient._compute_overlap(event_start, event_end, rec_start, rec_end)
        # Recording is 1 hour = 3600 seconds, fully inside event
        assert overlap == 3600.0

    def test_compute_overlap_partial(self):
        """Partial overlap between event and recording."""
        event_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        event_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        rec_start = datetime(2026, 3, 17, 10, 30, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 30, tzinfo=_TZ)

        overlap = CalendarClient._compute_overlap(event_start, event_end, rec_start, rec_end)
        # Overlap is 10:30 - 11:00 = 30 minutes = 1800 seconds
        assert overlap == 1800.0

    def test_compute_overlap_none(self):
        """Disjoint time ranges have zero overlap."""
        event_start = datetime(2026, 3, 17, 8, 0, tzinfo=_TZ)
        event_end = datetime(2026, 3, 17, 9, 0, tzinfo=_TZ)
        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)

        overlap = CalendarClient._compute_overlap(event_start, event_end, rec_start, rec_end)
        assert overlap == 0.0


# ---------------------------------------------------------------------------
# CalendarClient.fetch_event (async, mocked)
# ---------------------------------------------------------------------------

class TestFetchEvent:

    @pytest.fixture()
    def client(self):
        """CalendarClient with a mocked _list_events_sync."""
        c = CalendarClient.__new__(CalendarClient)
        c._cfg = _CFG
        c._service = None
        return c

    @pytest.mark.asyncio
    async def test_fetch_event_single_match(self, client):
        """Mock API returns one event, it is selected."""
        raw = _make_raw_event(
            start_dt="2026-03-17T10:00:00+09:00",
            end_dt="2026-03-17T11:00:00+09:00",
        )
        client._list_events_sync = MagicMock(return_value=[raw])

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        result = await client.fetch_event(rec_start, rec_end)

        assert result.event is not None
        assert result.event.title == "Weekly Standup"
        assert result.candidates_count == 1
        assert result.error is None
        assert result.fetch_duration_sec >= 0

    @pytest.mark.asyncio
    async def test_fetch_event_best_overlap(self, client):
        """Multiple events returned; the one with best overlap is selected."""
        # Event A: small overlap (only 15 min)
        event_a = _make_raw_event(
            summary="Morning Coffee",
            start_dt="2026-03-17T09:00:00+09:00",
            end_dt="2026-03-17T10:15:00+09:00",
            event_id="evt_a",
        )
        # Event B: large overlap (full 1 hour)
        event_b = _make_raw_event(
            summary="Sprint Planning",
            start_dt="2026-03-17T10:00:00+09:00",
            end_dt="2026-03-17T11:30:00+09:00",
            event_id="evt_b",
        )
        client._list_events_sync = MagicMock(return_value=[event_a, event_b])

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        result = await client.fetch_event(rec_start, rec_end)

        assert result.event is not None
        assert result.event.title == "Sprint Planning"
        assert result.candidates_count == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_event_no_match(self, client):
        """Mock API returns empty list."""
        client._list_events_sync = MagicMock(return_value=[])

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        result = await client.fetch_event(rec_start, rec_end)

        assert result.event is None
        assert result.candidates_count == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_event_api_error(self, client):
        """Mock API raises; returns graceful result with error message."""
        client._list_events_sync = MagicMock(
            side_effect=ConnectionError("Network unreachable")
        )

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        result = await client.fetch_event(rec_start, rec_end)

        assert result.event is None
        assert result.candidates_count == 0
        assert result.error is not None
        assert "Network unreachable" in result.error
        assert result.fetch_duration_sec >= 0

    @pytest.mark.asyncio
    async def test_fetch_event_never_raises(self, client):
        """Even unexpected errors (e.g., TypeError) return result, never raise."""
        client._list_events_sync = MagicMock(
            side_effect=TypeError("unexpected NoneType")
        )

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)

        # Must not raise
        result = await client.fetch_event(rec_start, rec_end)

        assert isinstance(result, CalendarFetchResult)
        assert result.event is None
        assert result.error is not None
        assert "NoneType" in result.error

    @pytest.mark.asyncio
    async def test_fetch_event_unparseable_events_skipped(self, client):
        """Events that fail to parse are skipped without error."""
        good = _make_raw_event(
            summary="Real Meeting",
            start_dt="2026-03-17T10:00:00+09:00",
            end_dt="2026-03-17T11:00:00+09:00",
        )
        bad = {"summary": "Bad", "start": {}, "end": {}, "id": "bad"}
        client._list_events_sync = MagicMock(return_value=[bad, good])

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        result = await client.fetch_event(rec_start, rec_end)

        assert result.event is not None
        assert result.event.title == "Real Meeting"
        assert result.candidates_count == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_event_tolerance_applied(self, client):
        """Verify that match_tolerance_minutes is applied to the query window."""
        call_args = {}

        def capture_list_events(time_min, time_max):
            call_args["time_min"] = time_min
            call_args["time_max"] = time_max
            return []

        client._list_events_sync = MagicMock(side_effect=capture_list_events)

        rec_start = datetime(2026, 3, 17, 10, 0, tzinfo=_TZ)
        rec_end = datetime(2026, 3, 17, 11, 0, tzinfo=_TZ)
        await client.fetch_event(rec_start, rec_end)

        tolerance = timedelta(minutes=_CFG.match_tolerance_minutes)
        assert call_args["time_min"] == rec_start - tolerance
        assert call_args["time_max"] == rec_end + tolerance
