"""Google Calendar API client for fetching events by recording time window."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from src.config import CalendarConfig
from src.transcriber import Segment

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


@dataclass(frozen=True)
class CalendarEvent:
    """Metadata from a Google Calendar event."""
    title: str
    attendees: list[str]
    description: str
    start: datetime
    end: datetime
    calendar_id: str
    event_id: str


@dataclass(frozen=True)
class CalendarFetchResult:
    """Result of a calendar lookup."""
    event: CalendarEvent | None
    candidates_count: int
    fetch_duration_sec: float
    error: str | None = None


def estimate_recording_window(
    segments: list[Segment],
    pipeline_start: datetime,
    timezone_str: str = "Asia/Tokyo",
) -> tuple[datetime, datetime]:
    """Estimate recording start/end from Whisper segments.

    Uses max(segment.end) for duration, anchored to pipeline execution time.
    Returns (estimated_start, estimated_end) as timezone-aware datetimes.
    """
    tz = ZoneInfo(timezone_str)
    if not segments:
        end = pipeline_start.replace(tzinfo=tz) if pipeline_start.tzinfo is None else pipeline_start.astimezone(tz)
        start = end - timedelta(hours=1)
        return start, end

    max_end = max(seg.end for seg in segments)
    duration = timedelta(seconds=max_end)

    end = pipeline_start.replace(tzinfo=tz) if pipeline_start.tzinfo is None else pipeline_start.astimezone(tz)
    start = end - duration
    return start, end


class CalendarClient:
    """Google Calendar API client for fetching events by time window.

    Never raises from fetch_event() - returns CalendarFetchResult with error field.
    """

    def __init__(self, cfg: CalendarConfig) -> None:
        self._cfg = cfg
        self._service: Any = None

    def _build_service(self) -> Any:
        """Build and cache the Google Calendar API v3 service."""
        if self._service is not None:
            return self._service
        creds_path = Path(self._cfg.credentials_path)
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials not found: {creds_path}")
        creds = Credentials.from_service_account_file(str(creds_path), scopes=_SCOPES)
        self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _list_events_sync(
        self, time_min: datetime, time_max: datetime
    ) -> list[dict]:
        """Call Events.list synchronously."""
        service = self._build_service()
        result = service.events().list(
            calendarId=self._cfg.calendar_id,
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=10,
        ).execute()
        return result.get("items", [])

    async def fetch_event(
        self,
        recording_start: datetime,
        recording_end: datetime,
    ) -> CalendarFetchResult:
        """Find the best-matching calendar event. Never raises."""
        t0 = time.monotonic()
        try:
            tolerance = timedelta(minutes=self._cfg.match_tolerance_minutes)
            time_min = recording_start - tolerance
            time_max = recording_end + tolerance

            events = await asyncio.to_thread(self._list_events_sync, time_min, time_max)

            if not events:
                elapsed = time.monotonic() - t0
                return CalendarFetchResult(event=None, candidates_count=0, fetch_duration_sec=elapsed)

            # Score events by overlap
            best_event = None
            best_overlap = 0.0
            for raw in events:
                event = self._parse_event(raw)
                if event is None:
                    continue
                overlap = self._compute_overlap(
                    event.start, event.end, recording_start, recording_end
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_event = event

            elapsed = time.monotonic() - t0
            return CalendarFetchResult(
                event=best_event,
                candidates_count=len(events),
                fetch_duration_sec=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning("Calendar fetch failed: %s", exc)
            return CalendarFetchResult(
                event=None, candidates_count=0,
                fetch_duration_sec=elapsed, error=str(exc),
            )

    @staticmethod
    def _compute_overlap(
        event_start: datetime,
        event_end: datetime,
        rec_start: datetime,
        rec_end: datetime,
    ) -> float:
        """Compute overlap in seconds between two time ranges."""
        overlap_start = max(event_start, rec_start)
        overlap_end = min(event_end, rec_end)
        overlap = (overlap_end - overlap_start).total_seconds()
        return max(0.0, overlap)

    @staticmethod
    def _parse_event(raw: dict) -> CalendarEvent | None:
        """Parse a Google Calendar API event resource."""
        try:
            # Handle dateTime vs date (all-day events)
            start_raw = raw.get("start", {})
            end_raw = raw.get("end", {})

            if "dateTime" in start_raw:
                start = datetime.fromisoformat(start_raw["dateTime"])
            elif "date" in start_raw:
                start = datetime.fromisoformat(start_raw["date"]).replace(
                    tzinfo=ZoneInfo("UTC")
                )
            else:
                return None

            if "dateTime" in end_raw:
                end = datetime.fromisoformat(end_raw["dateTime"])
            elif "date" in end_raw:
                end = datetime.fromisoformat(end_raw["date"]).replace(
                    tzinfo=ZoneInfo("UTC")
                )
            else:
                return None

            attendees = [
                a.get("displayName") or a.get("email", "")
                for a in raw.get("attendees", [])
            ]

            return CalendarEvent(
                title=raw.get("summary", ""),
                attendees=attendees,
                description=raw.get("description", ""),
                start=start,
                end=end,
                calendar_id=raw.get("organizer", {}).get("email", ""),
                event_id=raw.get("id", ""),
            )
        except (KeyError, ValueError, TypeError):
            return None
