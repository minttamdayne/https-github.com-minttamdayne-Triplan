from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any


WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class OpeningStatus:
    is_open: bool
    confidence: float
    source: str
    penalty: float = 0.0


@dataclass(frozen=True)
class VisitOpeningStatus:
    fits: bool
    source: str
    confidence: float
    arrival: datetime
    departure: datetime
    wait_minutes: int = 0


Window = tuple[int, int]


def parse_opening_schedule(opening_hours: list[Any] | None) -> tuple[dict[str, list[Window]], float]:
    """Parse POI opening hours into {mon: [(open_minute, close_minute), ...]}.

    Overnight rows keep close_minute below open_minute, e.g. 17:30-01:30 is
    stored as (1050, 90). Visit checks expand those windows across dates.
    """
    if not opening_hours:
        return {}, 0.0

    schedule: dict[str, list[Window]] = {}
    parsed_any = False
    for entry in opening_hours:
        days_text = str(getattr(entry, "days", "") or "")
        open_text = str(getattr(entry, "open", "") or "")
        close_text = str(getattr(entry, "close", "") or "")

        days = _parse_days(days_text)
        if not days:
            continue

        if _is_open_24_hours(open_text, close_text):
            open_min, close_min = 0, 24 * 60
        else:
            open_min = _parse_time(open_text)
            close_min = _parse_time(close_text)
            if open_min is None or close_min is None:
                continue

        for day in days:
            schedule.setdefault(day, []).append((open_min, close_min))
        parsed_any = True

    for windows in schedule.values():
        windows.sort(key=lambda item: item[0])
    return schedule, 0.9 if parsed_any else 0.0


def is_open(poi: Any, arrival_time: datetime, weekday: str | int | None = None) -> bool:
    status = opening_status(poi, arrival_time, weekday)
    return status.is_open


def opening_status(poi: Any, arrival_time: datetime, weekday: str | int | None = None) -> OpeningStatus:
    group = getattr(poi, "semantic_group_override", None) or ""
    schedule = getattr(poi, "opening_schedule", None) or {}
    minute = arrival_time.hour * 60 + arrival_time.minute

    if schedule:
        open_now = any(
            start <= arrival_time <= end
            for start, end in _candidate_windows_for_date(schedule, arrival_time, weekday)
        )
        return OpeningStatus(
            is_open=open_now,
            confidence=max(float(getattr(poi, "opening_hours_confidence", 0.9) or 0.9), 0.75),
            source="explicit",
            penalty=0.0 if open_now else 0.28,
        )

    assumed_open, type_penalty = _type_assumption(group, minute)
    return OpeningStatus(
        is_open=True,
        confidence=0.25,
        source="assumed",
        penalty=max(0.04, type_penalty),
    )


def visit_opening_status(
    poi: Any,
    arrival_time: datetime,
    departure_time: datetime,
    *,
    allow_wait: bool = True,
) -> VisitOpeningStatus:
    """Return whether a full visit fits within a listed opening window.

    Unknown hours are allowed, but explicitly marked as ``source='unknown'`` so
    callers can warn and score them below confirmed hours.
    """
    schedule = getattr(poi, "opening_schedule", None) or {}
    confidence = float(getattr(poi, "opening_hours_confidence", 0.0) or 0.0)
    duration = departure_time - arrival_time
    if duration <= timedelta(0):
        return VisitOpeningStatus(False, "invalid", confidence, arrival_time, departure_time)

    if not schedule:
        return VisitOpeningStatus(True, "unknown", 0.0, arrival_time, departure_time)

    windows = _candidate_windows_for_date(schedule, arrival_time)
    for window_start, window_end in windows:
        if arrival_time >= window_start and departure_time <= window_end:
            return VisitOpeningStatus(True, "explicit", max(confidence, 0.75), arrival_time, departure_time)

    if allow_wait:
        for window_start, window_end in windows:
            if arrival_time < window_start:
                waited_departure = window_start + duration
                if waited_departure <= window_end:
                    wait_minutes = round((window_start - arrival_time).total_seconds() / 60)
                    return VisitOpeningStatus(
                        True,
                        "explicit",
                        max(confidence, 0.75),
                        window_start,
                        waited_departure,
                        max(0, int(wait_minutes)),
                    )

    return VisitOpeningStatus(False, "explicit", max(confidence, 0.75), arrival_time, departure_time)


def _parse_days(days_text: str) -> list[str]:
    text = days_text.strip().lower()
    if not text or text == "every day":
        return list(WEEKDAYS)
    if text in DAY_INDEX:
        return [WEEKDAYS[DAY_INDEX[text]]]
    if " - " in text or "-" in text:
        left, right = re.split(r"\s*-\s*", text, maxsplit=1)
        if left in DAY_INDEX and right in DAY_INDEX:
            start = DAY_INDEX[left]
            end = DAY_INDEX[right]
            if start <= end:
                return list(WEEKDAYS[start : end + 1])
            return list(WEEKDAYS[start:]) + list(WEEKDAYS[: end + 1])
    return []


def _is_open_24_hours(open_text: str, close_text: str) -> bool:
    text = f"{open_text} {close_text}".lower()
    return "open 24 hours" in text or "24 hours" in text or text.strip() == "24"


def _parse_time(text: str) -> int | None:
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour == 24:
        return 24 * 60
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _weekday_key(arrival_time: datetime, weekday: str | int | None) -> str:
    if isinstance(weekday, int):
        return WEEKDAYS[weekday % 7]
    if isinstance(weekday, str) and weekday:
        normalized = weekday[:3].lower()
        if normalized in WEEKDAYS:
            return normalized
    return WEEKDAYS[arrival_time.weekday()]


def _candidate_windows_for_date(
    schedule: dict[str, list[Window]],
    arrival_time: datetime,
    weekday: str | int | None = None,
) -> list[tuple[datetime, datetime]]:
    """Return windows that could contain or follow arrival on its real date."""
    today_idx = WEEKDAYS.index(_weekday_key(arrival_time, weekday))
    candidates: list[tuple[datetime, datetime]] = []

    for offset in (-1, 0, 1):
        day_idx = (today_idx + offset) % 7
        day_key = WEEKDAYS[day_idx]
        window_date = arrival_time.date() + timedelta(days=offset)
        for open_min, close_min in schedule.get(day_key, []):
            start = datetime.combine(window_date, datetime.min.time()) + timedelta(minutes=open_min)
            if open_min == 0 and close_min >= 24 * 60:
                end = start + timedelta(days=1)
            elif close_min <= open_min:
                end = datetime.combine(window_date, datetime.min.time()) + timedelta(days=1, minutes=close_min)
            else:
                end = datetime.combine(window_date, datetime.min.time()) + timedelta(minutes=close_min)
            candidates.append((start, end))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _in_window(minute: int, open_min: int, close_min: int) -> bool:
    if close_min >= 24 * 60:
        close_min = 24 * 60
    if open_min == 0 and close_min >= 24 * 60:
        return True
    if close_min >= open_min:
        return open_min <= minute <= close_min
    return minute >= open_min or minute <= close_min


def _type_assumption(group: str, minute: int) -> tuple[bool, float]:
    if group == "culture":
        if minute >= 18 * 60 or minute < 8 * 60 + 30:
            return True, 0.22
        return True, 0.04
    if group == "coffee":
        if minute < 7 * 60 or minute > 23 * 60:
            return True, 0.16
        return True, 0.04
    if group == "nightlife":
        if minute < 17 * 60 + 30:
            return True, 0.30
        return True, 0.04
    if group == "food":
        if minute < 7 * 60 or minute > 22 * 60 + 30:
            return True, 0.20
        return True, 0.04
    return True, 0.04
