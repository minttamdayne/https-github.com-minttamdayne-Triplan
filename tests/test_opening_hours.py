from __future__ import annotations

from datetime import datetime

from src.agents.festival_agent import FestivalAgent
from src.models.poi import POI
from src.opening_hours import visit_opening_status
from src.preferences import apply_poi_semantic_overrides


def make_poi(opening_hours: list[dict] | None) -> POI:
    poi = POI(
        id="test",
        name="Test POI",
        latitude=10.0,
        longitude=106.0,
        primaryType="restaurant",
        types=["restaurant"],
        openingHours=opening_hours,
    )
    return apply_poi_semantic_overrides(poi)


def test_same_day_window_accepts_full_visit() -> None:
    poi = make_poi([{"days": "Monday - Saturday", "open": "10:00", "close": "22:00"}])

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 10, 30),
        datetime(2026, 5, 4, 11, 30),
        allow_wait=False,
    )

    assert status.fits
    assert status.source == "explicit"


def test_split_window_waits_for_next_valid_window() -> None:
    poi = make_poi(
        [
            {"days": "Every day", "open": "11:30", "close": "15:00"},
            {"days": "Every day", "open": "17:30", "close": "22:00"},
        ]
    )

    closed_gap = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 16, 0),
        datetime(2026, 5, 4, 17, 0),
        allow_wait=False,
    )
    waited = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 16, 0),
        datetime(2026, 5, 4, 17, 0),
        allow_wait=True,
    )

    assert not closed_gap.fits
    assert waited.fits
    assert waited.arrival == datetime(2026, 5, 4, 17, 30)
    assert waited.departure == datetime(2026, 5, 4, 18, 30)
    assert waited.wait_minutes == 90


def test_overnight_window_accepts_visit_after_midnight() -> None:
    poi = make_poi([{"days": "Friday - Saturday", "open": "17:30", "close": "01:30"}])

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 8, 23, 45),
        datetime(2026, 5, 9, 0, 45),
        allow_wait=False,
    )

    assert status.fits


def test_open_24_hours_accepts_any_time() -> None:
    poi = make_poi([{"days": "Every day", "open": "Open 24 hours"}])

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 3, 0),
        datetime(2026, 5, 4, 4, 0),
        allow_wait=False,
    )

    assert status.fits


def test_closed_when_no_matching_day() -> None:
    poi = make_poi([{"days": "Sunday", "open": "10:00", "close": "22:00"}])

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 10, 30),
        datetime(2026, 5, 4, 11, 30),
        allow_wait=False,
    )

    assert not status.fits
    assert status.source == "explicit"


def test_unknown_opening_hours_are_allowed_but_marked_unknown() -> None:
    poi = make_poi(None)

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 10, 30),
        datetime(2026, 5, 4, 11, 30),
        allow_wait=False,
    )

    assert status.fits
    assert status.source == "unknown"
    assert status.confidence == 0.0


def test_arrival_inside_but_departure_after_close_fails() -> None:
    poi = make_poi([{"days": "Every day", "open": "10:00", "close": "22:00"}])

    status = visit_opening_status(
        poi,
        datetime(2026, 5, 4, 21, 30),
        datetime(2026, 5, 4, 22, 30),
        allow_wait=False,
    )

    assert not status.fits


def test_festival_date_parser_handles_hcm_fest_formats() -> None:
    agent = FestivalAgent()

    assert agent._parse_time("26 - 29/3", 2026) == (
        datetime(2026, 3, 26).date(),
        datetime(2026, 3, 29).date(),
    )
    assert agent._parse_time("31/10 - 11/12/2025", 2026) == (
        datetime(2025, 10, 31).date(),
        datetime(2025, 12, 11).date(),
    )
    assert agent._parse_time("15/10/2025 - 28/2/2026", 2026) == (
        datetime(2025, 10, 15).date(),
        datetime(2026, 2, 28).date(),
    )
    assert agent._parse_time("6/11", 2026) == (
        datetime(2026, 11, 6).date(),
        datetime(2026, 11, 6).date(),
    )


def test_festival_style_opening_hours_become_explicit_when_preprocessed() -> None:
    poi = POI(
        id="fest",
        name="Festival",
        latitude=10.0,
        longitude=106.0,
        primaryType="tourist_attraction",
        types=["tourist_attraction"],
        openingHours=[{"days": "Every day", "open": "09:00", "close": "21:00"}],
        source="festival",
        dates=[datetime(2026, 3, 26).date()],
        event_start_date=datetime(2026, 3, 26).date(),
        event_end_date=datetime(2026, 3, 26).date(),
    )
    poi = apply_poi_semantic_overrides(poi)

    status = visit_opening_status(
        poi,
        datetime(2026, 3, 26, 10, 0),
        datetime(2026, 3, 26, 12, 0),
        allow_wait=False,
    )

    assert status.fits
    assert status.source == "explicit"
    assert poi.opening_hours_confidence >= 0.75
