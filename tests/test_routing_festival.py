from __future__ import annotations

from datetime import date, datetime

from src.agents.routing_agent import RoutingAgent
from src.models.poi import POI
from src.models.user_input import UserInput


def test_city_fallback_festival_is_allowed_with_low_confidence_warning() -> None:
    agent = RoutingAgent()
    trip_date = date(2026, 3, 26)
    user_input = UserInput(
        interests=["festival", "culture"],
        start_date=trip_date,
        end_date=trip_date,
        start_location=(10.7769, 106.7009),
    )
    start_dt = datetime(2026, 3, 26, 9, 0)
    end_dt = datetime(2026, 3, 26, 21, 0)
    state = agent._init_state(user_input.start_location, start_dt, end_dt, user_input)
    festival = POI(
        id="festival-low-confidence",
        name="Test Festival",
        latitude=10.7753,
        longitude=106.7039,
        primaryType="tourist_attraction",
        types=["tourist_attraction"],
        source="festival",
        geocode_confidence=0.35,
        geocode_method="city_fallback:Nguyen Hue Walking Street",
        estimated_visit_minutes=120,
        openingHours=[{"days": "Every day", "open": "09:00", "close": "21:00"}],
        dates=[trip_date],
        event_start_date=trip_date,
        event_end_date=trip_date,
    )

    ok, arrival, departure, _travel, _hours = agent._check_valid(festival, state, end_dt)

    assert ok
    assert arrival.date() == trip_date
    assert departure.date() == trip_date
    assert "Location confidence is low." in agent._stop_warnings(
        festival,
        user_input,
        arrival,
        departure,
    )
