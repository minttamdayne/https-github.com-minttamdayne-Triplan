from __future__ import annotations

from datetime import date, time
from typing import Optional

from pydantic import BaseModel, Field

from src.models.poi import POI


class ItineraryStop(BaseModel):
    """A single stop within a day."""

    order: int
    poi: POI
    arrival_time: Optional[time] = None
    departure_time: Optional[time] = None
    travel_minutes_from_prev: float = 0.0
    visit_minutes: int = 60
    notes: str = ""  # e.g. "Festival – check agenda"
    reason: str = ""
    matched_interests: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    """One day of the trip."""

    day_number: int
    date: date
    stops: list[ItineraryStop] = Field(default_factory=list)
    total_score: float = 0.0
    total_travel_minutes: float = 0.0
    total_visit_minutes: float = 0.0

    @property
    def total_time_minutes(self) -> float:
        return self.total_travel_minutes + self.total_visit_minutes


class Itinerary(BaseModel):
    """Complete multi-day trip plan – the final output."""

    city: str
    days: list[DayPlan] = Field(default_factory=list)
    total_score: float = 0.0
    metadata: dict = Field(default_factory=dict)  # any extra info
