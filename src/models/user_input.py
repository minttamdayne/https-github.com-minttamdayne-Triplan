from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class UserInput(BaseModel):
    """Structured user request for trip planning."""

    # Required
    interests: list[str]                  # e.g. ["Asian food", "museums", "nightlife"]
    start_date: date
    end_date: date
    start_location: tuple[float, float]   # (latitude, longitude)

    # Budget: 0=free, 1=cheap, 2=moderate, 3=expensive, 4=luxury
    budget_level: int = Field(2, ge=0, le=4)

    # LLM-core intent fields. These are intentionally soft signals: agents use
    # them for ranking and personalization instead of treating them as filters.
    free_text: str = ""
    themes: list[str] = Field(default_factory=list)
    vibes: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)
    time_preferences: dict[str, Any] = Field(default_factory=dict)

    # Optional overrides
    daily_hours: float = 10.0             # hours available per day
    max_places_per_day: int = 8
    start_time: str = "09:00"
    preferred_end_time: str = "21:00"

    # Personalization profile
    travel_group: Literal["solo", "couple", "family", "friends"] = "solo"
    group_type: Literal["solo", "couple", "family", "friends"] = "solo"
    pace: Literal["relaxed", "balanced", "intense"] = "balanced"
    mobility: Literal["normal", "limited"] = "normal"
    has_children: bool = False
    food_restrictions: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    nightlife_preference: Literal["auto", "avoid", "neutral", "like"] = "auto"
    food_priority: Literal["low", "normal", "high"] = "normal"
    culture_priority: Literal["low", "normal", "high"] = "normal"
    outdoor_priority: Literal["low", "normal", "high"] = "normal"
    food_preference: Literal["low", "normal", "high"] = "normal"
    culture_preference: Literal["low", "normal", "high"] = "normal"
    outdoor_preference: Literal["low", "normal", "high"] = "normal"
    gender: Literal["female", "male", "nonbinary", "unknown"] = "unknown"
    age: int | None = Field(None, ge=0, le=120)

    @model_validator(mode="before")
    @classmethod
    def _support_legacy_start_time(cls, values):
        if isinstance(values, dict) and "start_time" not in values and "preferred_start_time" in values:
            values = dict(values)
            values["start_time"] = values["preferred_start_time"]
        if isinstance(values, dict) and values.get("start_time") is None:
            values = dict(values)
            values["start_time"] = "09:00"
        if isinstance(values, dict):
            values = dict(values)
            if "travel_group" not in values and "group_type" in values:
                values["travel_group"] = values["group_type"]
            if "group_type" not in values and "travel_group" in values:
                values["group_type"] = values["travel_group"]
            for new_name, old_name in (
                ("food_preference", "food_priority"),
                ("culture_preference", "culture_priority"),
                ("outdoor_preference", "outdoor_priority"),
            ):
                if old_name not in values and new_name in values:
                    values[old_name] = values[new_name]
                if new_name not in values and old_name in values:
                    values[new_name] = values[old_name]
            if "negative_preferences" in values and "avoid" not in values:
                values["avoid"] = values["negative_preferences"]
        return values

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def city(self) -> str:
        """Placeholder – extend for multi-city support."""
        return "hcm"

    @property
    def preferred_start_time(self) -> str:
        """Backward-compatible alias used by older agents/scripts."""
        return self.start_time
