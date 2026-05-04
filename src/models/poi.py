from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OpeningHours(BaseModel):
    days: str
    open: str = ""
    close: str = ""


class POI(BaseModel):
    """Unified Point-of-Interest model.

    Both Google Maps POIs and enriched festivals are normalised into this
    schema so that every downstream agent works with a single type.
    """

    id: str
    name: str
    latitude: float
    longitude: float
    address: str = ""
    ward: Optional[str] = None
    district: Optional[str] = None
    province: str = ""
    primary_type: str = Field("", alias="primaryType")
    types: list[str] = Field(default_factory=list)
    rating: Optional[float] = None
    user_rating_count: Optional[int] = Field(None, alias="userRatingCount")
    timezone: str = "Asia/Ho_Chi_Minh"
    opening_hours: Optional[list[OpeningHours]] = Field(None, alias="openingHours")
    price_level: Optional[int] = Field(None, alias="priceLevel")
    price_range: Optional[str] = Field(None, alias="priceRange")

    # ── Extended fields (set by pipeline) ──
    source: str = "google_maps"  # "google_maps" | "festival"
    interest_fit: float = 0.0
    quality_score: float = 0.0
    budget_fit: float = 0.0
    proximity_score: float = 0.0
    interest_score: float = 0.0
    budget_score: float = 0.0
    hidden_gem_score: float = 0.0
    type_quality_score: float = 0.0
    composite_score: float = 0.0
    estimated_visit_minutes: int = 60
    geocode_confidence: float = 1.0
    geocode_method: str = "source"
    semantic_group_override: Optional[str] = None
    food_role: Optional[str] = None  # "meal" | "snack" | "cafe"
    opening_schedule: Optional[dict[str, list[tuple[int, int]]]] = None
    opening_hours_confidence: float = 0.0

    model_config = {"populate_by_name": True}
