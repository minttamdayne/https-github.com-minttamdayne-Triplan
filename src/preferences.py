from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import unicodedata

from src.models.user_input import UserInput
from src.opening_hours import parse_opening_schedule


FOOD_TYPES = {"restaurant", "food", "meal_takeaway", "meal_delivery", "bakery"}
CULTURE_TYPES = {"museum", "art_gallery", "tourist_attraction", "place_of_worship"}
COFFEE_TYPES = {"cafe", "coffee_shop"}
NIGHTLIFE_TYPES = {"bar", "night_club"}
OUTDOOR_TYPES = {"park", "zoo", "aquarium", "amusement_park"}
SHOPPING_TYPES = {"market", "shopping_mall"}

PRIORITY_MULTIPLIERS = {"low": 0.78, "normal": 1.0, "high": 1.22}


@dataclass(frozen=True)
class PreferenceProfile:
    """Soft personalization weights derived from explicit and inferred user preferences."""

    category_weights: dict[str, float]
    travel_penalty_multiplier: float
    buffer_minutes: int
    max_stops: int
    nightlife_preference: str
    family_mode: bool
    must_have_terms: tuple[str, ...]
    avoid_terms: tuple[str, ...]
    young_traveler: bool

    def category_weight(self, group: str) -> float:
        return self.category_weights.get(group, 1.0)


def build_preference_profile(user_input: UserInput) -> PreferenceProfile:
    family_mode = user_input.travel_group == "family" or user_input.has_children
    category_weights = {
        "food": PRIORITY_MULTIPLIERS[user_input.food_priority],
        "culture": PRIORITY_MULTIPLIERS[user_input.culture_priority],
        "outdoor": PRIORITY_MULTIPLIERS[user_input.outdoor_priority],
        "coffee": 1.0,
        "nightlife": 1.0,
        "shopping": 1.0,
        "festival": 1.0,
        "other": 1.0,
    }

    intent_terms = [
        user_input.free_text,
        *user_input.themes,
        *user_input.interests,
        *user_input.vibes,
        *user_input.must_have,
        *user_input.negative_preferences,
    ]
    text = normalize_text(" ".join(intent_terms))
    explicit_groups = _explicit_groups(text)
    for group in explicit_groups:
        category_weights[group] = max(category_weights.get(group, 1.0), 1.18)

    vibe_set = {normalize_text(vibe) for vibe in user_input.vibes}
    theme_set = {normalize_text(theme) for theme in user_input.themes}
    if "local" in vibe_set or "local" in theme_set:
        category_weights["food"] = max(category_weights["food"], 1.26)
        category_weights["shopping"] = max(category_weights["shopping"], 1.08)
    if "scenic view" in text or "scenic view" in vibe_set:
        category_weights["coffee"] = max(category_weights["coffee"], 1.18)
        category_weights["outdoor"] = max(category_weights["outdoor"], 1.16)
        category_weights["nightlife"] = max(category_weights["nightlife"], 1.08)
    if "chill" in vibe_set:
        category_weights["coffee"] = max(category_weights["coffee"], 1.16)
        category_weights["outdoor"] = max(category_weights["outdoor"], 1.14)
        category_weights["nightlife"] = max(category_weights["nightlife"], 0.92)
    if "first timer" in vibe_set or "first timer" in text:
        category_weights["culture"] = max(category_weights["culture"], 1.18)
        category_weights["food"] = max(category_weights["food"], 1.16)
        category_weights["coffee"] = max(category_weights["coffee"], 1.08)

    if family_mode:
        category_weights["outdoor"] = max(category_weights["outdoor"], 1.18)
        category_weights["culture"] = max(category_weights["culture"], 1.10)
        category_weights["coffee"] = max(category_weights["coffee"], 1.06)
        category_weights["nightlife"] *= 0.58

    nightlife_preference = _resolve_nightlife_preference(user_input, text)
    if nightlife_preference == "avoid":
        category_weights["nightlife"] *= 0.35
    elif nightlife_preference == "like":
        category_weights["nightlife"] = max(category_weights["nightlife"], 1.18)
    elif nightlife_preference == "neutral":
        category_weights["nightlife"] = max(category_weights["nightlife"], 0.85)

    pace_caps = {"relaxed": 5, "balanced": 7, "intense": 9}
    pace_buffers = {"relaxed": 25, "balanced": 15, "intense": 10}
    travel_penalty = {"relaxed": 1.18, "balanced": 1.0, "intense": 0.88}[user_input.pace]
    buffer_minutes = pace_buffers[user_input.pace]
    max_stops = pace_caps[user_input.pace]
    if "chill" in vibe_set or "too_many_stops" in {normalize_text(x) for x in user_input.negative_preferences}:
        max_stops = min(max_stops, 5)
        buffer_minutes = max(buffer_minutes, 25)
        travel_penalty = max(travel_penalty, 1.18)

    if user_input.mobility == "limited":
        travel_penalty += 0.45
        buffer_minutes += 10
        category_weights["outdoor"] *= 0.92
        category_weights["nightlife"] *= 0.85

    return PreferenceProfile(
        category_weights={k: _clamp(v, 0.2, 1.6) for k, v in category_weights.items()},
        travel_penalty_multiplier=travel_penalty,
        buffer_minutes=buffer_minutes,
        max_stops=max_stops,
        nightlife_preference=nightlife_preference,
        family_mode=family_mode,
        must_have_terms=tuple(normalize_text(term) for term in user_input.must_have if term.strip()),
        avoid_terms=tuple(
            normalize_text(term)
            for term in [*user_input.avoid, *user_input.negative_preferences]
            if term.strip()
        ),
        young_traveler=bool(user_input.age is not None and 18 <= user_input.age <= 30),
    )


def category_for_poi(poi: Any) -> str:
    override = getattr(poi, "semantic_group_override", None)
    if override:
        return override
    if getattr(poi, "source", "") == "festival":
        return "festival"
    keyword_group = infer_semantic_group_from_text(poi_text(poi))
    if keyword_group:
        return keyword_group
    poi_types = {getattr(poi, "primary_type", ""), *getattr(poi, "types", [])}
    if poi_types & COFFEE_TYPES:
        return "coffee"
    if poi_types & NIGHTLIFE_TYPES:
        return "nightlife"
    if poi_types & FOOD_TYPES:
        return "food"
    if poi_types & CULTURE_TYPES:
        return "culture"
    if poi_types & OUTDOOR_TYPES:
        return "outdoor"
    if poi_types & SHOPPING_TYPES:
        return "shopping"
    return "other"


def apply_poi_semantic_overrides(poi: Any) -> Any:
    group = infer_semantic_group_from_text(poi_text(poi))
    if group:
        try:
            poi.semantic_group_override = group
        except Exception:
            pass
    role = infer_food_role(poi)
    if role:
        try:
            poi.food_role = role
        except Exception:
            pass
    try:
        schedule, confidence = parse_opening_schedule(getattr(poi, "opening_hours", None))
        poi.opening_schedule = schedule or None
        poi.opening_hours_confidence = confidence
    except Exception:
        pass
    return poi


def infer_semantic_group_from_text(text: str) -> str | None:
    food_keywords = (
        "bun", "bún", "pho", "phở", "banh mi", "bánh mì", "com tam", "cơm tấm",
        "hu tieu", "hủ tiếu", "ramen", "restaurant", "nha hang", "nhà hàng",
        "quan an", "quán ăn", "bistro", "pizza", "yakiniku", "bbq", "sandwich",
    )
    culture_keywords = ("museum", "bao tang", "bảo tàng", "gallery", "galery", "art", "exhibition")
    coffee_keywords = ("coffee", "cafe", "ca phe", "cà phê", "kafe")
    outdoor_keywords = ("park", "cong vien", "công viên", "riverfront", "riverside", "embankment")
    shopping_keywords = ("market", "shopping", "mall", "boutique", "store", "souvenir")
    nightlife_keywords = ("bar", "pub", "club", "cocktail", "beer", "rooftop")
    if any(keyword in text for keyword in culture_keywords):
        return "culture"
    if any(keyword in text for keyword in food_keywords):
        return "food"
    if any(keyword in text for keyword in coffee_keywords):
        return "coffee"
    if any(keyword in text for keyword in outdoor_keywords):
        return "outdoor"
    if any(keyword in text for keyword in shopping_keywords):
        return "shopping"
    if any(keyword in text for keyword in nightlife_keywords):
        return "nightlife"
    return None


def infer_food_role(poi: Any) -> str | None:
    text = poi_text(poi)
    group = infer_semantic_group_from_text(text) or getattr(poi, "semantic_group_override", None)
    primary_type = getattr(poi, "primary_type", "")
    if group == "coffee" or primary_type in COFFEE_TYPES:
        return "cafe"
    snack_keywords = ("banh mi", "bánh mì", "banh trang", "bánh tráng", "bakery", "dessert", "snack", "juice")
    meal_keywords = ("pho", "phở", "bun", "bún", "com tam", "cơm tấm", "hu tieu", "hủ tiếu", "restaurant", "nha hang", "ramen", "yakiniku", "bbq", "pizza")
    if any(keyword in text for keyword in snack_keywords):
        return "snack"
    if group == "food" or primary_type in FOOD_TYPES:
        if any(keyword in text for keyword in meal_keywords):
            return "meal"
        if primary_type in {"restaurant", "food", "meal_delivery"}:
            return "meal"
        return "snack"
    return None


def poi_text(poi: Any) -> str:
    return normalize_text(" ".join([
        getattr(poi, "name", ""),
        getattr(poi, "primary_type", ""),
        *getattr(poi, "types", []),
        getattr(poi, "address", "") or "",
    ]))


def matches_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    normalized = normalize_text(text)
    return any(term and term in normalized for term in terms)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    ascii_text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"[a-z0-9À-ỹ]+", f"{lowered} {ascii_text}"))


def _resolve_nightlife_preference(user_input: UserInput, text: str) -> str:
    if user_input.nightlife_preference != "auto":
        return user_input.nightlife_preference
    if any(term in text for term in ("nightlife", "bar", "club", "cocktail", "pub", "beer", "rooftop")):
        return "like"
    if user_input.travel_group == "family" or user_input.has_children:
        return "avoid"
    return "neutral"


def _explicit_groups(text: str) -> set[str]:
    groups: set[str] = set()
    keywords = {
        "food": ("food", "restaurant", "street food", "local food", "vietnamese food", "lunch", "dinner"),
        "culture": ("museum", "gallery", "art", "history", "culture", "heritage", "attraction"),
        "coffee": ("coffee", "cafe", "brunch", "dessert"),
        "nightlife": ("nightlife", "bar", "club", "cocktail", "pub", "beer", "rooftop"),
        "outdoor": ("park", "zoo", "aquarium", "garden", "family", "kids", "children", "walking"),
        "shopping": ("market", "shopping", "mall", "souvenir"),
        "festival": ("festival", "event", "concert", "exhibition"),
    }
    for group, terms in keywords.items():
        if any(term in text for term in terms):
            groups.add(group)
    return groups


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
