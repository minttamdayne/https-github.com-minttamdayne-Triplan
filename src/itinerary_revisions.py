from __future__ import annotations

from dataclasses import dataclass

from src.models.itinerary import ItineraryStop
from src.models.poi import POI
from src.models.user_input import UserInput
from src.preferences import category_for_poi, normalize_text, poi_text
from src.tools.distance import haversine_km


@dataclass(frozen=True)
class AlternativeSuggestion:
    poi: POI
    distance_km: float
    score: float
    reason: str


def suggest_alternatives(
    *,
    target_stop: ItineraryStop,
    candidates: list[POI],
    user_input: UserInput,
    query: str = "",
    used_poi_ids: set[str] | None = None,
    limit: int = 5,
) -> list[AlternativeSuggestion]:
    """Suggest nearby replacements for a stop.

    If query is empty, alternatives share the original stop's semantic group.
    If query is provided, prioritize POIs whose name/type matches that query
    while still considering proximity and existing composite score.
    """
    used_poi_ids = used_poi_ids or set()
    target = target_stop.poi
    target_group = category_for_poi(target)
    query_terms = _query_terms(query)
    suggestions: list[AlternativeSuggestion] = []

    for poi in candidates:
        if poi.id == target.id or poi.id in used_poi_ids:
            continue
        if getattr(poi, "source", "") == "festival":
            continue
        group = category_for_poi(poi)
        text = poi_text(poi)
        query_match = _matches_query(text, query_terms) if query_terms else group == target_group
        if not query_match:
            continue

        distance = haversine_km(target.latitude, target.longitude, poi.latitude, poi.longitude)
        if distance > 5.0 and not query_terms:
            continue

        proximity_score = max(0.0, 1.0 - min(distance, 5.0) / 5.0)
        preference_score = 0.2 if _matches_query(text, _query_terms(" ".join(user_input.must_have))) else 0.0
        score = (
            0.55 * getattr(poi, "composite_score", 0.0)
            + 0.30 * proximity_score
            + 0.15 * getattr(poi, "quality_score", 0.0)
            + preference_score
        )
        reason = _reason(poi, query, target_group, distance)
        suggestions.append(AlternativeSuggestion(poi=poi, distance_km=distance, score=score, reason=reason))

    suggestions.sort(key=lambda item: (item.score, -item.distance_km), reverse=True)
    return suggestions[:limit]


def _query_terms(query: str) -> list[str]:
    text = normalize_text(query)
    if not text:
        return []
    aliases = {
        "pho": ["pho", "phở"],
        "banh mi": ["banh mi", "bánh mì", "sandwich"],
        "com tam": ["com tam", "cơm tấm"],
        "coffee": ["coffee", "cafe", "ca phe"],
        "cafe": ["coffee", "cafe", "ca phe"],
        "street food": ["street food", "banh trang", "banh mi", "pho", "com tam"],
    }
    terms = [text]
    if len(text.split()) == 1:
        terms.extend(text.split())
    for key, values in aliases.items():
        if key in text:
            for value in values:
                normalized = normalize_text(value)
                terms.append(value.lower())
                terms.append(normalized)
                if len(value.split()) == 1:
                    terms.extend(normalized.split())
    return list(dict.fromkeys(term for term in terms if term))


def _matches_query(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    return any(term in text for term in terms)


def _reason(poi: POI, query: str, target_group: str, distance: float) -> str:
    parts = []
    if query:
        parts.append(f"matches '{query}'")
    else:
        parts.append(f"same {target_group} vibe")
    if distance <= 1.0:
        parts.append("very close to the original stop")
    elif distance <= 3.0:
        parts.append("nearby")
    if getattr(poi, "quality_score", 0.0) >= 0.65:
        parts.append("strong review signal")
    return "; ".join(parts)
