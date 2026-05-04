from __future__ import annotations

import math
import re
import unicodedata

from src.agents.base import BaseAgent
from src.models.poi import POI
from src.models.user_input import UserInput
from src.preferences import build_preference_profile, category_for_poi, matches_any, normalize_text, poi_text
from src.tools.distance import haversine_km


WEIGHTS = {
    "interest": 0.42,
    "quality": 0.28,
    "budget": 0.15,
    "hidden_gem": 0.10,
    "type_quality": 0.05,
}

PRIOR_RATING = 4.2
PRIOR_COUNT = 50
VIETNAMESE_FOOD_INTERESTS = {
    "vietnamese food", "vietnamese cuisine", "vietnam food",
    "local food", "saigon food", "hcm food", "street food",
    "mon viet", "món việt", "am thuc viet", "ẩm thực việt",
}
VIETNAMESE_KEYWORDS = {
    "vietnamese", "phở", "pho", "bún", "bun",
    "bánh", "banh", "banhmi", "hủ tiếu", "hu tieu", "lẩu", "lau",
    "gỏi", "goi", "chả", "cha", "nem",
}
NON_VIETNAMESE_CUISINE_HINTS = {
    "japanese", "japan", "sushi", "ramen", "thai", "korean", "bbq",
    "pizza", "italian", "french", "mexican", "indian", "chinese",
    "fusion", "burger", "steak", "takoyaki",
}
FOOD_TYPES = {"restaurant", "food", "meal_takeaway", "meal_delivery", "bakery"}
CULTURE_TYPES = {"museum", "art_gallery", "tourist_attraction", "place_of_worship"}
COFFEE_TYPES = {"cafe", "coffee_shop"}
NIGHTLIFE_TYPES = {"bar", "night_club"}
FAMILY_FRIENDLY_TYPES = {
    "park", "zoo", "aquarium", "museum", "art_gallery", "tourist_attraction",
    "cafe", "restaurant", "bakery",
}
FAMILY_RISK_TYPES = {"bar", "night_club"}
LIMITED_MOBILITY_DISTANCE_PENALTY_START = 0.35
INVALID_TYPES = {
    "lodging", "hotel", "supermarket", "store", "baby_store", "toy_store",
    "apartment", "real_estate_agency", "real_estate", "spa", "gym",
    "hospital", "school", "office", "corporate_office",
    "local_government_office", "travel_agency", "insurance_agency", "bank",
    "atm", "car_dealer", "car_rental",
}
INVALID_NAME_PATTERNS = (
    "hotel", "khach san", "khách sạn", "supermarket", "sieu thi", "siêu thị",
    "baby", "mother", "me be", "mẹ bé", "toy", "do choi", "đồ chơi",
    "apartment", "lease", "cho thue", "cho thuê", "real estate",
    "bat dong san", "bất động sản", "spa", "gym", "office", "van phong",
    "văn phòng", "company", "corp", "co ltd", "co., ltd", "tnhh",
    "cong ty", "công ty", "travel agency", "du lich", "du lịch",
)


class ScoringAgent(BaseAgent):
    """Agent 3: deterministic MCDM scoring for candidate POIs."""

    name: str = "scoring"

    async def _execute(self, **kwargs: Any) -> list[POI]:
        candidates: list[POI] = kwargs["candidates"]
        user_input: UserInput = kwargs["user_input"]
        category_weights: dict[str, float] = kwargs.get("CATEGORY_WEIGHTS", {})

        if not candidates:
            return []

        proximity_scores = self._proximity_scores(candidates=candidates, start=user_input.start_location)
        raw_interest_scores = {
            id(poi): self._raw_interest_fit(
                poi=poi,
                user_interests=user_input.interests,
                category_weights=category_weights,
            )
            for poi in candidates
        }
        invalid_count = 0

        for poi in candidates:
            valid_entity = self._is_valid_entity(poi)
            if not valid_entity:
                invalid_count += 1

            poi.interest_score = self._interest_fit(raw_interest_scores[id(poi)])
            poi.interest_fit = poi.interest_score
            poi.quality_score = self._quality_score(
                rating=poi.rating,
                count=getattr(poi, "user_rating_count", None),
            )
            poi.budget_score = self._budget_fit(
                price_level=poi.price_level,
                user_budget=user_input.budget_level,
            )
            poi.budget_fit = poi.budget_score
            poi.proximity_score = proximity_scores.get(id(poi), 0.5)
            poi.hidden_gem_score = self._hidden_gem(
                rating=poi.rating,
                count=getattr(poi, "user_rating_count", None),
            )
            poi.type_quality_score = self._type_quality_score(poi)
            poi.composite_score = self._static_score(
                interest=poi.interest_score,
                quality=poi.quality_score,
                budget=poi.budget_score,
                hidden_gem=poi.hidden_gem_score,
                type_quality=poi.type_quality_score,
            )
            poi.composite_score = self._personalized_score(poi, poi.composite_score, user_input)
            poi.composite_score = self._intent_adjusted_score(poi, poi.composite_score, user_input)
            if not valid_entity:
                poi.composite_score = min(poi.composite_score, 0.05)

        ranked = sorted(candidates, key=lambda p: p.composite_score, reverse=True)
        self.memory.set("scored_candidates", ranked)
        self._log_proximity_distribution(ranked)
        self.logger.info(
            "Scored %d candidates. invalid_seen=%d score_range=%.3f-%.3f",
            len(ranked),
            invalid_count,
            min(p.composite_score for p in ranked),
            max(p.composite_score for p in ranked),
        )
        return ranked

    def _raw_interest_fit(
        self,
        poi: POI,
        user_interests: list[str],
        category_weights: dict[str, float],
    ) -> float:
        """Stable monotonic interest score derived from SemanticAgent output."""
        score = self._clamp01(getattr(poi, "interest_fit", 0.0) or 0.0)

        category_bonus = self._clamp01(category_weights.get(poi.primary_type, 0.0))
        if category_bonus > 0.0:
            score = max(score, category_bonus)

        score = self._apply_vietnamese_food_preference(poi, user_interests, score)
        return self._clamp01(score)

    @staticmethod
    def _interest_fit(raw_score: float) -> float:
        """Light power sharpening. No log/sigmoid time matching here."""
        return ScoringAgent._clamp01(ScoringAgent._clamp01(raw_score) ** 1.25)

    @staticmethod
    def _quality_score(rating: float | None, count: int | None) -> float:
        """Bayesian quality score on [0, 1], with m=50 and C=4.2."""
        safe_rating = ScoringAgent._safe_rating(rating, default=PRIOR_RATING)
        safe_count = ScoringAgent._safe_count(count, default=PRIOR_COUNT)
        bayes = (
            (safe_count / (safe_count + PRIOR_COUNT)) * safe_rating
            + (PRIOR_COUNT / (safe_count + PRIOR_COUNT)) * PRIOR_RATING
        )
        return ScoringAgent._clamp01((bayes - 3.0) / 2.0)

    @staticmethod
    def _budget_fit(price_level: int | None, user_budget: int | None) -> float:
        """Asymmetric fit: cheaper is fine, over-budget decays smoothly."""
        if price_level is None or user_budget is None:
            return 0.8
        diff = price_level - user_budget
        if diff <= 0:
            return 1.0
        scale = 2.2 if user_budget <= 1 else 1.5
        return math.exp(-scale * (diff ** 2))

    @staticmethod
    def _hidden_gem(
        rating: float | None,
        count: int | None,
    ) -> float:
        """Boost only genuinely strong low/medium review-count POIs."""
        safe_rating = ScoringAgent._safe_rating(rating, default=PRIOR_RATING)
        safe_count = ScoringAgent._safe_count(count, default=0)
        if safe_rating < 4.4 or not (20 <= safe_count <= 300):
            return 0.0
        rating_part = (safe_rating - 4.4) / 0.6
        count_part = 1.0 - abs(safe_count - 120) / 180
        return ScoringAgent._clamp01(0.65 * rating_part + 0.35 * max(0.0, count_part))

    @staticmethod
    def _static_score(
        interest: float,
        quality: float,
        budget: float,
        hidden_gem: float,
        type_quality: float,
    ) -> float:
        return ScoringAgent._clamp01(
            WEIGHTS["interest"] * ScoringAgent._clamp01(interest)
            + WEIGHTS["quality"] * ScoringAgent._clamp01(quality)
            + WEIGHTS["budget"] * ScoringAgent._clamp01(budget)
            + WEIGHTS["hidden_gem"] * ScoringAgent._clamp01(hidden_gem)
            + WEIGHTS["type_quality"] * ScoringAgent._clamp01(type_quality)
        )

    @staticmethod
    def _type_quality_score(poi: POI) -> float:
        group = ScoringAgent._category_group(poi)
        if getattr(poi, "source", "") == "festival":
            return 1.0
        if group == "culture":
            return 1.0
        if group == "nightlife":
            return 0.90
        if group == "coffee":
            return 0.82
        if group == "food":
            return 0.78
        return 0.35

    def _personalized_score(self, poi: POI, score: float, user_input: UserInput) -> float:
        profile = build_preference_profile(user_input)
        group = category_for_poi(poi)
        text = poi_text(poi)
        adjusted = score

        adjusted *= profile.category_weight(group)

        if profile.must_have_terms and matches_any(text, profile.must_have_terms):
            adjusted = min(1.0, adjusted + 0.16)
        if profile.avoid_terms and matches_any(text, profile.avoid_terms):
            adjusted *= 0.28

        if profile.family_mode and any(term in text for term in ("family", "kid", "kids", "children", "playground")):
            adjusted = min(1.0, adjusted + 0.08)

        if profile.young_traveler:
            young_terms = (
                "aesthetic", "instagram", "photo", "view", "rooftop", "boutique",
                "concept", "specialty coffee", "coffee", "cafe", "kafe",
                "riverside", "riverfront", "walking street", "nguyen hue",
                "market", "shopping", "street food", "banh mi", "pho", "bun",
            )
            if group in {"coffee", "food", "shopping", "outdoor", "nightlife"}:
                adjusted = min(1.0, adjusted + 0.05)
            if any(term in text for term in young_terms):
                adjusted = min(1.0, adjusted + 0.09)

        if user_input.mobility == "limited":
            proximity = getattr(poi, "proximity_score", 0.5)
            if proximity < LIMITED_MOBILITY_DISTANCE_PENALTY_START:
                adjusted *= 0.78 + 0.22 * max(0.0, proximity / LIMITED_MOBILITY_DISTANCE_PENALTY_START)

        if getattr(poi, "source", "") == "festival":
            confidence = max(0.0, min(1.0, getattr(poi, "geocode_confidence", 1.0)))
            if confidence <= 0.35:
                adjusted *= 0.45
            elif confidence < 0.65:
                adjusted *= 0.75

        if self._violates_food_restrictions(poi, user_input.food_restrictions):
            adjusted *= 0.35

        return self._clamp01(adjusted)

    def _intent_adjusted_score(self, poi: POI, score: float, user_input: UserInput) -> float:
        text = poi_text(poi)
        group = category_for_poi(poi)
        vibes = {self._normalize_token(vibe) for vibe in user_input.vibes}
        themes = {self._normalize_token(theme) for theme in user_input.themes}
        adjusted = score
        must_have_text = normalize_text(" ".join(user_input.must_have))

        if "chill" in vibes:
            if group in {"coffee", "outdoor"}:
                adjusted += 0.08
            if any(term in text for term in ("lounge", "garden", "riverside", "riverfront", "park", "cafe", "coffee")):
                adjusted += 0.06
            if group == "culture" and any(term in text for term in ("war", "museum")):
                adjusted -= 0.03

        if "less touristy" in vibes or "hidden gem" in vibes:
            review_count = getattr(poi, "user_rating_count", None) or 0
            iconic = self._is_iconic_or_touristy(poi)
            if 20 <= review_count <= 350:
                adjusted += 0.08
            if any(term in text for term in ("local", "quan", "quán", "hem", "hẻm", "market")):
                adjusted += 0.06
            if iconic and not any(term in must_have_text for term in ("iconic", "bieu tuong", "biểu tượng", "landmark")):
                adjusted *= 0.78

        if "scenic view" in vibes:
            if any(term in text for term in ("view", "rooftop", "sky", "river", "riverside", "riverfront", "landmark", "nguyen hue")):
                adjusted += 0.12
            elif group in {"coffee", "nightlife", "outdoor"}:
                adjusted += 0.05

        if "local" in vibes or "local" in themes:
            if group == "food":
                adjusted += 0.10
            if any(term in text for term in ("pho", "phở", "bun", "bún", "com", "cơm", "banh", "bánh", "quan", "quán", "market")):
                adjusted += 0.08
            if self._has_non_vietnamese_cuisine_hint(poi):
                adjusted *= 0.72

        if "first timer" in vibes or "iconic" in themes:
            if self._is_iconic_or_touristy(poi):
                adjusted += 0.10
            if group in {"food", "coffee"}:
                adjusted += 0.06
            if group == "nightlife" and user_input.nightlife_preference != "avoid":
                adjusted += 0.04

        if user_input.nightlife_preference == "like" and any(
            term in normalize_text(user_input.free_text)
            for term in ("nightlife nhe", "nightlife nhẹ", "light nightlife", "rooftop", "lounge")
        ):
            if group == "nightlife":
                adjusted += 0.08
            if poi.primary_type == "night_club" and "light nightlife" in normalize_text(user_input.free_text):
                adjusted *= 0.86

        return self._clamp01(adjusted)

    @staticmethod
    def _normalize_token(text: str) -> str:
        return normalize_text(text).replace("_", " ")

    @staticmethod
    def _is_iconic_or_touristy(poi: POI) -> bool:
        text = poi_text(poi)
        return any(
            term in text
            for term in (
                "tourist attraction", "landmark", "war remnants", "independence",
                "ben thanh", "notre dame", "post office", "nguyen hue", "opera house",
                "bitexco",
            )
        )

    @staticmethod
    def _violates_food_restrictions(poi: POI, restrictions: list[str]) -> bool:
        if not restrictions or not ScoringAgent._is_food_poi(poi):
            return False
        text = ScoringAgent._poi_text(poi)
        normalized = [ScoringAgent._normalize_text(r).strip() for r in restrictions]
        restriction_map = {
            "vegetarian": ("steak", "bbq", "grill", "seafood", "meat", "beef", "pork", "chicken"),
            "vegan": ("steak", "bbq", "grill", "seafood", "meat", "beef", "pork", "chicken", "milk", "cheese"),
            "halal": ("pork", "bar", "beer", "wine", "cocktail"),
            "no seafood": ("seafood", "fish", "sushi", "oyster", "snail"),
            "seafood allergy": ("seafood", "fish", "sushi", "oyster", "snail"),
        }
        for restriction in normalized:
            keywords = restriction_map.get(restriction, (restriction,))
            if any(keyword in text for keyword in keywords):
                return True
        return False

    @staticmethod
    def _proximity_scores(
        candidates: list[POI],
        start: tuple[float, float],
    ) -> dict[int, float]:
        distances: dict[int, float] = {}
        for poi in candidates:
            if poi.latitude is None or poi.longitude is None:
                distances[id(poi)] = float("inf")
                continue
            distances[id(poi)] = haversine_km(
                start[0], start[1], poi.latitude, poi.longitude
            )

        finite_distances = [d for d in distances.values() if math.isfinite(d)]
        if not finite_distances:
            return {id(poi): 0.5 for poi in candidates}

        max_distance = max(finite_distances)
        if max_distance <= 0:
            return {id(poi): 1.0 for poi in candidates}

        proximity: dict[int, float] = {}
        for poi_id, dist in distances.items():
            if not math.isfinite(dist):
                proximity[poi_id] = 0.0
                continue
            scaled = 1.0 - (dist / max_distance)
            proximity[poi_id] = ScoringAgent._clamp01(max(0.02, scaled))
        return proximity

    @staticmethod
    def _category_group(poi: POI) -> str:
        return category_for_poi(poi)

    def _log_proximity_distribution(self, pois: list[POI]) -> None:
        bins = {
            "0.00-0.20": 0,
            "0.20-0.40": 0,
            "0.40-0.60": 0,
            "0.60-0.80": 0,
            "0.80-1.00": 0,
        }
        for poi in pois:
            prox = self._clamp01(getattr(poi, "proximity_score", 0.0))
            if prox < 0.2:
                bins["0.00-0.20"] += 1
            elif prox < 0.4:
                bins["0.20-0.40"] += 1
            elif prox < 0.6:
                bins["0.40-0.60"] += 1
            elif prox < 0.8:
                bins["0.60-0.80"] += 1
            else:
                bins["0.80-1.00"] += 1
        self.logger.info("Proximity histogram: %s", bins)

    def _apply_vietnamese_food_preference(
        self,
        poi: POI,
        user_interests: list[str],
        current_score: float,
    ) -> float:
        if not self._wants_vietnamese_food(user_interests):
            return current_score
        if not self._is_food_poi(poi):
            return current_score

        if self._is_vietnamese_food(poi):
            return max(current_score, 0.92)
        if self._has_non_vietnamese_cuisine_hint(poi):
            return min(current_score, 0.46)
        return min(current_score, 0.62)

    @staticmethod
    def _wants_vietnamese_food(user_interests: list[str]) -> bool:
        normalized = {ScoringAgent._normalize_text(i) for i in user_interests}
        targets = {ScoringAgent._normalize_text(i) for i in VIETNAMESE_FOOD_INTERESTS}
        return bool(normalized & targets)

    @staticmethod
    def _is_food_poi(poi: POI) -> bool:
        non_food_primary = {"lodging", "hotel", "spa", "shopping_mall", "supermarket"}
        return poi.primary_type not in non_food_primary and bool(
            ScoringAgent._meaningful_types(poi) & FOOD_TYPES
        )

    @staticmethod
    def _is_vietnamese_food(poi: POI) -> bool:
        text = ScoringAgent._poi_text(poi)
        tokens = set(re.findall(r"[\w]+", text))
        for keyword in VIETNAMESE_KEYWORDS:
            if " " in keyword:
                if keyword in text:
                    return True
            elif keyword in tokens:
                return True
        return False

    @staticmethod
    def _has_non_vietnamese_cuisine_hint(poi: POI) -> bool:
        text = ScoringAgent._poi_text(poi)
        return any(keyword in text for keyword in NON_VIETNAMESE_CUISINE_HINTS)

    @staticmethod
    def _meaningful_types(poi: POI) -> set[str]:
        generic = {"establishment", "point_of_interest"}
        return ({poi.primary_type, *poi.types} - generic) - {""}

    @staticmethod
    def _poi_text(poi: POI) -> str:
        return ScoringAgent._normalize_text(
            " ".join([poi.name, poi.primary_type, *poi.types])
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        decomposed = unicodedata.normalize("NFD", lowered)
        ascii_text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return f"{lowered} {ascii_text}"

    @staticmethod
    def _safe_rating(rating: float | None, default: float) -> float:
        if rating is None:
            return default
        try:
            return min(max(float(rating), 1.0), 5.0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_count(count: int | None, default: int = 0) -> int:
        if count is None:
            return default
        try:
            return max(0, int(count))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            if math.isnan(float(value)):
                return 0.0
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _is_valid_entity(cls, poi: POI) -> bool:
        if not poi.name or not poi.primary_type or not poi.types:
            return False
        if poi.source == "festival":
            return True

        poi_types = {poi.primary_type.lower(), *(t.lower() for t in poi.types)}
        if poi_types & INVALID_TYPES:
            return False
        normalized_for_invalid = cls._normalize_text(poi.name)
        if any(pattern in normalized_for_invalid for pattern in INVALID_NAME_PATTERNS):
            return False

        name = poi.name.strip()
        if len(name) < 3 or len(name) > 90:
            return False
        if re.search(r"[\uFFFD\x00-\x1F]", name):
            return False

        letters = re.findall(r"[A-Za-zÀ-ỹ]", name)
        if not letters:
            return False

        tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]+", name)
        if not tokens:
            return False

        normalized = " ".join(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", name.lower()))
        generic_bad = {"unknown", "test", "placeholder", "cổng", "cong"}
        if normalized in generic_bad or re.fullmatch(r"(gate|cong|cổng)\s*\d+", normalized):
            return False

        return True
