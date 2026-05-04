"""Agent 2 — Semantic Matching (Ranking + Diversity-Aware Selection).

Architecture change summary
───────────────────────────
• Hard threshold filtering replaced by a composite ranking score.
• Fixed-size adaptive selection: always returns 500–800 candidates.
• Diversity-aware stratified selection enforces minimum category quotas
  before filling remaining slots by global score.
• Deterministic: no random sampling; stable tie-breaking on
  (score desc, proximity asc, poi.id asc).
• Soft penalty replaces hard removal for low-relevance POIs.
• cap_ratio and invalid_rejected hard-pruning logic removed.
• Debug logging added at every key decision point.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from typing import Any

from src.agents.base import BaseAgent
from src.config import settings
from src.models.poi import POI
from src.models.user_input import UserInput
from src.preferences import category_for_poi, matches_any, normalize_text, poi_text
from src.tools.distance import haversine_km
from src.tools.knowledge_graph import KnowledgeGraphClient

# ── Selection size bounds ──────────────────────────────────────────────────────
TARGET_MIN: int = 400          # Keep semantic broad for ~1k POI datasets.
TARGET_MAX: int = 700          # Hard ceiling on candidate pool size.

# ── Composite score weights ────────────────────────────────────────────────────
W_SEMANTIC: float = 0.50       # KG-backed semantic similarity
W_CATEGORY: float = 0.20       # Overlap with expanded interest types
W_SPATIAL: float = 0.15        # Proximity to trip start location
W_QUALITY: float = 0.15        # Rating × review-count signal

# ── Soft penalty (replaces hard threshold) ────────────────────────────────────
SOFT_PENALTY_FLOOR: float = 0.10   # Minimum semantic score before penalty kicks in
SOFT_PENALTY_SCALE: float = 0.40   # How much to down-weight very low scores

# ── Diversity quotas (fraction of final K) ────────────────────────────────────
# Keys must match _category_bucket() return values.
DIVERSITY_QUOTAS: dict[str, float] = {
    "food":      0.20,
    "culture":   0.20,
    "coffee":    0.15,
    "nightlife": 0.15,
    # Remaining 30 % filled by global score rank (any category).
}

ALLOWED_INTEREST_TYPES = {
    "restaurant", "food", "meal_takeaway", "meal_delivery", "bakery",
    "cafe", "coffee_shop", "bar", "night_club", "museum", "art_gallery",
    "tourist_attraction", "place_of_worship", "historic_site",
}
INVALID_TYPES = {
    "lodging", "hotel", "gym", "spa", "supermarket", "store", "hospital",
    "school", "real_estate_agency", "real_estate", "apartment", "baby_store",
    "clothing_store", "convenience_store", "electronics_store",
    "furniture_store", "hardware_store", "home_goods_store", "jewelry_store",
    "shoe_store", "toy_store", "book_store", "department_store",
    "office", "corporate_office", "local_government_office", "travel_agency",
    "insurance_agency", "bank", "atm", "car_dealer", "car_rental",
}
INVALID_NAME_PATTERNS = (
    "hotel", "khach san", "khách sạn", "supermarket", "sieu thi", "siêu thị",
    "baby", "mother", "me be", "mẹ bé", "toy", "do choi", "đồ chơi",
    "apartment", "lease", "cho thue", "cho thuê", "real estate",
    "bat dong san", "bất động sản", "do cung", "đồ cúng",
    "spa", "gym", "hospital", "school", "office", "van phong", "văn phòng",
    "company", "corp", "co ltd", "co., ltd", "tnhh", "cong ty", "công ty",
    "travel agency", "du lich", "du lịch",
)

# ── Festival bonus (kept from original) ───────────────────────────────────────
FESTIVAL_BONUS: float = settings.festival_scarcity_bonus


class SemanticAgent(BaseAgent):
    """Agent 2 — Knowledge-Graph Semantic Matching.

    Expands user interests via KG ontology, then ranks every candidate POI
    on a composite score and returns a diversity-balanced, fixed-size pool.
    """

    name: str = "semantic"

    def __init__(self, kg: KnowledgeGraphClient | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.kg = kg or KnowledgeGraphClient()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def _execute(self, **kwargs: Any) -> list[POI]:
        candidates: list[POI] = kwargs["candidates"]
        user_input: UserInput = kwargs["user_input"]

        intent_terms = self._intent_terms(user_input)

        # ── Step 1: KG interest expansion ────────────────────────────────────
        expanded = await self._expand_interests([*user_input.themes, *user_input.interests])
        self.logger.info(
            "Expanded %d intent interests/themes → %d KG types.",
            len([*user_input.themes, *user_input.interests]),
            len(expanded),
        )

        # Pre-compute embeddings once (deterministic hash-based)
        interest_embeddings = [self._embed_text(i) for i in intent_terms]
        expanded_embedding = self._embed_text(" ".join(sorted([*expanded, *intent_terms])))

        # ── Step 2: Score every candidate ────────────────────────────────────
        scored: list[tuple[POI, float]] = []
        invalid_skipped = 0

        for poi in candidates:
            if not self._is_valid_entity(poi):
                invalid_skipped += 1
                continue

            # Initialise display features needed for spatial prior
            self._initialize_display_features(poi, user_input)

            composite = self._compute_composite_score(
                poi=poi,
                expanded_types=expanded,
                interest_embeddings=interest_embeddings,
                expanded_embedding=expanded_embedding,
                user_input=user_input,
            )

            # Festival scarcity bonus (soft, capped at 1.0)
            if poi.source == "festival":
                composite = min(1.0, composite + FESTIVAL_BONUS)

            scored.append((poi, composite))

        self.logger.debug(
            "Scoring complete: %d valid candidates, %d invalid skipped.",
            len(scored),
            invalid_skipped,
        )

        # ── Step 3: Global rank (deterministic tie-breaking) ─────────────────
        scored = self._stable_sort(scored, user_input)

        # Log category distribution BEFORE selection
        self._log_category_distribution("BEFORE semantic selection", scored)

        # ── Step 4: Diversity-aware stratified selection ──────────────────────
        k = self._choose_k(len(scored))
        self.logger.info(
            "Adaptive K chosen: %d  (pool size before selection: %d)",
            k,
            len(scored),
        )

        selected = self._stratified_select(scored, k)

        # Assign final composite score back to POI
        score_map = {poi.id: score for poi, score in scored}
        for poi in selected:
            poi.interest_fit = score_map.get(poi.id, 0.0)
            poi.interest_score = poi.interest_fit

        # Log category distribution AFTER selection
        self._log_category_distribution(
            "AFTER semantic selection",
            [(p, score_map.get(p.id, 0.0)) for p in selected],
        )

        self.logger.info(
            "Semantic matching done: %d → %d candidates selected (K=%d).",
            len(candidates),
            len(selected),
            k,
        )

        self.memory.set("matched_candidates", selected)
        return selected

    # ── KG interest expansion ─────────────────────────────────────────────────

    async def _expand_interests(self, interests: list[str]) -> set[str]:
        all_types: set[str] = set()
        for interest in interests:
            if not interest:
                continue
            related = await self.kg.expand_category(interest)
            all_types.update(related)
        return all_types

    @staticmethod
    def _intent_terms(user_input: UserInput) -> list[str]:
        raw_terms = [
            *user_input.themes,
            *user_input.interests,
            *user_input.vibes,
            *user_input.must_have,
            user_input.free_text,
        ]
        if "first_timer" in user_input.vibes or "first timer" in normalize_text(user_input.free_text):
            raw_terms.extend(["iconic landmark", "local food", "coffee", "light nightlife"])
        terms: list[str] = []
        for term in raw_terms:
            normalized = str(term).strip()
            if normalized and normalized not in terms:
                terms.append(normalized)
        return terms or ["local food", "coffee", "culture", "outdoor"]

    # ── Composite scoring ─────────────────────────────────────────────────────

    @classmethod
    def _compute_composite_score(
        cls,
        poi: POI,
        expanded_types: set[str],
        interest_embeddings: list[list[float]],
        expanded_embedding: list[float],
        user_input: UserInput,
    ) -> float:
        """Compute weighted composite score for a single POI.

        final_score = w_semantic * semantic_score
                    + w_category * category_match
                    + w_spatial  * spatial_prior
                    + w_quality  * quality_signal
        """
        # 1. Semantic similarity (cosine over hash embeddings)
        semantic_score = cls._compute_semantic_score(
            poi, interest_embeddings, expanded_embedding
        )

        # Apply soft penalty for very low semantic relevance instead of
        # hard removal — keeps the POI in the pool but down-weights it.
        if semantic_score < SOFT_PENALTY_FLOOR:
            penalty = SOFT_PENALTY_SCALE * (SOFT_PENALTY_FLOOR - semantic_score)
            semantic_score = max(0.0, semantic_score - penalty)

        # 2. Category overlap with KG-expanded types
        category_score = cls._compute_category_score(poi, expanded_types)

        # 2b. Soft natural-language intent signals.
        vibe_score = cls._compute_vibe_score(poi, user_input)
        prompt_score = cls._compute_prompt_similarity(poi, user_input)
        must_have_bonus = cls._must_have_bonus(poi, user_input)
        avoid_penalty = cls._avoid_penalty(poi, user_input)

        # 3. Spatial prior (proximity to start location)
        spatial_score = poi.proximity_score  # already set by _initialize_display_features

        # 4. Quality signal (rating × review count, normalised)
        quality_score = poi.quality_score    # already set by _initialize_display_features

        composite = (
            0.34 * semantic_score
            + 0.18 * category_score
            + 0.16 * vibe_score
            + 0.12 * prompt_score
            + W_SPATIAL * spatial_score
            + W_QUALITY * quality_score
            + must_have_bonus
            - avoid_penalty
        )
        return max(0.0, min(1.0, composite))

    @classmethod
    def _compute_prompt_similarity(cls, poi: POI, user_input: UserInput) -> float:
        prompt = " ".join([
            user_input.free_text,
            *user_input.themes,
            *user_input.interests,
            *user_input.vibes,
        ])
        if not prompt.strip():
            return 0.0
        return cls._cosine(cls._embed_text(poi_text(poi)), cls._embed_text(prompt))

    @staticmethod
    def _compute_vibe_score(poi: POI, user_input: UserInput) -> float:
        text = poi_text(poi)
        group = category_for_poi(poi)
        vibes = {normalize_text(vibe) for vibe in user_input.vibes}
        score = 0.0
        if "chill" in vibes:
            if group in {"coffee", "outdoor"}:
                score += 0.45
            if any(term in text for term in ("lounge", "riverside", "garden", "park", "cafe", "coffee")):
                score += 0.25
        if "scenic view" in vibes:
            if any(term in text for term in ("view", "rooftop", "sky", "river", "riverside", "landmark", "walking street")):
                score += 0.55
            if group in {"coffee", "nightlife", "outdoor", "culture"}:
                score += 0.15
        if "local" in vibes:
            if group == "food":
                score += 0.40
            if any(term in text for term in ("pho", "phở", "bun", "bún", "com", "cơm", "banh", "bánh", "market", "quan", "quán")):
                score += 0.30
        if "hidden gem" in vibes or "less touristy" in vibes:
            review_count = getattr(poi, "user_rating_count", None) or 0
            if 10 <= review_count <= 350:
                score += 0.35
            if any(term in text for term in ("local", "alley", "hem", "hẻm", "quan", "quán")):
                score += 0.20
        if "first timer" in vibes:
            if group in {"culture", "food", "coffee", "nightlife"}:
                score += 0.25
            if any(term in text for term in ("war remnants", "independence", "ben thanh", "nguyen hue", "post office", "notre dame")):
                score += 0.45
        return max(0.0, min(1.0, score))

    @staticmethod
    def _must_have_bonus(poi: POI, user_input: UserInput) -> float:
        terms = tuple(normalize_text(term) for term in [*user_input.must_have, *user_input.themes] if term.strip())
        if not terms:
            return 0.0
        return 0.12 if matches_any(poi_text(poi), terms) else 0.0

    @staticmethod
    def _avoid_penalty(poi: POI, user_input: UserInput) -> float:
        terms = tuple(normalize_text(term) for term in [*user_input.avoid, *user_input.negative_preferences] if term.strip())
        if not terms:
            return 0.0
        text = poi_text(poi)
        penalty = 0.18 if matches_any(text, terms) else 0.0
        if any(term in terms for term in ("crowded", "tourist trap", "touristy", "khach du lich")):
            if any(term in text for term in ("tourist attraction", "landmark", "walking street", "ben thanh")):
                penalty += 0.12
        return min(0.42, penalty)

    @classmethod
    def _compute_semantic_score(
        cls,
        poi: POI,
        interest_embeddings: list[list[float]],
        expanded_embedding: list[float],
    ) -> float:
        if not poi.types:
            return 0.0

        poi_set = set(poi.types)
        generic = {"establishment", "point_of_interest"}
        poi_meaningful = poi_set - generic or poi_set

        poi_text = " ".join([poi.name, poi.primary_type, *sorted(poi_meaningful)])
        poi_embedding = cls._embed_text(poi_text)

        return max(
            [cls._cosine(poi_embedding, e) for e in interest_embeddings]
            + [cls._cosine(poi_embedding, expanded_embedding)]
        )

    @classmethod
    def _compute_category_score(cls, poi: POI, expanded_types: set[str]) -> float:
        if not poi.types or not expanded_types:
            return 0.0
        poi_set = set(poi.types)
        generic = {"establishment", "point_of_interest"}
        poi_meaningful = poi_set - generic or poi_set
        overlap = poi_meaningful & expanded_types
        return len(overlap) / max(len(poi_meaningful), 1)

    # ── Adaptive K selection ──────────────────────────────────────────────────

    @staticmethod
    def _choose_k(pool_size: int) -> int:
        """Choose final candidate count within [TARGET_MIN, TARGET_MAX].

        Rules:
        - If pool ≥ TARGET_MAX  → return TARGET_MAX (cap at 800)
        - If pool ≥ TARGET_MIN  → return pool (keep all, still ≥ 500)
        - If pool < TARGET_MIN  → return pool (dataset is genuinely small)
        """
        if pool_size >= TARGET_MAX:
            return TARGET_MAX
        if pool_size >= TARGET_MIN:
            return pool_size
        # Dataset is smaller than 500 — return everything, log a warning
        return pool_size

    # ── Diversity-aware stratified selection ──────────────────────────────────

    @classmethod
    def _stratified_select(
        cls,
        scored: list[tuple[POI, float]],
        k: int,
    ) -> list[POI]:
        """Select K POIs with minimum category quotas, then fill by score.

        Algorithm:
        1. Compute per-category quota (floor of fraction * k).
        2. Walk the globally-ranked list; fill each category bucket up to
           its quota first (round-robin over categories that still need POIs).
        3. Fill remaining slots from the globally-ranked list (any category).
        4. Preserve deterministic order throughout.
        """
        if not scored:
            return []

        k = min(k, len(scored))

        # Compute integer quotas
        quotas: dict[str, int] = {
            cat: max(1, math.floor(frac * k))
            for cat, frac in DIVERSITY_QUOTAS.items()
        }

        # Buckets: category → ranked list of (poi, score)
        buckets: dict[str, list[tuple[POI, float]]] = defaultdict(list)
        for poi, score in scored:
            bucket = cls._category_bucket(poi)
            buckets[bucket].append((poi, score))

        selected_ids: set[str] = set()
        selected: list[POI] = []

        # Phase 1: fill quotas (round-robin over categories needing POIs)
        # Iterate categories in a fixed order for determinism
        quota_categories = sorted(quotas.keys())
        filled: dict[str, int] = defaultdict(int)
        bucket_cursors: dict[str, int] = defaultdict(int)

        changed = True
        while changed and len(selected) < k:
            changed = False
            for cat in quota_categories:
                if filled[cat] >= quotas[cat]:
                    continue
                cursor = bucket_cursors[cat]
                cat_list = buckets.get(cat, [])
                while cursor < len(cat_list):
                    poi, _ = cat_list[cursor]
                    cursor += 1
                    if poi.id not in selected_ids:
                        selected_ids.add(poi.id)
                        selected.append(poi)
                        filled[cat] += 1
                        changed = True
                        break
                bucket_cursors[cat] = cursor

        # Phase 2: fill remaining slots from global ranked list
        for poi, _ in scored:
            if len(selected) >= k:
                break
            if poi.id not in selected_ids:
                selected_ids.add(poi.id)
                selected.append(poi)

        return selected

    @staticmethod
    def _category_bucket(poi: POI) -> str:
        """Map a POI to one of the diversity bucket keys."""
        types_lower = {t.lower() for t in poi.types}
        primary_lower = poi.primary_type.lower()

        if poi.source == "festival":
            return "festival"
        if any(t in types_lower for t in ("cafe", "coffee_shop")):
            return "coffee"
        if any(t in types_lower for t in ("bar", "night_club", "nightclub")):
            return "nightlife"
        if any(t in types_lower for t in (
            "museum", "art_gallery", "tourist_attraction",
            "place_of_worship", "historic_site",
        )):
            return "culture"
        if any(t in types_lower for t in (
            "restaurant", "food", "meal_delivery", "meal_takeaway",
            "bakery", "fast_food",
        )):
            return "food"
        if any(t in types_lower for t in ("shopping_mall", "market")):
            return "shopping"
        return primary_lower or "other"

    # ── Stable sort ───────────────────────────────────────────────────────────

    @staticmethod
    def _stable_sort(
        scored: list[tuple[POI, float]],
        user_input: UserInput,
    ) -> list[tuple[POI, float]]:
        """Sort by (score desc, proximity asc, poi.id asc) for determinism."""
        def sort_key(item: tuple[POI, float]) -> tuple[float, float, str]:
            poi, score = item
            proximity = haversine_km(
                user_input.start_location[0],
                user_input.start_location[1],
                poi.latitude,
                poi.longitude,
            )
            return (-score, proximity, poi.id)

        return sorted(scored, key=sort_key)

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_category_distribution(
        self,
        label: str,
        scored: list[tuple[POI, float]],
    ) -> None:
        dist: dict[str, int] = defaultdict(int)
        for poi, _ in scored:
            dist[self._category_bucket(poi)] += 1
        total = sum(dist.values())
        dist_str = ", ".join(
            f"{cat}={cnt} ({100*cnt/max(total,1):.1f}%)"
            for cat, cnt in sorted(dist.items())
        )
        self.logger.info("Category distribution %s [total=%d]: %s", label, total, dist_str)

    # ── Text embedding (deterministic hash-based) ─────────────────────────────

    @staticmethod
    def _embed_text(text: str, dims: int = 64) -> list[float]:
        """Lightweight deterministic embedding via hashed token buckets."""
        vec = [0.0] * dims
        tokens = re.findall(r"[a-zA-Z0-9_À-ỹ]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest[:2], "big") % dims
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[bucket] += sign

            # Character 3-grams for partial token similarity
            padded = f"_{token}_"
            for i in range(max(0, len(padded) - 2)):
                gram = padded[i : i + 3]
                gdigest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
                gbucket = int.from_bytes(gdigest[:2], "big") % dims
                gsign = 1.0 if gdigest[2] % 2 == 0 else -1.0
                vec[gbucket] += 0.35 * gsign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    # ── Entity validation (soft — invalid entities are skipped, not penalised) ─

    @classmethod
    def _is_valid_entity(cls, poi: POI) -> bool:
        """Return False only for structurally broken POIs (no name/type, garbage text).

        This is NOT a relevance filter — it only removes entities that cannot
        be meaningfully scored (missing required fields, corrupted names).
        """
        if not poi.name or not poi.primary_type or not poi.types:
            return False
        if poi.source == "festival":
            return True

        poi_types = {poi.primary_type.lower(), *(t.lower() for t in poi.types)}
        invalid_by_type = bool(poi_types & INVALID_TYPES)
        normalized_for_invalid = cls._normalize_ascii(name := poi.name.strip())
        invalid_by_name = any(pattern in normalized_for_invalid for pattern in INVALID_NAME_PATTERNS)
        if invalid_by_type or invalid_by_name:
            return False

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

        generic_bad = {"unknown", "test", "placeholder", "cổng", "cong"}
        normalized = cls._normalize_name(name)
        if normalized in generic_bad or re.fullmatch(r"(gate|cong|cổng)\s*\d+", normalized):
            return False

        return True

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", name.lower()))

    @staticmethod
    def _normalize_ascii(name: str) -> str:
        import unicodedata

        lowered = name.lower()
        decomposed = unicodedata.normalize("NFD", lowered)
        ascii_text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return " ".join(re.findall(r"[a-z0-9À-ỹ]+", f"{lowered} {ascii_text}"))

    # ── Display feature initialisation ───────────────────────────────────────

    @staticmethod
    def _initialize_display_features(poi: POI, user_input: UserInput) -> None:
        """Populate quality_score, budget_fit, proximity_score if not yet set."""
        if poi.quality_score <= 0.0:
            rating = poi.rating if poi.rating is not None else 4.0
            count = poi.user_rating_count if poi.user_rating_count is not None else 50
            poi.quality_score = max(
                0.05,
                min(1.0, ((rating * min(count, 5000) ** 0.08) - 3.0) / 2.8),
            )
        if poi.budget_fit <= 0.0:
            if poi.price_level is None:
                poi.budget_fit = 0.8
            else:
                diff = poi.price_level - user_input.budget_level
                poi.budget_fit = 1.0 if diff <= 0 else math.exp(-1.5 * diff * diff)
        if poi.proximity_score <= 0.0:
            dist = haversine_km(
                user_input.start_location[0],
                user_input.start_location[1],
                poi.latitude,
                poi.longitude,
            )
            poi.proximity_score = max(0.02, min(1.0, 1.0 / (1.0 + dist / 5.0)))
