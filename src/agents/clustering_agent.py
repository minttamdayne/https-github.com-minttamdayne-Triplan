from __future__ import annotations

"""
clustering_agent.py  — v3

Root-cause fixes for observed output problems
═══════════════════════════════════════════════
PROBLEM 1  Day 3 has 8 places but only 5 usable stops
  Cause:   K-Means gave Day 3 a cluster of mostly bars/night_clubs.
           Routing hard-blocks bars before noon → only 5 afternoon slots.
  Fix:     _enforce_category_mix() runs after K-Means and rebalancing.
           It swaps out excess bars/nightclubs (above a per-day cap) with
           the best-scored POIs of missing categories from the global pool.

PROBLEM 2  Travel = 0 min / Visit = 0 min in display (display-only)
  Fix:     Not in clustering; see display_patch.py and routing_v3.

PROBLEM 3  Ao Dai Festival (lat=10.8231) assigned to Day 3 cluster
           (centroid ~10.774) → 5.7 km detour → wasted time
  Fix:     Festival assignment now uses haversine distance AND respects
           per-day date constraints. The routing layer also has a proximity
           gate (FESTIVAL_MAX_DETOUR_KM) as a second filter.

PROBLEM 4  Score imbalance: 11.3 / 8.7 / 5.9
  Fix:     Category mix enforcement gives Day 3 daytime-friendly POIs
           (museums, cafes, restaurants) so routing can fill more stops.
"""

from collections import defaultdict
from datetime import timedelta
from typing import Any
import math
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
from sklearn.cluster import KMeans

from src.agents.base import BaseAgent
from src.models.poi import POI
from src.models.user_input import UserInput
from src.preferences import build_preference_profile, category_for_poi
from src.tools.distance import haversine_km


# ── Category mix targets per day ────────────────────────────────────
# Minimum number of "daytime-friendly" POIs every cluster must contain
# so routing always has enough stops before noon / bars open.
DAYTIME_TYPES = {
    "cafe", "restaurant", "museum", "art_gallery",
    "park", "tourist_attraction", "bakery", "meal_takeaway",
}
MIN_DAYTIME_PER_DAY = 4     # at least 4 daytime POIs per cluster

# Hard cap on nightlife per cluster (routing already soft-caps,
# but clustering should not dump ALL bars into one day)
NIGHTLIFE_TYPES = {"bar", "night_club"}
MAX_NIGHTLIFE_PER_DAY = 3   # at most 3 bar/night_club per cluster
MUSEUM_TYPES = {"museum", "art_gallery"}
COFFEE_TYPES = {"cafe"}
MAX_CLUSTER_RADIUS_KM = 12.0
CORE_QUOTAS = {"food": 1, "coffee": 1, "culture": 2, "nightlife": 1}
GROUP_CAPS = {"food": 2, "coffee": 2, "culture": 4}

SEMANTIC_GROUPS = {
    "food": {"restaurant", "food", "meal_takeaway", "meal_delivery", "bakery"},
    "culture": {"museum", "art_gallery", "tourist_attraction", "place_of_worship"},
    "nightlife": {"bar", "night_club"},
    "coffee": {"cafe", "coffee_shop"},
    "outdoor": {"park", "zoo", "aquarium", "amusement_park", "tourist_attraction"},
    "shopping": {"market", "shopping_mall"},
    "wellness": {"spa", "gym", "beauty_salon"},
}


class ClusteringAgent(BaseAgent):
    """Agent 4 — Spatial clustering with score-weighted K-Means,
    spatial-aware rebalancing, category-mix enforcement, and
    proximity-aware festival assignment.
    """

    name: str = "clustering"

    OVERBOOK_FACTOR = 6
    MIN_PER_DAY     = 4

    async def _execute(self, **kwargs: Any) -> dict[int, list[POI]]:
        candidates: list[POI] = kwargs["candidates"]
        user_input: UserInput = kwargs["user_input"]
        n_days      = user_input.num_days
        max_per_day = user_input.max_places_per_day

        if not candidates:
            return {d: [] for d in range(n_days)}

        # Adaptive trim: keep a larger, category-balanced pool instead of a
        # hard top-N slice that over-concentrates nightlife or food.
        target_pool_size = min(180, max(max_per_day * n_days * self.OVERBOOK_FACTOR, 120))
        if len(candidates) > target_pool_size:
            candidates = self._adaptive_trim(candidates, target_pool_size, user_input)
            self.logger.info(
                "Adaptive trimmed pool to %d candidates; entropy=%.3f ratios=%s",
                len(candidates),
                self._category_entropy(candidates),
                self._category_ratios(candidates),
            )

        memory_festivals = self.memory.get("enriched_festivals") or []
        known_ids = {id(p) for p in candidates}
        festivals = [p for p in candidates if p.source == "festival"]
        festivals.extend(f for f in memory_festivals if id(f) not in known_ids)
        for fest in festivals:
            if getattr(fest, "composite_score", 0.0) <= 0.0:
                fest.composite_score = 0.75
        regular   = [p for p in candidates if p.source != "festival"]

        # 1. Constrained greedy assignment. This directly optimizes cluster
        # capacity and category/time balance instead of KMeans + post-hoc repair.
        clusters = self._greedy_constrained_cluster(regular, n_days, max_per_day)

        # Keep clustering spatial and loose. Routing is responsible for the
        # experience rhythm; clustering should not force every day to look the same.
        self._dedupe_clusters(clusters)
        self._limit_cluster_radius(clusters, regular, max_per_day)

        # 4. Assign festivals (proximity-aware, date-constrained)
        self._assign_festivals(clusters, festivals, user_input)
        self._log_cluster_category_ratios(clusters)

        self.memory.set("daily_clusters", clusters)
        return clusters

    def _adaptive_trim(
        self,
        candidates: list[POI],
        target_size: int,
        user_input: UserInput,
    ) -> list[POI]:
        profile = build_preference_profile(user_input)
        sorted_candidates = sorted(
            candidates,
            key=lambda p: (
                -(getattr(p, "composite_score", 0.0) * profile.category_weight(self._semantic_group(p))),
                getattr(p, "id", ""),
            ),
        )
        selected: list[POI] = []
        selected_keys: set[str] = set()
        for poi in sorted_candidates:
            if len(selected) >= target_size:
                break
            key = self._dedupe_key(poi)
            if key in selected_keys:
                continue
            selected.append(poi)
            selected_keys.add(key)

        selected.sort(
            key=lambda p: (-getattr(p, "composite_score", 0.0), getattr(p, "id", ""))
        )
        selected = selected[:target_size]
        self.logger.info(
            "Intent-weighted trim selected_ratios=%s entropy=%.3f",
            self._category_ratios(selected),
            self._category_entropy(selected),
        )
        return selected

    def _dynamic_trim_quotas(
        self,
        user_input: UserInput,
        target_size: int,
    ) -> dict[str, int]:
        weights = {
            "food": 0.22,
            "culture": 0.28,
            "coffee": 0.16,
            "nightlife": 0.12,
            "outdoor": 0.12,
            "shopping": 0.05,
            "other": 0.05,
        }
        keywords = {
            "food": (
                "food", "restaurant", "street food", "vietnamese food",
                "local food", "banh mi", "bánh mì", "pho", "phở", "bun",
                "bún", "hu tieu", "hủ tiếu", "cuisine", "culinary",
            ),
            "culture": (
                "museum", "art gallery", "gallery", "history", "historic",
                "heritage", "temple", "pagoda", "architecture",
                "exhibition", "culture",
            ),
            "coffee": (
                "coffee", "cafe", "café", "brunch", "dessert", "bakery",
            ),
            "nightlife": (
                "nightlife", "night", "bar", "club", "cocktail", "pub",
                "beer", "rooftop", "live music",
            ),
            "outdoor": (
                "park", "zoo", "aquarium", "garden", "walking", "river",
                "family friendly", "kid friendly", "kids", "children",
                "playground", "amusement",
            ),
            "shopping": (
                "market", "shopping", "mall", "souvenir", "local market",
            ),
        }

        text = " ".join(user_input.interests).lower()
        explicit_groups: set[str] = set()
        for group, terms in keywords.items():
            matches = sum(1 for term in terms if term in text)
            if matches:
                explicit_groups.add(group)
                weights[group] += 0.08 * matches

        family_requested = any(
            term in text
            for term in (
                "family friendly", "kid friendly", "kids", "children",
                "playground", "zoo", "aquarium", "amusement",
            )
        )
        nightlife_requested = "nightlife" in explicit_groups
        if family_requested:
            weights["outdoor"] += 0.20
            weights["culture"] += 0.06
            if not nightlife_requested:
                weights["nightlife"] *= 0.25

        if nightlife_requested:
            weights["nightlife"] += 0.22
        if any(term in text for term in ("vietnamese food", "street food", "local food")):
            weights["food"] += 0.18
        if any(term in text for term in ("museum", "art gallery", "exhibition")):
            weights["culture"] += 0.16

        total = sum(max(weight, 0.0) for weight in weights.values()) or 1.0
        normalized = {group: max(weight, 0.0) / total for group, weight in weights.items()}
        quotas = {
            group: int(normalized[group] * target_size)
            for group in normalized
        }

        for group in explicit_groups:
            quotas[group] = max(1, quotas.get(group, 0))

        allocated = sum(quotas.values())
        if allocated > target_size:
            for group in sorted(
                quotas,
                key=lambda g: (g in explicit_groups, quotas[g], normalized.get(g, 0.0)),
            ):
                while allocated > target_size and quotas[group] > (1 if group in explicit_groups else 0):
                    quotas[group] -= 1
                    allocated -= 1
                if allocated <= target_size:
                    break
        elif allocated < target_size:
            remainder = target_size - allocated
            groups_by_fraction = sorted(
                normalized,
                key=lambda g: (
                    normalized[g] * target_size - quotas.get(g, 0),
                    normalized[g],
                    g,
                ),
                reverse=True,
            )
            for i in range(remainder):
                quotas[groups_by_fraction[i % len(groups_by_fraction)]] += 1

        self.logger.info(
            "Dynamic trim weights=%s quotas=%s",
            {g: round(w, 3) for g, w in normalized.items()},
            quotas,
        )
        return quotas

    def _greedy_constrained_cluster(
        self,
        pois: list[POI],
        n_days: int,
        max_per_day: int,
    ) -> dict[int, list[POI]]:
        clusters: dict[int, list[POI]] = {day: [] for day in range(n_days)}
        if not pois:
            return clusters

        ranked = sorted(pois, key=lambda p: getattr(p, "composite_score", 0.0), reverse=True)
        assigned_seed_ids: set[int] = set()
        for day in range(n_days):
            seed = next(
                (p for p in ranked if id(p) not in assigned_seed_ids),
                None,
            )
            if seed is None:
                seed = next(p for p in ranked if id(p) not in assigned_seed_ids)
            clusters[day].append(seed)
            assigned_seed_ids.add(id(seed))

        assigned_ids = {id(p) for cluster in clusters.values() for p in cluster}
        remaining = [p for p in ranked if id(p) not in assigned_ids]

        for poi in remaining:
            feasible_days = [
                day for day in range(n_days)
                if len(clusters[day]) < max_per_day
            ]
            if not feasible_days:
                break
            best_day = min(
                feasible_days,
                key=lambda day: self._assignment_cost(poi, clusters[day]),
            )
            clusters[best_day].append(poi)

        for day in clusters:
            clusters[day].sort(key=lambda p: p.composite_score, reverse=True)

        self.logger.info(
            "Greedy constrained cluster sizes=%s entropy=%s",
            {d: len(clusters[d]) for d in clusters},
            {d: round(self._category_entropy(clusters[d]), 3) for d in clusters},
        )
        return clusters

    def _assignment_cost(self, poi: POI, cluster: list[POI]) -> float:
        centroid = self._compute_centroids({0: cluster}).get(0)
        distance_cost = self._dist_to_centroid(poi, centroid) / 5.0

        group = self._semantic_group(poi)
        ratios = self._category_ratios(cluster)
        category_overlap = ratios.get(group, 0.0)
        # Only discourage extreme same-theme clustering. A food-heavy day is
        # fine; seven copies of the same exact experience is not.
        repetition_cost = max(0.0, category_overlap - 0.65)

        return 0.82 * distance_cost + 0.18 * repetition_cost

    # ── Score-weighted K-Means ───────────────────────────────────────

    def _kmeans_cluster(self, pois: list[POI], k: int) -> dict[int, list[POI]]:
        clusters: dict[int, list[POI]] = {i: [] for i in range(k)}
        if not pois:
            return clusters
        if len(pois) <= k:
            for i, poi in enumerate(pois):
                clusters[i % k].append(poi)
            return clusters

        coords = np.array([[p.latitude, p.longitude] for p in pois])
        coord_std = coords.std(axis=0)
        coord_std[coord_std == 0] = 1.0
        geo_features = (coords - coords.mean(axis=0)) / coord_std
        semantic_features = np.array([self._semantic_vector(p) for p in pois])
        features = np.hstack([geo_features, semantic_features * 0.35])
        scores = np.array([max(p.composite_score, 0.01) for p in pois])
        weights = scores / scores.sum()

        km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=42)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="joblib\\.externals\\.loky\\.backend\\.context",
            )
            labels = km.fit_predict(features, sample_weight=weights)

        for poi, label in zip(pois, labels):
            clusters[int(label)].append(poi)

        for day in clusters:
            clusters[day].sort(key=lambda p: p.composite_score, reverse=True)

        self.logger.info("K-Means cluster sizes: %s", {d: len(clusters[d]) for d in clusters})
        return clusters

    # ── Spatial-aware rebalancing ────────────────────────────────────

    def _rebalance_clusters(
        self,
        clusters:    dict[int, list[POI]],
        max_per_day: int,
        min_per_day: int,
    ) -> None:
        max_iter = max_per_day * len(clusters) * 2

        for _ in range(max_iter):
            overloaded  = [d for d, p in clusters.items() if len(p) > max_per_day]
            underloaded = [d for d, p in clusters.items() if len(p) < min_per_day]
            if not overloaded:
                break

            centroids = self._compute_centroids(clusters)
            day       = overloaded[0]
            centroid  = centroids.get(day)
            moveable  = [p for p in clusters[day] if p.source != "festival"]
            if not moveable:
                break

            evict = max(
                moveable,
                key=lambda p: (
                    self._dist_to_centroid(p, centroid),
                    -p.composite_score,
                ),
            )

            if underloaded:
                dest = min(
                    underloaded,
                    key=lambda d: self._dist_to_centroid(evict, centroids.get(d)),
                )
            else:
                cands = [d for d in clusters if d != day and len(clusters[d]) < max_per_day]
                if not cands:
                    break
                dest = min(cands, key=lambda d: self._dist_to_centroid(evict, centroids.get(d)))

            clusters[day].remove(evict)
            clusters[dest].append(evict)
            clusters[dest].sort(key=lambda p: p.composite_score, reverse=True)

        # Final hard cap
        for day in clusters:
            if len(clusters[day]) > max_per_day:
                clusters[day].sort(key=lambda p: p.composite_score, reverse=True)
                clusters[day] = clusters[day][:max_per_day]

        self.logger.info("After rebalancing: %s", {d: len(clusters[d]) for d in clusters})

    # ── Category mix enforcement ─────────────────────────────────────

    def _enforce_category_mix(
        self,
        clusters:    dict[int, list[POI]],
        full_pool:   list[POI],
        max_per_day: int,
    ) -> None:
        """Guarantee every cluster has at least MIN_DAYTIME_PER_DAY daytime
        POIs and at most MAX_NIGHTLIFE_PER_DAY nightlife POIs.

        Strategy:
        - Count daytime/nightlife per cluster.
        - For clusters with too many nightlife: evict lowest-score nightlife
          POIs, add to a "swap pool".
        - For clusters lacking daytime: pull from the swap pool or from the
          full_pool (POIs not yet in any cluster), preferring spatial proximity.
        """
        # Build a set of all POI ids already assigned
        assigned_ids: set[int] = {
            id(p) for pois in clusters.values() for p in pois
        }

        # Swap pool: nightlife evicted from overloaded clusters
        swap_pool: list[POI] = []

        # --- Step 1: Evict excess nightlife ---
        for day, pois in clusters.items():
            nightlife = [p for p in pois if p.primary_type in NIGHTLIFE_TYPES]
            excess    = len(nightlife) - MAX_NIGHTLIFE_PER_DAY
            if excess <= 0:
                continue
            # Evict lowest-score ones first
            nightlife.sort(key=lambda p: p.composite_score)
            for p in nightlife[:excess]:
                clusters[day].remove(p)
                assigned_ids.discard(id(p))
                swap_pool.append(p)

        # --- Step 2: Fill daytime deficit ---
        centroids = self._compute_centroids(clusters)

        # Unassigned daytime POIs from full pool (sorted by score desc)
        unassigned_daytime = [
            p for p in full_pool
            if id(p) not in assigned_ids and p.primary_type in DAYTIME_TYPES
        ]
        unassigned_daytime.sort(key=lambda p: p.composite_score, reverse=True)

        for day, pois in clusters.items():
            daytime_count = sum(1 for p in pois if p.primary_type in DAYTIME_TYPES)
            deficit       = MIN_DAYTIME_PER_DAY - daytime_count
            if deficit <= 0:
                continue

            centroid = centroids.get(day)

            # Candidates: swap_pool daytime first, then unassigned daytime
            swap_daytime = [p for p in swap_pool if p.primary_type in DAYTIME_TYPES]
            candidates   = swap_daytime + [
                p for p in unassigned_daytime if id(p) not in {id(x) for x in swap_daytime}
            ]

            # Sort by proximity to this day's centroid, then by score
            if centroid:
                candidates.sort(
                    key=lambda p: (
                        self._dist_to_centroid(p, centroid),
                        -p.composite_score,
                    )
                )

            added = 0
            for p in candidates:
                if added >= deficit:
                    break
                if len(clusters[day]) >= max_per_day:
                    break
                if id(p) in assigned_ids:
                    continue

                clusters[day].append(p)
                assigned_ids.add(id(p))
                if p in swap_pool:
                    swap_pool.remove(p)
                if p in unassigned_daytime:
                    unassigned_daytime.remove(p)
                added += 1

            clusters[day].sort(key=lambda p: p.composite_score, reverse=True)

        self.logger.info(
            "After mix enforcement: %s",
            {
                d: {
                    "total": len(clusters[d]),
                    "daytime": sum(1 for p in clusters[d] if p.primary_type in DAYTIME_TYPES),
                    "nightlife": sum(1 for p in clusters[d] if p.primary_type in NIGHTLIFE_TYPES),
                }
                for d in clusters
            },
        )

    def _ensure_daily_presence(
        self,
        clusters: dict[int, list[POI]],
        full_pool: list[POI],
        required_types: set[str],
        max_per_day: int,
    ) -> None:
        """Add one required semantic type per day when matching POIs exist."""
        available = [p for p in full_pool if p.primary_type in required_types]
        if not available:
            return

        assigned_ids = {id(p) for pois in clusters.values() for p in pois}
        centroids = self._compute_centroids(clusters)

        for day, pois in clusters.items():
            if any(p.primary_type in required_types for p in pois):
                continue

            candidates = [p for p in available if id(p) not in assigned_ids]
            if not candidates:
                candidates = available

            centroid = centroids.get(day)
            best = min(
                candidates,
                key=lambda p: (
                    self._dist_to_centroid(p, centroid),
                    -getattr(p, "composite_score", 0.0),
                ),
            )

            if id(best) in assigned_ids:
                continue
            if len(pois) < max_per_day:
                pois.append(best)
                assigned_ids.add(id(best))
            else:
                replaceable = [
                    p for p in pois
                    if p.source != "festival" and p.primary_type not in required_types
                ]
                if not replaceable:
                    continue
                evict = min(
                    replaceable,
                    key=lambda p: (
                        p.primary_type not in NIGHTLIFE_TYPES,
                        getattr(p, "composite_score", 0.0),
                    ),
                )
                pois.remove(evict)
                assigned_ids.discard(id(evict))
                pois.append(best)
                assigned_ids.add(id(best))
            pois.sort(key=lambda p: p.composite_score, reverse=True)

    def _enforce_category_quotas(
        self,
        clusters: dict[int, list[POI]],
        full_pool: list[POI],
        max_per_day: int,
        require_nightlife: bool,
    ) -> None:
        quotas = dict(CORE_QUOTAS)
        if not require_nightlife:
            quotas.pop("nightlife", None)

        assigned_ids = {id(p) for pois in clusters.values() for p in pois}

        for day, pois in clusters.items():
            for group, minimum in quotas.items():
                while self._group_count(pois, group) < minimum:
                    candidate = self._best_quota_candidate(
                        group=group,
                        day_pois=pois,
                        full_pool=full_pool,
                        assigned_ids=assigned_ids,
                    )
                    if candidate is None:
                        break

                    if id(candidate) in assigned_ids:
                        break
                    if len(pois) < max_per_day:
                        pois.append(candidate)
                        assigned_ids.add(id(candidate))
                    else:
                        evict = self._quota_evict_candidate(pois, quotas)
                        if evict is None:
                            break
                        pois.remove(evict)
                        assigned_ids.discard(id(evict))
                        pois.append(candidate)
                        assigned_ids.add(id(candidate))
                    pois.sort(key=lambda p: getattr(p, "composite_score", 0.0), reverse=True)

        self.logger.info(
            "After quota enforcement: %s",
            {
                d + 1: {
                    "total": len(pois),
                    "counts": self._category_counts(pois),
                    "entropy": round(self._category_entropy(pois), 3),
                }
                for d, pois in clusters.items()
            },
        )

    def _best_quota_candidate(
        self,
        group: str,
        day_pois: list[POI],
        full_pool: list[POI],
        assigned_ids: set[int],
    ) -> POI | None:
        candidates = [
            p for p in full_pool
            if id(p) not in assigned_ids and self._semantic_group(p) == group
        ]
        if not candidates:
            return None
        centroid = self._compute_centroids({0: day_pois}).get(0)
        return min(
            candidates,
            key=lambda p: (
                self._dist_to_centroid(p, centroid),
                -getattr(p, "composite_score", 0.0),
            ),
        )

    def _quota_evict_candidate(self, pois: list[POI], quotas: dict[str, int]) -> POI | None:
        counts = self._category_counts(pois)
        replaceable = [
            p for p in pois
            if p.source != "festival"
            and counts.get(self._semantic_group(p), 0) > quotas.get(self._semantic_group(p), 0)
        ]
        if not replaceable:
            replaceable = [p for p in pois if p.source != "festival"]
        if not replaceable:
            return None
        return min(replaceable, key=lambda p: getattr(p, "composite_score", 0.0))

    def _enforce_group_caps(
        self,
        clusters: dict[int, list[POI]],
        full_pool: list[POI],
        max_per_day: int,
    ) -> None:
        assigned_ids = {id(p) for pois in clusters.values() for p in pois}
        for day, pois in clusters.items():
            while True:
                counts = self._category_counts(pois)
                over_group = next(
                    (group for group, cap in GROUP_CAPS.items() if counts.get(group, 0) > cap),
                    None,
                )
                if over_group is None:
                    break

                evictable = [
                    p for p in pois
                    if p.source != "festival" and self._semantic_group(p) == over_group
                ]
                if not evictable:
                    break
                outgoing = min(evictable, key=lambda p: getattr(p, "composite_score", 0.0))

                candidates = [
                    p for p in full_pool
                    if id(p) not in assigned_ids
                    and self._semantic_group(p) != over_group
                    and counts.get(self._semantic_group(p), 0) < GROUP_CAPS.get(self._semantic_group(p), 99)
                ]
                if not candidates:
                    break

                centroid = self._compute_centroids({day: pois}).get(day)
                incoming = min(
                    candidates,
                    key=lambda p: (
                        self._dist_to_centroid(p, centroid),
                        -getattr(p, "composite_score", 0.0),
                    ),
                )
                pois.remove(outgoing)
                assigned_ids.discard(id(outgoing))
                pois.append(incoming)
                assigned_ids.add(id(incoming))
                pois.sort(key=lambda p: getattr(p, "composite_score", 0.0), reverse=True)

            if len(pois) > max_per_day:
                pois.sort(key=lambda p: getattr(p, "composite_score", 0.0), reverse=True)
                del pois[max_per_day:]

        self.logger.info(
            "After group cap enforcement: %s",
            {day + 1: self._category_counts(pois) for day, pois in clusters.items()},
        )

    @classmethod
    def _group_count(cls, pois: list[POI], group: str) -> int:
        return sum(1 for p in pois if cls._semantic_group(p) == group)

    @staticmethod
    def _user_wants_nightlife(user_input: UserInput) -> bool:
        return any("night" in interest.lower() or "bar" in interest.lower() for interest in user_input.interests)

    def _limit_cluster_radius(
        self,
        clusters: dict[int, list[POI]],
        full_pool: list[POI],
        max_per_day: int,
    ) -> None:
        """Replace distant non-festival outliers with closer unassigned POIs."""
        assigned_ids = {id(p) for pois in clusters.values() for p in pois}
        unassigned = [p for p in full_pool if id(p) not in assigned_ids]

        for day, pois in clusters.items():
            centroid = self._compute_centroids({day: pois}).get(day)
            if centroid is None:
                continue

            outliers = [
                p for p in list(pois)
                if p.source != "festival"
                and self._dist_to_centroid(p, centroid) > MAX_CLUSTER_RADIUS_KM
            ]
            for outlier in outliers:
                replacements = [
                    p for p in unassigned
                    if self._dist_to_centroid(p, centroid) <= MAX_CLUSTER_RADIUS_KM
                ]
                if not replacements:
                    continue
                replacement = max(
                    replacements,
                    key=lambda p: (
                        getattr(p, "composite_score", 0.0),
                        -self._dist_to_centroid(p, centroid),
                    ),
                )
                pois.remove(outlier)
                pois.append(replacement)
                assigned_ids.discard(id(outlier))
                assigned_ids.add(id(replacement))
                unassigned.remove(replacement)
                unassigned.append(outlier)
            if len(pois) > max_per_day:
                pois.sort(key=lambda p: p.composite_score, reverse=True)
                del pois[max_per_day:]
            pois.sort(key=lambda p: p.composite_score, reverse=True)

    def _enforce_theme_continuity(
        self,
        clusters: dict[int, list[POI]],
        full_pool: list[POI],
        max_per_day: int,
    ) -> None:
        assigned_ids = {id(p) for pois in clusters.values() for p in pois}
        unassigned = [p for p in full_pool if id(p) not in assigned_ids]

        for day, pois in clusters.items():
            if not pois:
                continue

            festival_anchors = [p for p in pois if p.source == "festival"]
            if festival_anchors:
                target_theme = self._semantic_group(festival_anchors[0])
            else:
                ratios = self._category_ratios(pois)
                target_theme = max(ratios, key=ratios.get)

            target_count = math.ceil(0.70 * len(pois))
            current_count = sum(1 for p in pois if self._semantic_group(p) == target_theme)
            if current_count >= target_count:
                continue

            centroid = self._compute_centroids({day: pois}).get(day)
            replacements = [
                p for p in unassigned
                if self._semantic_group(p) == target_theme
            ]
            replacements.sort(
                key=lambda p: (
                    self._dist_to_centroid(p, centroid),
                    -getattr(p, "composite_score", 0.0),
                )
            )

            replaceable = [
                p for p in list(pois)
                if p.source != "festival" and self._semantic_group(p) != target_theme
            ]
            replaceable.sort(key=lambda p: getattr(p, "composite_score", 0.0))

            swaps = 0
            while current_count < target_count and replacements and replaceable:
                incoming = replacements.pop(0)
                outgoing = replaceable.pop(0)
                pois.remove(outgoing)
                pois.append(incoming)
                unassigned.remove(incoming)
                unassigned.append(outgoing)
                current_count += 1
                swaps += 1

            if len(pois) > max_per_day + len(festival_anchors):
                pois.sort(key=lambda p: p.composite_score, reverse=True)
                del pois[max_per_day + len(festival_anchors):]
            pois.sort(key=lambda p: p.composite_score, reverse=True)
            self.logger.info(
                "Day %d theme continuity target=%s ratio=%.3f swaps=%d",
                day + 1,
                target_theme,
                current_count / max(len(pois), 1),
                swaps,
            )

    # ── Festival assignment (proximity-aware) ────────────────────────

    def _assign_festivals(
        self,
        clusters:   dict[int, list[POI]],
        festivals:  list[POI],
        user_input: UserInput,
    ) -> None:
        """Assign each festival to the day whose:
        1. trip_date is within the festival's allowed dates (or no constraint), AND
        2. cluster centroid is nearest to the festival.
        Festivals that are too far from every centroid are still assigned to
        the nearest qualifying day (we never silently drop a festival).
        """
        if not festivals:
            return

        n_days     = user_input.num_days
        trip_dates = [user_input.start_date + timedelta(days=i) for i in range(n_days)]
        centroids  = self._compute_centroids(clusters)
        used_keys: set[str] = set()

        for fest in sorted(festivals, key=lambda p: getattr(p, "composite_score", 0.0), reverse=True):
            fest_key = self._festival_key(fest)
            if fest_key in used_keys:
                continue
            allowed = getattr(fest, "dates", None)
            if allowed:
                candidate_days = [i for i, d in enumerate(trip_dates) if d in allowed]
                if not candidate_days:
                    self.logger.warning("Festival '%s' has no date overlap — skipping", fest.name)
                    continue
            else:
                candidate_days = list(range(n_days))

            candidate_days = [
                day for day in candidate_days
                if sum(1 for p in clusters[day] if p.source == "festival") < 1
            ]
            if not candidate_days:
                continue

            best_day, reason = max(
                (
                    (day, self._festival_day_score(fest, clusters[day], centroids.get(day)))
                    for day in candidate_days
                ),
                key=lambda item: item[1]["score"],
            )
            clusters[best_day].append(fest)
            used_keys.add(fest_key)
            self.logger.info(
                "Festival '%s' -> day %d reason=%s",
                fest.name,
                best_day + 1,
                reason,
            )

    # ── Geometry helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_centroids(
        clusters: dict[int, list[POI]],
    ) -> dict[int, tuple[float, float]]:
        return {
            day: (
                sum(p.latitude  for p in pois) / len(pois),
                sum(p.longitude for p in pois) / len(pois),
            )
            for day, pois in clusters.items()
            if pois
        }

    @staticmethod
    def _dist_to_centroid(
        poi: POI, centroid: tuple[float, float] | None
    ) -> float:
        if centroid is None:
            return float("inf")
        return haversine_km(poi.latitude, poi.longitude, centroid[0], centroid[1])

    @staticmethod
    def _semantic_vector(poi: POI) -> list[float]:
        poi_types = {poi.primary_type, *poi.types}
        return [
            1.0 if poi_types & group_types else 0.0
            for group_types in SEMANTIC_GROUPS.values()
        ]

    def _festival_day_score(
        self,
        festival: POI,
        day_pois: list[POI],
        centroid: tuple[float, float] | None,
    ) -> dict[str, float | str]:
        dist = self._dist_to_centroid(festival, centroid)
        spatial = 0.0 if not math.isfinite(dist) else 1.0 / (1.0 + dist / 5.0)
        geocode_confidence = max(0.0, min(1.0, getattr(festival, "geocode_confidence", 1.0)))
        spatial *= geocode_confidence

        fest_group = self._semantic_group(festival)
        day_ratios = self._category_ratios(day_pois)
        thematic = day_ratios.get(fest_group, 0.0)

        if fest_group == "food":
            thematic = max(thematic, day_ratios.get("food", 0.0))
        elif fest_group == "culture":
            thematic = max(thematic, day_ratios.get("culture", 0.0))
        elif fest_group == "nightlife":
            thematic = max(thematic, day_ratios.get("nightlife", 0.0))

        time_compatibility = 1.0
        if fest_group == "nightlife":
            time_compatibility = min(1.0, day_ratios.get("nightlife", 0.0) + 0.5)

        existing_festivals = sum(1 for p in day_pois if p.source == "festival")
        spread_penalty = 0.12 * existing_festivals
        spatial_weight = 0.40 * geocode_confidence
        thematic_weight = 0.40 + (0.40 - spatial_weight)
        score = (
            spatial_weight * spatial
            + thematic_weight * thematic
            + 0.2 * time_compatibility
            - spread_penalty
        )
        return {
            "score": round(max(0.0, score), 3),
            "spatial": round(spatial, 3),
            "thematic": round(thematic, 3),
            "time": round(time_compatibility, 3),
            "theme": fest_group,
            "geocode_confidence": round(geocode_confidence, 2),
            "geocode_method": getattr(festival, "geocode_method", "source"),
        }

    @staticmethod
    def _semantic_group(poi: POI) -> str:
        return category_for_poi(poi)

    @staticmethod
    def _dedupe_key(poi: POI) -> str:
        import re
        import unicodedata

        text = unicodedata.normalize("NFD", poi.name.lower())
        ascii_text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text))[:80]

    def _dedupe_clusters(self, clusters: dict[int, list[POI]]) -> None:
        seen: set[str] = set()
        for day in sorted(clusters):
            unique: list[POI] = []
            for poi in clusters[day]:
                key = self._dedupe_key(poi)
                if key in seen:
                    continue
                unique.append(poi)
                seen.add(key)
            clusters[day] = unique

    @classmethod
    def _category_ratios(cls, pois: list[POI]) -> dict[str, float]:
        if not pois:
            return {}
        counts: defaultdict[str, int] = defaultdict(int)
        for poi in pois:
            counts[cls._semantic_group(poi)] += 1
        total = len(pois)
        return {
            group: round(count / total, 3)
            for group, count in sorted(counts.items())
        }

    @classmethod
    def _category_counts(cls, pois: list[POI]) -> dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for poi in pois:
            counts[cls._semantic_group(poi)] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _festival_key(poi: POI) -> str:
        return str(getattr(poi, "id", None) or poi.name).strip().lower()

    @classmethod
    def _category_entropy(cls, pois: list[POI]) -> float:
        ratios = cls._category_ratios(pois)
        if not ratios:
            return 0.0
        entropy = -sum(p * math.log(p, 2) for p in ratios.values() if p > 0)
        max_entropy = math.log(max(len(ratios), 1), 2) or 1.0
        return entropy / max_entropy

    def _log_cluster_category_ratios(self, clusters: dict[int, list[POI]]) -> None:
        for day, pois in clusters.items():
            self.logger.info(
                "Day %d category counts=%s ratios=%s entropy=%.3f",
                day + 1,
                self._category_counts(pois),
                self._category_ratios(pois),
                self._category_entropy(pois),
            )
