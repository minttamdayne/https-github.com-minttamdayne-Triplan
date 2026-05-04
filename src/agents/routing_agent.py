from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any

from src.agents.base import BaseAgent
from src.models.itinerary import DayPlan, Itinerary, ItineraryStop
from src.models.poi import POI
from src.models.user_input import UserInput
from src.opening_hours import opening_status, visit_opening_status
from src.preferences import build_preference_profile, category_for_poi, infer_food_role, matches_any, poi_text
from src.tools.distance import haversine_km

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

MAX_STOPS   = 7
BUFFER_MIN  = 15
BEAM_WIDTH  = 20
PACE_MAX_STOPS = {"relaxed": 5, "balanced": 7, "intense": 9}
PACE_BUFFER_MIN = {"relaxed": 25, "balanced": 15, "intense": 10}

AVG_SPEED_KMH  = 20.0
RUSH_SPEED_KMH = 12.0
MIN_TRAVEL_MIN = 8.0
TRAVEL_OVERHEAD_MIN = 6.0
RUSH_HOURS = ((7, 9), (16, 19))
MAX_FOOD_GAP_MIN = 120
TRAVEL_PENALTY_BASE = 0.55
END_HEURISTIC_WEIGHT = 0.35

VISIT_DURATION: dict[str, int] = {
    "museum":             90,
    "art_gallery":        60,
    "restaurant":         60,
    "cafe":               30,
    "bakery":             20,
    "bar":                45,
    "night_club":         60,
    "park":               50,
    "tourist_attraction": 60,
}

DAILY_HARD_CAP: dict[str, int] = {
    "bar":         3,
    "night_club":  2,
    "cafe":        3,
    "bakery":      2,
    "museum":      3,
    "art_gallery": 3,
    "market":      2,
}

SOFT_CAP: dict[str, int] = {
    "bar":         2,
    "night_club":  1,
    "cafe":        2,
    "bakery":      1,
    "museum":      2,
    "art_gallery": 2,
    "market":      1,
}
SOFT_CAP_PENALTY = 0.35

MAX_FESTIVALS_PER_DAY  = 1
FESTIVAL_BOOST         = 1.35
FESTIVAL_MAX_DETOUR_KM = 8.0

# Fallback: if cluster has < this many candidates, borrow from global pool
MIN_CANDIDATES_PER_DAY = 18
# Max extra candidates pulled from global pool (sorted by proximity to centroid)
MAX_FALLBACK_CANDIDATES = 24

END_OF_DAY_HOUR = 23
NIGHTLIFE_MIN_HOUR = {"bar": 17, "night_club": 17}
FOOD_TYPES = {"restaurant", "food", "meal_takeaway", "meal_delivery", "bakery"}
MUSEUM_TYPES = {"museum", "art_gallery"}
CULTURE_TYPES = {"museum", "art_gallery", "tourist_attraction"}
COFFEE_TYPES = {"cafe", "coffee_shop"}
NIGHTLIFE_TYPES = {"bar", "night_club"}
INVALID_TYPES = {
    "lodging", "hotel", "supermarket", "store", "baby_store", "toy_store",
    "apartment", "real_estate_agency", "real_estate", "shopping_mall",
    "spa", "gym", "hospital", "school", "office", "corporate_office",
    "local_government_office", "travel_agency", "insurance_agency", "bank",
    "atm", "car_dealer", "car_rental",
}
INVALID_NAME_PATTERNS = (
    "hotel", "khach san", "khách sạn", "supermarket", "sieu thi", "siêu thị",
    "baby", "mother", "me be", "mẹ bé", "toy", "do choi", "đồ chơi",
    "apartment", "lease", "cho thue", "cho thuê", "real estate",
    "bat dong san", "bất động sản", "do cung", "đồ cúng",
    "spa", "gym", "hospital", "school", "travel agency", "cong ty du lich",
    "công ty du lịch", "office", "van phong", "văn phòng", "company",
    "corp", "co ltd", "co., ltd", "tnhh", "cong ty", "công ty",
    "du lich", "du lịch",
)

# Nightlife graduated penalty (proposal #1 adjusted)
NIGHTLIFE_EARLY_PENALTY = {
    "bar":        {(0, 16): -99.0, (16, 17): -0.8, (17, 24): 0.0},
    "night_club": {(0, 16): -99.0, (16, 17): -0.8, (17, 18): -0.2, (18, 24): 0.0},
}

# Vietnamese food keywords (proposal #2)
VIETNAMESE_KEYWORDS = {
    "việt", "viet", "vietnamese", "phở", "pho", "bún", "bun",
    "cơm", "com", "bánh", "banh", "hủ tiếu", "hu tieu", "lẩu", "lau",
    "nhà hàng", "nha hang", "quán", "quan", "saigon", "sài gòn",
}
VIETNAMESE_MEAL_BONUS = 0.40
MEAL_WINDOWS = ((690, 810), (1080, 1260))  # 11:30-13:30 and 18:00-21:00


# ─────────────────────────────────────────────────────────────────────
# ROUTING AGENT
# ─────────────────────────────────────────────────────────────────────

class RoutingAgent(BaseAgent):
    """Agent 5 — Beam Search routing (v5).

    Root cause fix vs v4
    ─────────────────────
    Clustering trims to 72 total candidates then rebalances to 7-8/day.
    After 4 stops the cluster is exhausted → beam dies → only 4 stops/day
    despite 5+ hours remaining.

    Fix: _build_candidate_list() merges the cluster POIs (spatially coherent,
    high-score) with a fallback pool drawn from ALL scored candidates in
    memory, sorted by (proximity to cluster centroid, composite_score).
    The fallback is only used when cluster size < MIN_CANDIDATES_PER_DAY,
    and is capped at MAX_FALLBACK_CANDIDATES extra POIs.

    This guarantees the beam always has enough candidates to fill a 10-hour
    day without changing any scoring or constraint logic.
    """

    name: str = "routing"

    async def _execute(self, **kwargs: Any) -> Itinerary:
        clusters: dict[int, list[POI]]  = kwargs["daily_clusters"]
        user_input: UserInput           = kwargs["user_input"]
        # Full scored pool for fallback (set by ScoringAgent)
        all_scored: list[POI] = self.memory.get("scored_candidates") or []

        used_festival_keys: set[str] = set()
        used_poi_ids: set[str] = set()
        days: list[DayPlan]   = []

        for day_idx in sorted(clusters.keys()):
            trip_date   = user_input.start_date + timedelta(days=day_idx)
            day_pois    = clusters[day_idx]
            centroid    = self._centroid(day_pois)

            day_festival = self._cluster_festival(day_pois, used_festival_keys)
            allowed_festival_id = getattr(day_festival, "id", None)
            day_pois = [
                p for p in day_pois
                if getattr(p, "source", "") != "festival"
                or getattr(p, "id", None) == allowed_festival_id
            ]

            # Augment cluster with fallback if too small
            full_pool = self._build_candidate_list(day_pois, all_scored, centroid)
            full_pool = [
                p for p in full_pool
                if str(getattr(p, "id", id(p))) not in used_poi_ids
            ]

            day_plan = self._solve_day(
                day_num       = day_idx + 1,
                candidates    = full_pool,
                day_festival  = day_festival,
                start         = user_input.start_location,
                start_time    = user_input.start_time,
                trip_date     = trip_date,
                daily_minutes = user_input.daily_hours * 60,
                user_input    = user_input,
            )

            for stop in day_plan.stops:
                used_poi_ids.add(str(getattr(stop.poi, "id", id(stop.poi))))
                if getattr(stop.poi, "source", "") == "festival":
                    used_festival_keys.add(self._festival_key(stop.poi))

            days.append(day_plan)

        total_score = sum(d.total_score for d in days)
        total_stops = sum(len(d.stops)  for d in days)

        itinerary = Itinerary(
            city        = user_input.city,
            days        = days,
            total_score = total_score,
            metadata    = {
                "day_summaries": [self._day_summary(day) for day in days],
                "objective": (
                    "interest_alignment + entity_quality + spatial_efficiency "
                    "+ diversity_entropy - repetition_penalty"
                ),
            },
        )
        self.memory.set("itinerary", itinerary)
        self.logger.info(
            "Itinerary: %d days, %d stops, score=%.3f",
            len(days), total_stops, total_score,
        )
        return itinerary

    # ─────────────────────────────────────────────────────────────────
    # CANDIDATE POOL BUILDER (key fix)
    # ─────────────────────────────────────────────────────────────────

    def _build_candidate_list(
        self,
        cluster:   list[POI],
        all_scored: list[POI],
        centroid:  tuple[float, float] | None,
    ) -> list[POI]:
        """Return cluster POIs + fallback from global pool when cluster is small."""
        if len(cluster) >= MIN_CANDIDATES_PER_DAY:
            return list(cluster)

        cluster_ids = {id(p) for p in cluster}
        needed      = MIN_CANDIDATES_PER_DAY - len(cluster)
        needed      = min(needed, MAX_FALLBACK_CANDIDATES)

        # Sort non-cluster POIs by proximity to centroid then score
        extras = [p for p in all_scored if id(p) not in cluster_ids]
        if centroid:
            extras.sort(
                key=lambda p: (
                    haversine_km(p.latitude, p.longitude, centroid[0], centroid[1]),
                    -p.composite_score,
                )
            )
        else:
            extras.sort(key=lambda p: -p.composite_score)

        result = list(cluster) + extras[:needed]
        self.logger.debug(
            "Cluster %d POIs + %d fallback = %d total candidates",
            len(cluster), min(needed, len(extras)), len(result),
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # FESTIVAL SELECTION
    # ─────────────────────────────────────────────────────────────────

    def _pick_festival(
        self,
        festivals:  list[POI],
        used_ids:   set,
        trip_date:  date,
        centroid:   tuple[float, float] | None,
    ) -> POI | None:
        candidates = [
            f for f in festivals
            if getattr(f, "id", None) not in used_ids
            and (
                getattr(f, "dates", None) is None
                or trip_date in getattr(f, "dates", set())
            )
        ]
        if not candidates:
            return None

        if centroid is not None:
            nearby = [
                f for f in candidates
                if haversine_km(f.latitude, f.longitude, centroid[0], centroid[1])
                   <= FESTIVAL_MAX_DETOUR_KM
            ]
            pool = nearby if nearby else candidates
        else:
            pool = candidates

        return max(pool, key=lambda f: getattr(f, "composite_score", 0.0))

    def _cluster_festival(self, day_pois: list[POI], used_keys: set[str]) -> POI | None:
        festivals = [
            p for p in day_pois
            if getattr(p, "source", "") == "festival"
            and self._festival_key(p) not in used_keys
        ]
        if not festivals:
            return None
        return max(
            festivals,
            key=lambda p: (
                getattr(p, "composite_score", 0.0),
                getattr(p, "geocode_confidence", 1.0),
            ),
        )

    @staticmethod
    def _centroid(pois: list[POI]) -> tuple[float, float] | None:
        if not pois:
            return None
        return (
            sum(p.latitude  for p in pois) / len(pois),
            sum(p.longitude for p in pois) / len(pois),
        )

    # ─────────────────────────────────────────────────────────────────
    # BEAM SEARCH
    # ─────────────────────────────────────────────────────────────────

    def _solve_day(
        self,
        day_num:        int,
        candidates:     list[POI],
        day_festival:   POI | None,
        start:          tuple[float, float],
        start_time:     str,
        trip_date:      date,
        daily_minutes:  float,
        user_input:     UserInput,
    ) -> DayPlan:
        h, m      = map(int, start_time.split(":"))
        start_dt  = datetime.combine(trip_date, time(h, m))
        end_dt    = start_dt + timedelta(minutes=daily_minutes)

        seen_ids: set          = set()
        full_candidates: list[POI] = []

        if day_festival is not None:
            full_candidates.append(day_festival)
            seen_ids.add(id(day_festival))

        regular = sorted(
            [
                p for p in candidates
                if self._is_valid_entity(p)
                if getattr(p, "source", "") != "festival"
                or (
                    day_festival is not None
                    and getattr(p, "id", None) == getattr(day_festival, "id", None)
                )
            ],
            key=lambda p: getattr(p, "composite_score", 0.0),
            reverse=True,
        )
        for p in regular:
            pid = id(p)
            if pid not in seen_ids:
                full_candidates.append(p)
                seen_ids.add(pid)

        if not full_candidates:
            return DayPlan(day_number=day_num, date=trip_date, stops=[], total_score=0.0)

        beam: list[dict]                               = [self._init_state(start, start_dt, end_dt, user_input)]
        best_sequences: list[tuple[list, float, int]]  = []
        max_stops = self._max_stops(user_input)

        for _step in range(max_stops):
            next_beam: list[dict] = []

            for state in beam:
                for poi in full_candidates:
                    if id(poi) in state["visited_ids"]:
                        continue
                    ok, arr, dep, trav, hours = self._check_valid(poi, state, end_dt)
                    if not ok:
                        continue
                    next_beam.append(self._expand_state(state, poi, arr, dep, trav, hours))

                if state["route"]:
                    best_sequences.append((
                        list(state["route"]),
                        state["score"],
                        len(state["route"]),
                    ))

            if not next_beam:
                break

            next_beam.sort(key=lambda s: s["score"], reverse=True)
            beam = next_beam[:BEAM_WIDTH]

        for state in beam:
            if state["route"]:
                best_sequences.append((
                    list(state["route"]),
                    state["score"],
                    len(state["route"]),
                ))

        if not best_sequences:
            for poi in full_candidates:
                dist  = haversine_km(start[0], start[1], poi.latitude, poi.longitude)
                trav  = self._travel_time(dist, start_dt)
                arr   = start_dt + timedelta(minutes=trav)
                dep   = arr + timedelta(minutes=self._visit_duration(poi))
                hours = visit_opening_status(poi, arr, dep, allow_wait=True)
                arr, dep = hours.arrival, hours.departure
                if hours.fits and dep <= end_dt and not self._hard_time_invalid(poi, arr):
                    entry = {
                        "poi": poi, "arrival": arr, "departure": dep,
                        "travel": trav, "visit": self._visit_duration(poi),
                        "opening_source": hours.source,
                        "opening_wait": hours.wait_minutes,
                    }
                    return self._build_plan([entry], day_num, trip_date,
                                            getattr(poi, "composite_score", 0.5), user_input)
            return DayPlan(day_number=day_num, date=trip_date, stops=[], total_score=0.0)

        best_route, best_score, _ = self._select_best_sequence(
            best_sequences,
            full_candidates,
            day_festival,
            user_input,
        )
        best_route = self._ensure_nightlife_stop(best_route, full_candidates, end_dt, user_input)

        return self._build_plan(best_route, day_num, trip_date, best_score, user_input)

    # ─────────────────────────────────────────────────────────────────
    # STATE
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _init_state(start: tuple[float, float], start_dt: datetime, end_dt: datetime, user_input: UserInput | None = None) -> dict:
        return {
            "route":           [],
            "visited_ids":     set(),
            "pos":             start,
            "time":            start_dt,
            "start_dt":         start_dt,
            "end_dt":           end_dt,
            "end_pos":          start,
            "score":           0.0,
            "festival_count":  0,
            "history":         [],
            "food_before_17":  0,
            "lunch_count":     0,
            "dinner_count":    0,
            "category_counts": defaultdict(int),
            "user_input":      user_input,
        }

    def _expand_state(
        self,
        state: dict,
        poi:   POI,
        arr:   datetime,
        dep:   datetime,
        trav:  float,
        hours: Any,
    ) -> dict:
        visit_min  = self._visit_duration(poi)
        step_score = self._score(poi, trav, state, arr)
        new_counts = defaultdict(int, state["category_counts"])
        new_counts[poi.primary_type] += 1
        food_before_17 = state["food_before_17"]
        if self._is_food_type(poi) and arr.hour < 17:
            food_before_17 += 1
        lunch_count = state.get("lunch_count", 0)
        dinner_count = state.get("dinner_count", 0)
        if self._food_role(poi) == "meal":
            minute = arr.hour * 60 + arr.minute
            if 11 * 60 <= minute <= 14 * 60:
                lunch_count += 1
            elif minute >= 17 * 60:
                dinner_count += 1

        return {
            "route": state["route"] + [{
                "poi":       poi,
                "arrival":   arr,
                "departure": dep,
                "travel":    trav,
                "visit":     visit_min,
                "opening_source": getattr(hours, "source", "unknown"),
                "opening_wait": getattr(hours, "wait_minutes", 0),
            }],
            "visited_ids":    state["visited_ids"] | {id(poi)},
            "pos":            (poi.latitude, poi.longitude),
            "time":           dep + timedelta(minutes=self._buffer_min(state.get("user_input"))),
            "start_dt":       state["start_dt"],
            "end_dt":         state["end_dt"],
            "end_pos":        state["end_pos"],
            "score":          state["score"] + step_score,
            "festival_count": state["festival_count"] + (
                1 if getattr(poi, "source", "") == "festival" else 0
            ),
            "history":        state["history"] + [self._semantic_group(poi)],
            "food_before_17": food_before_17,
            "lunch_count":    lunch_count,
            "dinner_count":   dinner_count,
            "category_counts": new_counts,
            "user_input":      state.get("user_input"),
        }

    # ─────────────────────────────────────────────────────────────────
    # CONSTRAINTS
    # ─────────────────────────────────────────────────────────────────

    def _check_valid(
        self,
        poi:    POI,
        state:  dict,
        end_dt: datetime,
    ) -> tuple[bool, datetime, datetime, float, Any]:
        dist  = haversine_km(
            state["pos"][0], state["pos"][1],
            poi.latitude, poi.longitude,
        )
        trav  = self._travel_time(dist, state["time"])
        arr   = state["time"] + timedelta(minutes=trav)
        dep   = arr + timedelta(minutes=self._visit_duration(poi))
        FAIL  = (False, arr, dep, trav, None)

        if dep > end_dt:                return FAIL
        if arr.hour >= END_OF_DAY_HOUR: return FAIL

        user_input: UserInput | None = state.get("user_input")
        if getattr(poi, "source", "") == "festival":
            if state["festival_count"] >= MAX_FESTIVALS_PER_DAY:
                return FAIL
            if getattr(poi, "geocode_confidence", 1.0) <= 0.35:
                return FAIL

        ptype = poi.primary_type
        visit_min = self._visit_duration(poi)
        if visit_min <= 0:
            return FAIL

        # Keep nightlife in evening slots only. With children, night clubs are
        # treated as inappropriate; bars remain a soft preference decision.
        if user_input and user_input.has_children and ptype == "night_club":
            return FAIL
        if ptype in NIGHTLIFE_TYPES and (arr.hour * 60 + arr.minute) < 17 * 60 + 30:
            arr = datetime.combine(arr.date(), time(17, 30))
            dep = arr + timedelta(minutes=self._visit_duration(poi))
            if dep > end_dt:
                return FAIL
        if ptype in NIGHTLIFE_TYPES and arr.hour < 16:
            return FAIL
        if ptype in NIGHTLIFE_TYPES and (arr.hour * 60 + arr.minute) < 17 * 60 + 30:
            return FAIL
        if ptype in NIGHTLIFE_MIN_HOUR and arr.hour < NIGHTLIFE_MIN_HOUR[ptype]:
            return FAIL

        if ptype in COFFEE_TYPES and arr.hour >= 18 and state["category_counts"]["cafe"] > 1:
            return FAIL

        hours = visit_opening_status(poi, arr, dep, allow_wait=True)
        if not hours.fits:
            return FAIL
        arr, dep = hours.arrival, hours.departure
        if dep > end_dt or arr.hour >= END_OF_DAY_HOUR:
            return FAIL

        if not self._food_sequence_ok(poi, state, arr):
            return FAIL

        if self._hard_time_invalid(poi, arr):
            return FAIL

        if self._semantic_group(poi) == "food":
            if self._food_role(poi) == "meal":
                minute = arr.hour * 60 + arr.minute
                if 11 * 60 <= minute <= 14 * 60 and state.get("lunch_count", 0) >= 1:
                    return FAIL
            for entry in reversed(state["route"]):
                if self._semantic_group(entry["poi"]) == "food":
                    delta = (arr - entry["arrival"]).total_seconds() / 60.0
                    if delta < 75 and self._food_role(poi) == "meal":
                        return FAIL
                    break

        # Daily hard cap
        if state["category_counts"][ptype] >= DAILY_HARD_CAP.get(ptype, 99):
            return FAIL

        # No 3 consecutive identical categories
        h = state["history"]
        group = self._semantic_group(poi)
        if len(h) >= 2 and h[-1] == group and h[-2] == group:
            return FAIL
        if group == "culture" and len(h) >= 2 and h[-1] == "culture" and h[-2] == "culture":
            return FAIL

        if user_input and user_input.pace == "relaxed" and len(state["route"]) >= self._max_stops(user_input):
            return FAIL

        return (True, arr, dep, trav, hours)

    # ─────────────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────────────

    def _score(self, poi: POI, travel_min: float, state: dict, arrival: datetime) -> float:
        diversity = self._route_diversity_gain(poi, state["category_counts"])
        repetition_penalty = self._repeat_type_penalty(poi, state["category_counts"])
        travel_penalty = self._travel_penalty(travel_min, state)
        end_penalty = self._end_location_penalty(poi, arrival, state)
        fatigue_penalty = self._fatigue_penalty(poi, arrival, state)

        base = (
            0.40 * getattr(poi, "composite_score", 0.5)
            + 0.25 * getattr(poi, "quality_score", 0.5)
            + 0.25 * diversity
            - repetition_penalty
            - travel_penalty
            - end_penalty
            - fatigue_penalty
        )

        base += self._temporal_bonus(poi, arrival)
        base += self._experience_rhythm_bonus(poi, state, arrival)
        base -= self._energy_fatigue_penalty(poi, state)
        base -= self._opening_hours_penalty(poi, arrival)
        base -= self._unrealistic_meal_penalty(poi, arrival)
        base += self._profile_bonus(poi, travel_min, state, arrival)
        base += self._required_category_bonus(poi, state["category_counts"])
        base += self._daytime_food_bonus(poi, state)
        base += self._flow_bonus(poi, state["history"])
        base -= self._soft_penalty(poi, state["category_counts"])

        if getattr(poi, "source", "") == "festival":
            confidence = max(0.0, min(1.0, getattr(poi, "geocode_confidence", 1.0)))
            if confidence <= 0.35:
                base -= 2.0
            else:
                base += FESTIVAL_BOOST * (0.75 + 0.25 * confidence)

        return base

    @staticmethod
    def _profile_bonus(poi: POI, travel_min: float, state: dict, arrival: datetime) -> float:
        user_input: UserInput | None = state.get("user_input")
        if user_input is None:
            return 0.0
        profile = build_preference_profile(user_input)
        bonus = 0.0
        group = RoutingAgent._semantic_group(poi)
        text = poi_text(poi)

        bonus += (profile.category_weight(group) - 1.0) * 1.2
        if profile.must_have_terms and matches_any(text, profile.must_have_terms):
            bonus += 0.85
        if profile.avoid_terms and matches_any(text, profile.avoid_terms):
            bonus -= 1.8

        if profile.family_mode:
            if group == "nightlife":
                bonus -= 0.9 if profile.nightlife_preference != "like" else 0.25
            if poi.primary_type in {"park", "zoo", "aquarium", "museum", "art_gallery", "cafe"}:
                bonus += 0.28
            if any(term in text for term in ("family", "kid", "children", "playground")):
                bonus += 0.25
        if profile.young_traveler:
            if group in {"coffee", "shopping", "outdoor", "nightlife"}:
                bonus += 0.22
            if any(term in text for term in ("photo", "view", "rooftop", "boutique", "concept", "riverside", "riverfront", "walking street", "nguyen hue", "market")):
                bonus += 0.28
        if user_input.mobility == "limited":
            bonus -= min(travel_min / 35.0, 1.0) * 0.75
            if len(state["route"]) >= 4:
                bonus -= 0.35
        if user_input.pace == "relaxed" and len(state["route"]) >= 4:
            bonus -= 0.55
        if user_input.pace == "balanced" and len(state["route"]) >= 6:
            bonus -= 0.22
        if user_input.pace == "intense" and len(state["route"]) >= 5:
            bonus += 0.18
        minute = arrival.hour * 60 + arrival.minute
        if group == "food" and 11 * 60 + 30 <= minute <= 13 * 60 + 30:
            bonus += 0.45
        if group == "food" and 14 * 60 <= minute < 17 * 60:
            bonus -= 1.2
        if user_input.food_restrictions and RoutingAgent._violates_food_restrictions(poi, user_input.food_restrictions):
            bonus -= 2.2
        return bonus

    def _temporal_bonus(self, poi: POI, current_dt: datetime) -> float:
        hour  = current_dt.hour
        ptype = poi.primary_type
        bonus = 0.0

        minute = hour * 60 + current_dt.minute

        if ptype in COFFEE_TYPES:
            if 8 * 60 <= minute <= 10 * 60 + 30:
                bonus += 0.55
            elif 15 * 60 <= minute <= 17 * 60:
                bonus += 0.40
            elif hour >= 18:
                bonus -= 0.55
        if ptype in CULTURE_TYPES:
            if 9 <= hour < 17:
                bonus += 0.45
            else:
                bonus -= 0.45
        if ptype == "park":
            if 6  <= hour <= 10: bonus += 0.20
            if 16 <= hour <= 19: bonus += 0.15
        if ptype == "tourist_attraction":
            if 8  <= hour <= 16: bonus += 0.15

        if ptype in FOOD_TYPES:
            if 11 * 60 + 30 <= minute <= 13 * 60 + 30:
                bonus += 0.55
            elif 17 * 60 + 30 <= minute <= 19 * 60 + 30:
                bonus += 0.35
            if self._is_vietnamese(poi):
                if any(lo <= minute <= hi for lo, hi in MEAL_WINDOWS):
                    bonus += VIETNAMESE_MEAL_BONUS

        # Graduated nightlife penalty
        if ptype in NIGHTLIFE_EARLY_PENALTY:
            for (lo, hi), pen in NIGHTLIFE_EARLY_PENALTY[ptype].items():
                if lo <= hour < hi:
                    bonus += pen
                    break

        return bonus

    @staticmethod
    def _experience_rhythm_bonus(poi: POI, state: dict, arrival: datetime) -> float:
        group = RoutingAgent._semantic_group(poi)
        role = RoutingAgent._food_role(poi)
        minute = arrival.hour * 60 + arrival.minute
        bonus = 0.0

        # Morning: light start, aesthetic cafe, walking/photo-friendly places.
        if 8 * 60 <= minute < 11 * 60:
            if group in {"coffee", "outdoor"}:
                bonus += 0.55
            elif group == "food" and role == "snack":
                bonus += 0.22
            elif group == "nightlife":
                bonus -= 1.6

        # Midday: one core activity or one real meal. Avoid heavy chains.
        if 11 * 60 <= minute < 14 * 60:
            if group == "food" and role == "meal":
                bonus += 0.65
            elif group == "culture":
                bonus += 0.30

        # Afternoon: flex/rest/photo/shopping. Heavy museums are okay, but not endlessly.
        if 14 * 60 <= minute < 17 * 60:
            if group in {"outdoor", "shopping", "coffee"}:
                bonus += 0.42
            elif group == "culture":
                bonus += 0.08

        # Evening: food/nightlife/walkable social energy.
        if minute >= 17 * 60:
            if group == "food":
                bonus += 0.55 if role == "meal" else 0.25
            elif group == "nightlife":
                bonus += 0.62
            elif group == "culture":
                bonus -= 0.55

        if state["route"]:
            prev_group = RoutingAgent._semantic_group(state["route"][-1]["poi"])
            if prev_group != group:
                bonus += 0.18
        return bonus

    @staticmethod
    def _energy_fatigue_penalty(poi: POI, state: dict) -> float:
        group = RoutingAgent._semantic_group(poi)
        route = state.get("route", [])
        if not route:
            return 0.0

        prev = route[-1]["poi"]
        prev_group = RoutingAgent._semantic_group(prev)
        penalty = 0.0

        heavy_groups = {"culture"}
        if group in heavy_groups and prev_group in heavy_groups:
            penalty += 0.45
            if len(route) >= 2 and RoutingAgent._semantic_group(route[-2]["poi"]) in heavy_groups:
                penalty += 1.2

        if group == "food" and prev_group == "food":
            role = RoutingAgent._food_role(poi)
            prev_role = RoutingAgent._food_role(prev)
            if prev_role == "meal" or role == "meal":
                penalty += 0.85
            else:
                penalty += 0.22

        if group == "coffee" and prev_group == "coffee":
            penalty += 0.50

        if len(route) >= 5 and group == "culture":
            penalty += 0.25
        return penalty

    @staticmethod
    def _opening_hours_penalty(poi: POI, arrival: datetime) -> float:
        status = opening_status(poi, arrival, arrival.weekday())
        group = RoutingAgent._semantic_group(poi)
        penalty = status.penalty

        if status.source == "explicit" and not status.is_open:
            if group == "culture" and arrival.hour >= 18:
                penalty += 0.12
            if group == "nightlife" and arrival.hour < 17:
                penalty += 0.10
            return min(0.40, penalty)

        # Missing/weak hours: assume open, but nudge away from suspicious slots.
        if status.source == "assumed":
            return min(0.30, penalty)
        return 0.0

    @staticmethod
    def _unrealistic_meal_penalty(poi: POI, arrival: datetime) -> float:
        group = RoutingAgent._semantic_group(poi)
        role = RoutingAgent._food_role(poi)
        minute = arrival.hour * 60 + arrival.minute
        if group != "food":
            return 0.0
        if role == "meal":
            lunch = 11 * 60 <= minute <= 14 * 60
            dinner = 17 * 60 <= minute <= 21 * 60
            breakfastish = 7 * 60 <= minute <= 10 * 60 and any(
                term in poi_text(poi)
                for term in ("pho", "phở", "bun", "bún", "banh mi", "bánh mì", "breakfast")
            )
            if not (lunch or dinner or breakfastish):
                return 0.24
        if role == "snack" and minute < 7 * 60:
            return 0.16
        return 0.0

    @staticmethod
    def _hard_time_invalid(poi: POI, arrival: datetime) -> bool:
        group = RoutingAgent._semantic_group(poi)
        minute = arrival.hour * 60 + arrival.minute
        if group == "nightlife" and minute < 16 * 60:
            return True
        if group == "culture" and minute >= 20 * 60:
            return True
        if group == "food" and RoutingAgent._food_role(poi) == "meal":
            if minute < 6 * 60 or minute > 22 * 60 + 30:
                return True
        status = opening_status(poi, arrival, arrival.weekday())
        if status.source == "explicit" and status.confidence >= 0.85 and not status.is_open:
            if group == "culture" and minute >= 18 * 60 + 30:
                return True
            if group == "nightlife" and minute < 17 * 60:
                return True
        return False

    @staticmethod
    def _is_vietnamese(poi: POI) -> bool:
        return any(kw in poi.name.lower() for kw in VIETNAMESE_KEYWORDS)

    @staticmethod
    def _flow_bonus(poi: POI, history: list[str]) -> float:
        if not history:
            return 0.0
        prev = history[-1]
        curr = RoutingAgent._semantic_group(poi)
        if prev in ("food", "coffee") and curr == "culture":
            return 0.22
        if prev == "culture" and curr in ("food", "coffee"):
            return 0.18
        if prev != "nightlife" and curr == "nightlife":
            return 0.25
        if prev == curr:
            return -0.35
        return 0.0

    @staticmethod
    def _travel_penalty(travel_min: float, state: dict) -> float:
        elapsed = (state["time"] - state["start_dt"]).total_seconds() / 60.0
        total = max((state["end_dt"] - state["start_dt"]).total_seconds() / 60.0, 1.0)
        fatigue_weight = TRAVEL_PENALTY_BASE + 0.35 * max(0.0, min(1.0, elapsed / total))
        user_input: UserInput | None = state.get("user_input")
        profile_multiplier = (
            build_preference_profile(user_input).travel_penalty_multiplier
            if user_input is not None
            else 1.0
        )
        return min(travel_min / 45.0, 1.0) * fatigue_weight * profile_multiplier

    @staticmethod
    def _end_location_penalty(poi: POI, arrival: datetime, state: dict) -> float:
        end_pos = state.get("end_pos")
        if end_pos is None:
            return 0.0
        remaining = (state["end_dt"] - arrival).total_seconds() / 60.0
        total = max((state["end_dt"] - state["start_dt"]).total_seconds() / 60.0, 1.0)
        day_progress = 1.0 - max(0.0, min(1.0, remaining / total))
        dist_back = haversine_km(poi.latitude, poi.longitude, end_pos[0], end_pos[1])
        return min(dist_back / 8.0, 1.0) * END_HEURISTIC_WEIGHT * day_progress

    @staticmethod
    def _fatigue_penalty(poi: POI, arrival: datetime, state: dict) -> float:
        if arrival.hour < 15:
            return 0.0
        group = RoutingAgent._semantic_group(poi)
        visit_min = RoutingAgent._visit_duration(poi)
        route_len = len(state["route"])
        if group == "nightlife":
            return 0.0
        heavy = group == "culture" or visit_min >= 75
        if not heavy:
            return 0.08 if route_len >= 5 else 0.0
        return 0.22 + 0.05 * max(0, route_len - 4)

    @staticmethod
    def _soft_penalty(poi: POI, counts: defaultdict) -> float:
        cap = SOFT_CAP.get(poi.primary_type)
        if cap is None:
            return 0.0
        excess = max(0, counts[poi.primary_type] - cap + 1)
        return excess * SOFT_CAP_PENALTY

    @staticmethod
    def _repeat_type_penalty(poi: POI, counts: defaultdict) -> float:
        return 0.18 * counts[poi.primary_type]

    @staticmethod
    def _route_diversity_gain(poi: POI, counts: defaultdict) -> float:
        same_group = sum(
            count for ptype, count in counts.items()
            if RoutingAgent._semantic_group_from_type(ptype) == RoutingAgent._semantic_group(poi)
        )
        return 1.0 / (1.0 + same_group)

    @staticmethod
    def _semantic_group_from_type(primary_type: str) -> str:
        if primary_type in COFFEE_TYPES:
            return "coffee"
        if primary_type in NIGHTLIFE_TYPES:
            return "nightlife"
        if primary_type in CULTURE_TYPES:
            return "culture"
        if primary_type in FOOD_TYPES:
            return "food"
        return "other"

    def _daytime_food_bonus(self, poi: POI, state: dict) -> float:
        if self._is_food_type(poi) and state["time"].hour < 17 and state["food_before_17"] < 2:
            return 0.45
        return 0.0

    @staticmethod
    def _required_category_bonus(poi: POI, counts: defaultdict) -> float:
        return 0.0

    @staticmethod
    def _select_best_sequence(
        best_sequences: list[tuple[list, float, int]],
        candidates: list[POI],
        day_festival: POI | None,
        user_input: UserInput | None = None,
    ) -> tuple[list, float, int]:
        profile = build_preference_profile(user_input) if user_input is not None else None
        wants_nightlife = user_input is not None and RoutingAgent._user_wants_nightlife(user_input)
        has_nightlife_candidate = wants_nightlife and any(p.primary_type in NIGHTLIFE_TYPES for p in candidates)

        def contains_type(route: list, types: set[str]) -> bool:
            return any(entry["poi"].primary_type in types for entry in route)

        def contains_festival(route: list) -> bool:
            return any(
                getattr(entry["poi"], "source", "") == "festival"
                for entry in route
            )

        pool = best_sequences
        if day_festival is not None:
            festival_pool = [seq for seq in pool if contains_festival(seq[0])]
            if festival_pool:
                pool = festival_pool

        if has_nightlife_candidate:
            nightlife_pool = [seq for seq in pool if contains_type(seq[0], NIGHTLIFE_TYPES)]
            if nightlife_pool:
                seven_stop_pool = [seq for seq in nightlife_pool if seq[2] >= 6]
                pool = seven_stop_pool or nightlife_pool

        pool.sort(
            key=lambda x: (
                x[1],
                RoutingAgent._route_balance_score(x[0], candidates, user_input),
                -abs(x[2] - RoutingAgent._preferred_stop_count(user_input)),
            ),
            reverse=True,
        )
        return pool[0]

    def _ensure_nightlife_stop(
        self,
        route: list[dict],
        candidates: list[POI],
        end_dt: datetime,
        user_input: UserInput | None = None,
    ) -> list[dict]:
        if user_input is None or not self._user_wants_nightlife(user_input):
            return route
        if any(entry["poi"].primary_type in NIGHTLIFE_TYPES for entry in route):
            return route
        nightlife = [
            p for p in candidates
            if p.primary_type in NIGHTLIFE_TYPES
            and id(p) not in {id(entry["poi"]) for entry in route}
        ]
        if not nightlife or len(route) < 2:
            return route

        if len(route) < self._max_stops(user_input):
            state = self._state_from_route(route, user_input)
            for poi in sorted(nightlife, key=lambda p: getattr(p, "composite_score", 0.0), reverse=True):
                ok, arr, dep, trav, hours = self._check_valid(poi, state, end_dt)
                if ok and arr.hour >= 17:
                    return route + [{
                        "poi": poi,
                        "arrival": arr,
                        "departure": dep,
                        "travel": trav,
                        "visit": self._visit_duration(poi),
                        "opening_source": getattr(hours, "source", "unknown"),
                        "opening_wait": getattr(hours, "wait_minutes", 0),
                    }]

        prefix = route[:-1]
        if getattr(route[-1]["poi"], "source", "") == "festival":
            return route

        state = self._state_from_route(prefix, user_input)
        for poi in sorted(nightlife, key=lambda p: getattr(p, "composite_score", 0.0), reverse=True):
            ok, arr, dep, trav, hours = self._check_valid(poi, state, end_dt)
            if ok and arr.hour >= 17:
                replacement = {
                    "poi": poi,
                    "arrival": arr,
                    "departure": dep,
                    "travel": trav,
                    "visit": self._visit_duration(poi),
                    "opening_source": getattr(hours, "source", "unknown"),
                    "opening_wait": getattr(hours, "wait_minutes", 0),
                }
                return prefix + [replacement]
        return route

    @staticmethod
    def _state_from_route(route: list[dict], user_input: UserInput | None = None) -> dict:
        counts: defaultdict[str, int] = defaultdict(int)
        history: list[str] = []
        visited_ids: set[int] = set()
        food_before_17 = 0
        festival_count = 0
        for entry in route:
            poi = entry["poi"]
            counts[poi.primary_type] += 1
            history.append(RoutingAgent._semantic_group(poi))
            visited_ids.add(id(poi))
            if RoutingAgent._is_food_type(poi) and entry["arrival"].hour < 17:
                food_before_17 += 1
            if getattr(poi, "source", "") == "festival":
                festival_count += 1
        last = route[-1]
        return {
            "route": route,
            "visited_ids": visited_ids,
            "pos": (last["poi"].latitude, last["poi"].longitude),
            "time": last["departure"] + timedelta(minutes=RoutingAgent._buffer_min(user_input)),
            "start_dt": route[0]["arrival"],
            "end_dt": route[-1]["departure"] + timedelta(hours=4),
            "end_pos": (last["poi"].latitude, last["poi"].longitude),
            "score": 0.0,
            "festival_count": festival_count,
            "history": history,
            "food_before_17": food_before_17,
            "lunch_count": sum(
                1 for entry in route
                if RoutingAgent._food_role(entry["poi"]) == "meal"
                and 11 <= entry["arrival"].hour <= 14
            ),
            "dinner_count": sum(
                1 for entry in route
                if RoutingAgent._food_role(entry["poi"]) == "meal"
                and entry["arrival"].hour >= 17
            ),
            "category_counts": counts,
            "user_input": user_input,
        }

    @staticmethod
    def _route_balance_score(route: list, candidates: list[POI], user_input: UserInput | None = None) -> float:
        profile = build_preference_profile(user_input) if user_input is not None else None
        available_food = sum(1 for p in candidates if RoutingAgent._is_food_type(p))
        target_food = min(2, available_food)
        food_before_17 = sum(
            1 for entry in route
            if RoutingAgent._is_food_type(entry["poi"]) and entry["arrival"].hour < 17
        )
        museums_midday = sum(
            1 for entry in route
            if RoutingAgent._semantic_group(entry["poi"]) == "culture"
            and 10 <= entry["arrival"].hour <= 15
        )
        museum_total = sum(
            1 for entry in route if RoutingAgent._semantic_group(entry["poi"]) == "culture"
        )
        nightlife_evening = sum(
            1 for entry in route
            if RoutingAgent._semantic_group(entry["poi"]) == "nightlife" and entry["arrival"].hour >= 17
        )
        type_counts: defaultdict[str, int] = defaultdict(int)
        for entry in route:
            type_counts[RoutingAgent._semantic_group(entry["poi"])] += 1
        repeat_penalty = sum(max(0, count - 1) for count in type_counts.values()) * 0.15

        score = 0.0
        if target_food:
            food_weight = profile.category_weight("food") if profile else 1.0
            score += 0.25 * food_weight * min(food_before_17, target_food) / target_food
        culture_weight = profile.category_weight("culture") if profile else 1.0
        score += 0.20 * culture_weight * min(1.0, museum_total / 2)
        if museum_total:
            score += 0.15 * culture_weight * min(1.0, museums_midday / museum_total)
        if profile is None or profile.nightlife_preference == "like":
            score += 0.45 * min(1.0, nightlife_evening / 1)
        transitions = sum(
            1 for left, right in zip(route, route[1:])
            if RoutingAgent._semantic_group(left["poi"]) != RoutingAgent._semantic_group(right["poi"])
        )
        if len(route) > 1:
            score += 0.35 * transitions / (len(route) - 1)
        score -= repeat_penalty
        return score

    @staticmethod
    def _preferred_stop_count(user_input: UserInput | None) -> int:
        if user_input is None:
            return 6
        return {"relaxed": 4, "balanced": 6, "intense": 8}.get(user_input.pace, 6)

    @staticmethod
    def _is_food_type(poi: POI) -> bool:
        return RoutingAgent._semantic_group(poi) in {"food", "coffee"}

    @staticmethod
    def _semantic_group(poi: POI) -> str:
        return category_for_poi(poi)

    @staticmethod
    def _food_role(poi: POI) -> str | None:
        return getattr(poi, "food_role", None) or infer_food_role(poi)

    @staticmethod
    def _food_sequence_ok(poi: POI, state: dict, arrival: datetime) -> bool:
        group = RoutingAgent._semantic_group(poi)
        if group not in {"food", "coffee"}:
            return True
        if not state["route"]:
            return True
        prev = state["route"][-1]["poi"]
        prev_group = RoutingAgent._semantic_group(prev)
        if prev_group not in {"food", "coffee"}:
            return True

        role = RoutingAgent._food_role(poi)
        prev_role = RoutingAgent._food_role(prev)
        if prev_role in {"cafe", "snack"} and role == "meal":
            return True
        if prev_role == "meal" and role in {"cafe", "snack"}:
            delta = (arrival - state["route"][-1]["arrival"]).total_seconds() / 60.0
            return delta >= 90
        return False

    @staticmethod
    def _family_mode(user_input: UserInput) -> bool:
        return user_input.travel_group == "family" or user_input.has_children

    @staticmethod
    def _user_wants_nightlife(user_input: UserInput) -> bool:
        return build_preference_profile(user_input).nightlife_preference == "like"

    @staticmethod
    def _max_stops(user_input: UserInput | None) -> int:
        if user_input is None:
            return MAX_STOPS
        profile_cap = build_preference_profile(user_input).max_stops
        return max(3, min(max(user_input.max_places_per_day, 1), profile_cap))

    @staticmethod
    def _buffer_min(user_input: UserInput | None) -> int:
        if user_input is None:
            return BUFFER_MIN
        return build_preference_profile(user_input).buffer_minutes

    @staticmethod
    def _violates_food_restrictions(poi: POI, restrictions: list[str]) -> bool:
        if not restrictions or not RoutingAgent._is_food_type(poi):
            return False
        text = RoutingAgent._normalize_ascii(" ".join([poi.name, poi.primary_type, *poi.types]))
        restriction_map = {
            "vegetarian": ("steak", "bbq", "grill", "seafood", "meat", "beef", "pork", "chicken"),
            "vegan": ("steak", "bbq", "grill", "seafood", "meat", "beef", "pork", "chicken", "milk", "cheese"),
            "halal": ("pork", "bar", "beer", "wine", "cocktail"),
            "no seafood": ("seafood", "fish", "sushi", "oyster", "snail"),
            "seafood allergy": ("seafood", "fish", "sushi", "oyster", "snail"),
        }
        for restriction in restrictions:
            normalized = RoutingAgent._normalize_ascii(restriction)
            keywords = restriction_map.get(normalized, (normalized,))
            if any(keyword in text for keyword in keywords):
                return True
        return False

    @staticmethod
    def _is_valid_entity(poi: POI) -> bool:
        if getattr(poi, "source", "") == "festival":
            return True
        if not poi.name or not poi.primary_type or not poi.types:
            return False
        poi_types = {poi.primary_type.lower(), *(t.lower() for t in poi.types)}
        if poi_types & INVALID_TYPES:
            return False
        normalized = RoutingAgent._normalize_ascii(poi.name)
        if any(pattern in normalized for pattern in INVALID_NAME_PATTERNS):
            return False
        name = poi.name.strip()
        if len(name) < 3 or len(name) > 90:
            return False
        if any(ord(ch) < 32 for ch in name):
            return False
        return True

    @staticmethod
    def _normalize_ascii(name: str) -> str:
        import re
        import unicodedata

        lowered = name.lower()
        decomposed = unicodedata.normalize("NFD", lowered)
        ascii_text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return " ".join(re.findall(r"[a-z0-9À-ỹ]+", f"{lowered} {ascii_text}"))

    @staticmethod
    def _festival_key(poi: POI) -> str:
        return str(getattr(poi, "id", None) or poi.name).strip().lower()

    @staticmethod
    def _day_summary(day: DayPlan) -> dict:
        type_counts: defaultdict[str, int] = defaultdict(int)
        reasons: list[str] = []
        for stop in day.stops:
            ptype = stop.poi.primary_type
            type_counts[ptype] += 1
            if getattr(stop.poi, "source", "") == "festival":
                reasons.append(f"festival anchor: {stop.poi.name}")
            elif ptype in MUSEUM_TYPES:
                reasons.append(f"culture stop: {stop.poi.name}")
            elif ptype == "cafe":
                reasons.append(f"coffee break: {stop.poi.name}")
            elif RoutingAgent._is_food_type(stop.poi):
                reasons.append(f"meal/local food: {stop.poi.name}")
        return {
            "day": day.day_number,
            "score": round(day.total_score, 3),
            "stops": len(day.stops),
            "category_counts": dict(type_counts),
            "top_reasons": reasons[:5],
        }

    # ─────────────────────────────────────────────────────────────────
    # TRAVEL & DURATION
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _travel_time(dist_km: float, current_dt: datetime) -> float:
        hour    = current_dt.hour
        is_rush = any(s <= hour < e for s, e in RUSH_HOURS)
        speed   = RUSH_SPEED_KMH if is_rush else AVG_SPEED_KMH
        in_vehicle = (dist_km / speed) * 60.0
        signal_delay = min(8.0, dist_km * 1.4)
        variability = ((int(dist_km * 1000) + current_dt.hour * 7 + current_dt.minute) % 5) * 0.8
        return max(MIN_TRAVEL_MIN, TRAVEL_OVERHEAD_MIN + in_vehicle + signal_delay + variability)

    @staticmethod
    def _visit_duration(poi: POI) -> int:
        if getattr(poi, "source", "") == "festival":
            return getattr(poi, "estimated_visit_minutes", 120)
        return VISIT_DURATION.get(poi.primary_type, 45)

    # ─────────────────────────────────────────────────────────────────
    # BUILD PLAN — travel_minutes_from_prev tính đúng cho stop[0]
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_plan(
        route:       list[dict],
        day_num:     int,
        trip_date:   date,
        total_score: float,
        user_input:  UserInput | None = None,
    ) -> DayPlan:
        stops: list[ItineraryStop] = []
        for i, entry in enumerate(route):
            poi = entry["poi"]
            warnings = RoutingAgent._stop_warnings(
                poi,
                user_input,
                entry["arrival"],
                entry["departure"],
            )
            notes = RoutingAgent._opening_notes(entry)
            if getattr(poi, "source", "") == "festival":
                festival_note = (
                    f"Festival event; geocode={getattr(poi, 'geocode_method', 'source')} "
                    f"confidence={getattr(poi, 'geocode_confidence', 1.0):.2f}"
                )
                notes = f"{festival_note} {notes}".strip()
            stops.append(ItineraryStop(
                order                    = i + 1,
                poi                      = poi,
                arrival_time             = entry["arrival"].time(),
                departure_time           = entry["departure"].time(),
                travel_minutes_from_prev = round(entry["travel"], 1),  # always set
                visit_minutes            = entry["visit"],
                notes                    = notes,
                reason                   = RoutingAgent._stop_reason(poi, entry, user_input),
                matched_interests        = RoutingAgent._matched_interests(poi, user_input),
                warnings                 = warnings,
            ))
        return DayPlan(
            day_number  = day_num,
            date        = trip_date,
            stops       = stops,
            total_score = total_score,
            total_travel_minutes = sum(stop.travel_minutes_from_prev for stop in stops),
            total_visit_minutes = sum(stop.visit_minutes for stop in stops),
        )

    @staticmethod
    def _opening_notes(entry: dict) -> str:
        source = entry.get("opening_source")
        notes: list[str] = []
        if source == "explicit":
            notes.append("Opening hours checked.")
        else:
            notes.append("Opening hours are not confirmed; verify before going.")
        wait = int(round(float(entry.get("opening_wait", 0) or 0)))
        if wait > 0:
            notes.append(f"Includes {wait} minutes waiting for opening time.")
        return " ".join(notes)

    @staticmethod
    def _matched_interests(poi: POI, user_input: UserInput | None) -> list[str]:
        if user_input is None:
            return []
        poi_text = RoutingAgent._normalize_ascii(" ".join([poi.name, poi.primary_type, *poi.types]))
        matches: list[str] = []
        group = RoutingAgent._semantic_group(poi)
        group_terms = {
            "food": ("food", "restaurant", "street food", "local food", "vietnamese food", "lunch", "dinner"),
            "coffee": ("coffee", "cafe", "brunch", "dessert"),
            "culture": ("museum", "art", "gallery", "culture", "history", "attraction"),
            "nightlife": ("nightlife", "bar", "club", "cocktail"),
            "other": ("park", "family", "shopping", "market"),
            "festival": ("festival", "event", "culture"),
        }
        for interest in user_input.interests:
            normalized = RoutingAgent._normalize_ascii(interest)
            if normalized and (normalized in poi_text or any(term in normalized for term in group_terms.get(group, ()))):
                matches.append(interest)
        return matches[:4]

    @staticmethod
    def _stop_reason(poi: POI, entry: dict, user_input: UserInput | None) -> str:
        parts: list[str] = []
        matches = RoutingAgent._matched_interests(poi, user_input)
        text = poi_text(poi)
        if matches:
            parts.append("matches " + ", ".join(matches[:2]))
        if user_input is not None:
            profile = build_preference_profile(user_input)
            if profile.must_have_terms and matches_any(text, profile.must_have_terms):
                parts.append("explicit must-have fit")
        if getattr(poi, "composite_score", 0.0) >= 0.65:
            parts.append("high overall score")
        if getattr(poi, "quality_score", 0.0) >= 0.65:
            parts.append("strong review signal")
        if entry.get("travel", 0.0) <= 18:
            parts.append("keeps travel short")
        arrival = entry["arrival"]
        group = RoutingAgent._semantic_group(poi)
        minute = arrival.hour * 60 + arrival.minute
        if group == "coffee" and arrival.hour < 11:
            parts.append("fits a gentle morning start")
        elif group == "food" and 11 * 60 + 30 <= minute <= 13 * 60 + 30:
            parts.append("lands in the lunch window")
        elif group == "culture" and 9 <= arrival.hour < 17:
            parts.append("works well in the daytime")
        elif group == "nightlife" and arrival.hour >= 17:
            parts.append("placed in the evening")
        if user_input and RoutingAgent._family_mode(user_input) and group != "nightlife":
            parts.append("fits a family-oriented day")
        if getattr(poi, "source", "") == "festival":
            parts.append("date-specific festival option")
        return "; ".join(parts[:4]) or "selected for score, timing, and route fit"

    @staticmethod
    def _stop_warnings(
        poi: POI,
        user_input: UserInput | None,
        arrival: datetime | None = None,
        departure: datetime | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        text = poi_text(poi)
        confidence = getattr(poi, "geocode_confidence", 1.0)
        if getattr(poi, "source", "") == "festival" and confidence < 0.65:
            warnings.append("Festival location is inferred; confirm venue before going.")
        if confidence <= 0.35:
            warnings.append("Location confidence is low.")
        if user_input and RoutingAgent._family_mode(user_input) and RoutingAgent._semantic_group(poi) == "nightlife":
            warnings.append("May not be ideal for a family trip.")
        if user_input and RoutingAgent._violates_food_restrictions(poi, user_input.food_restrictions):
            warnings.append("May conflict with food restrictions.")
        if user_input:
            profile = build_preference_profile(user_input)
            if profile.avoid_terms and matches_any(text, profile.avoid_terms):
                warnings.append("This matches something you asked to avoid; it stayed only because the route fit was strong.")
        if getattr(poi, "opening_hours_confidence", 0.0) <= 0.0:
            warnings.append("Opening hours are not confirmed; verify before going.")
        elif arrival is not None and departure is not None:
            status = visit_opening_status(poi, arrival, departure, allow_wait=False)
            if status.source == "explicit" and not status.fits:
                warnings.append("This stop may be outside listed opening hours; verify before going.")
        return warnings
