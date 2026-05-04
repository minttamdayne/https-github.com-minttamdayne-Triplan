"""Run the 5-agent travel-planning pipeline.

Examples:
    python scripts/test_agents.py --days 5 --interests "street food, park, museum" --budget 2 --group family --pace relaxed --children true --start-time 10:00
    python scripts/test_agents.py --input examples/family_hcm.json --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.clustering_agent import ClusteringAgent
from src.agents.festival_agent import FestivalAgent
from src.agents.routing_agent import RoutingAgent
from src.agents.scoring_agent import ScoringAgent
from src.agents.semantic_agent import SemanticAgent
from src.config import settings
from src.itinerary_revisions import AlternativeSuggestion, suggest_alternatives
from src.memory.store import MemoryStore
from src.models.itinerary import Itinerary
from src.models.poi import POI
from src.models.user_input import UserInput
from src.natural_language import parse_travel_prompt_llm
from src.opening_hours import visit_opening_status
from src.preferences import apply_poi_semantic_overrides, category_for_poi
from src.tools.knowledge_graph import KnowledgeGraphClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_agents")


@dataclass
class PipelineResults:
    user_input: UserInput
    pois: list[POI]
    raw_festivals: list[dict]
    enriched_festivals: list[POI]
    candidates: list[POI]
    matched: list[POI]
    scored: list[POI]
    daily_clusters: dict[int, list[POI]]
    itinerary: Itinerary


def load_pois() -> list[POI]:
    poi_path = settings.data_dir / "hcm_poi.json"
    with open(poi_path, encoding="utf-8") as f:
        raw = json.load(f)
    pois: list[POI] = []
    for entry in raw:
        try:
            pois.append(apply_poi_semantic_overrides(POI(**entry)))
        except Exception as exc:
            logger.debug("Skipping POI: %s", exc)
    return pois


def load_festivals() -> list[dict]:
    fest_path = settings.data_dir / "HCM_FEST.json"
    if not fest_path.exists():
        return []
    with open(fest_path, encoding="utf-8") as f:
        return json.load(f)


async def run_pipeline(user_input: UserInput) -> PipelineResults:
    """Run the existing agent pipeline and return intermediate results."""
    memory = MemoryStore()
    kg = KnowledgeGraphClient()
    try:
        pois = load_pois()
        raw_festivals = load_festivals()

        fest_agent = FestivalAgent(memory=memory)
        enriched_festivals = await fest_agent.run(
            raw_festivals=raw_festivals,
            user_input=user_input,
        )

        candidates = pois + enriched_festivals

        sem_agent = SemanticAgent(kg=kg, memory=memory)
        matched = await sem_agent.run(candidates=candidates, user_input=user_input)

        score_agent = ScoringAgent(memory=memory)
        scored = await score_agent.run(candidates=matched, user_input=user_input)

        cluster_agent = ClusteringAgent(memory=memory)
        daily_clusters = await cluster_agent.run(candidates=scored, user_input=user_input)

        route_agent = RoutingAgent(memory=memory)
        itinerary = await route_agent.run(daily_clusters=daily_clusters, user_input=user_input)

        return PipelineResults(
            user_input=user_input,
            pois=pois,
            raw_festivals=raw_festivals,
            enriched_festivals=enriched_festivals,
            candidates=candidates,
            matched=matched,
            scored=scored,
            daily_clusters=daily_clusters,
            itinerary=itinerary,
        )
    finally:
        await kg.close()


def render_debug_report(results: PipelineResults) -> None:
    user_input = results.user_input
    print_header("DEBUG: USER INPUT")
    print(f"  Interests:     {user_input.interests}")
    print(f"  Trip:          {user_input.start_date} -> {user_input.end_date} ({user_input.num_days} days)")
    print(f"  Profile:       group={user_input.travel_group}, pace={user_input.pace}, mobility={user_input.mobility}, children={user_input.has_children}")
    print(f"  Customer:      gender={user_input.gender}, age={user_input.age or 'unknown'}")
    print(f"  Budget level:  {user_input.budget_level}")
    print(f"  Start:         {user_input.start_location} at {user_input.start_time}")
    print(f"  Food limits:   {user_input.food_restrictions or 'none'}")
    print(f"  Must have:     {user_input.must_have or 'none'}")
    print(f"  Avoid:         {user_input.avoid or 'none'}")
    print(f"  Preferences:   nightlife={user_input.nightlife_preference}, food={user_input.food_priority}, culture={user_input.culture_priority}, outdoor={user_input.outdoor_priority}")

    print_header("DEBUG: PIPELINE")
    print(f"  POIs loaded:              {len(results.pois)}")
    print(f"  Festivals loaded:         {len(results.raw_festivals)}")
    print(f"  Festivals enriched:       {len(results.enriched_festivals)}")
    low_conf = [p for p in results.enriched_festivals if p.geocode_confidence <= 0.35]
    print(f"  Low-confidence festivals: {len(low_conf)}")
    for poi in low_conf[:8]:
        print(f"    - {poi.name[:52]} confidence={poi.geocode_confidence:.2f} method={poi.geocode_method}")
    print(f"  Candidate pool:           {len(results.candidates)}")
    print(f"  Semantic matches:         {len(results.matched)}")
    print(f"  Scored candidates:        {len(results.scored)}")

    print_poi_table(results.scored, limit=15, title="Top 15 scored candidates")

    print_header("DEBUG: CLUSTERS")
    for day_idx in sorted(results.daily_clusters):
        cluster = results.daily_clusters[day_idx]
        counts = Counter(category_group(p) for p in cluster)
        festivals = sum(1 for p in cluster if p.source == "festival")
        avg_score = sum(p.composite_score for p in cluster) / len(cluster) if cluster else 0.0
        print(f"  Day {day_idx + 1}: {len(cluster):>3} places | avg_score={avg_score:.3f} | festivals={festivals} | mix={dict(counts)}")

    print_header("DEBUG: FINAL ITINERARY")
    print(f"  Total score: {results.itinerary.total_score:.3f}")
    print(f"  Total stops: {sum(len(day.stops) for day in results.itinerary.days)}")


def render_user_itinerary(itinerary: Itinerary) -> None:
    print_header("YOUR TRAVEL PLAN")
    print(f"  City: {itinerary.city.upper()} | Total score: {itinerary.total_score:.2f}")

    for day in itinerary.days:
        theme = day_theme(day)
        density = density_label(day)
        family = family_suitability(day)
        area = main_area(day)

        print(f"\nNgày {day.day_number}: {theme}" + (f" quanh {area}" if area else ""))
        print("Summary:")
        print(f"- Chủ đề ngày: {theme}")
        print(f"- Tổng số điểm: {len(day.stops)}")
        print(f"- Tổng thời gian di chuyển ước tính: {day.total_travel_minutes:.0f} phút")
        print(f"- Mức độ dày: {density}")
        print(f"- Phù hợp gia đình: {family}")

        print("\nTimeline:")
        for stop in day.stops:
            arrival = stop.arrival_time.strftime("%H:%M") if stop.arrival_time else "?"
            departure = stop.departure_time.strftime("%H:%M") if stop.departure_time else "?"
            poi = stop.poi
            hours_label = "unknown"
            if stop.arrival_time and stop.departure_time:
                arrival_dt = datetime.combine(day.date, stop.arrival_time)
                departure_dt = datetime.combine(day.date, stop.departure_time)
                if departure_dt <= arrival_dt:
                    departure_dt += timedelta(days=1)
                status = visit_opening_status(
                    poi,
                    arrival_dt,
                    departure_dt,
                    allow_wait=False,
                )
                hours_label = "confirmed" if status.source == "explicit" and status.fits else status.source
            print(f"{arrival} - {departure}  {poi.name}")
            print(f"  Loại: {display_type(poi)} | Ở lại: {stop.visit_minutes} phút | Di chuyển: {stop.travel_minutes_from_prev:.0f} phút | Giờ mở cửa: {hours_label}")
            if stop.matched_interests:
                print(f"  Khớp sở thích: {', '.join(stop.matched_interests)}")
            if stop.reason:
                print(f"  Lý do chọn: {stop.reason}")
            if stop.notes:
                print(f"  Ghi chú: {stop.notes}")
            if hours_label == "explicit":
                print("  Lưu ý: Stop is outside listed opening hours.")
            for warning in stop.warnings:
                print(f"  Lưu ý: {warning}")
        print()


async def build_user_input_from_args(args: argparse.Namespace) -> UserInput:
    if args.input:
        return user_input_from_json(Path(args.input))

    if args.prompt:
        fallback_start = date.fromisoformat(args.start_date)
        parsed = await parse_travel_prompt_llm(
            args.prompt,
            fallback_start_date=fallback_start,
            fallback_location=tuple(args.start_location),
        )
        return merge_prompt_overrides(parsed, args)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date) if args.end_date else start + timedelta(days=args.days - 1)
    return UserInput(
        interests=parse_interests(args.interests),
        start_date=start,
        end_date=end,
        start_location=tuple(args.start_location),
        budget_level=args.budget,
        daily_hours=args.daily_hours,
        max_places_per_day=args.max_places_per_day,
        start_time=args.start_time,
        travel_group=args.group,
        pace=args.pace,
        mobility=args.mobility,
        has_children=parse_bool(args.children),
        food_restrictions=parse_interests(args.food_restrictions),
        must_have=parse_interests(args.must_have),
        avoid=parse_interests(args.avoid),
        nightlife_preference=args.nightlife,
        food_priority=args.food_priority,
        culture_priority=args.culture_priority,
        outdoor_priority=args.outdoor_priority,
    )


def user_input_from_json(path: Path) -> UserInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start = payload.pop("trip_start", payload.get("start_date", None))
    end = payload.pop("trip_end", payload.get("end_date", None))
    if start is not None:
        payload["start_date"] = start
    if end is not None:
        payload["end_date"] = end
    if "start_location" in payload:
        payload["start_location"] = tuple(payload["start_location"])
    return UserInput(**payload)


def merge_prompt_overrides(parsed: UserInput, args: argparse.Namespace) -> UserInput:
    """Apply only CLI flags the user explicitly supplied on top of --prompt."""
    raw = set(sys.argv[1:])
    updates: dict[str, Any] = {}
    flag_map = {
        "--budget": ("budget_level", args.budget),
        "--daily-hours": ("daily_hours", args.daily_hours),
        "--max-places-per-day": ("max_places_per_day", args.max_places_per_day),
        "--group": ("travel_group", args.group),
        "--pace": ("pace", args.pace),
        "--mobility": ("mobility", args.mobility),
        "--children": ("has_children", parse_bool(args.children)),
        "--food-restrictions": ("food_restrictions", parse_interests(args.food_restrictions)),
        "--must-have": ("must_have", parse_interests(args.must_have)),
        "--avoid": ("avoid", parse_interests(args.avoid)),
        "--nightlife": ("nightlife_preference", args.nightlife),
        "--food-priority": ("food_priority", args.food_priority),
        "--culture-priority": ("culture_priority", args.culture_priority),
        "--outdoor-priority": ("outdoor_priority", args.outdoor_priority),
        "--start-time": ("start_time", args.start_time),
    }
    for flag, (field, value) in flag_map.items():
        if flag in raw:
            updates[field] = value
    if "--group" in raw:
        updates["group_type"] = args.group
    if "--food-priority" in raw:
        updates["food_preference"] = args.food_priority
    if "--culture-priority" in raw:
        updates["culture_preference"] = args.culture_priority
    if "--outdoor-priority" in raw:
        updates["outdoor_preference"] = args.outdoor_priority
    if "--interests" in raw:
        updates["interests"] = parse_interests(args.interests)
    if "--days" in raw and "--end-date" not in raw:
        updates["end_date"] = parsed.start_date + timedelta(days=args.days - 1)
    if "--end-date" in raw:
        updates["end_date"] = date.fromisoformat(args.end_date)
    return parsed.model_copy(update=updates)


def render_alternative_suggestions(
    results: PipelineResults,
    day_number: int,
    stop_order: int,
    query: str,
    limit: int,
) -> None:
    day = next((item for item in results.itinerary.days if item.day_number == day_number), None)
    if day is None:
        print(f"\nNo day {day_number} in itinerary.")
        return
    stop = next((item for item in day.stops if item.order == stop_order), None)
    if stop is None:
        print(f"\nNo stop {stop_order} on day {day_number}.")
        return

    used_ids = {s.poi.id for d in results.itinerary.days for s in d.stops}
    scored_ids = {poi.id for poi in results.scored}
    candidate_pool = list(results.scored) + [poi for poi in results.candidates if poi.id not in scored_ids]
    suggestions = suggest_alternatives(
        target_stop=stop,
        candidates=candidate_pool,
        user_input=results.user_input,
        query=query,
        used_poi_ids=used_ids,
        limit=limit,
    )
    render_suggestions(stop, suggestions, query)


def render_suggestions(stop, suggestions: list[AlternativeSuggestion], query: str) -> None:
    print_header("REPLACEMENT SUGGESTIONS")
    target = stop.poi
    print(f"  Replace: {target.name} ({pretty_type(target.primary_type)})")
    if query:
        print(f"  Looking for: {query}")
    if not suggestions:
        print("  No close matching alternatives found in the current candidate pool.")
        return
    for idx, item in enumerate(suggestions, start=1):
        poi = item.poi
        print(f"\n  {idx}. {poi.name}")
        print(f"     Type: {display_type(poi)} | Distance from original: {item.distance_km:.1f} km | Score: {item.score:.3f}")
        print(f"     Why: {item.reason}")
        if poi.address:
            print(f"     Address: {poi.address}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Triplan travel assistant pipeline.",
        epilog=(
            'Examples:\n'
            '  python scripts/test_agents.py --prompt "Tôi muốn đi chill, ít khách du lịch, có view đẹp"\n'
            '  python scripts/test_agents.py --prompt "Tôi lần đầu đến Sài Gòn, muốn ăn local và đi các điểm biểu tượng"\n'
            '  python scripts/test_agents.py --prompt "Tôi đi với gia đình, có trẻ nhỏ, không muốn đi quá nhiều"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", help="Path to a JSON request file.")
    parser.add_argument("--prompt", help="Casual natural-language travel request, e.g. 'Tôi là nữ 22 tuổi muốn ở lại TPHCM 3 ngày từ 4/5/2026 tới 7/5/2026'.")
    parser.add_argument("--interests", default="Vietnamese food, park, art gallery, street food, family friendly")
    parser.add_argument("--start-date", default="2026-03-26")
    parser.add_argument("--end-date")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--start-location", nargs=2, type=float, default=[10.7769, 106.7009])
    parser.add_argument("--daily-hours", type=float, default=10.0)
    parser.add_argument("--max-places-per-day", type=int, default=8)
    parser.add_argument("--group", choices=["solo", "couple", "family", "friends"], default="family")
    parser.add_argument("--pace", choices=["relaxed", "balanced", "intense"], default="balanced")
    parser.add_argument("--mobility", choices=["normal", "limited"], default="normal")
    parser.add_argument("--children", default="false")
    parser.add_argument("--food-restrictions", default="")
    parser.add_argument("--must-have", default="", help="Comma-separated explicit must-have preferences.")
    parser.add_argument("--avoid", default="", help="Comma-separated terms/categories to avoid.")
    parser.add_argument("--nightlife", choices=["auto", "avoid", "neutral", "like"], default="auto")
    parser.add_argument("--food-priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--culture-priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--outdoor-priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--start-time", default="09:00")
    parser.add_argument("--debug", action="store_true", help="Also print intermediate agent diagnostics.")
    parser.add_argument("--alternatives-day", type=int, help="Suggest replacements for this itinerary day number.")
    parser.add_argument("--alternatives-stop", type=int, help="Suggest replacements for this stop order within the day.")
    parser.add_argument("--replace-with", default="", help="Desired replacement vibe/query, e.g. 'pho', 'banh mi', 'coffee'. Empty means same category nearby.")
    parser.add_argument("--alternatives-limit", type=int, default=5)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    user_input = await build_user_input_from_args(args)
    print_header("PARSED INTENT")
    print(json.dumps(user_input.model_dump(mode="json"), ensure_ascii=False, indent=2))
    results = await run_pipeline(user_input)
    render_user_itinerary(results.itinerary)
    if args.alternatives_day and args.alternatives_stop:
        render_alternative_suggestions(results, args.alternatives_day, args.alternatives_stop, args.replace_with, args.alternatives_limit)
    if args.debug:
        render_debug_report(results)


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_poi_table(pois: list[POI], limit: int = 10, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    print(f"  {'Name':<35} {'Type':<20} {'Score':>6} {'IntFit':>6} {'Qual':>6} {'Budg':>6} {'Prox':>6}")
    print(f"  {'-'*35} {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for poi in pois[:limit]:
        src_tag = " [F]" if poi.source == "festival" else ""
        print(
            f"  {(poi.name[:32] + src_tag):<35} {poi.primary_type:<20} "
            f"{poi.composite_score:>6.3f} {poi.interest_fit:>6.3f} {poi.quality_score:>6.3f} "
            f"{poi.budget_fit:>6.3f} {poi.proximity_score:>6.3f}"
        )


def parse_interests(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def category_group(poi: POI) -> str:
    return category_for_poi(poi)


def pretty_type(primary_type: str) -> str:
    return primary_type.replace("_", " ").title() if primary_type else "Place"


def display_type(poi: POI) -> str:
    labels = {
        "food": "Food",
        "coffee": "Cafe",
        "culture": "Culture",
        "outdoor": "Outdoor",
        "shopping": "Shopping",
        "nightlife": "Nightlife",
        "festival": "Festival",
    }
    group = category_group(poi)
    return labels.get(group, pretty_type(poi.primary_type))


def day_theme(day) -> str:
    counts = Counter(category_group(stop.poi) for stop in day.stops)
    if not counts:
        return "Ngày nhẹ"
    labels = {
        "food": "Ẩm thực địa phương",
        "coffee": "Cafe và nghỉ nhẹ",
        "culture": "Văn hóa và tham quan",
        "nightlife": "Buổi tối sôi động",
        "outdoor": "Không gian xanh",
        "festival": "Lễ hội và sự kiện",
    }
    top = [labels.get(group, pretty_type(group)) for group, _ in counts.most_common(2)]
    return " + ".join(top)


def density_label(day) -> str:
    stops = len(day.stops)
    total = day.total_time_minutes
    if stops <= 4 or total <= 360:
        return "Relaxed"
    if stops >= 7 or total >= 540:
        return "Intense"
    return "Balanced"


def family_suitability(day) -> str:
    if not day.stops:
        return "Trung bình"
    nightlife = sum(1 for stop in day.stops if category_group(stop.poi) == "nightlife")
    familyish = sum(1 for stop in day.stops if category_group(stop.poi) in {"culture", "outdoor", "coffee", "food", "festival"})
    if nightlife:
        return "Thấp" if nightlife > 1 else "Trung bình"
    return "Cao" if familyish / len(day.stops) >= 0.7 else "Trung bình"


def main_area(day) -> str:
    districts = [stop.poi.district for stop in day.stops if stop.poi.district]
    if districts:
        return Counter(districts).most_common(1)[0][0]
    return ""


if __name__ == "__main__":
    asyncio.run(main())
