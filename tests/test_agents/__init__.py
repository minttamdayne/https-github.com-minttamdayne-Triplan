"""
display_patch_v3.py  —  Drop vào scripts/test_agents.py

Fixes
─────
1. Travel=0, Visit=0  →  tính đúng từ stops list
2. Normalised score   →  hiển thị (proposal #4)
3. Cleaner per-stop format với travel thực tế
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.itinerary import DayPlan, Itinerary


def print_day_plan(day: "DayPlan") -> None:
    total_travel = round(sum(s.travel_minutes_from_prev for s in day.stops))
    total_visit  = round(sum(s.visit_minutes            for s in day.stops))
    n_stops      = len(day.stops)
    bar = "─" * 65

    print(f"\n  ┌─── Day {day.day_number} ({day.date}) {bar[:47]}┐")
    print(
        f"  │  Stops: {n_stops}"
        f"  |  Travel: {total_travel} min"
        f"  |  Visit: {total_visit} min"
        f"  |  Score: {day.total_score:.3f}"
    )
    print(f"  ├{bar}┤")
    for stop in day.stops:
        is_festival = getattr(stop.poi, "source", "") == "festival"
        tag  = " 🎪" if is_festival else "   "
        name = stop.poi.name[:42]
        arr  = stop.arrival_time.strftime("%H:%M")
        dep  = stop.departure_time.strftime("%H:%M")
        trav = round(stop.travel_minutes_from_prev)
        visit = stop.visit_minutes
        print(
            f"  │  {stop.order:>2}. [{arr}-{dep}] {name:<44}{tag}"
            f"  +{trav}m travel, {visit}m visit"
        )
    print(f"  └{bar}┘")


def print_itinerary_summary(itinerary: "Itinerary") -> None:
    """Print full itinerary with normalised score (proposal #4)."""
    total_stops    = sum(len(d.stops) for d in itinerary.days)
    n_days         = len(itinerary.days)
    expected_total = n_days * 8           # assume 8 stops/day target
    fill_rate      = total_stops / max(expected_total, 1)

    # Normalised score: quality × fill rate
    max_score      = total_stops * 1.0    # rough ceiling (score ~1.0/stop)
    quality_rate   = itinerary.total_score / max(max_score, 0.001)
    normalised     = round(min(quality_rate * fill_rate, 1.0) * 100, 1)

    print(f"\n  Total itinerary score: {itinerary.total_score:.3f}")
    print(f"  Normalised score:      {normalised}%  "
          f"(quality={quality_rate:.2f} × fill={fill_rate:.2f})")
    print(f"  Total stops:           {total_stops} / {expected_total} expected\n")

    for day in itinerary.days:
        print_day_plan(day)