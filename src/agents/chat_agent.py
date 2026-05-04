from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from src.agents.base import BaseAgent
from src.config import settings
from src.models.itinerary import Itinerary
from src.models.user_input import UserInput

_BUDGET_LABELS = {0: "Free", 1: "Budget", 2: "Moderate", 3: "Expensive", 4: "Luxury"}

_SYSTEM_TEMPLATE = """\
You are Triplan, a friendly and knowledgeable AI travel assistant. \
You have just planned a trip itinerary for a traveler and have their full plan below. \
Help them understand their trip, answer questions about specific places and timings, \
suggest tips, and provide practical travel advice. Be conversational and specific.

== TRIP OVERVIEW ==
City: {city}
Dates: {start_date} → {end_date} ({num_days} days)
Interests: {interests}
Budget: {budget_label}

== FULL ITINERARY ==
{itinerary_text}

If asked about something not in the itinerary, use your general travel knowledge to help. \
Keep answers concise unless the user asks for more detail."""


class ChatAgent(BaseAgent):
    """Conversational agent that discusses the completed trip itinerary with the user.

    Two modes:
    - chat()        → returns the full reply as a string (non-streaming)
    - chat_stream() → async-yields text tokens as they arrive (SSE-friendly)
    """

    name: str = "chat"

    async def _execute(self, **kwargs: Any) -> Any:
        """Not used directly; call chat() or chat_stream() instead."""

    # ── Public interface ──

    async def chat(
        self,
        messages: list[dict],
        itinerary: Itinerary,
        user_input: UserInput,
    ) -> str:
        """Single-turn non-streaming chat. Returns the full assistant reply."""
        full_messages = self._build_messages(messages, itinerary, user_input)
        client = self._make_llm_client()
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=full_messages,  # type: ignore[arg-type]
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict],
        itinerary: Itinerary,
        user_input: UserInput,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat. Yields text token chunks as they arrive."""
        full_messages = self._build_messages(messages, itinerary, user_input)
        client = self._make_llm_client()
        stream = await client.chat.completions.create(  # type: ignore[call-overload]
            model=settings.chat_model,
            messages=full_messages,  # type: ignore[arg-type]
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Helpers ──

    def _build_messages(
        self,
        messages: list[dict],
        itinerary: Itinerary,
        user_input: UserInput,
    ) -> list[dict]:
        system_prompt = _SYSTEM_TEMPLATE.format(
            city=itinerary.city.upper(),
            start_date=user_input.start_date,
            end_date=user_input.end_date,
            num_days=user_input.num_days,
            interests=", ".join(user_input.interests),
            budget_label=_BUDGET_LABELS.get(user_input.budget_level, "Moderate"),
            itinerary_text=self._format_itinerary(itinerary),
        )
        return [{"role": "system", "content": system_prompt}] + messages

    @staticmethod
    def _format_itinerary(itinerary: Itinerary) -> str:
        lines: list[str] = []
        for day in itinerary.days:
            lines.append(f"\nDay {day.day_number} — {day.date}")
            for stop in day.stops:
                p = stop.poi
                arrival = stop.arrival_time.strftime("%H:%M") if stop.arrival_time else "?"
                departure = stop.departure_time.strftime("%H:%M") if stop.departure_time else "?"
                notes = f"  [{stop.notes}]" if stop.notes else ""
                lines.append(
                    f"  {stop.order}. {p.name}"
                    f" ({arrival}–{departure}, ~{stop.visit_minutes}min)"
                    f" | {p.primary_type}{notes}"
                )
                if p.address:
                    parts = filter(None, [p.address, p.district, p.province])
                    lines.append(f"     {', '.join(parts)}")
                if p.price_range and p.price_range != "Free":
                    lines.append(f"     Price: {p.price_range}")
                if stop.reason:
                    lines.append(f"     Why: {stop.reason}")
                if stop.matched_interests:
                    lines.append(f"     Matches: {', '.join(stop.matched_interests)}")
                for warning in stop.warnings:
                    lines.append(f"     Note: {warning}")
            lines.append(
                f"  Summary: {int(day.total_visit_minutes)}min visiting"
                f" + {int(day.total_travel_minutes)}min travel"
            )
        return "\n".join(lines)
