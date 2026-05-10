from __future__ import annotations

from datetime import date, timedelta
import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from src.config import settings
from src.models.user_input import UserInput
from src.preferences import normalize_text


HCM_CENTER = (10.7769, 106.7009)
logger = logging.getLogger(__name__)


INTENT_SYSTEM_PROMPT = """
You extract structured travel-planning intent from Vietnamese or English text.
Return ONLY one strict JSON object. Do not include markdown.

Infer soft preferences even when exact place categories are not mentioned.
Use concise English canonical tokens where possible.

Schema:
{
  "themes": ["food", "coffee", "culture", "outdoor", "nightlife", "shopping", "iconic", "local"],
  "interests": ["local food", "scenic cafe", "rooftop bar"],
  "vibes": ["chill", "hidden_gem", "scenic_view", "local", "less_touristy", "first_timer", "family_friendly"],
  "negative_preferences": ["crowded", "tourist_trap", "too_many_stops"],
  "must_have": ["iconic landmarks"],
  "avoid": ["crowded places"],
  "time_preferences": {"morning": ["coffee"], "evening": ["light nightlife"]},
  "pace": "relaxed|balanced|intense",
  "group_type": "solo|couple|family|friends",
  "budget_level": 0-4,
  "nightlife_preference": "auto|avoid|neutral|like",
  "food_preference": "low|normal|high",
  "culture_preference": "low|normal|high",
  "outdoor_preference": "low|normal|high",
  "has_children": true|false,
  "mobility": "normal|limited",
  "food_restrictions": []
}

Guidance:
- "chill", "nhẹ", "không đi quá nhiều" => relaxed pace.
- "ít khách du lịch", "ít đông", "hidden gem" => less_touristy/hidden_gem vibes and avoid crowded/tourist_trap.
- "view đẹp", "đẹp để chụp ảnh" => scenic_view vibe; interests may include rooftop, riverside, scenic cafe.
- "local", "ăn local", "authentic" => local vibe; food_preference high.
- "lần đầu đến Sài Gòn" / first time => first_timer vibe; include iconic, local food, coffee, one light evening option unless nightlife is rejected.
- "nightlife nhẹ" => nightlife_preference like, vibe relaxed, interests include light nightlife/lounge/rooftop.
"""


async def parse_travel_prompt_llm(
    prompt: str,
    *,
    fallback_start_date: date | None = None,
    fallback_location: tuple[float, float] = HCM_CENTER,
) -> UserInput:
    """Parse free-form travel text with the configured LLM, falling back safely."""
    fallback = parse_travel_prompt(
        prompt,
        fallback_start_date=fallback_start_date,
        fallback_location=fallback_location,
    )
    client = _make_llm_client()
    user_payload = {
        "prompt": prompt,
        "fallback_start_date": str(fallback_start_date or date.today()),
        "fallback_trip_dates": {
            "start_date": str(fallback.start_date),
            "end_date": str(fallback.end_date),
        },
        "fallback_location": fallback_location,
    }
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=settings.llm_timeout,
        )
        raw = resp.choices[0].message.content or "{}"
        payload = _extract_json_object(raw)
        return _intent_payload_to_user_input(
            payload=payload,
            prompt=prompt,
            fallback=fallback,
            fallback_location=fallback_location,
        )
    except Exception as exc:
        logger.warning("LLM intent parsing failed; using fallback parser: %s", exc)
        return fallback


def parse_travel_prompt(
    prompt: str,
    *,
    fallback_start_date: date | None = None,
    fallback_location: tuple[float, float] = HCM_CENTER,
) -> UserInput:
    """Parse a casual Vietnamese/English travel request into UserInput.

    This is intentionally deterministic and local. It is a bridge for CLI/API
    UX; a future LLM parser can replace it while keeping the same output model.
    """
    normalized = normalize_text(prompt)
    start_date, end_date = _parse_dates(prompt.lower(), fallback_start_date or date.today())
    explicit_days = _parse_num_days(normalized)
    if explicit_days and not _has_date_range(normalized):
        end_date = start_date + timedelta(days=max(1, explicit_days) - 1)

    interests = _parse_interests(normalized)
    themes = _parse_themes(normalized, interests)
    vibes = _parse_vibes(normalized)
    negative_preferences = _parse_negative_preferences(normalized)
    must_have = _parse_must_have(normalized)
    avoid = _parse_avoid(normalized)
    gender = _parse_gender(normalized)
    age = _parse_age(normalized)
    food_priority = "high" if any(term in normalized for term in ("local", "an local", "food tour", "an ngon", "am thuc", "mon ngon", "foodie")) else "normal"
    culture_priority = "high" if any(term in normalized for term in ("lan dau", "lần đầu", "diem bieu tuong", "biểu tượng", "bao tang", "museum", "lich su", "van hoa")) else "normal"
    outdoor_priority = "high" if any(term in normalized for term in ("view dep", "view đẹp", "scenic", "riverside", "song", "cong vien", "park", "di dao", "ngoai troi")) else "normal"

    return UserInput(
        free_text=prompt,
        themes=themes,
        interests=interests or ["local food", "coffee", "museum", "park"],
        vibes=vibes,
        negative_preferences=negative_preferences,
        start_date=start_date,
        end_date=end_date,
        start_location=fallback_location,
        budget_level=_parse_budget(normalized),
        daily_hours=10.0,
        max_places_per_day=_parse_max_places(normalized),
        start_time=_parse_start_time(normalized),
        travel_group=_parse_group(normalized),
        group_type=_parse_group(normalized),
        pace=_parse_pace(normalized),
        mobility="limited" if any(term in normalized for term in ("di lai it", "it di bo", "mobility limited", "han che di chuyen")) else "normal",
        has_children=any(term in normalized for term in ("tre em", "tre nho", "trẻ nhỏ", "con nho", "kids", "children")),
        food_restrictions=_parse_food_restrictions(normalized),
        must_have=must_have,
        avoid=avoid or negative_preferences,
        time_preferences=_parse_time_preferences(normalized),
        nightlife_preference=_parse_nightlife(normalized),
        food_priority=food_priority,
        culture_priority=culture_priority,
        outdoor_priority=outdoor_priority,
        food_preference=food_priority,
        culture_preference=culture_priority,
        outdoor_preference=outdoor_priority,
        gender=gender,
        age=age,
    )


def _make_llm_client():
    from openai import AsyncOpenAI

    if settings.llm_provider == "ollama":
        return AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    return AsyncOpenAI(api_key=settings.openai_api_key)


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Intent parser returned non-object JSON")
    return parsed


def _intent_payload_to_user_input(
    *,
    payload: dict[str, Any],
    prompt: str,
    fallback: UserInput,
    fallback_location: tuple[float, float],
) -> UserInput:
    data = {
        "free_text": prompt,
        "themes": _clean_list(payload.get("themes")),
        "interests": _clean_list(payload.get("interests")),
        "vibes": _clean_list(payload.get("vibes")),
        "negative_preferences": _clean_list(payload.get("negative_preferences")),
        "must_have": _clean_list(payload.get("must_have")),
        "avoid": _clean_list(payload.get("avoid")),
        "time_preferences": payload.get("time_preferences") if isinstance(payload.get("time_preferences"), dict) else {},
        "pace": payload.get("pace") or fallback.pace,
        "travel_group": payload.get("travel_group") or payload.get("group_type") or fallback.travel_group,
        "group_type": payload.get("group_type") or payload.get("travel_group") or fallback.group_type,
        "budget_level": payload.get("budget_level", fallback.budget_level),
        "nightlife_preference": payload.get("nightlife_preference") or fallback.nightlife_preference,
        "food_priority": payload.get("food_priority") or payload.get("food_preference") or fallback.food_priority,
        "culture_priority": payload.get("culture_priority") or payload.get("culture_preference") or fallback.culture_priority,
        "outdoor_priority": payload.get("outdoor_priority") or payload.get("outdoor_preference") or fallback.outdoor_priority,
        "food_preference": payload.get("food_preference") or payload.get("food_priority") or fallback.food_preference,
        "culture_preference": payload.get("culture_preference") or payload.get("culture_priority") or fallback.culture_preference,
        "outdoor_preference": payload.get("outdoor_preference") or payload.get("outdoor_priority") or fallback.outdoor_preference,
        "mobility": payload.get("mobility") or fallback.mobility,
        "has_children": payload.get("has_children", fallback.has_children),
        "food_restrictions": _clean_list(payload.get("food_restrictions")) or fallback.food_restrictions,
        "start_date": fallback.start_date,
        "end_date": fallback.end_date,
        "start_location": fallback_location,
        "daily_hours": fallback.daily_hours,
        "max_places_per_day": fallback.max_places_per_day,
        "start_time": fallback.start_time,
        "gender": fallback.gender,
        "age": fallback.age,
    }
    if not data["interests"]:
        data["interests"] = _interests_from_intent(data["themes"], data["vibes"]) or fallback.interests
    if not data["themes"]:
        data["themes"] = _parse_themes(normalize_text(prompt), data["interests"])
    if not data["avoid"]:
        data["avoid"] = data["negative_preferences"] or fallback.avoid
    data = _coerce_intent_values(data, fallback)
    try:
        return UserInput(**data)
    except ValidationError as exc:
        logger.warning("LLM intent payload failed validation; using fallback parser: %s", exc)
        return fallback


def _coerce_intent_values(data: dict[str, Any], fallback: UserInput) -> dict[str, Any]:
    coerced = dict(data)
    enum_defaults = {
        "pace": ({"relaxed", "balanced", "intense"}, fallback.pace),
        "travel_group": ({"solo", "couple", "family", "friends"}, fallback.travel_group),
        "group_type": ({"solo", "couple", "family", "friends"}, fallback.group_type),
        "mobility": ({"normal", "limited"}, fallback.mobility),
        "nightlife_preference": ({"auto", "avoid", "neutral", "like"}, fallback.nightlife_preference),
        "food_priority": ({"low", "normal", "high"}, fallback.food_priority),
        "culture_priority": ({"low", "normal", "high"}, fallback.culture_priority),
        "outdoor_priority": ({"low", "normal", "high"}, fallback.outdoor_priority),
        "food_preference": ({"low", "normal", "high"}, fallback.food_preference),
        "culture_preference": ({"low", "normal", "high"}, fallback.culture_preference),
        "outdoor_preference": ({"low", "normal", "high"}, fallback.outdoor_preference),
    }
    aliases = {
        "medium": "normal",
        "moderate": "normal",
        "yes": "like",
        "no": "avoid",
        "light": "like",
        "slow": "relaxed",
        "easy": "relaxed",
        "fast": "intense",
        "busy": "intense",
        "group": "friends",
    }
    for field, (allowed, default) in enum_defaults.items():
        value = str(coerced.get(field, default)).strip().lower()
        value = aliases.get(value, value)
        coerced[field] = value if value in allowed else default
    try:
        coerced["budget_level"] = max(0, min(4, int(coerced.get("budget_level", fallback.budget_level))))
    except (TypeError, ValueError):
        coerced["budget_level"] = fallback.budget_level
    return coerced


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[,;]", value)
    elif isinstance(value, list):
        items = value
    else:
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _interests_from_intent(themes: list[str], vibes: list[str]) -> list[str]:
    interests: list[str] = []
    mapping = {
        "food": "local food",
        "local": "local food",
        "coffee": "coffee",
        "culture": "museum",
        "outdoor": "park",
        "nightlife": "rooftop bar",
        "iconic": "tourist attraction",
        "shopping": "market",
    }
    for term in [*themes, *vibes]:
        interest = mapping.get(term)
        if interest and interest not in interests:
            interests.append(interest)
    if "scenic_view" in vibes and "rooftop bar" not in interests:
        interests.append("rooftop bar")
    if "first_timer" in vibes:
        for interest in ("tourist attraction", "local food", "coffee"):
            if interest not in interests:
                interests.append(interest)
    return interests


def _parse_dates(text: str, fallback: date) -> tuple[date, date]:
    matches = re.findall(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
    parsed: list[date] = []
    current_year = fallback.year
    for day_s, month_s, year_s in matches[:2]:
        year = int(year_s) if year_s else current_year
        if year < 100:
            year += 2000
        try:
            parsed.append(date(year, int(month_s), int(day_s)))
        except ValueError:
            continue
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    if len(parsed) == 1:
        days = _parse_num_days(text) or 3
        return parsed[0], parsed[0] + timedelta(days=days - 1)
    days = _parse_num_days(text) or 3
    return fallback, fallback + timedelta(days=days - 1)


def _has_date_range(text: str) -> bool:
    return any(term in text for term in (" tu ", " toi ", " den ", "from", "to"))


def _parse_num_days(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(ngay|days?)\b", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_interests(text: str) -> list[str]:
    mapping = [
        ("banh mi", "banh mi"),
        ("pho", "pho"),
        ("com tam", "com tam"),
        ("street food", "street food"),
        ("an vat", "street food"),
        ("am thuc", "local food"),
        ("mon ngon", "local food"),
        ("local food", "local food"),
        ("coffee", "coffee"),
        ("cafe", "coffee"),
        ("ca phe", "coffee"),
        ("bao tang", "museum"),
        ("museum", "museum"),
        ("art", "art gallery"),
        ("gallery", "art gallery"),
        ("cong vien", "park"),
        ("park", "park"),
        ("cho dem", "night market"),
        ("night market", "night market"),
        ("bar", "bar"),
        ("rooftop", "rooftop bar"),
        ("shopping", "shopping"),
        ("mua sam", "shopping"),
        ("local", "local food"),
        ("hidden gem", "hidden gem"),
        ("view dep", "scenic view"),
        ("view đẹp", "scenic view"),
        ("scenic", "scenic view"),
        ("riverside", "riverside"),
        ("song", "riverside"),
        ("rooftop", "rooftop bar"),
        ("lan dau", "tourist attraction"),
        ("lần đầu", "tourist attraction"),
        ("bieu tuong", "tourist attraction"),
        ("biểu tượng", "tourist attraction"),
        ("nightlife nhe", "light nightlife"),
        ("nightlife nhẹ", "light nightlife"),
    ]
    interests: list[str] = []
    for needle, interest in mapping:
        if _contains_term(text, needle) and interest not in interests:
            interests.append(interest)
    return interests


def _parse_themes(text: str, interests: list[str]) -> list[str]:
    themes: list[str] = []
    mapping = {
        "food": ("food", "local food", "am thuc", "mon ngon", "an local", "restaurant", "street food"),
        "coffee": ("coffee", "cafe", "ca phe"),
        "culture": ("museum", "bao tang", "lich su", "van hoa", "art", "gallery"),
        "outdoor": ("park", "cong vien", "riverside", "song", "view dep", "scenic"),
        "nightlife": ("nightlife", "bar", "club", "rooftop", "lounge"),
        "shopping": ("shopping", "mua sam", "market", "cho"),
        "iconic": ("lan dau", "lần đầu", "bieu tuong", "biểu tượng", "landmark", "iconic"),
        "local": ("local", "authentic", "ban dia", "bản địa"),
    }
    haystack = " ".join([text, *interests])
    for theme, needles in mapping.items():
        if any(_contains_term(haystack, needle) for needle in needles):
            themes.append(theme)
    return themes


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _parse_vibes(text: str) -> list[str]:
    mapping = {
        "chill": ("chill", "nhe", "nhẹ", "thu thai", "thư thái", "relaxed"),
        "less_touristy": ("it khach du lich", "ít khách du lịch", "it dong", "ít đông", "not touristy"),
        "hidden_gem": ("hidden gem", "it nguoi biet", "ít người biết", "local hidden"),
        "scenic_view": ("view dep", "view đẹp", "scenic", "photo", "chup anh", "chụp ảnh", "rooftop", "riverside"),
        "local": ("local", "authentic", "ban dia", "bản địa"),
        "first_timer": ("lan dau", "lần đầu", "first time", "first_timer"),
        "family_friendly": ("gia dinh", "family", "tre em", "con nho", "kids", "children"),
    }
    return [vibe for vibe, needles in mapping.items() if any(needle in text for needle in needles)]


def _parse_negative_preferences(text: str) -> list[str]:
    negatives: list[str] = []
    mapping = {
        "crowded": ("dong", "đông", "crowded"),
        "tourist_trap": ("khach du lich", "khách du lịch", "tourist trap", "touristy"),
        "too_many_stops": ("khong qua nhieu", "không quá nhiều", "it di", "ít đi", "di qua nhieu", "đi quá nhiều"),
        "bars": ("khong bar", "không bar", "avoid bar"),
    }
    for item, needles in mapping.items():
        if any(needle in text for needle in needles):
            negatives.append(item)
    return negatives


def _parse_time_preferences(text: str) -> dict[str, list[str]]:
    prefs: dict[str, list[str]] = {}
    if any(term in text for term in ("sang", "morning", "buoi sang", "buổi sáng")):
        prefs.setdefault("morning", []).append("light start")
    if any(term in text for term in ("toi", "tối", "evening", "nightlife", "bar", "rooftop")):
        prefs.setdefault("evening", []).append("nightlife" if "nightlife" in text else "evening activity")
    return prefs


def _parse_must_have(text: str) -> list[str]:
    terms = []
    for marker in ("muon", "thich", "must have", "uu tien"):
        if marker in text:
            terms.extend(_parse_interests(text))
            break
    return terms[:5]


def _parse_avoid(text: str) -> list[str]:
    avoid: list[str] = []
    patterns = (r"(?:khong muon|khong thich|tranh|avoid)\s+([^,.]+)",)
    for pattern in patterns:
        for match in re.findall(pattern, text):
            avoid.extend(_parse_interests(match) or [match.strip()])
    return avoid[:5]


def _parse_gender(text: str) -> str:
    if re.search(r"\b(nu|female|girl|woman)\b", text):
        return "female"
    if re.search(r"\b(nam|male|boy|man)\b", text):
        return "male"
    return "unknown"


def _parse_age(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(tuoi|years? old)\b", text)
    return int(match.group(1)) if match else None


def _parse_budget(text: str) -> int:
    if any(term in text for term in ("tiet kiem", "giá rẻ", "gia re", "cheap", "budget")) or re.search(r"\bre\b", text):
        return 1
    if any(term in text for term in ("sang", "luxury", "cao cap", "fine dining")):
        return 4
    if any(term in text for term in ("mac", "expensive")):
        return 3
    return 2


def _parse_max_places(text: str) -> int:
    if any(term in text for term in ("chill", "nhe", "nhẹ", "thu thai", "thư thái", "relaxed", "khong muon di qua nhieu", "không muốn đi quá nhiều")):
        return 5
    if any(term in text for term in ("nhieu noi", "di duoc nhieu", "intense")):
        return 9
    return 7


def _parse_start_time(text: str) -> str:
    match = re.search(r"(?:bat dau|start)\s*(?:luc|at)?\s*(\d{1,2})(?::|h)?(\d{2})?", text)
    if not match:
        return "09:00"
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return f"{hour:02d}:{minute:02d}"


def _parse_group(text: str) -> str:
    if any(term in text for term in ("gia dinh", "gia đình", "family", "tre em", "tre nho", "trẻ nhỏ", "con nho")):
        return "family"
    if any(term in text for term in ("nguoi yeu", "couple", "ban trai", "ban gai")):
        return "couple"
    if any(term in text for term in ("ban be", "friends", "nhom ban")):
        return "friends"
    return "solo"


def _parse_pace(text: str) -> str:
    if any(term in text for term in ("chill", "nhe", "nhẹ", "thu thai", "thư thái", "relaxed", "khong qua day", "khong muon di qua nhieu", "không muốn đi quá nhiều")):
        return "relaxed"
    if any(term in text for term in ("nhieu noi", "di duoc nhieu", "intense", "day dac")):
        return "intense"
    return "balanced"


def _parse_nightlife(text: str) -> str:
    if any(term in text for term in ("khong bar", "khong nightlife", "tranh bar", "avoid nightlife")):
        return "avoid"
    if any(term in text for term in ("bar", "club", "nightlife", "rooftop", "cho dem", "night market")):
        return "like"
    return "auto"


def _parse_food_restrictions(text: str) -> list[str]:
    restrictions = []
    for term in ("vegetarian", "vegan", "halal", "no seafood", "seafood allergy"):
        if term in text:
            restrictions.append(term)
    if "an chay" in text:
        restrictions.append("vegetarian")
    if "di ung hai san" in text:
        restrictions.append("seafood allergy")
    return restrictions
