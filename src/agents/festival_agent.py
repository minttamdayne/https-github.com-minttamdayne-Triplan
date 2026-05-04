from __future__ import annotations
import json
import re
import asyncio
import hashlib
from datetime import date
from typing import Any

from src.agents.base import BaseAgent
from src.models.poi import POI, OpeningHours
from src.models.user_input import UserInput
from src.models.festival import RawFestival
from src.preferences import apply_poi_semantic_overrides
from src.tools.geocoder import geocode_address
from src.tools.web_search import search_web
from src.config import settings

CITY_FALLBACK_LOCATIONS: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "hcm": [
        ("Nguyen Hue Walking Street", (10.7753, 106.7039)),
        ("Saigon Opera House", (10.7766, 106.7031)),
        ("Tao Dan Park", (10.7763, 106.6920)),
        ("Ho Chi Minh City Museum", (10.7761, 106.6993)),
        ("September 23 Park", (10.7696, 106.6926)),
    ],
    "ho chi minh": [
        ("Nguyen Hue Walking Street", (10.7753, 106.7039)),
        ("Saigon Opera House", (10.7766, 106.7031)),
        ("Tao Dan Park", (10.7763, 106.6920)),
        ("Ho Chi Minh City Museum", (10.7761, 106.6993)),
        ("September 23 Park", (10.7696, 106.6926)),
    ],
}
class FestivalAgent(BaseAgent):
    """Agent 1 — Festival Enrichment (High-Performance Edition)."""

    name: str = "festival"

    async def _execute(self, **kwargs: Any) -> list[POI]:
        raw_festivals: list[dict] = kwargs["raw_festivals"]
        user_input: UserInput = kwargs["user_input"]
        year: int = user_input.start_date.year

        # Bước 1+2: Phân tích thời gian và lọc các lễ hội trùng ngày chuyến đi
        temporally_valid = []
        for raw in raw_festivals:
            fest = RawFestival(**raw)
            start, end = self._parse_time(fest.time, year)
            if start and self._overlaps(start, end, user_input.start_date, user_input.end_date):
                temporally_valid.append((fest, start, end))

        self.logger.info(f"Temporal filter: {len(temporally_valid)}/{len(raw_festivals)} festivals overlap.")

        # Bước 3-6: Làm giàu dữ liệu SONG SONG (Tối ưu hóa I/O cho HPC)
        # Sử dụng asyncio.gather để gọi API hàng loạt thay vì đợi từng cái một
        tasks = [
            self._enrich_festival(fest, start, end, user_input)
            for fest, start, end in temporally_valid
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Loại bỏ các kết quả None (do lỗi địa chỉ hoặc API)
        enriched_pois = [poi for poi in results if poi is not None]

        self.memory.set("enriched_festivals", enriched_pois)
        return enriched_pois

    # ── Xử lý Regex Parsing (Đã sửa lỗi IndexError) ──
    def _parse_time(self, time_str: str, default_year: int) -> tuple[date | None, date | None]:
        if not time_str.strip(): return None, None
        ts = time_str.strip()

        patterns = [
            # 1. Định dạng: 15/10/2025 - 28/2/2026 (6 nhóm)
            (r"(\d+)/(\d+)/(\d{4})\s*-\s*(\d+)/(\d+)/(\d{4})", 
             lambda m: (date(int(m[2]), int(m[1]), int(m[0])), date(int(m[5]), int(m[4]), int(m[3])))),

            # 2. Định dạng: 26 - 29/3 (3 nhóm: d1, d2, month) - ĐÃ SỬA CHỈ SỐ
            (r"(\d+)\s*-\s*(\d+)/(\d+)$", 
             lambda m: (date(default_year, int(m[2]), int(m[0])), date(default_year, int(m[2]), int(m[1])))),

            # 3. Định dạng: 6/11 (2 nhóm: day, month)
            (r"^(\d+)/(\d+)$", 
             lambda m: (date(default_year, int(m[1]), int(m[0])), date(default_year, int(m[1]), int(m[0])))),

            # 4. Định dạng: 24/2 - 31/3 (4 nhóm: d1, m1, d2, m2)
            (r"(\d+)/(\d+)\s*-\s*(\d+)/(\d+)$", 
             lambda m: (date(default_year, int(m[1]), int(m[0])), date(default_year, int(m[3]), int(m[2]))))
        ]

        for pattern, factory in patterns:
            m = re.match(pattern, ts)
            if m:
                try: 
                    return factory(m.groups())
                except (ValueError, IndexError): 
                    continue
        
        return None, None

    @staticmethod
    def _overlaps(fest_start: date, fest_end: date, trip_start: date, trip_end: date) -> bool:
        return fest_start <= trip_end and fest_end >= trip_start

    # ── Làm giàu dữ liệu (Parallel Enrichment) ──
    async def _enrich_festival(self, fest: RawFestival, start_dt: date, end_dt: date, user_input: UserInput) -> POI | None:
        # Kiểm tra Cache trước
        cached = self.memory.cache_get("festivals", fest.name)
        if cached: return POI(**cached["value"])

        full_address = f"{fest.ward or ''}, {fest.commune or ''}, {fest.province}, Vietnam".strip(", ")

        search_task = search_web(f"{fest.name} {fest.province} {start_dt.year} agenda schedule")
        geocode_task = self._resolve_location(fest, full_address, user_input)
        
        search_results, location = await asyncio.gather(search_task, geocode_task)
        lat, lng, confidence, method = location

        # Phân loại bằng LLM (Sử dụng đúng model llama3.2 từ settings)
        types, visit_min, _ = await self._categorise(fest, search_results)

        slug = re.sub(r"[^a-z0-9]+", "_", fest.name.lower()).strip("_")[:80]
        poi = POI(
            id=f"FEST_{start_dt.isoformat()}_{slug}_{lat:.5f}_{lng:.5f}",
            name=fest.name,
            latitude=lat,
            longitude=lng,
            address=full_address,
            primaryType="tourist_attraction",
            types=types,
            openingHours=[OpeningHours(days=f"{start_dt} - {end_dt}", open="09:00", close="21:00")],
            source="festival",
            estimated_visit_minutes=visit_min,
            geocode_confidence=confidence,
            geocode_method=method,
        )

        self.logger.info(
            "Festival geocode: '%s' method=%s confidence=%.2f lat=%.5f lng=%.5f",
            fest.name,
            method,
            confidence,
            lat,
            lng,
        )

        poi = apply_poi_semantic_overrides(poi)
        self.memory.cache_set("festivals", fest.name, poi.model_dump())
        return poi

    async def _resolve_location(
        self,
        fest: RawFestival,
        full_address: str,
        user_input: UserInput,
    ) -> tuple[float, float, float, str]:
        """Resolve festival coordinates without collapsing all failures to trip start."""
        queries = [
            full_address,
            f"{fest.name}, {fest.province}, Vietnam",
            f"{fest.name}, Ho Chi Minh City, Vietnam",
            f"{fest.ward or fest.commune}, {fest.province}, Vietnam",
        ]
        seen: set[str] = set()
        for idx, query in enumerate(queries):
            query = re.sub(r"\s+", " ", query.strip(" ,"))
            if not query or len(query) < 8 or query.lower() in seen:
                continue
            seen.add(query.lower())
            lat, lng = await geocode_address(query)
            if lat and lng:
                if idx == 0:
                    return lat, lng, 0.95, "exact_address"
                if idx in (1, 2):
                    return lat, lng, 0.78, "inferred_name_city"
                return lat, lng, 0.62, "ward_or_district"

        search_text = await search_web(f"{fest.name} venue address Ho Chi Minh City", max_results=3)
        secondary_query = self._extract_secondary_location_query(fest, search_text)
        if secondary_query:
            lat, lng = await geocode_address(secondary_query)
            if lat and lng:
                return lat, lng, 0.72, "secondary_search"

        label, coords = self._city_fallback_location(fest, user_input)
        self.logger.warning(
            "Could not geocode festival '%s'; using city-level fallback '%s' instead of trip start.",
            fest.name,
            label,
        )
        return coords[0], coords[1], 0.35, f"city_fallback:{label}"

    @staticmethod
    def _extract_secondary_location_query(fest: RawFestival, search_text: str) -> str | None:
        if not search_text:
            return None
        match = re.search(
            r"((?:at|venue|address|location|địa điểm|tai|tại)\s*[:\-]?\s*)([^.\n]{8,90})",
            search_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        venue = re.sub(r"[\[\]\(\)]", " ", match.group(2)).strip(" ,;-")
        if not venue:
            return None
        return f"{venue}, {fest.province or 'Ho Chi Minh City'}, Vietnam"

    @staticmethod
    def _city_fallback_location(fest: RawFestival, user_input: UserInput) -> tuple[str, tuple[float, float]]:
        text = f"{fest.name} {fest.description}".lower()
        keyword_anchors = [
            (("ao dai", "áo dài", "parade"), ("Nguyen Hue Walking Street", (10.7753, 106.7039))),
            (("culinary", "food", "ẩm thực", "am thuc"), ("September 23 Park", (10.7696, 106.6926))),
            (("exhibition", "museum", "triển lãm", "trien lam"), ("Ho Chi Minh City Museum", (10.7761, 106.6993))),
            (("music", "concert", "night"), ("Saigon Opera House", (10.7766, 106.7031))),
        ]
        for keywords, anchor in keyword_anchors:
            if any(keyword in text for keyword in keywords):
                return anchor

        city_key = user_input.city.lower()
        province_key = (fest.province or "").lower()
        anchors = CITY_FALLBACK_LOCATIONS.get(city_key) or CITY_FALLBACK_LOCATIONS.get(province_key)
        if not anchors:
            return "trip city centroid", user_input.start_location
        digest = hashlib.blake2b(fest.name.encode("utf-8", errors="ignore"), digest_size=2).digest()
        idx = int.from_bytes(digest, "big") % len(anchors)
        return anchors[idx]

    async def _categorise(self, fest: RawFestival, search_results: str) -> tuple[list[str], int, str]:
        """Sử dụng LLM để phân loại và ước tính thời gian (Dùng model llama3.2)."""
        system = (
            "You are a travel expert. Return ONLY a JSON object with 'types' (list), "
            "'estimated_visit_minutes' (int), and 'agenda' (string)."
        )
        user = f"Festival: {fest.name}. Search results: {search_results[:1000]}"

        try:
            # Ép model llama3.2 từ settings để tránh lỗi 404 gpt-4o
            raw = await self.llm_call(system, user, model=settings.llm_model, temperature=0.1)
            # Dọn dẹp markdown nếu có
            raw_clean = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw_clean)
            return (
                data.get("types", ["tourist_attraction"]),
                data.get("estimated_visit_minutes", 120),
                data.get("agenda", ""),
            )
        except Exception:
            return self._fallback_categorise(fest)

    @staticmethod
    def _fallback_categorise(fest: RawFestival) -> tuple[list[str], int, str]:
        desc = (fest.name + " " + (fest.description or "")).lower()
        types = ["tourist_attraction"]
        visit_min = 120
        
        keyword_map = {
            "food": "food", "music": "night_club", "art": "art_gallery", 
            "flower": "park", "market": "market", "museum": "museum"
        }

        for keyword, gtype in keyword_map.items():
            if keyword in desc and gtype not in types:
                types.append(gtype)
        return types, visit_min, ""
