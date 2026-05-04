from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agents.clustering_agent import ClusteringAgent
from src.agents.festival_agent import FestivalAgent
from src.agents.routing_agent import RoutingAgent
from src.agents.scoring_agent import ScoringAgent
from src.agents.semantic_agent import SemanticAgent
from src.config import settings
from src.memory.store import MemoryStore
from src.models.itinerary import Itinerary
from src.models.poi import POI
from src.models.user_input import UserInput
from src.preferences import apply_poi_semantic_overrides
from src.tools.knowledge_graph import KnowledgeGraphClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """Master agent that coordinates the full trip-planning pipeline.

    Lifecycle of a single plan_trip() call:
        0. Load data
        1. Festival Agent   → enriched festival POIs
        2. Merge POIs + festivals
        3. Semantic Agent   → interest-filtered candidates
        4. Scoring Agent    → composite-scored candidates
        5. Clustering Agent → per-day groups
        6. Routing Agent    → ordered itinerary
        7. Return
    """

    def __init__(self) -> None:
        persist_path = settings.data_dir / ".memory_cache.json"
        self.memory = MemoryStore(persist_path=persist_path)
        self.kg = KnowledgeGraphClient()

        # Instantiate agents with shared memory
        self.festival_agent = FestivalAgent(memory=self.memory)
        self.semantic_agent = SemanticAgent(kg=self.kg, memory=self.memory)
        self.scoring_agent = ScoringAgent(memory=self.memory)
        self.clustering_agent = ClusteringAgent(memory=self.memory)
        self.routing_agent = RoutingAgent(memory=self.memory)

    async def plan_trip(self, user_input: UserInput) -> Itinerary:
        """End-to-end trip planning pipeline."""
        self.memory.clear_short_term()
        self.memory.set("user_input", user_input.model_dump())

        # ── Phase 0: Load raw data ──
        pois = self._load_pois(user_input.city)
        raw_festivals = self._load_festivals(user_input.city)
        logger.info("Loaded %d POIs and %d raw festivals.", len(pois), len(raw_festivals))

        # ── Phase 1: Festival enrichment ──
        enriched_festivals: list[POI] = []
        if raw_festivals:
            enriched_festivals = await self.festival_agent.run(
                raw_festivals=raw_festivals,
                user_input=user_input,
            )
        logger.info("Enriched %d festivals for the trip window.", len(enriched_festivals))

        # ── Phase 2: Merge into unified candidate pool ──
        candidates = pois + enriched_festivals
        logger.info("Total candidate pool: %d", len(candidates))

        # ── Phase 3: Semantic matching ──
        matched = await self.semantic_agent.run(
            candidates=candidates,
            user_input=user_input,
        )

        # ── Phase 4: MCDM scoring ──
        scored = await self.scoring_agent.run(
            candidates=matched,
            user_input=user_input,
        )

        # ── Phase 5: Spatial clustering ──
        daily_clusters = await self.clustering_agent.run(
            candidates=scored,
            user_input=user_input,
        )

        # ── Phase 6: Route optimization ──
        itinerary = await self.routing_agent.run(
            daily_clusters=daily_clusters,
            user_input=user_input,
        )

        logger.info(
            "Itinerary complete: %d days, total score %.2f",
            len(itinerary.days),
            itinerary.total_score,
        )
        return itinerary

    # ── Data loaders ──

    def _load_pois(self, city: str) -> list[POI]:
        path = self._data_path(city, "poi")
        if not path.exists():
            logger.warning("POI file not found: %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [apply_poi_semantic_overrides(POI(**item)) for item in raw]

    def _load_festivals(self, city: str) -> list[dict]:
        path = self._data_path(city, "fest")
        if not path.exists():
            logger.info("No festival file for city '%s'.", city)
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _data_path(city: str, kind: str) -> Path:
        """Map city code + kind to file path.

        Convention:  data/{city}_poi.json  /  data/{city}_FEST.json
        """
        name_map = {
            "poi": f"{city}_poi.json",
            "fest": f"HCM_FEST.json" if city == "hcm" else f"{city.upper()}_FEST.json",
        }
        return settings.data_dir / name_map.get(kind, f"{city}_{kind}.json")
