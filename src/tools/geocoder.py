"""Geocoding tool — resolves address text to (lat, lng) via Google Maps API."""

from __future__ import annotations

import logging

from src.config import settings

logger = logging.getLogger(__name__)


async def geocode_address(address: str) -> tuple[float | None, float | None]:
    """Geocode an address string to (latitude, longitude).

    Uses Google Maps Geocoding API.  Returns (None, None) on failure.
    """
    if not address.strip():
        return None, None

    try:
        import httpx

        params = {
            "address": address,
            "key": settings.google_maps_api_key,
            "region": "vn",
            "language": "vi",
            "components": "country:VN",
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.warning("Geocoding returned no results for '%s'.", address)
            return None, None

        location = results[0]["geometry"]["location"]
        return location["lat"], location["lng"]

    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", address, e)
        return None, None
