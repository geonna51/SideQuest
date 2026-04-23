import json
import os
import logging

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
_CACHE_PATHS = [
    os.path.join(_DATA_DIR, "google_places", "places_data.json"),
    os.path.join(_DATA_DIR, "places_cache", "places_data.json"),
]

_cache: dict = {}
_loaded = False


def _load_cache():
    global _cache, _loaded
    if _loaded:
        return
    _loaded = True
    for cache_path in _CACHE_PATHS:
        if not os.path.exists(cache_path):
            continue
        try:
            with open(cache_path, encoding="utf-8") as f:
                _cache = json.load(f)
            logger.info("places_enrichment: loaded %d entries from cache %s", len(_cache), cache_path)
            return
        except Exception as e:
            logger.warning("places_enrichment: failed to load cache %s: %s", cache_path, e)
    logger.info("places_enrichment: no cache file found in %s — enrichment disabled", _CACHE_PATHS)


def get_places_data(doc_id: str, name: str = "", lat: float = None, lon: float = None) -> dict | None:
    """
    Return pre-fetched Places data for a document, or None if not cached.

    Args:
        doc_id: The document ID as stored in SEARCH_DOCS (e.g. "osm:357545313")
        name:   Place name (unused — kept for API compatibility)
        lat:    Latitude (unused — kept for API compatibility)
        lon:    Longitude (unused — kept for API compatibility)
    """
    _load_cache()
    return _cache.get(doc_id)
