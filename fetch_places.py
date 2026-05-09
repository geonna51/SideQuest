import csv
import json
import os
import sys
import time
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY")

CACHE_PATH = os.path.join("data", "google_places", "places_data.json")

DELAY_SECONDS = 0.12 

BUSINESS_SUBCATEGORIES = {
    "restaurant", "cafe", "fast_food", "bar", "pub", "food_court",
    "ice_cream", "biergarten", "bakery", "coffee",
    "convenience", "supermarket", "hairdresser", "clothes", "books",
    "electronics", "hardware", "florist", "gift", "jewelry", "optician",
    "sports", "toys", "music", "beauty", "chemist", "department_store",
    "general", "mall", "variety_store", "wine",
    "hotel", "hostel", "motel", "guest_house",
    "cinema", "theatre", "nightclub", "arts_centre", "casino",
    "bowling_alley", "amusement_arcade",
    "pharmacy", "clinic", "dentist", "doctors", "optometrist",
    "bank",
    "attraction", "museum", "gallery", "theme_park", "viewpoint",
    "aquarium", "zoo",
    "gym", "sports_centre", "fitness_centre", "swimming_pool",
    "park", "playground", "nature_reserve", "dog_park",
    "car_wash", "car_repair", "laundry", "dry_cleaning",
    "post_office", "library",
}

SKIP_CATEGORIES = {"highway", "building", "public_transport"}

ENRICHABLE_SOURCES = {"osm", "dining"}

_PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": "",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

_FIELDS = ",".join([
    "places.id",
    "places.websiteUri",
    "places.nationalPhoneNumber",
    "places.regularOpeningHours",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.photos",
    "places.reviews",
    "places.primaryType",
    "places.types",
    "places.businessStatus",
    "places.location",
    "places.servesVegetarianFood",
    "places.servesBreakfast",
    "places.servesBrunch",
    "places.servesLunch",
    "places.servesDinner",
    "places.servesDessert",
    "places.servesBeer",
    "places.servesWine",
    "places.servesCocktails",
    "places.outdoorSeating",
    "places.delivery",
    "places.takeout",
    "places.dineIn",
    "places.goodForGroups",
    "places.goodForChildren",
])


def load_csv_places(csv_path, source_name):
    """Yield dicts with id, name, lat, lon, subcategory for each CSV row."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                lat = float(row.get("lat") or row.get("latitude") or 0) or None
                lon = float(row.get("lon") or row.get("longitude") or 0) or None
            except (ValueError, TypeError):
                lat, lon = None, None

            yield {
                "id": f"{source_name}:{row.get('id', name)}",
                "name": name,
                "lat": lat,
                "lon": lon,
                "category": row.get("category", "").strip().lower(),
                "subcategory": row.get("subcategory", "").strip().lower(),
            }


def is_business(place):
    """Return True if this place is worth looking up on Google Places."""
    if place["category"] in SKIP_CATEGORIES:
        return False
    sub = place["subcategory"]
    cat = place["category"]

    if cat == "shop":
        return True
    if cat == "tourism" and sub not in ("camp_pitch", "picnic_site", "information"):
        return True
    if cat in ("amenity", "leisure") and sub in BUSINESS_SUBCATEGORIES:
        return True
    if "dining" in place["id"]:
        return True
    return False


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _call_search(body):
    try:
        resp = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": _FIELDS,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("places", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return "ERROR"


def fetch_place(name, lat, lon):
    """Call Google Places Text Search and return enrichment dict or None.

    Two-phase: first a HARD locationRestriction (10km box) to avoid famous-name
    mismatches like Bryant Park=NYC. If that returns nothing, fall back to a
    soft locationBias that DOES include CLOSED_PERMANENTLY places, then verify
    the result is geographically near the expected pin to discard NYC-style
    drift. The fallback is what surfaces closed local businesses so we can
    tag them as permanently closed in the UI.
    """
    body = {"textQuery": name, "maxResultCount": 1}
    if lat and lon:
        import math
        d_lat = 10000.0 / 111_000.0
        d_lon = 10000.0 / (111_000.0 * max(math.cos(math.radians(lat)), 0.01))
        body["locationRestriction"] = {
            "rectangle": {
                "low": {"latitude": lat - d_lat, "longitude": lon - d_lon},
                "high": {"latitude": lat + d_lat, "longitude": lon + d_lon},
            }
        }

    places = _call_search(body)
    if places == "ERROR":
        return "ERROR"

    if not places and lat and lon:
        # Fallback: soft bias picks up CLOSED_PERMANENTLY places (which
        # locationRestriction silently filters out).
        body.pop("locationRestriction", None)
        body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 15000.0}}
        places = _call_search(body)
        if places == "ERROR":
            return "ERROR"
        # Verify the bias result is actually local — Google's relevance ranking
        # can override the bias for famous names, so reject results > 15km away.
        if places:
            loc = places[0].get("location") or {}
            r_lat, r_lon = loc.get("latitude"), loc.get("longitude")
            if r_lat is None or r_lon is None or _haversine_km(lat, lon, r_lat, r_lon) > 15.0:
                places = []

    if not places:
        return None

    p = places[0]

    photo_name = None
    photos = p.get("photos", [])
    if photos:
        pname = photos[0].get("name", "")
        if pname:
            photo_name = pname

    hours = None
    oh = p.get("regularOpeningHours", {})
    if oh.get("weekdayDescriptions"):
        hours = oh["weekdayDescriptions"]

    reviews = []
    for r in p.get("reviews", [])[:3]:
        text = r.get("text", {})
        reviews.append({
            "author": r.get("authorAttribution", {}).get("displayName", ""),
            "rating": r.get("rating"),
            "text": text.get("text", "") if isinstance(text, dict) else "",
            "relative_time": r.get("relativePublishTimeDescription", ""),
        })

    attrs = {
        k: p.get(k) for k in (
            "servesVegetarianFood", "servesBreakfast", "servesBrunch",
            "servesLunch", "servesDinner", "servesDessert",
            "servesBeer", "servesWine", "servesCocktails",
            "outdoorSeating", "delivery", "takeout", "dineIn",
            "goodForGroups", "goodForChildren",
        ) if k in p
    }

    result = {
        "website": p.get("websiteUri") or None,
        "phone": p.get("nationalPhoneNumber") or None,
        "hours": hours,
        "rating": p.get("rating"),
        "rating_count": p.get("userRatingCount"),
        "price_level": _PRICE_LEVEL_MAP.get(p.get("priceLevel", ""), None),
        "reviews": reviews or None,
        "primary_type": p.get("primaryType") or None,
        "types": p.get("types") or None,
        "attributes": attrs or None,
    }
    # Only carry business_status when it's NOT operational — keeps the cache
    # quiet for the common case and makes "is closed" a simple truthy check.
    status = p.get("businessStatus")
    if status and status != "OPERATIONAL":
        result["business_status"] = status
    # Only include photo_name when Google actually returned one — avoids
    # leaving photo_name: null residue in the cache for photo-less places.
    if photo_name:
        result["photo_name"] = photo_name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print count only, don't fetch")
    args = parser.parse_args()

    if not GOOGLE_PLACES_API_KEY and not args.dry_run:
        print("ERROR: GOOGLE_PLACES_API_KEY not set in .env")
        sys.exit(1)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} existing entries from cache.")
    else:
        cache = {}

    sources = [
        ("data/open_street_map/osm_places.csv", "osm"),
        ("data/cornell_dining/dining.csv", "dining"),
    ]

    candidates = []
    for csv_path, source in sources:
        for place in load_csv_places(csv_path, source):
            if is_business(place):
                candidates.append(place)

    def needs_fetch(p):
        if p["id"] not in cache:
            return True
        cached = cache[p["id"]]
        # Backfill: re-fetch any cached dict that's missing fields we now request.
        # Update the key list when adding new fields to the API request.
        if not isinstance(cached, dict):
            return False
        if "primary_type" not in cached or "attributes" not in cached:
            return True
        # Re-try entries that lack any photo — Google may have added one since
        # the last fetch. photo_path means a downloaded file exists; photo_name
        # means we have an unprocessed reference. Either is sufficient.
        if not cached.get("photo_path") and not cached.get("photo_name"):
            return True
        return False

    to_fetch = [p for p in candidates if needs_fetch(p)]

    print(f"\nTotal businesses identified: {len(candidates)}")
    print(f"Already cached:              {len(candidates) - len(to_fetch)}")
    print(f"Need to fetch:               {len(to_fetch)}")

    if args.dry_run:
        print("\n-- DRY RUN: sample of what would be fetched --")
        for p in to_fetch[:10]:
            print(f"  {p['id']:<40}  {p['name']}")
        return

    print(f"\nStarting fetch (≈{len(to_fetch) * DELAY_SECONDS:.0f}s estimated)...\n")

    found = 0
    not_found = 0
    errors = 0

    for i, place in enumerate(to_fetch, 1):
        print(f"[{i}/{len(to_fetch)}] {place['name']}", end=" ... ", flush=True)

        data = fetch_place(place["name"], place["lat"], place["lon"])

        if data == "ERROR":
            errors += 1
            # Don't touch the cache — leave any existing entry intact for retry later.
            if errors >= 5 and found == 0:
                print("\nAborting: 5+ errors with no successes. Check API key / Places API (New) enablement.")
                break
        elif data:
            # Preserve photo_path if it was already populated by download_photos.py;
            # the new fetch only carries photo_name (raw Google reference) and would
            # otherwise overwrite the local-file path the frontend depends on.
            existing = cache.get(place["id"])
            if isinstance(existing, dict) and existing.get("photo_path"):
                data["photo_path"] = existing["photo_path"]
                data.pop("photo_name", None)
            cache[place["id"]] = data
            found += 1
            print(f"OK  (rating={data.get('rating')}, has_website={bool(data.get('website'))})")
        else:
            cache[place["id"]] = None  # Cache misses too, so we don't retry
            not_found += 1
            print("not found")

        if i % 25 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2)
            print(f"  [saved checkpoint at {i}]\n")

        time.sleep(DELAY_SECONDS)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nDone.")
    print(f"  Found:     {found}")
    print(f"  Not found: {not_found}")
    print(f"  Errors:    {errors}")
    print(f"  Cache saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
