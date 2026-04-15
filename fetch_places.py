import csv
import json
import os
import sys
import time
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

CACHE_PATH = os.path.join("data", "places_cache", "places_data.json")

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
    "gym", "sports_centre",
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
    if cat == "amenity" and sub in BUSINESS_SUBCATEGORIES:
        return True
    if "dining" in place["id"]:
        return True
    return False


def fetch_place(name, lat, lon):
    """Call Google Places Text Search and return enrichment dict or None."""
    body = {"textQuery": name, "maxResultCount": 1}
    if lat and lon:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 500.0,
            }
        }

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
        places = resp.json().get("places", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    if not places:
        return None

    p = places[0]

    photo_url = None
    photos = p.get("photos", [])
    if photos:
        pname = photos[0].get("name", "")
        if pname:
            photo_url = (
                f"https://places.googleapis.com/v1/{pname}/media"
                f"?maxHeightPx=400&maxWidthPx=600&key={GOOGLE_PLACES_API_KEY}"
            )

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

    return {
        "website": p.get("websiteUri") or None,
        "phone": p.get("nationalPhoneNumber") or None,
        "hours": hours,
        "rating": p.get("rating"),
        "rating_count": p.get("userRatingCount"),
        "price_level": _PRICE_LEVEL_MAP.get(p.get("priceLevel", ""), None),
        "photo_url": photo_url,
        "reviews": reviews or None,
    }


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

    to_fetch = [p for p in candidates if p["id"] not in cache]

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

        if data:
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
