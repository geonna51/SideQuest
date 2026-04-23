"""
Re-fetch Google Places data for entries that are null or missing reviews.
Updates data/google_places/places_data.json in place.
"""
import csv
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY")
DATA_PATH = os.path.join("data", "google_places", "places_data.json")
DELAY_SECONDS = 0.2

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
    if not os.path.exists(csv_path):
        return {}
    places = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if not name:
                continue
            pid = f"{source_name}:{row.get('id', name)}"
            try:
                lat = float(row.get("lat") or row.get("latitude") or 0) or None
                lon = float(row.get("lon") or row.get("longitude") or 0) or None
            except (ValueError, TypeError):
                lat, lon = None, None
            places[pid] = {"name": name, "lat": lat, "lon": lon}
    return places


def fetch_place(name, lat, lon):
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
    if not GOOGLE_PLACES_API_KEY:
        print("ERROR: GOOGLE_PLACES_API_KEY not set in .env")
        sys.exit(1)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Identify entries to refetch: null or no reviews
    to_refetch_ids = {k for k, v in data.items() if v is None or not v.get("reviews")}
    print(f"Entries to refetch: {len(to_refetch_ids)}")
    print(f"  - Null (prior API miss): {sum(1 for k in to_refetch_ids if data[k] is None)}")
    print(f"  - No reviews:            {sum(1 for k in to_refetch_ids if data[k] is not None)}")

    # Load CSV sources to get names/coordinates
    csv_places = {}
    csv_places.update(load_csv_places("data/open_street_map/osm_places.csv", "osm"))
    csv_places.update(load_csv_places("data/cornell_dining/dining.csv", "dining"))

    candidates = {pid: csv_places[pid] for pid in to_refetch_ids if pid in csv_places}
    not_in_csv = to_refetch_ids - set(candidates.keys())
    if not_in_csv:
        print(f"\nSkipping (not found in CSVs): {not_in_csv}")

    print(f"\nFetching {len(candidates)} places...\n")

    updated = 0
    improved = 0
    still_missing = 0

    for i, (pid, info) in enumerate(candidates.items(), 1):
        old = data.get(pid)
        old_status = "null" if old is None else f"no_reviews (rating={old.get('rating')})"
        print(f"[{i}/{len(candidates)}] {info['name']} ({old_status})", end=" ... ", flush=True)

        new_data = fetch_place(info["name"], info["lat"], info["lon"])
        time.sleep(DELAY_SECONDS)

        if new_data is None:
            still_missing += 1
            print("not found (keeping existing)")
            # Don't overwrite existing partial data with null
            continue

        has_reviews = bool(new_data.get("reviews"))
        # For entries that had partial data (no reviews), only update if we got reviews
        # For null entries, always update if we got anything
        if old is None or has_reviews or new_data.get("rating") is not None:
            data[pid] = new_data
            updated += 1
            if has_reviews:
                improved += 1
            print(f"OK  (rating={new_data.get('rating')}, reviews={len(new_data.get('reviews') or [])})")
        else:
            print(f"no improvement, skipping (rating={new_data.get('rating')})")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone.")
    print(f"  Updated:       {updated}")
    print(f"  With reviews:  {improved}")
    print(f"  Still missing: {still_missing}")
    print(f"  Saved to: {DATA_PATH}")


if __name__ == "__main__":
    main()
