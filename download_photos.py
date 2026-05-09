"""
Download Google Places photos locally so the deployed app needs no API key at runtime.

Run once before building/deploying:
    python download_photos.py

Requires GOOGLE_PLACES_API_KEY (or GOOGLE_API_KEY) in .env or environment.
Photos are saved to frontend/public/photos/ and places_data.json is updated
to reference the static paths instead of the photo resource names.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("Set GOOGLE_PLACES_API_KEY in .env before running this script.")

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "google_places" / "places_data.json"
PHOTOS_DIR = BASE_DIR / "frontend" / "public" / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

# Also link any manually-saved photos: entries with neither photo_name nor
# photo_path get their photo_path filled in if a matching file already exists
# under frontend/public/photos/. Checks common image extensions.
MANUAL_EXTS = (".jpg", ".jpeg", ".png", ".webp")
linked_manual = 0
for doc_id, entry in data.items():
    if not isinstance(entry, dict):
        continue
    if entry.get("photo_path") or entry.get("photo_name"):
        continue
    base = doc_id.replace(":", "-").replace("/", "-")
    for ext in MANUAL_EXTS:
        if (PHOTOS_DIR / f"{base}{ext}").exists():
            entry["photo_path"] = f"/photos/{base}{ext}"
            linked_manual += 1
            break
if linked_manual:
    print(f"Linked {linked_manual} manually-saved photos.")

to_download = [
    (doc_id, entry)
    for doc_id, entry in data.items()
    if isinstance(entry, dict) and entry.get("photo_name") and not entry.get("photo_path")
]

print(f"{len(to_download)} photos to download")

ok = skipped = failed = 0

for i, (doc_id, entry) in enumerate(to_download, 1):
    # Use the doc id as the filename: "osm:123" -> "osm-123.jpg"
    filename = doc_id.replace(":", "-").replace("/", "-") + ".jpg"
    dest = PHOTOS_DIR / filename

    if dest.exists():
        entry["photo_path"] = f"/photos/{filename}"
        del entry["photo_name"]
        skipped += 1
        continue

    # skipHttpRedirect=true returns JSON with photoUri; parse it to get the real image URL
    metadata_url = (
        f"https://places.googleapis.com/v1/{entry['photo_name']}/media"
        f"?maxHeightPx=400&maxWidthPx=600&key={API_KEY}&skipHttpRedirect=true"
    )

    try:
        with urllib.request.urlopen(metadata_url, timeout=15) as resp:
            photo_uri = json.loads(resp.read()).get("photoUri")
        with urllib.request.urlopen(photo_uri, timeout=15) as resp:
            dest.write_bytes(resp.read())
        entry["photo_path"] = f"/photos/{filename}"
        del entry["photo_name"]
        ok += 1
        if i % 50 == 0:
            print(f"  {i}/{len(to_download)} done...")
        time.sleep(0.05)  # stay well under quota
    except Exception as e:
        print(f"  FAILED {doc_id}: {e}")
        failed += 1

with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Done. Downloaded: {ok}, already existed: {skipped}, failed: {failed}")
print(f"Photos saved to {PHOTOS_DIR}")
