import os
import csv
import requests

def fetch_trails():
    print("Querying Overpass API for Ithaca Trails...")
    url = "http://overpass-api.de/api/interpreter"
    
    # Query for all designated paths, footways, cycleways, tracks, and hiking routes 
    # anywhere within the greater Tompkins County, specifically requiring a "name" 
    # to filter out random campus sidewalks and focus on real trails.
    query = """
    [out:json][timeout:900];
    (
      way["highway"="path"]["name"](42.25,-76.75, 42.65,-76.20);
      way["highway"="footway"]["name"](42.25,-76.75, 42.65,-76.20);
      way["highway"="track"]["name"](42.25,-76.75, 42.65,-76.20);
      way["highway"="cycleway"]["name"](42.25,-76.75, 42.65,-76.20);
      relation["route"="hiking"]["name"](42.25,-76.75, 42.65,-76.20);
      relation["route"="foot"]["name"](42.25,-76.75, 42.65,-76.20);
    );
    out center;
    """
    
    response = requests.post(url, data={'data': query})
    if response.status_code != 200:
        print("Failed to fetch trails data", response.text)
        return
        
    data = response.json().get("elements", [])
    
    os.makedirs("data/ithaca_trails", exist_ok=True)
    csv_path = "data/ithaca_trails/trails.csv"
    
    records = []
    seen_names = set()
    
    for item in data:
        tags = item.get("tags", {})
        name = tags.get("name", "").strip()
        
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        
        surface = tags.get("surface", "Unknown surface")
        sac_scale = tags.get("sac_scale", "") # Hiking difficulty metric
        
        desc = f"Ithaca Hiking/Walking Trail. Surface: {surface}."
        if sac_scale:
            desc += f" Scale: {sac_scale}."
            
        # Get location center
        if item["type"] == "node":
            lon = item["lon"]
            lat = item["lat"]
        elif "center" in item:
            lon = item["center"]["lon"]
            lat = item["center"]["lat"]
        else:
            continue
            
        records.append({
            "id": f"trail_{item.get('id', '')}",
            "type": "Trail",
            "lon": lon,
            "lat": lat,
            "name": name,
            "category": "Ithaca Trails",
            "subcategory": "Hiking / Footway",
            "description": desc,
            "location": "Greater Ithaca / Tompkins Area",
            "start_time": "",
            "end_time": "",
            "website": f"https://www.openstreetmap.org/{item.get('type')}/{item.get('id')}"
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "type", "lon", "lat", "name", "category", "subcategory", "description", "location", "start_time", "end_time", "website"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Successfully saved {len(records)} unique Ithaca Trails to {csv_path}")

if __name__ == "__main__":
    fetch_trails()
