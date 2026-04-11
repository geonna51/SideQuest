import os
import csv
import requests

def fetch_dining():
    print("Querying Overpass API for Cornell Dining...")
    url = "http://overpass-api.de/api/interpreter"
    
    # Query for all eateries specifically operated by "Cornell Dining" 
    # as well as any major Campus "Dining", "Eatery", "Cafe" structures.
    query = """
    [out:json][timeout:900];
    (
      node["operator"="Cornell Dining"](42.43,-76.51, 42.46,-76.43);
      way["operator"="Cornell Dining"](42.43,-76.51, 42.46,-76.43);
      relation["operator"="Cornell Dining"](42.43,-76.51, 42.46,-76.43);
      
      node["name"~"Dining|Eatery|Cafe"](42.43,-76.51, 42.46,-76.43);
      way["name"~"Dining|Eatery|Cafe"](42.43,-76.51, 42.46,-76.43);
    );
    out center;
    """
    
    response = requests.post(url, data={'data': query})
    if response.status_code != 200:
        print("Failed to fetch dining data", response.text)
        return
        
    data = response.json().get("elements", [])
    
    os.makedirs("data/cornell_dining", exist_ok=True)
    csv_path = "data/cornell_dining/dining.csv"
    
    records = []
    seen_names = set()
    
    for item in data:
        tags = item.get("tags", {})
        name = tags.get("name", "").strip()
        
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        
        category = tags.get("amenity", "restaurant")
        cuisine = tags.get("cuisine", "Mixed")
        operator = tags.get("operator", "Cornell Dining")
        
        desc = f"Operated by: {operator}. Cuisine layout: {cuisine}."
        if "opening_hours" in tags:
            desc += f" Typical hours: {tags['opening_hours']}."
            
        if item["type"] == "node":
            lon = item["lon"]
            lat = item["lat"]
        elif "center" in item:
            lon = item["center"]["lon"]
            lat = item["center"]["lat"]
        else:
            continue
            
        records.append({
            "id": f"dining_{item.get('id', '')}",
            "type": "Dining Location",
            "lon": lon,
            "lat": lat,
            "name": name,
            "category": "Cornell Dining",
            "subcategory": category.capitalize() if category else "Food",
            "description": desc,
            "location": "Cornell Campus Area",
            "start_time": "",
            "end_time": "",
            "website": tags.get("website", "")
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "type", "lon", "lat", "name", "category", "subcategory", "description", "location", "start_time", "end_time", "website"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Successfully saved {len(records)} Cornell Dining locations to {csv_path}")

if __name__ == "__main__":
    fetch_dining()
