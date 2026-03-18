import requests
import csv
import os

def fetch_osm_data():
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Ithaca + Surrounding State Parks Bounding Box: 
    # (Captures Cornell, Downtown Ithaca, Taughannock Falls, Buttermilk Falls, Treman State Park)
    overpass_query = """
    [out:json][timeout:900];
    (
      nwr["amenity"](42.38,-76.62, 42.56,-76.40);
      nwr["shop"](42.38,-76.62, 42.56,-76.40);
      nwr["leisure"](42.38,-76.62, 42.56,-76.40);
      nwr["tourism"](42.38,-76.62, 42.56,-76.40);
      nwr["historic"](42.38,-76.62, 42.56,-76.40);
      nwr["sport"](42.38,-76.62, 42.56,-76.40);
      nwr["craft"](42.38,-76.62, 42.56,-76.40);
      nwr["office"](42.38,-76.62, 42.56,-76.40);
      nwr["building"]["name"](42.38,-76.62, 42.56,-76.40);
      nwr["public_transport"](42.38,-76.62, 42.56,-76.40);
    );
    out center;
    """
    
    print("Querying Overpass API... (This might take a minute)")
    response = requests.post(overpass_url, data={'data': overpass_query})
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.text}")
        return
        
    data = response.json()
    
    os.makedirs("data/open_street_map", exist_ok=True)
    csv_file_path = "data/open_street_map/osm_places.csv"
    
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "osm_type", "lon", "lat", "name", "category", "subcategory", "address", "website"])
        
        elements = data.get("elements", [])
        count = 0
        for element in elements:
            tags = element.get("tags", {})
            name = tags.get("name", tags.get("brand", ""))
            
            # NO FILTERING: Save absolutely everything as requested.
            category = ""
            subcategory = ""
            for cat in ["amenity", "shop", "leisure", "tourism", "historic", "sport", "craft", "office", "building", "highway", "natural", "landuse", "waterway", "public_transport"]:
                if cat in tags:
                    category = cat
                    subcategory = tags[cat]
                    break
            
            if element["type"] == "node":
                lon = element["lon"]
                lat = element["lat"]
            elif "center" in element:
                lon = element["center"]["lon"]
                lat = element["center"]["lat"]
            else:
                continue
                
            addr_parts = []
            if "addr:housenumber" in tags:
                addr_parts.append(tags["addr:housenumber"])
            if "addr:street" in tags:
                addr_parts.append(tags["addr:street"])
            addr = " ".join(addr_parts)
            website = tags.get("website", "")
            
            writer.writerow([
                element["id"],
                element["type"],
                lon,
                lat,
                name,
                category,
                subcategory,
                addr,
                website
            ])
            count += 1
            
    print(f"Successfully saved {count} records to {csv_file_path}")

if __name__ == "__main__":
    fetch_osm_data()
