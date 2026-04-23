import os
import csv
import requests

def fetch_recs():
    print("Querying Cornell Recs (via Uplift API)...")
    url = "https://uplift-backend.cornellappdev.com/graphql"
    
    # We deeply extract all Gyms, their Facilities, Amenities, Activities, 
    # and all individualized Group Fitness Classes (with times, instructors, locations) 
    query = """
    query {
      getAllGyms {
        id
        name
        address
        latitude
        longitude
        imageUrl
        classes {
          id
          location
          instructor
          isCanceled
          isVirtual
          startTime
          endTime
          class_ {
            name
            description
          }
        }
        activities {
          name
          needsReserve
          pricing {
            cost
            name
          }
        }
        facilities {
          name
        }
        amenities {
          type
        }
      }
    }
    """
    
    try:
        response = requests.post(url, json={"query": query}, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print("Failed to fetch data", exc)
        return

    data = response.json().get("data", {}).get("getAllGyms", [])
    
    os.makedirs("data/cornell_recs", exist_ok=True)
    csv_path = "data/cornell_recs/recs.csv"
    
    records = []
    
    for gym in data:
        # 1. Base Gym Record (Helen Newman, Noyes, etc.)
        gym_id = gym.get("id", "")
        gym_name = gym.get("name", "")
        
        amenities = [a.get("type") for a in gym.get("amenities", []) if a.get("type")]
        facilities = [f.get("name") for f in gym.get("facilities", []) if f.get("name")]
        activities = [a.get("name") for a in gym.get("activities", []) if a.get("name")]
        
        desc_parts = []
        if facilities: desc_parts.append(f"Facilities: {', '.join(facilities)}")
        if activities: desc_parts.append(f"Activities: {', '.join(activities)}")
        if amenities: desc_parts.append(f"Amenities: {', '.join(amenities)}")
        
        gym_desc = " | ".join(desc_parts)

        records.append({
            "id": f"gym_{gym_id}",
            "type": "Gym / Fitness Center",
            "lon": gym.get("longitude", ""),
            "lat": gym.get("latitude", ""),
            "name": gym_name,
            "category": "Cornell Rec",
            "subcategory": "Athletics Facility",
            "description": gym_desc,
            "location": gym.get("address", ""),
            "start_time": "",
            "end_time": "",
            "website": gym.get("imageUrl", "")
        })
        
        # 2. Extract every single scheduled Group Fitness Class
        for inst in gym.get("classes", []):
            cls_info = inst.get("class_", {})
            if not cls_info:
                continue
                
            cls_name = cls_info.get("name", "")
            cls_desc = cls_info.get("description", "")
            
            canceled_note = " (CANCELED)" if inst.get("isCanceled") else ""
            virtual_note = " (Virtual)" if inst.get("isVirtual") else ""
            
            inst_id = inst.get("id", "")
            instructor = inst.get("instructor", "")
            location = inst.get("location", gym_name)
            start = inst.get("startTime", "")
            end_date = inst.get("endTime", "")
            
            records.append({
                "id": f"class_{inst_id}",
                "type": "Group Fitness Class",
                "lon": gym.get("longitude", ""),
                "lat": gym.get("latitude", ""),
                "name": f"{cls_name}{virtual_note}{canceled_note}",
                "category": "Cornell Rec class",
                "subcategory": "Group Fitness",
                "description": f"Instructor: {instructor}. {cls_desc}",
                "location": location,
                "start_time": start,
                "end_time": end_date,
                "website": gym.get("imageUrl", "")
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "type", "lon", "lat", "name", "category", "subcategory", "description", "location", "start_time", "end_time", "website"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Successfully saved {len(records)} Cornell Rec records to {csv_path}")

if __name__ == "__main__":
    fetch_recs()
