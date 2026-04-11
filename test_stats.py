import json, os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import preprocessing
from src.app import load_campusgroups_documents, load_reddit_documents, load_generic_csv_documents, osm_csv_path, recs_csv_path, trails_csv_path, dining_csv_path

docs = []
docs.extend(load_campusgroups_documents())
docs.extend(load_reddit_documents())
docs.extend(load_generic_csv_documents(osm_csv_path, "osm"))
docs.extend(load_generic_csv_documents(recs_csv_path, "recs"))
docs.extend(load_generic_csv_documents(trails_csv_path, "trails"))
docs.extend(load_generic_csv_documents(dining_csv_path, "dining"))

p = preprocessing.process_documents(docs)
print(json.dumps(p, default=str))
