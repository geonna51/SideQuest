import json
import math
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing
import logging
import numpy as np
from datetime import datetime, timezone

from collections import Counter, defaultdict
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from places_enrichment import get_places_data
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

try:
    from infosci_spark_client import LLMClient
except ImportError:
    from llm_routes import LLMClient

load_dotenv()

# -----------------------------
# Paths
# -----------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

data_directory = os.path.join(project_root, "data")
campusgroups_directory = os.path.join(data_directory, "CampusGroup")
reddit_directory = os.path.join(data_directory, "reddit")

campusgroups_json_path = os.path.join(campusgroups_directory, "cornell_events_clean.json")
reddit_conversations_path = os.path.join(reddit_directory, "conversations.json")
reddit_utterances_path = os.path.join(reddit_directory, "utterances.jsonl")
reddit_corpus_path = os.path.join(reddit_directory, "corpus.json")

osm_directory = os.path.join(data_directory, "open_street_map")
osm_csv_path = os.path.join(osm_directory, "osm_places.csv")

recs_directory = os.path.join(data_directory, "cornell_recs")
recs_csv_path = os.path.join(recs_directory, "recs.csv")

trails_directory = os.path.join(data_directory, "ithaca_trails")
trails_csv_path = os.path.join(trails_directory, "trails.csv")

dining_directory = os.path.join(data_directory, "cornell_dining")
dining_csv_path = os.path.join(dining_directory, "dining.csv")

libraries_directory = os.path.join(data_directory, "cornell_libraries")
libraries_csv_path = os.path.join(libraries_directory, "libraries.csv")

cafes_directory = os.path.join(data_directory, "downtown_ithaca")
cafes_csv_path = os.path.join(cafes_directory, "cafes.csv")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(project_root, "frontend", "dist"),
    static_url_path=""
)
CORS(app)
logger = logging.getLogger(__name__)

# -----------------------------
# Search index globals
# -----------------------------
SEARCH_DOCS = []
IDF = {}
VOCAB = set()
PREPROCESSING_STATS = {}
VOCAB_TERMS = []
TERM_INDEX = {}
INVERTED_INDEX = {}  # term -> [doc_idx, ...] for fast candidate pre-filtering
SVD_COMPONENTS = None
SVD_SINGULAR_VALUES = None
DOC_LATENT_MATRIX = None
DOC_LATENT_NORMS = None
DIMENSION_LABELS = []
LATENT_DOC_INDEX = {}
SVD_STATUS = {
    "enabled": False,
    "components": 0,
    "message": "SVD index has not been built yet.",
}


def get_llm_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key or LLMClient is None:
        return None, api_key
    return LLMClient(api_key=api_key), api_key


def get_llm_status():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        return False, "AI features are unavailable because SPARK_API_KEY is not set."
    if LLMClient is None:
        return False, "AI features are unavailable because the Spark client dependency is not installed."
    return True, None


# -----------------------------
# Helpers
# -----------------------------
def as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(as_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(as_text(v) for v in value.values() if v is not None)
    return str(value).strip()


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", as_text(text)).strip()


def parse_event_date(start_time_str: str):
    """Parse start_time string into a date-only datetime.
    Handles format: 'Tuesday, 20 January 2026 At 8:00 AM, EST (GMT-5)'
    Returns a datetime (midnight) or None if unparseable.
    """
    if not start_time_str:
        return None
    match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', start_time_str)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(), '%d %B %Y')
    except ValueError:
        return None


def parse_event_datetime(start_time_str: str):
    """Parse a CampusGroups-style start_time into a full datetime when possible."""
    if not start_time_str:
        return None

    match = re.search(
        r'(\d{1,2})\s+(\w+)\s+(\d{4})(?:.*?\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b)?',
        start_time_str,
        re.IGNORECASE,
    )
    if not match:
        return None

    try:
        if match.group(4):
            minute = match.group(5) or "00"
            meridiem = match.group(6).upper()
            return datetime.strptime(
                f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}:{minute} {meridiem}",
                "%d %B %Y %I:%M %p",
            )
        return datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%d %B %Y")
    except ValueError:
        return parse_event_date(start_time_str)


def first_nonempty(record, *keys):
    for key in keys:
        if key in record:
            value = normalize_whitespace(record.get(key))
            if value:
                return value
    return ""


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", normalize_whitespace(text).lower())


GENERIC_EVENT_TITLES = {
    "board",
    "meeting",
    "general meeting",
    "weekly meeting",
    "exec board",
    "e board",
}

FOOD_ITEM_TERMS = {
    "bagel",
    "bbq",
    "breakfast",
    "brunch",
    "burger",
    "burrito",
    "chicken",
    "coffee",
    "dumpling",
    "lunch",
    "pizza",
    "ramen",
    "sandwich",
    "sushi",
    "taco",
    "tea",
    "wings",
}

# Maps a cuisine query token to substrings expected in a place's Google
# primary_type / types array. Used by is_cuisine_match to filter out food
# places whose actual Google-categorized type doesn't match the queried
# cuisine, even if the cuisine word appears in their reviews.
# Coffee/tea omitted — covered by is_coffee_relevant_doc. Generic terms
# (chicken, brunch, breakfast, lunch) omitted — too broad to filter on.
CUISINE_TYPE_HINTS = {
    "pizza": ("pizza",),
    "sushi": ("sushi", "japanese"),
    "ramen": ("ramen", "japanese", "noodle"),
    "burger": ("hamburger", "burger"),
    "taco": ("mexican", "taco"),
    "burrito": ("mexican", "burrito"),
    "dumpling": ("chinese", "asian", "dumpling"),
    "wings": ("wing", "chicken"),
    "bbq": ("barbecue", "bbq"),
    "bagel": ("bagel",),
    "sandwich": ("sandwich", "deli"),
}

# Dietary query tokens → the Google Place attribute that must be True for the
# place to be a hard match. "vegan" maps to servesVegetarianFood as a proxy
# (Google has no servesVegan flag), with a name-match fallback for places
# whose name explicitly says "vegan".
DIETARY_ATTR_HINTS = {
    "vegetarian": "servesVegetarianFood",
    "veggie": "servesVegetarianFood",
    "vegan": "servesVegetarianFood",
}

MOVIE_QUERY_TERMS = {"movie", "movies", "film", "films"}
MOVIE_QUERY_EXPANSIONS = {"cinema", "theater", "theatre", "screening"}
COFFEE_LOCATION_TERMS = {
    "cafe",
    "cafes",
    "coffee",
    "espresso",
    "latte",
    "roaster",
    "roasters",
    "tea",
}
GAMING_QUERY_TERMS = {
    "arcade",
    "esport",
    "esports",
    "game",
    "gaming",
    "tabletop",
}
GAMING_LOCATION_TERMS = {
    "arcade",
    "board",
    "esport",
    "esports",
    "fight",
    "fighter",
    "game",
    "gaming",
    "tabletop",
}
SPORT_FITNESS_TERMS = {
    "athletic",
    "athletics",
    "badminton",
    "basketball",
    "climb",
    "climbing",
    "court",
    "courts",
    "fitness",
    "gym",
    "hoop",
    "hoops",
    "pickup",
    "pickleball",
    "play",
    "recreation",
    "rec",
    "soccer",
    "sport",
    "sports",
    "squash",
    "swim",
    "swimming",
    "tennis",
    "volleyball",
    "workout",
}
SPORT_LOCATION_TERMS = {
    "athletic",
    "athletics",
    "badminton",
    "basketball",
    "court",
    "courts",
    "fitness",
    "gym",
    "gymnasium",
    "hoops",
    "pickup",
    "recreation",
    "sport",
    "sports",
    "sports_centre",
    "volleyball",
}
CAMPUS_LOCATION_HINTS = {
    "campus",
    "campus road",
    "central campus",
    "cornell",
    "cradit farm",
    "east ave",
    "north campus",
    "tower road",
    "university",
    "west ave",
    "west campus",
}
SPORT_ACTIVITY_TERMS = {
    "badminton",
    "basketball",
    "court",
    "courts",
    "hoop",
    "hoops",
    "pickup",
    "play",
    "soccer",
    "squash",
    "swim",
    "swimming",
    "tennis",
    "volleyball",
}
LOW_SIGNAL_OSM_CATEGORIES = {"building", "highway", "public_transport"}
LOW_SIGNAL_OSM_CATEGORY_SUBCATEGORY = {
    ("amenity", "bicycle_parking"),
    ("amenity", "bench"),
    ("amenity", "parking_entrance"),
    ("amenity", "parking_space"),
    ("amenity", "waste_basket"),
    ("amenity", "vending_machine"),
}
NOISY_CAMPUSGROUP_CATEGORY_STRINGS = {
    "open meeting, esports, gaming, free, fun",
    "open event, gaming, free, fun, community, food/drink",
}


def normalize_for_key(text):
    cleaned = normalize_whitespace(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", "", cleaned)
    return cleaned.strip()


def append_unique_text(base_text, extra_text):
    base_text = normalize_whitespace(base_text)
    extra_text = normalize_whitespace(extra_text)
    if not extra_text:
        return base_text
    if not base_text:
        return extra_text
    if extra_text.lower() in base_text.lower():
        return base_text
    return f"{base_text} | {extra_text}"


def is_generic_location(text):
    normalized = normalize_whitespace(text).lower()
    return normalized in {"", "ithaca area", "cornell campus area", "greater ithaca / tompkins area"}


def get_doc_lat_lon(doc):
    raw = doc.get("raw", {})
    try:
        doc_lat = float(raw.get("lat") or raw.get("latitude") or 0) or None
        doc_lon = float(raw.get("lon") or raw.get("longitude") or 0) or None
    except (ValueError, TypeError):
        return None, None
    return doc_lat, doc_lon


def is_probably_on_campus(doc):
    text_blob = " ".join([
        doc.get("title", ""),
        doc.get("location", ""),
        doc.get("description", ""),
        doc.get("category", ""),
    ]).lower()
    if any(token in text_blob for token in CAMPUS_LOCATION_HINTS):
        return True

    doc_lat, doc_lon = get_doc_lat_lon(doc)
    if doc_lat is None or doc_lon is None:
        return False

    return 42.435 <= doc_lat <= 42.4615 and -76.4925 <= doc_lon <= -76.458


def is_in_area(area_val, doc):
    if area_val == "any":
        return True

    text_blob = " ".join([
        doc.get("location", ""),
        doc.get("description", ""),
        doc.get("category", ""),
    ]).lower()

    if area_val == "campus":
        return is_probably_on_campus(doc) or doc.get("source") in {"libraries", "dining", "recs"}
    if area_val == "collegetown":
        return "collegetown" in text_blob or "college ave" in text_blob or "dryden" in text_blob or "eddy" in text_blob
    if area_val == "downtown":
        return any(token in text_blob for token in ["downtown", "commons", "state st", "aurora", "seneca", "tioga"])
    if area_val == "nature":
        return any(token in text_blob for token in ["trail", "park", "gorge", "waterfall", "nature"]) or doc.get("source") == "trails"
    return True


def passes_temporal_filters(item, future_only=True, date_from=None, date_to=None):
    if item.get("source") != "campusgroups":
        return True

    start_time = item.get("start_time", "")

    if future_only:
        event_dt = parse_event_datetime(start_time)
        if event_dt is not None and event_dt < datetime.now():
            return False

    if date_from or date_to:
        event_date = parse_event_date(start_time)
        if event_date is not None:
            if date_from and event_date < date_from:
                return False
            if date_to and event_date > date_to:
                return False

    return True


def is_low_signal_osm_record(category, subcategory):
    category = normalize_whitespace(category).lower()
    subcategory = normalize_whitespace(subcategory).lower()
    if category in LOW_SIGNAL_OSM_CATEGORIES:
        return True
    return (category, subcategory) in LOW_SIGNAL_OSM_CATEGORY_SUBCATEGORY


def parse_hour_from_text(text):
    if not text:
        return None
    text = normalize_whitespace(text).lower()

    meridiem_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text)
    if meridiem_match:
        hour = int(meridiem_match.group(1)) % 12
        minute = int(meridiem_match.group(2) or 0)
        if meridiem_match.group(3) == "pm":
            hour += 12
        return hour + (minute / 60.0)

    twenty_four_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if twenty_four_match:
        hour = int(twenty_four_match.group(1))
        minute = int(twenty_four_match.group(2))
        return hour + (minute / 60.0)

    return None


def normalized_query_tokens(query, restrict_to_vocab=False):
    cleaned_query = preprocessing.normalize_text(query, aggressive=False)
    tokens = tokenize(cleaned_query)
    if restrict_to_vocab:
        return [term for term in tokens if term in VOCAB]
    return tokens


def expand_query_tokens(query_tokens):
    expanded = list(query_tokens)
    token_set = set(query_tokens)
    query_intents = infer_query_intents(token_set)

    if query_intents["study_friendly"] and not query_intents["social"]:
        expanded.extend(["library", "quiet_study", "study_friendly"])
    if query_intents["late_night"] and not query_intents["social"]:
        expanded.extend(["late_night", "open_late"])
    has_specific_food = bool(token_set & FOOD_ITEM_TERMS)
    if query_intents["food"] and not query_intents["social"]:
        if has_specific_food:
            expanded.extend(["food", "restaurant"])
        else:
            expanded.extend(["food", "restaurant", "cafe", "dining"])
    if query_intents["cheap"]:
        expanded.extend(["cheap", "budget", "budget_friendly"])
    if query_intents["gaming"]:
        expanded.extend(["gaming", "game", "esports", "tabletop", "club"])
    if query_intents["fitness"]:
        expanded.extend(["fitness", "group_fitness", "athletics", "gym"])
        if token_set & SPORT_ACTIVITY_TERMS:
            expanded.extend(["basketball", "court", "sports", "sports_centre"])
        else:
            expanded.extend(["recreation"])
    if token_set & MOVIE_QUERY_TERMS:
        expanded.extend(sorted(MOVIE_QUERY_EXPANSIONS))

    return expanded


def derive_intent_signals(doc):
    """
    Build lightweight intent features from existing metadata so ranking can reason
    about budget/late-night/study even when raw docs are sparse.
    """
    raw = doc.get("raw", {}) if isinstance(doc.get("raw"), dict) else {}
    text_blob = " ".join([
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("category", ""),
        doc.get("location", ""),
        doc.get("organization", ""),
        raw.get("subcategory", ""),
        raw.get("type", ""),
    ]).lower()

    cheap_score = 0.0
    late_score = 0.0
    study_score = 0.0

    if any(token in text_blob for token in ["free", "low cost", "low-cost", "budget", "cheap", "affordable"]):
        cheap_score += 0.8
    if any(token in text_blob for token in ["fast food", "deli", "food court", "bagel", "pizza", "burger"]):
        cheap_score += 0.3
    if doc.get("source") == "dining" and any(token in text_blob for token in ["cafe", "deli", "food court"]):
        cheap_score += 0.1

    if any(token in text_blob for token in ["late night", "late-night", "open late", "after dark", "night"]):
        late_score += 0.7
    start_hour = parse_hour_from_text(doc.get("start_time", ""))
    end_hour = parse_hour_from_text(doc.get("end_time", ""))
    if start_hour is not None and start_hour >= 20:
        late_score += 0.35
    if end_hour is not None and end_hour >= 22:
        late_score += 0.45

    if any(token in text_blob for token in ["study", "library", "quiet", "reading", "workspace", "cowork", "bookstore"]):
        study_score += 0.8
    if any(token in text_blob for token in ["cafe", "coffee", "tea", "espresso"]):
        study_score += 0.15
    if raw.get("subcategory", "").lower() in {"library", "cafe"}:
        study_score += 0.6
    if "cocktail" in text_blob or "bar" in raw.get("subcategory", "").lower():
        study_score -= 0.15

    signals = {
        "cheap": round(max(0.0, min(1.0, cheap_score)), 4),
        "late_night": round(max(0.0, min(1.0, late_score)), 4),
        "study_friendly": round(max(0.0, min(1.0, study_score)), 4),
    }

    intent_tags = []
    if signals["cheap"] >= 0.45:
        intent_tags.append("budget budget_friendly cheap")
    if signals["late_night"] >= 0.45:
        intent_tags.append("late_night open_late")
    if signals["study_friendly"] >= 0.45:
        intent_tags.append("study_friendly quiet_study")

    return signals, " ".join(intent_tags).strip()


def infer_query_intents(query_tokens):
    study_intent = bool({"study", "quiet", "library", "reading", "focus"} & query_tokens)
    late_intent = bool({"late", "night", "midnight", "open", "after", "dark"} & query_tokens)
    social_intent = bool({"social", "group", "club", "event", "meet", "meeting", "people"} & query_tokens)
    gaming_intent = bool(query_tokens & GAMING_QUERY_TERMS) or ("board" in query_tokens and "game" in query_tokens)
    return {
        "food": bool({"food", "dining", "restaurant", "eat", "lunch", "dinner", "breakfast", "coffee", "cafe"} & query_tokens) or bool(query_tokens & FOOD_ITEM_TERMS) or bool(query_tokens & DIETARY_ATTR_HINTS.keys()),
        "coffee": bool({"coffee", "espresso", "latte", "cafe"} & query_tokens),
        "cheap": bool({"cheap", "budget", "affordable", "low", "cost", "free"} & query_tokens),
        "gaming": gaming_intent,
        "fitness": bool(query_tokens & SPORT_FITNESS_TERMS) or bool({"exercise", "active"} & query_tokens),
        "late_night": late_intent,
        "study_friendly": study_intent,
        "social": social_intent,
        "nightlife": bool({"karaoke", "music", "concert", "open_mic", "mic", "bar", "nightlife"} & query_tokens),
        "place_like": bool({"place", "spot", "where", "library", "cafe", "coffee", "near"} & query_tokens) or (study_intent and not social_intent),
    }


def compute_intent_alignment_adjustment(query_intents, doc):
    signals = doc.get("_intent_signals") or {}
    cheap_signal = float(signals.get("cheap", 0.0))
    late_signal = float(signals.get("late_night", 0.0))
    study_signal = float(signals.get("study_friendly", 0.0))
    source = doc.get("source", "")
    doc_type = doc.get("doc_type", "")

    adjustment = 0.0

    if query_intents["cheap"]:
        adjustment += (cheap_signal * 0.22) - 0.04
        if query_intents["social"] and source == "campusgroups":
            adjustment += 0.06
    if query_intents["social"] and source == "campusgroups" and doc_type == "event":
        adjustment += 0.12
    if query_intents["gaming"]:
        text_blob = " ".join([
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("category", ""),
            doc.get("organization", ""),
        ]).lower()
        if source == "campusgroups" and any(token in text_blob for token in GAMING_LOCATION_TERMS):
            adjustment += 0.22
        elif source in {"osm", "dining", "libraries", "cafes", "trails", "recs"}:
            adjustment -= 0.14
    if query_intents["food"] and query_intents["social"]:
        category = normalize_whitespace(doc.get("category", "")).lower()
        if source == "campusgroups" and ("food/drink" in category or "social" in category):
            adjustment += 0.14
        if source == "dining":
            adjustment -= 0.08
    if query_intents["coffee"]:
        text_blob = " ".join([
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("category", ""),
        ]).lower()
        if any(token in text_blob for token in ["coffee", "espresso", "latte", "cafe"]):
            adjustment += 0.16
        elif query_intents["food"] and source in {"dining", "cafes"}:
            adjustment += 0.04
        else:
            adjustment -= 0.08
    if query_intents["fitness"]:
        text_blob = " ".join([
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("category", ""),
            doc.get("location", ""),
        ]).lower()
        has_sports_signal = any(token in text_blob for token in SPORT_LOCATION_TERMS)
        looks_like_class = "class" in text_blob or "instructor:" in text_blob
        if source == "recs":
            adjustment += 0.2
            if query_intents["place_like"] and looks_like_class:
                adjustment -= 0.18
        elif source == "osm" and has_sports_signal:
            adjustment += 0.14
        elif source == "osm":
            adjustment -= 0.16
        elif source in {"dining", "libraries", "cafes"}:
            adjustment -= 0.16
    if query_intents["late_night"]:
        adjustment += (late_signal * 0.2) - 0.02
    if query_intents["study_friendly"]:
        adjustment += (study_signal * 0.26) - 0.05
        if source == "campusgroups" and doc_type == "event" and study_signal < 0.45:
            adjustment -= 0.12
        if source in {"osm", "dining", "libraries", "cafes"} and study_signal >= 0.40:
            adjustment += 0.15
        if query_intents["place_like"] and doc_type == "event":
            adjustment -= 0.18
        if query_intents["place_like"] and source in {"osm", "dining", "libraries", "cafes"}:
            adjustment += 0.15
        if source == "libraries":
            adjustment += 0.22  # Libraries are inherently strong study spots
        if query_intents["late_night"] and doc_type == "event":
            adjustment -= 0.12
    if query_intents["nightlife"]:
        if source in {"cafes", "campusgroups"}:
            adjustment += 0.18
        if source == "libraries":
            adjustment -= 0.22
        if source == "osm" and "library" in normalize_whitespace(doc.get("title", "")).lower():
            adjustment -= 0.18

    return max(-0.25, min(0.35, adjustment))


def is_food_relevant_doc(doc):
    source = doc.get("source", "")
    if source in {"dining", "cafes"}:
        return True
    if source != "osm":
        return False

    raw = doc.get("raw", {}) if isinstance(doc.get("raw"), dict) else {}
    subcategory = normalize_whitespace(raw.get("subcategory", "")).lower()
    if subcategory in {
        "restaurant", "cafe", "fast_food", "food_court", "bar", "pub",
        "ice_cream", "bakery", "coffee_shop",
    }:
        return True

    name = normalize_whitespace(doc.get("title", "")).lower()
    return any(token in name for token in [
        "cafe", "coffee", "restaurant", "pizza", "grill", "deli", "bagel",
        "eatery", "kitchen", "bistro", "bakery", "burger", "tea",
    ])


def is_cuisine_match(query_tokens, doc):
    """
    Return False when the query names a specific cuisine and this doc is a
    food place whose Google Place types clearly don't match. Other docs
    (non-food queries, non-food docs, places without Google type data, or
    places whose name itself contains the cuisine) pass through.
    """
    cuisines = query_tokens & CUISINE_TYPE_HINTS.keys()
    if not cuisines:
        return True
    if not is_food_relevant_doc(doc):
        return True

    title = normalize_whitespace(doc.get("title", "")).lower()
    if any(c in title for c in cuisines):
        return True

    google_types = doc.get("google_types") or []
    if not google_types:
        return True  # No Google data to filter on — keep based on lexical match.

    type_blob = " ".join(google_types).lower()
    expected = {hint for c in cuisines for hint in CUISINE_TYPE_HINTS[c]}
    return any(hint in type_blob for hint in expected)


def is_dietary_match(query_tokens, doc):
    """
    Return False when the query asks for a specific dietary preference and this
    doc is a food place that doesn't satisfy it. Pass-through cases:
    non-dietary queries, non-food docs, places without attribute data, and
    places whose name itself contains the dietary term.
    """
    diets = query_tokens & DIETARY_ATTR_HINTS.keys()
    if not diets:
        return True
    if not is_food_relevant_doc(doc):
        return True

    title = normalize_whitespace(doc.get("title", "")).lower()
    if any(d in title for d in diets):
        return True

    attrs = doc.get("place_attributes") or {}
    if not attrs:
        return True  # No Google attribute data — keep based on lexical match.

    return any(attrs.get(DIETARY_ATTR_HINTS[d]) is True for d in diets)


def is_coffee_relevant_doc(doc):
    source = doc.get("source", "")
    if source == "cafes":
        return True

    raw = doc.get("raw", {}) if isinstance(doc.get("raw"), dict) else {}
    subcategory = normalize_whitespace(raw.get("subcategory", "")).lower()
    text_blob = " ".join([
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("category", ""),
        doc.get("location", ""),
    ]).lower()

    if subcategory in {"cafe", "coffee", "coffee_shop", "bakery"}:
        return True
    return any(token in text_blob for token in COFFEE_LOCATION_TERMS)


def compute_metadata_adjustment(query_tokens, doc):
    """
    Applies small ranking nudges based on metadata quality and intent alignment.
    The adjustment is bounded to keep core retrieval behavior stable.
    """
    adjustment = 0.0
    title = normalize_whitespace(doc.get("title", "")).lower()
    category = normalize_whitespace(doc.get("category", "")).lower()
    location = normalize_whitespace(doc.get("location", "")).lower()
    source = doc.get("source", "")

    if category.startswith("private"):
        adjustment -= 0.15

    if title in GENERIC_EVENT_TITLES:
        adjustment -= 0.1
        if query_tokens & GAMING_QUERY_TERMS or ("board" in query_tokens and "game" in query_tokens):
            adjustment -= 0.08
    elif len(title.split()) <= 1:
        adjustment -= 0.1

    if {"food", "dining", "restaurant", "eat", "lunch", "dinner"} & query_tokens:
        if source == "dining":
            adjustment += 0.14
        if "food" in category or "restaurant" in category:
            adjustment += 0.08
        if source == "campusgroups" and "food/drink" in category:
            # Avoid event announcements outranking actual places to eat.
            adjustment -= 0.12

    if {"study", "quiet", "library"} & query_tokens:
        if "library" in title or "library" in category:
            adjustment += 0.12
        if "quiet" in doc.get("description", "").lower():
            adjustment += 0.06
        if source == "campusgroups" and "study" not in title and "study" not in doc.get("description", "").lower():
            adjustment -= 0.15

    if {"hike", "trail", "waterfall", "nature", "outdoor"} & query_tokens:
        if source in {"trails", "osm"}:
            adjustment += 0.1
        if "trail" in title or "waterfall" in title or "nature" in title:
            adjustment += 0.08

    if query_tokens & SPORT_FITNESS_TERMS:
        text_blob = " ".join([
            title,
            category,
            location,
            normalize_whitespace(doc.get("description", "")).lower(),
            normalize_whitespace(doc.get("organization", "")).lower(),
        ])
        has_sports_signal = any(token in text_blob for token in SPORT_LOCATION_TERMS)
        if has_sports_signal:
            adjustment += 0.16
            if query_tokens & {"basketball", "court", "courts", "hoop", "hoops"} and (
                "basketball" in text_blob or "court" in text_blob
            ):
                adjustment += 0.08
        elif source == "osm":
            adjustment -= 0.16
        elif source in {"dining", "libraries", "cafes"}:
            adjustment -= 0.12

    if {"collegetown", "downtown", "campus", "ithaca"} & query_tokens:
        overlap = sum(1 for token in query_tokens if token in location)
        adjustment += min(0.12, overlap * 0.04)

    return max(-0.25, min(0.35, adjustment))


def compute_overlap_boost(query_tokens, doc):
    """
    Boosts documents that share explicit query terms in key fields.
    This keeps ranking anchored to user wording even when SVD is active.
    """
    if not query_tokens:
        return 0.0

    title_tokens = set(tokenize(doc.get("title", "")))
    category_tokens = set(tokenize(doc.get("category", "")))
    location_tokens = set(tokenize(doc.get("location", "")))
    desc_tokens = set(tokenize(doc.get("description", "")))

    title_overlap = len(query_tokens & title_tokens)
    category_overlap = len(query_tokens & category_tokens)
    location_overlap = len(query_tokens & location_tokens)
    desc_overlap = len(query_tokens & desc_tokens)

    boost = (
        title_overlap * 0.06
        + category_overlap * 0.03
        + location_overlap * 0.02
        + desc_overlap * 0.01
    )
    return min(0.24, boost)


def format_non_llm_summary(query, results):
    if not results:
        return "No strong matches found yet. Try adding location or intent terms."

    top = results[:3]
    lines = [f"Top matches for \"{query}\":"]
    for idx, item in enumerate(top, start=1):
        meta = []
        if item.get("category"):
            meta.append(item["category"])
        if item.get("location"):
            meta.append(item["location"])
        meta_text = " - ".join(meta[:2]) if meta else item.get("source", "result")
        lines.append(f"{idx}. {item['title']} ({meta_text})")
    return " ".join(lines)


def load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_records(payload):
    """
    Supports:
    - [ {...}, {...} ]
    - { "events": [...] }
    - { "activities": [...] }
    - { "data": [...] }
    - { "items": [...] }
    - { "records": [...] }
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("events", "activities", "data", "items", "records"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]

    return []


def build_lookup_from_json(payload):
    """
    Tries to turn conversations.json-like payloads into:
    { "<id>": { ...record... } }
    """
    lookup = {}

    if payload is None:
        return lookup

    if isinstance(payload, dict):
        # Case 1: already looks like {id: {...}, id2: {...}}
        if payload and all(isinstance(v, dict) for v in payload.values()):
            for k, v in payload.items():
                record = dict(v)
                record.setdefault("id", k)
                lookup[str(record["id"])] = record
            return lookup

        # Case 2: nested list
        records = extract_json_records(payload)
        for i, record in enumerate(records):
            record_id = first_nonempty(record, "id", "conversation_id", "uuid") or str(i)
            lookup[str(record_id)] = record
        return lookup

    if isinstance(payload, list):
        for i, record in enumerate(payload):
            if not isinstance(record, dict):
                continue
            record_id = first_nonempty(record, "id", "conversation_id", "uuid") or str(i)
            lookup[str(record_id)] = record

    return lookup


# -----------------------------
# CampusGroups loading
# -----------------------------
def clean_campusgroups_category(title, description, organization, category):
    category_text = normalize_whitespace(category)
    if not category_text:
        return ""

    category_key = category_text.lower()
    if category_key not in NOISY_CAMPUSGROUP_CATEGORY_STRINGS:
        return category_text

    text_blob = " ".join([title, description, organization]).lower()
    if any(token in text_blob for token in GAMING_LOCATION_TERMS):
        return category_text

    # Some CampusGroups exports contain repeated misaligned gaming tags on
    # unrelated events; drop them so the IR index stays grounded.
    return ""


def normalize_campusgroup_record(record, idx):
    title = first_nonempty(record, "title", "name", "event_name", "summary")
    description = first_nonempty(record, "description", "descr", "details", "body", "content", "about")
    organization = first_nonempty(record, "organization", "org", "group", "club", "host", "organization_name")
    category = first_nonempty(record, "category", "categories", "tags", "tag", "event_type", "type")
    category = clean_campusgroups_category(title, description, organization, category)
    location = first_nonempty(record, "location", "place", "venue", "room", "building")
    start_time = first_nonempty(record, "start_time", "start", "date", "start_date", "datetime", "event_date")
    end_time = first_nonempty(record, "end_time", "end", "end_date")
    url = first_nonempty(record, "url", "link", "registration_url", "event_url")
    record_id = first_nonempty(record, "id", "event_id", "uuid") or f"campusgroups:{idx}"

    if not any([title, description, organization, category, location, start_time]):
        return None

    search_text = " ".join(
        part for part in [
            title,
            description,
            organization,
            category,
            location,
            start_time,
            end_time
        ] if part
    )

    return {
        "id": record_id,
        "title": title or f"CampusGroups Event {idx}",
        "description": description,
        "organization": organization,
        "category": category or "campusgroups_event",
        "location": location,
        "start_time": start_time,
        "end_time": end_time,
        "url": url,
        "source": "campusgroups",
        "doc_type": "event",
        "search_text": search_text,
        "raw": record,
    }


def load_campusgroups_documents():
    docs = []
    payload = load_json_if_exists(campusgroups_json_path)
    records = extract_json_records(payload)

    for idx, record in enumerate(records):
        doc = normalize_campusgroup_record(record, idx)
        if doc:
            docs.append(doc)

    return docs


# -----------------------------
# Reddit loading
# -----------------------------
def normalize_reddit_thread(conversation_id, conversation_meta, utterances):
    title = first_nonempty(
        conversation_meta,
        "title", "subject", "name", "link_title", "submission_title"
    )

    subreddit = first_nonempty(
        conversation_meta,
        "subreddit", "community", "forum"
    )

    created = first_nonempty(
        conversation_meta,
        "created_at", "created_utc", "timestamp", "date"
    )

    url = first_nonempty(
        conversation_meta,
        "url", "permalink", "link"
    )

    texts = []
    speakers = []

    for utt in utterances:
        text = first_nonempty(utt, "text", "body", "content")
        speaker = first_nonempty(utt, "speaker", "author", "username", "user")
        if text:
            texts.append(text)
        if speaker:
            speakers.append(speaker)

    if not title and texts:
        title = texts[0][:120]

    snippet = " ".join(texts[:3])
    full_text = " ".join(texts)
    speaker_text = " ".join(speakers[:20])

    if not any([title, full_text, subreddit]):
        return None

    search_text = " ".join(
        part for part in [
            title,
            subreddit,
            speaker_text,
            full_text
        ] if part
    )

    return {
        "id": f"reddit:{conversation_id}",
        "title": title or f"Reddit Thread {conversation_id}",
        "description": snippet,
        "organization": subreddit,
        "category": "reddit_thread",
        "location": "",
        "start_time": created,
        "end_time": "",
        "url": url,
        "source": "reddit",
        "doc_type": "thread",
        "search_text": search_text,
        "raw": {
            "conversation": conversation_meta,
            "utterance_count": len(utterances)
        },
    }


def load_reddit_documents():
    docs = []

    conversation_lookup = build_lookup_from_json(load_json_if_exists(reddit_conversations_path))
    corpus_lookup = build_lookup_from_json(load_json_if_exists(reddit_corpus_path))

    utterances_by_conversation = defaultdict(list)

    if os.path.exists(reddit_utterances_path):
        with open(reddit_utterances_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    utt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                conversation_id = first_nonempty(utt, "conversation_id", "thread_id", "root")
                if not conversation_id:
                    conversation_id = f"unknown_{line_num}"

                utterances_by_conversation[str(conversation_id)].append(utt)

    # Prefer conversations.json metadata, fall back to corpus.json metadata
    all_conversation_ids = set(utterances_by_conversation.keys()) | set(conversation_lookup.keys()) | set(corpus_lookup.keys())

    for conversation_id in all_conversation_ids:
        meta = conversation_lookup.get(conversation_id) or corpus_lookup.get(conversation_id) or {}
        utterances = utterances_by_conversation.get(conversation_id, [])
        doc = normalize_reddit_thread(conversation_id, meta, utterances)
        if doc:
            docs.append(doc)

    return docs


# -----------------------------
# CSV Datasets Loading
# -----------------------------
_OSM_TEMPLATES = {
    ("amenity", "restaurant"):        "{n} is a restaurant in Ithaca offering dine-in meals.",
    ("amenity", "cafe"):              "{n} is a café in Ithaca serving coffee, drinks, and light meals.",
    ("amenity", "fast_food"):         "{n} is a fast food spot in Ithaca for quick, affordable meals.",
    ("amenity", "bar"):               "{n} is a bar in Ithaca serving drinks and cocktails.",
    ("amenity", "pub"):               "{n} is a pub in Ithaca with drinks and a relaxed atmosphere.",
    ("amenity", "ice_cream"):         "{n} is an ice cream shop in Ithaca.",
    ("amenity", "bakery"):            "{n} is a bakery in Ithaca serving fresh baked goods.",
    ("amenity", "place_of_worship"):  "{n} is a place of worship in the Ithaca area.",
    ("amenity", "school"):            "{n} is a school in Ithaca.",
    ("amenity", "clinic"):            "{n} is a medical clinic in Ithaca.",
    ("amenity", "dentist"):           "{n} is a dental office in Ithaca.",
    ("amenity", "veterinary"):        "{n} is a veterinary clinic in Ithaca.",
    ("amenity", "pharmacy"):          "{n} is a pharmacy in Ithaca.",
    ("amenity", "bank"):              "{n} is a bank in Ithaca.",
    ("amenity", "library"):           "{n} is a library in Ithaca with books and study resources.",
    ("amenity", "theatre"):           "{n} is a theatre in Ithaca for performances and events.",
    ("amenity", "community_centre"):  "{n} is a community center in Ithaca hosting local events and programs.",
    ("amenity", "marketplace"):       "{n} is a marketplace in Ithaca.",
    ("amenity", "fuel"):              "{n} is a gas station in Ithaca.",
    ("amenity", "grave_yard"):        "{n} is a cemetery in Ithaca.",
    ("amenity", "townhall"):          "{n} is a town hall or government building in Ithaca.",
    ("amenity", "fire_station"):      "{n} is a fire station in Ithaca.",
    ("amenity", "police"):            "{n} is a police station in Ithaca.",
    ("leisure", "park"):              "{n} is a park in Ithaca, great for outdoor relaxation and recreation.",
    ("leisure", "playground"):        "{n} is a playground in Ithaca for outdoor play.",
    ("leisure", "pitch"):             "{n} is an outdoor sports court or field in Ithaca.",
    ("leisure", "swimming_pool"):     "{n} is a swimming pool in Ithaca.",
    ("leisure", "fitness_centre"):    "{n} is a fitness center in Ithaca with gym and workout facilities.",
    ("leisure", "sports_centre"):     "{n} is a sports and recreation center in Ithaca.",
    ("leisure", "nature_reserve"):    "{n} is a nature reserve in the Ithaca area with trails and wildlife.",
    ("leisure", "garden"):            "{n} is a garden in Ithaca, a peaceful outdoor spot.",
    ("leisure", "golf_course"):       "{n} is a golf course in the Ithaca area.",
    ("leisure", "stadium"):           "{n} is a stadium in Ithaca hosting sports and events.",
    ("leisure", "slipway"):           "{n} is a boat launch near Ithaca.",
    ("tourism", "hotel"):             "{n} is a hotel in Ithaca offering overnight accommodation.",
    ("tourism", "motel"):             "{n} is a motel in Ithaca offering overnight accommodation.",
    ("tourism", "guest_house"):       "{n} is a guest house in Ithaca offering lodging.",
    ("tourism", "museum"):            "{n} is a museum in Ithaca with exhibits and cultural programming.",
    ("tourism", "attraction"):        "{n} is a local attraction in the Ithaca area worth visiting.",
    ("tourism", "viewpoint"):         "{n} is a scenic viewpoint in the Ithaca area.",
    ("tourism", "camp_pitch"):        "{n} is a camping spot in the Ithaca area for outdoor overnight stays.",
    ("building", "university"):       "{n} is a Cornell University academic or administrative building.",
    ("building", "dormitory"):        "{n} is a Cornell student dormitory.",
    ("shop", "convenience"):          "{n} is a convenience store in Ithaca for everyday essentials.",
    ("shop", "supermarket"):          "{n} is a supermarket in Ithaca for grocery shopping.",
    ("shop", "bakery"):               "{n} is a bakery in Ithaca with fresh breads and pastries.",
    ("shop", "alcohol"):              "{n} is a liquor and wine shop in Ithaca.",
    ("shop", "books"):                "{n} is a bookstore in Ithaca.",
    ("shop", "sports"):               "{n} is a sporting goods store in Ithaca.",
    ("shop", "hairdresser"):          "{n} is a hair salon in Ithaca.",
    ("shop", "clothes"):              "{n} is a clothing store in Ithaca.",
    ("shop", "bicycle"):              "{n} is a bicycle shop in Ithaca for sales and repairs.",
    ("shop", "cannabis"):             "{n} is a cannabis dispensary in Ithaca.",
    ("shop", "department_store"):     "{n} is a department store in Ithaca.",
    ("shop", "supermarket"):          "{n} is a supermarket in Ithaca.",
    ("highway", "bus_stop"):          "{n} is a bus stop in Ithaca.",
    ("historic", "memorial"):         "{n} is a historic memorial or landmark in Ithaca.",
    ("office", "government"):         "{n} is a government office in Ithaca.",
    ("office", "research"):           "{n} is a research office or institute in Ithaca.",
}

def _osm_template_description(name, category, subcategory):
    template = _OSM_TEMPLATES.get((category, subcategory))
    n = name if name else f"This {subcategory.replace('_', ' ') or category}"
    if template:
        return template.format(n=n)
    label = subcategory.replace("_", " ") if subcategory else category
    return f"{n} is a {label} in Ithaca." if name else f"A {label} in Ithaca."


def load_generic_csv_documents(csv_path, source_name):
    docs = []
    if not os.path.exists(csv_path):
        return docs
    
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            title = row.get("name", "").strip()
            if not title:
                title = f"{source_name.capitalize()} {row.get('id', idx)}"
            
            category = row.get("category", "").strip()
            subcategory = row.get("subcategory", "").strip()
            address = row.get("address", "").strip() or row.get("location", "").strip()
            db_desc = row.get("description", "").strip()

            if source_name == "osm" and is_low_signal_osm_record(category, subcategory):
                continue

            if not db_desc and source_name == "osm":
                db_desc = _osm_template_description(title, category, subcategory)

            meta_parts = []
            if category: meta_parts.append(f"Category: {category}")
            if subcategory: meta_parts.append(f"Type: {subcategory}")
            if address: meta_parts.append(f"Address: {address}")
            meta_line = " | ".join(meta_parts)
            description = "\n".join(filter(None, [db_desc, meta_line]))
            
            search_text = " ".join([title, category, subcategory, address, db_desc])
            
            start_time = row.get("start_time", "").strip()
            end_time = row.get("end_time", "").strip()
            
            docs.append({
                "id": f"{source_name}:{row.get('id', idx)}",
                "title": title,
                "description": description,
                "organization": row.get("operator", source_name.capitalize()),
                "category": category or source_name,
                "location": address or "Ithaca Area",
                "start_time": start_time,
                "end_time": end_time,
                "url": row.get("website", ""),
                "source": source_name,
                "doc_type": row.get("type", "location"),
                "search_text": search_text,
                "raw": row,
            })
    return docs


def merge_official_place_metadata(base_docs, official_docs):
    """
    Merge curated Cornell/Ithaca metadata into matching OSM/dining documents when
    titles line up, and add unmatched curated records as standalone documents.
    """
    title_index = defaultdict(list)
    for doc in base_docs:
        title_index[normalize_for_key(doc.get("title", ""))].append(doc)

    merged_ids = set()
    unmatched_docs = []

    for official_doc in official_docs:
        title_key = normalize_for_key(official_doc.get("title", ""))
        candidates = title_index.get(title_key, [])
        matched_doc = None

        for candidate in candidates:
            candidate_source = candidate.get("source")
            official_source = official_doc.get("source")

            if official_source == "libraries" and candidate_source == "osm":
                matched_doc = candidate
                break
            if official_source == "cafes" and candidate_source in {"osm", "dining"}:
                matched_doc = candidate
                break

        if matched_doc is None:
            unmatched_docs.append(official_doc)
            continue

        merged_ids.add(matched_doc.get("id"))
        matched_doc["description"] = append_unique_text(
            matched_doc.get("description", ""),
            official_doc.get("description", ""),
        )
        if is_generic_location(matched_doc.get("location", "")):
            matched_doc["location"] = official_doc.get("location", "")
        else:
            matched_doc["location"] = append_unique_text(
                matched_doc.get("location", ""),
                official_doc.get("location", ""),
            )
        matched_doc["organization"] = append_unique_text(
            matched_doc.get("organization", ""),
            official_doc.get("organization", ""),
        )
        matched_doc["category"] = append_unique_text(
            matched_doc.get("category", ""),
            official_doc.get("category", ""),
        )

        if official_doc.get("start_time") and not matched_doc.get("start_time"):
            matched_doc["start_time"] = official_doc["start_time"]
        if official_doc.get("end_time"):
            matched_doc["end_time"] = official_doc["end_time"]
        if official_doc.get("url") and not matched_doc.get("url"):
            matched_doc["url"] = official_doc["url"]

        matched_doc["search_text"] = " ".join(
            part for part in [
                matched_doc.get("search_text", ""),
                official_doc.get("search_text", ""),
                official_doc.get("category", ""),
            ] if part
        ).strip()

        if isinstance(matched_doc.get("raw"), dict):
            matched_doc["raw"]["official_overlay"] = official_doc.get("raw", {})

    return base_docs + unmatched_docs

# -----------------------------
# TF-IDF / cosine similarity
# -----------------------------
def compute_idf(num_docs, df_counter):
    """
    Smoothed IDF:
        idf(t) = log((N + 1) / (df(t) + 1)) + 1
    """
    idf = {}
    for term, df in df_counter.items():
        idf[term] = math.log((num_docs + 1) / (df + 1)) + 1.0
    return idf


def compute_tfidf_vector(token_counts, idf_map):
    """
    Log-scaled TF-IDF:
        tf(t,d) = 1 + log(count)
        tfidf(t,d) = tf(t,d) * idf(t)
    """
    weights = {}
    for term, count in token_counts.items():
        if count <= 0 or term not in idf_map:
            continue
        tf = 1.0 + math.log(count)
        weights[term] = tf * idf_map[term]
    return weights


def vector_norm(weights):
    return math.sqrt(sum(weight * weight for weight in weights.values()))


def dot_product_sparse(vec_a, vec_b):
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())


def sparse_vector_to_matrix(weights):
    if not TERM_INDEX or not weights:
        return csr_matrix((1, len(VOCAB_TERMS)), dtype=float)

    cols = []
    values = []
    for term, weight in weights.items():
        idx = TERM_INDEX.get(term)
        if idx is None:
            continue
        cols.append(idx)
        values.append(weight)

    if not cols:
        return csr_matrix((1, len(VOCAB_TERMS)), dtype=float)

    rows = np.zeros(len(cols), dtype=int)
    return csr_matrix((values, (rows, cols)), shape=(1, len(VOCAB_TERMS)), dtype=float)


def summarize_dimension(component_weights, vocab_terms, top_n=4):
    indexed_weights = list(enumerate(component_weights))
    indexed_weights.sort(key=lambda item: item[1], reverse=True)

    positive_terms = [
        vocab_terms[idx]
        for idx, weight in indexed_weights
        if weight > 0
    ][:top_n]

    negative_terms = [
        vocab_terms[idx]
        for idx, weight in sorted(indexed_weights, key=lambda item: item[1])
        if weight < 0
    ][:top_n]

    return {
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
    }


def build_svd_index(doc_term_matrix, vocab_terms, requested_components=18):
    global SVD_COMPONENTS, SVD_SINGULAR_VALUES, DOC_LATENT_MATRIX
    global DOC_LATENT_NORMS, DIMENSION_LABELS, SVD_STATUS

    num_docs, vocab_size = doc_term_matrix.shape
    max_rank = min(num_docs - 1, vocab_size - 1)

    if max_rank < 2:
        SVD_COMPONENTS = None
        SVD_SINGULAR_VALUES = None
        DOC_LATENT_MATRIX = None
        DOC_LATENT_NORMS = None
        DIMENSION_LABELS = []
        SVD_STATUS = {
            "enabled": False,
            "components": 0,
            "message": "Not enough data to compute a meaningful SVD index.",
        }
        return

    component_count = min(requested_components, max_rank)

    try:
        u, singular_values, vt = svds(doc_term_matrix, k=component_count)

        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        vt = vt[order]
        u = u[:, order]

        doc_latent = u * singular_values
        doc_norms = np.linalg.norm(doc_latent, axis=1)

        dimension_labels = []
        for dim_idx, component in enumerate(vt):
            label = summarize_dimension(component, vocab_terms)
            label["dimension"] = dim_idx
            label["singular_value"] = round(float(singular_values[dim_idx]), 6)
            dimension_labels.append(label)

        SVD_COMPONENTS = vt
        SVD_SINGULAR_VALUES = singular_values
        DOC_LATENT_MATRIX = doc_latent
        DOC_LATENT_NORMS = doc_norms
        DIMENSION_LABELS = dimension_labels
        SVD_STATUS = {
            "enabled": True,
            "components": int(component_count),
            "message": "SVD index built successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to build SVD index")
        SVD_COMPONENTS = None
        SVD_SINGULAR_VALUES = None
        DOC_LATENT_MATRIX = None
        DOC_LATENT_NORMS = None
        DIMENSION_LABELS = []
        SVD_STATUS = {
            "enabled": False,
            "components": 0,
            "message": f"SVD build failed: {exc}",
        }


def build_query_latent_vector(query_tfidf):
    if SVD_COMPONENTS is None:
        return None, 0.0

    query_matrix = sparse_vector_to_matrix(query_tfidf)
    latent = query_matrix.dot(SVD_COMPONENTS.T)
    latent = np.asarray(latent).reshape(-1)
    latent_norm = float(np.linalg.norm(latent))
    return latent, latent_norm


def cosine_similarity_dense(vec_a, norm_a, vec_b, norm_b):
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def explain_svd_alignment(query_latent, doc_latent, top_n=3):
    alignments = []

    for dim_idx, (query_weight, doc_weight) in enumerate(zip(query_latent, doc_latent)):
        contribution = float(query_weight * doc_weight)
        if contribution <= 0:
            continue

        label = DIMENSION_LABELS[dim_idx] if dim_idx < len(DIMENSION_LABELS) else {
            "positive_terms": [],
            "negative_terms": [],
        }
        direction = "positive" if query_weight > 0 else "negative"
        alignments.append({
            "dimension": dim_idx,
            "direction": direction,
            "query_weight": round(float(query_weight), 6),
            "document_weight": round(float(doc_weight), 6),
            "alignment": round(contribution, 6),
            "positive_terms": label.get("positive_terms", []),
            "negative_terms": label.get("negative_terms", []),
        })

    alignments.sort(key=lambda item: abs(item["alignment"]), reverse=True)
    return alignments[:top_n]


def summarize_query_latent_profile(query_latent, top_n=3):
    if query_latent is None or not len(query_latent):
        return {"positive": [], "negative": []}

    indexed = list(enumerate(query_latent))
    positive = sorted(
        [item for item in indexed if item[1] > 0],
        key=lambda item: item[1],
        reverse=True
    )[:top_n]
    negative = sorted(
        [item for item in indexed if item[1] < 0],
        key=lambda item: item[1]
    )[:top_n]

    def to_payload(items, direction):
        payload = []
        for dim_idx, weight in items:
            label = DIMENSION_LABELS[dim_idx] if dim_idx < len(DIMENSION_LABELS) else {
                "positive_terms": [],
                "negative_terms": [],
            }
            payload.append({
                "dimension": dim_idx,
                "direction": direction,
                "weight": round(float(weight), 6),
                "positive_terms": label.get("positive_terms", []),
                "negative_terms": label.get("negative_terms", []),
            })
        return payload

    return {
        "positive": to_payload(positive, "positive"),
        "negative": to_payload(negative, "negative"),
    }


def build_search_index():
    global SEARCH_DOCS, IDF, VOCAB, VOCAB_TERMS, TERM_INDEX, LATENT_DOC_INDEX, INVERTED_INDEX

    docs = []
    docs.extend(load_campusgroups_documents())
    docs.extend(load_reddit_documents())
    docs.extend(load_generic_csv_documents(osm_csv_path, "osm"))
    docs.extend(load_generic_csv_documents(recs_csv_path, "recs"))
    docs.extend(load_generic_csv_documents(trails_csv_path, "trails"))
    docs.extend(load_generic_csv_documents(dining_csv_path, "dining"))
    official_docs = []
    official_docs.extend(load_generic_csv_documents(libraries_csv_path, "libraries"))
    official_docs.extend(load_generic_csv_documents(cafes_csv_path, "cafes"))
    docs = merge_official_place_metadata(docs, official_docs)

    # Enrich OSM/dining search text with Google Places review content so review
    # keywords (cuisine terms, atmosphere, etc.) are factored into ranking
    for doc in docs:
        if doc["source"] not in {"osm", "dining"}:
            continue
        places_data = get_places_data(doc["id"])
        if not places_data:
            continue
        review_texts = " ".join(r.get("text", "") for r in (places_data.get("reviews") or []))
        if review_texts:
            doc["search_text"] = doc["search_text"] + " " + review_texts

        # Inject Google Place types as tokens so cuisine queries match the actual
        # establishment type, not just incidental review mentions. Snake_case types
        # like "pizza_restaurant" decompose into individual searchable tokens.
        google_types = list(places_data.get("types") or [])
        primary = places_data.get("primary_type")
        if primary and primary not in google_types:
            google_types.append(primary)
        if google_types:
            type_tokens = " ".join(t.replace("_", " ") for t in google_types)
            doc["search_text"] = doc["search_text"] + " " + type_tokens
            doc["google_types"] = google_types
            doc["primary_type"] = primary

        # Boolean attribute flags (servesVegetarianFood, outdoorSeating, etc.)
        # used for dietary/feature hard filters. Not added to search_text — TF-IDF
        # over snake_case attribute names would mostly create noise.
        attrs = places_data.get("attributes")
        if attrs:
            doc["place_attributes"] = attrs

    # Pass docs through standard preprocessing pipeline
    processed = preprocessing.process_documents(docs)
    deduped = processed["docs"]
    
    global PREPROCESSING_STATS
    PREPROCESSING_STATS = processed
    
    print(f"Preprocessing Report:")
    print(f" - Original docs: {processed['original_count']}")
    print(f" - Empty Removed: {processed['empty_removed']}")
    print(f" - Duplicates Removed: {processed['duplicates_removed']}")
    print(f" - Clean Index Size: {processed['cleaned_count']}")

    df_counter = Counter()
    indexed_docs = []

    enriched_docs = []
    for doc in deduped:
        intent_signals, intent_tag_text = derive_intent_signals(doc)
        enriched_doc = dict(doc)
        enriched_doc["_intent_signals"] = intent_signals
        if intent_tag_text:
            enriched_doc["search_text"] = f"{enriched_doc['search_text']} {intent_tag_text}".strip()
        enriched_docs.append(enriched_doc)

    for doc in enriched_docs:
        tokens = tokenize(doc["search_text"])
        token_counts = Counter(tokens)

        if doc["source"] != "reddit":
            for term in token_counts.keys():
                df_counter[term] += 1

        indexed_doc = dict(doc)
        indexed_doc["_tokens"] = tokens
        indexed_doc["_token_counts"] = token_counts
        indexed_docs.append(indexed_doc)

    num_docs = sum(1 for doc in indexed_docs if doc["source"] != "reddit")
    if num_docs == 0:
        num_docs = len(indexed_docs)
    idf_map = compute_idf(num_docs, df_counter)

    for doc in indexed_docs:
        tfidf = compute_tfidf_vector(doc["_token_counts"], idf_map)
        doc["_tfidf"] = tfidf
        doc["_norm"] = vector_norm(tfidf)

    SEARCH_DOCS = indexed_docs
    IDF = idf_map
    VOCAB = set(df_counter.keys())
    VOCAB_TERMS = sorted(VOCAB)
    TERM_INDEX = {term: idx for idx, term in enumerate(VOCAB_TERMS)}
    LATENT_DOC_INDEX = {}

    # Build inverted index: term -> sorted list of doc indices that contain it.
    # Lets search_documents skip zero-scoring docs without scoring them at all.
    inv = defaultdict(list)
    for i, doc in enumerate(indexed_docs):
        for term in doc["_token_counts"]:
            inv[term].append(i)
    INVERTED_INDEX = dict(inv)

    if indexed_docs and VOCAB_TERMS:
        structured_doc_rows = []
        rows = []
        cols = []
        values = []
        for doc_idx, doc in enumerate(indexed_docs):
            if doc["source"] == "reddit":
                continue
            latent_row_idx = len(structured_doc_rows)
            structured_doc_rows.append(doc_idx)
            LATENT_DOC_INDEX[doc_idx] = latent_row_idx
            for term, weight in doc["_tfidf"].items():
                rows.append(latent_row_idx)
                cols.append(TERM_INDEX[term])
                values.append(weight)

        doc_term_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(structured_doc_rows), len(VOCAB_TERMS)),
            dtype=float,
        )
        build_svd_index(doc_term_matrix, VOCAB_TERMS)
    else:
        build_svd_index(csr_matrix((0, 0), dtype=float), [], requested_components=0)

    print(f"Indexed {len(SEARCH_DOCS)} total docs")
    print(f" - CampusGroups: {sum(1 for d in SEARCH_DOCS if d['source'] == 'campusgroups')}")
    print(f" - Reddit: {sum(1 for d in SEARCH_DOCS if d['source'] == 'reddit')}")
    print(f" - OSM: {sum(1 for d in SEARCH_DOCS if d['source'] == 'osm')}")
    print(f" - Recs: {sum(1 for d in SEARCH_DOCS if d['source'] == 'recs')}")
    print(f" - Trails: {sum(1 for d in SEARCH_DOCS if d['source'] == 'trails')}")
    print(f" - Dining: {sum(1 for d in SEARCH_DOCS if d['source'] == 'dining')}")
    print(f" - Libraries: {sum(1 for d in SEARCH_DOCS if d['source'] == 'libraries')}")
    print(f" - Cafes: {sum(1 for d in SEARCH_DOCS if d['source'] == 'cafes')}")
    print(f" - SVD Enabled: {SVD_STATUS['enabled']} ({SVD_STATUS['components']} dimensions)")


def build_query_vector(query):
    # Preprocess query text the same way as documents
    # Use non-aggressive preprocessing initially to preserve intents like "campus" or "club",
    # but still apply intent expansion and restrict to vocabulary
    cleaned_query = preprocessing.normalize_text(query, aggressive=False)
    query_tokens = tokenize(cleaned_query)
    
    # Expand tokens based on intents
    expanded_tokens = expand_query_tokens(query_tokens)
    
    # Restrict to vocabulary for TF-IDF computation
    vocab_tokens = [term for term in expanded_tokens if term in VOCAB]
    query_counts = Counter(vocab_tokens)
    query_tfidf = compute_tfidf_vector(query_counts, IDF)
    query_norm = vector_norm(query_tfidf)
    return query_tfidf, query_norm


def blend_sparse_vectors(primary_vec, secondary_vec, primary_weight=0.75, secondary_weight=0.25):
    blended = {}
    for term, weight in primary_vec.items():
        blended[term] = blended.get(term, 0.0) + (primary_weight * weight)
    for term, weight in secondary_vec.items():
        blended[term] = blended.get(term, 0.0) + (secondary_weight * weight)
    return blended


def cosine_similarity(query_vec, query_norm, doc_vec, doc_norm):
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return dot_product_sparse(query_vec, doc_vec) / (query_norm * doc_norm)


def keyword_fallback_search(
    query,
    top_k=10,
    allowed_sources=None,
    expanded_query=None,
    include_reddit=True,
    area="any",
    future_only=True,
    date_from=None,
    date_to=None,
    query_intents=None,
):
    query_tokens = set(normalized_query_tokens(query, restrict_to_vocab=False))
    expanded_query_tokens = set(expand_query_tokens(list(query_tokens)))
    if expanded_query:
        expanded_query_tokens.update(normalized_query_tokens(expanded_query, restrict_to_vocab=False))
    if not expanded_query_tokens:
        return []

    query_text = normalize_whitespace(query).lower()
    results = []

    for doc in SEARCH_DOCS:
        source = doc.get("source")
        if source == "reddit" and not include_reddit:
            continue
        if allowed_sources and source not in allowed_sources and source != "reddit":
            continue
        if not is_in_area(area, doc):
            continue
        if not passes_temporal_filters(doc, future_only=future_only, date_from=date_from, date_to=date_to):
            continue
        if query_intents and query_intents.get("food") and source in {"osm", "libraries"} and not is_food_relevant_doc(doc):
            continue
        if query_intents and query_intents.get("coffee") and source in {"osm", "dining", "cafes"} and not is_coffee_relevant_doc(doc):
            continue

        title_tokens = set(tokenize(doc.get("title", "")))
        category_tokens = set(tokenize(doc.get("category", "")))
        location_tokens = set(tokenize(doc.get("location", "")))
        description_tokens = set(tokenize(doc.get("description", "")))
        organization_tokens = set(tokenize(doc.get("organization", "")))

        raw_title_overlap = len(query_tokens & title_tokens)
        raw_category_overlap = len(query_tokens & category_tokens)
        raw_location_overlap = len(query_tokens & location_tokens)
        raw_description_overlap = len(query_tokens & description_tokens)
        raw_organization_overlap = len(query_tokens & organization_tokens)

        expansion_only_tokens = expanded_query_tokens - query_tokens
        expanded_title_overlap = len(expansion_only_tokens & title_tokens)
        expanded_category_overlap = len(expansion_only_tokens & category_tokens)
        expanded_location_overlap = len(expansion_only_tokens & location_tokens)
        expanded_description_overlap = len(expansion_only_tokens & description_tokens)
        expanded_organization_overlap = len(expansion_only_tokens & organization_tokens)

        score = (
            raw_title_overlap * 0.65
            + raw_category_overlap * 0.35
            + raw_location_overlap * 0.18
            + raw_description_overlap * 0.14
            + raw_organization_overlap * 0.12
            + expanded_title_overlap * 0.2
            + expanded_category_overlap * 0.14
            + expanded_location_overlap * 0.08
            + expanded_description_overlap * 0.05
            + expanded_organization_overlap * 0.04
        )

        title_text = normalize_whitespace(doc.get("title", "")).lower()
        description_text = normalize_whitespace(doc.get("description", "")).lower()
        category_text = normalize_whitespace(doc.get("category", "")).lower()

        if query_text and query_text in title_text:
            score += 0.45
        elif query_text and query_text in description_text:
            score += 0.2

        if expanded_query_tokens & MOVIE_QUERY_TERMS and (
            "cinema" in title_text
            or "cinema" in category_text
            or "theater" in title_text
            or "theatre" in title_text
        ):
            score += 0.35
            if source != "reddit":
                score += 0.45

        if source in {"dining", "cafes"}:
            score += 0.08
        elif source == "reddit":
            score -= 0.02

        has_raw_match = any([
            raw_title_overlap,
            raw_category_overlap,
            raw_location_overlap,
            raw_description_overlap,
            raw_organization_overlap,
        ])
        if query_tokens and not has_raw_match and query_text not in title_text and query_text not in description_text:
            score -= 0.12

        if score <= 0:
            continue

        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"],
            "organization": doc["organization"],
            "category": doc["category"],
            "location": doc["location"],
            "start_time": doc["start_time"],
            "end_time": doc["end_time"],
            "url": doc["url"],
            "source": doc["source"],
            "doc_type": doc["doc_type"],
            "score": round(score, 6),
            "reddit_snippet": None,
            "search_mode": "keyword_fallback",
            "matched_dimensions": [],
            "lat": get_doc_lat_lon(doc)[0],
            "lon": get_doc_lat_lon(doc)[1],
            "places_data": None,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]

def _enrich_results_with_places_data(results):
    """Attach cached Google Places data (rating, hours, photo, business_status, etc.)
    to osm/dining results, and synthesize a minimal places_data for recs entries
    whose 'website' field is actually an image URL on a Cornell CDN. Mutates in
    place. Used by both the main and keyword_fallback search paths so result
    cards render consistently regardless of which retrieval path was taken.
    """
    PLACES_SOURCES = {"osm", "dining"}
    for result in results:
        if result.get("places_data") or result.get("source") not in PLACES_SOURCES:
            continue
        places_data = get_places_data(result["id"])
        if places_data:
            result["places_data"] = places_data
            if places_data.get("website") and not result.get("url"):
                result["url"] = places_data["website"]
            rating = places_data.get("rating")
            rating_count = places_data.get("rating_count", 0)
            if rating and rating_count and rating_count >= 5:
                confidence = min(rating_count, 100) / 100.0
                rating_bonus = (rating - 3.0) / 2.0 * 0.05 * confidence
                result["score"] = round(result["score"] + rating_bonus, 6)

    for result in results:
        if result.get("source") != "recs" or result.get("places_data"):
            continue
        url = result.get("url") or ""
        if url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            result["places_data"] = {"photo_path": url}
            result["url"] = None


def search_documents(query, top_k=10, source="all", mode="svd", future_only=True, date_from=None, date_to=None, original_query=None, include_reddit=True, area="any"):
    query = query.strip()
    original_query = (original_query or query).strip()
    if not query or not SEARCH_DOCS:
        return [], {"positive": [], "negative": []}, "tfidf"

    source_mapping = {
        "all": {"campusgroups", "osm", "recs", "trails", "dining", "libraries", "cafes"},
        "events": {"campusgroups"},
        "places": {"osm", "libraries", "cafes"},
        "food": {"dining", "osm", "cafes"},
        "outdoors": {"trails", "osm"},
        "fitness": {"recs"},
    }
    allowed_sources = source_mapping.get(source, source_mapping["all"])
    query_tokens = set(normalized_query_tokens(query, restrict_to_vocab=False))
    query_intents = infer_query_intents(query_tokens)
    if source == "all" and query_intents["food"] and not query_intents["social"]:
        # Specific food-item queries like "chicken" or "ramen" behave much
        # better when we keep retrieval focused on dining/place sources.
        allowed_sources = {"dining", "osm", "cafes"}
    elif source == "all" and query_intents["fitness"] and not query_intents["social"]:
        # Sport/activity queries like "play basketball" should stay focused on
        # recreation facilities and sports locations rather than generic campus places.
        allowed_sources = {"recs", "osm"}

    primary_query_vec, primary_query_norm = build_query_vector(original_query)
    expanded_query_vec, expanded_query_norm = build_query_vector(query)
    has_primary_vocab_overlap = bool(primary_query_vec)
    has_expanded_vocab_overlap = bool(expanded_query_vec)

    if query != original_query:
        query_vec = blend_sparse_vectors(primary_query_vec, expanded_query_vec)
        query_norm = vector_norm(query_vec)
    else:
        query_vec = primary_query_vec
        query_norm = primary_query_norm

    if query_norm == 0.0 or not (has_primary_vocab_overlap or has_expanded_vocab_overlap):
        fallback_results = keyword_fallback_search(
            original_query,
            top_k=top_k,
            allowed_sources=allowed_sources,
            expanded_query=query if query != original_query else None,
            include_reddit=include_reddit,
            area=area,
            future_only=future_only,
            date_from=date_from,
            date_to=date_to,
            query_intents=query_intents,
        )
        if fallback_results:
            _enrich_results_with_places_data(fallback_results)
            return fallback_results, {"positive": [], "negative": []}, "keyword_fallback"

    query_terms = set(query_vec.keys())
    short_keyword_query = len(query_terms) <= 2

    use_svd = mode == "svd" and SVD_STATUS["enabled"]
    query_latent = None
    query_latent_norm = 0.0
    query_profile = {"positive": [], "negative": []}

    # Short keyword searches such as "pizza" or "zumba" behave better with
    # lexical ranking than with latent similarity, which can overgeneralize.
    if short_keyword_query:
        use_svd = False

    if use_svd:
        query_latent, query_latent_norm = build_query_latent_vector(query_vec)
        if query_latent is None or query_latent_norm == 0.0:
            use_svd = False
        else:
            query_profile = summarize_query_latent_profile(query_latent)


    structured_candidates = []
    reddit_results = []

    # Use the inverted index to get only docs that share at least one query term.
    # Any doc with no term overlap scores 0 anyway, so we can safely skip them.
    candidate_doc_indices = set()
    for term in query_terms:
        candidate_doc_indices.update(INVERTED_INDEX.get(term, []))

    for doc_idx in candidate_doc_indices:
        doc = SEARCH_DOCS[doc_idx]
        if not is_in_area(area, doc):
            continue

        if query_intents.get("food") and doc.get("source") in {"osm", "libraries"} and not is_food_relevant_doc(doc):
            continue
        if query_intents.get("coffee") and doc.get("source") in {"osm", "dining", "cafes"} and not is_coffee_relevant_doc(doc):
            continue
        if not is_cuisine_match(query_tokens, doc):
            continue
        if not is_dietary_match(query_tokens, doc):
            continue

        primary_lexical_score = cosine_similarity(primary_query_vec, primary_query_norm, doc["_tfidf"], doc["_norm"])
        expanded_lexical_score = cosine_similarity(expanded_query_vec, expanded_query_norm, doc["_tfidf"], doc["_norm"])
        if query != original_query:
            lexical_score = (0.75 * primary_lexical_score) + (0.25 * expanded_lexical_score)
        else:
            lexical_score = primary_lexical_score

        overlap_terms = query_terms.intersection(doc["_token_counts"].keys())

        if doc["source"] == "reddit":
            if not include_reddit:
                continue
            if lexical_score > 0:
                reddit_results.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "organization": doc["organization"],
                    "category": doc["category"],
                    "location": doc["location"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "url": doc["url"],
                    "source": doc["source"],
                    "doc_type": doc["doc_type"],
                    "score": round(lexical_score, 6),
                    "reddit_snippet": None,
                    "search_mode": "tfidf",
                    "matched_dimensions": [],
                    "lat": None,
                    "lon": None,
                    "places_data": None,
                })
            continue

        if doc["source"] not in allowed_sources or lexical_score <= 0:
            continue

        lexical_score += compute_metadata_adjustment(query_tokens, doc)
        lexical_score += compute_overlap_boost(query_tokens, doc)
        lexical_score += compute_intent_alignment_adjustment(query_intents, doc)

        if lexical_score <= 0:
            continue

        overlap_weight = sum(IDF.get(term, 0.0) for term in overlap_terms)
        structured_candidates.append({
            "doc_idx": doc_idx,
            "doc": doc,
            "lexical_score": lexical_score,
            "overlap_count": len(overlap_terms),
            "overlap_weight": overlap_weight,
        })

    top_lexical_score = max((candidate["lexical_score"] for candidate in structured_candidates), default=0.0)
    if use_svd and (len(structured_candidates) < max(top_k, 5) or top_lexical_score < 0.3):
        use_svd = False
        query_profile = {"positive": [], "negative": []}

    if use_svd:
        # Treat SVD as a reranker over a lexical shortlist instead of a
        # standalone retriever. This keeps the semantic signal useful without
        # letting broad latent clusters dominate obviously relevant matches.
        candidate_pool_size = max(top_k * 5, 25)
        structured_candidates.sort(
            key=lambda item: (
                item["lexical_score"],
                item["overlap_weight"],
                item["overlap_count"],
            ),
            reverse=True,
        )
        rerank_candidates = structured_candidates[:candidate_pool_size]
        structured_results = []

        for candidate in rerank_candidates:
            doc_idx = candidate["doc_idx"]
            doc = candidate["doc"]
            lexical_score = candidate["lexical_score"]
            svd_score = 0.0
            matched_dimensions = []
            doc_lat, doc_lon = get_doc_lat_lon(doc)

            if doc_idx in LATENT_DOC_INDEX:
                latent_idx = LATENT_DOC_INDEX[doc_idx]
                svd_score = max(0.0, cosine_similarity_dense(
                    query_latent,
                    query_latent_norm,
                    DOC_LATENT_MATRIX[latent_idx],
                    float(DOC_LATENT_NORMS[latent_idx]),
                ))
                if svd_score > 0:
                    matched_dimensions = explain_svd_alignment(query_latent, DOC_LATENT_MATRIX[latent_idx])

            score = (0.75 * lexical_score) + (0.25 * svd_score)
            structured_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "organization": doc["organization"],
                "category": doc["category"],
                "location": doc["location"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "url": doc["url"],
                "source": doc["source"],
                "doc_type": doc["doc_type"],
                "score": round(score, 6),
                "reddit_snippet": None,
                "search_mode": "svd",
                "matched_dimensions": matched_dimensions,
                "lat": doc_lat,
                "lon": doc_lon,
                "places_data": None,
            })
    else:
        structured_results = [
            {
                "id": candidate["doc"]["id"],
                "title": candidate["doc"]["title"],
                "description": candidate["doc"]["description"],
                "organization": candidate["doc"]["organization"],
                "category": candidate["doc"]["category"],
                "location": candidate["doc"]["location"],
                "start_time": candidate["doc"]["start_time"],
                "end_time": candidate["doc"]["end_time"],
                "url": candidate["doc"]["url"],
                "source": candidate["doc"]["source"],
                "doc_type": candidate["doc"]["doc_type"],
                "score": round(candidate["lexical_score"], 6),
                "reddit_snippet": None,
                "search_mode": "tfidf",
                "matched_dimensions": [],
                "lat": get_doc_lat_lon(candidate["doc"])[0],
                "lon": get_doc_lat_lon(candidate["doc"])[1],
                "places_data": None,
            }
            for candidate in structured_candidates
        ]

    structured_results.sort(key=lambda x: x["score"], reverse=True)
    reddit_results.sort(key=lambda x: x["score"], reverse=True)

    structured_results = [
        r for r in structured_results
        if passes_temporal_filters(r, future_only=future_only, date_from=date_from, date_to=date_to)
    ]

    top_structured = structured_results[:top_k]
    
    if include_reddit:
        for s_res in top_structured:
            best_reddit = None
            best_r_score = 0
            s_tokens = set(tokenize(s_res["title"] + " " + s_res["category"]))
            
            for r_res in reddit_results[:15]:
                r_tokens = set(tokenize(r_res["title"] + " " + r_res["description"]))
                overlap = len(s_tokens.intersection(r_tokens))
                if overlap > best_r_score:
                    best_r_score = overlap
                    best_reddit = r_res
                    
            if best_reddit and best_r_score >= 1:
                snippet = best_reddit["description"]
                s_res["reddit_snippet"] = f"Community mention: \"{snippet}\""
                s_res["score"] = round(s_res["score"] + 0.05, 6)

    # Remove near-duplicate cards so users see diverse options.
    top_structured.sort(key=lambda x: x["score"], reverse=True)
    
    deduped = []
    seen_keys = set()
    for result in top_structured:
        title_key = normalize_for_key(result.get("title", ""))
        location_key = normalize_for_key(result.get("location", ""))
        dedupe_key = f"{title_key}|{location_key}"
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(result)
        if len(deduped) >= top_k:
            break

    _enrich_results_with_places_data(deduped)

    deduped.sort(key=lambda x: x["score"], reverse=True)

    if not deduped and reddit_results:
        return reddit_results[:top_k], query_profile, ("svd" if use_svd else "tfidf")

    return deduped, query_profile, ("svd" if use_svd else "tfidf")

def reformulate_query_for_ir(query):
    """
    RAG Step 1: Transform a natural-language user query into retrieval-
    optimized keywords for the IR system.  Returns the rewritten string,
    or the original query unchanged on failure.
    """
    client, api_key = get_llm_client()
    if not api_key or client is None:
        return query
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query reformulator for SideQuest, an Ithaca and Cornell activity discovery app. "
                "Translate the user's conversational request into a concise set of retrieval keywords. "
                "The corpus contains: OpenStreetMap places (restaurants, parks, shops, cafes), "
                "Cornell Dining halls, Ithaca hiking trails and gorges, CampusGroups student events, "
                "Cornell libraries, downtown Ithaca cafes, and Cornell recreation/fitness facilities. "
                "Extract core entities and intents. Expand vague concepts with corpus-relevant synonyms "
                "(e.g., 'nature' -> 'trail gorge park preserve', 'food' -> 'dining restaurant cafe', "
                "'study spot' -> 'library cafe quiet wifi', 'workout' -> 'gym fitness recreation'). "
                "IMPORTANT: For specific food items (e.g., 'pizza', 'ramen', 'sushi', 'burger', 'coffee'), "
                "keep the specific term and add 'restaurant' but do NOT add generic 'dining' or 'cafe' — "
                "those terms match campus dining halls, not specific food establishments. "
                "Return ONLY space-separated keywords. No sentences, no quotes, no punctuation, no numbering."
            ),
        },
        {
            "role": "user",
            "content": f"Rewrite this query for better search retrieval: {query}",
        },
    ]

    try:
        response = client.chat(messages)
        modified_query = (response.get("content") or "").strip()
        if not modified_query:
            return query
        # Strip stray quotes and punctuation
        modified_query = modified_query.replace('"', '').replace("'", "").strip()
        # If the LLM returned a sentence (contains a colon preamble), extract the tail
        if ":" in modified_query:
            modified_query = modified_query.split(":")[-1].strip()
        # Cap at 30 words to prevent prompt-injection or runaway output
        words = modified_query.split()
        if len(words) > 30:
            modified_query = " ".join(words[:30])
        return modified_query
    except Exception as exc:
        logger.exception("Failed to reformulate query")
    return query


def build_result_context(results):
    context_blocks = []

    for index, result in enumerate(results, start=1):
        lines = [
            f"Result {index}",
            f"Title: {result['title']}",
            f"Source: {result['source']}",
            f"Category: {result['category']}",
        ]

        desc = result.get("description", "")
        if desc:
            if len(desc) > 300:
                cut = desc[:300].rfind(" ")
                desc = desc[:cut] + "..." if cut > 0 else desc[:300] + "..."
            lines.append(f"Snippet: {desc}")
        if result.get("location"):
            lines.append(f"Location: {result['location']}")
            
        places_data = result.get("places_data")
        if places_data:
            if places_data.get("rating"):
                lines.append(f"Rating: {places_data['rating']} ({places_data.get('rating_count', 0)} reviews)")
            if places_data.get("price_level"):
                lines.append(f"Price: {places_data['price_level']}")
            if places_data.get("hours") and isinstance(places_data["hours"], list):
                lines.append(f"Hours: {', '.join(places_data['hours'][:2])}...")

        context_blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(context_blocks)


def synthesize_search_answer(query, results, rewritten_query=None):
    """RAG Step 3: Generate a grounded answer from both the user's original
    query and the retrieved IR results. Optionally receives a rewritten
    interpretation so the LLM can better understand the user's intent."""
    if not results:
        return {
            "answer": "I couldn't find relevant results for that query in the current dataset.",
            "warning": None,
        }

    client, api_key = get_llm_client()
    if not api_key:
        return {
            "answer": format_non_llm_summary(query, results),
            "warning": "LLM synthesis is unavailable because SPARK_API_KEY is not set. Showing a rules-based summary instead.",
        }
    if client is None:
        return {
            "answer": format_non_llm_summary(query, results),
            "warning": "LLM synthesis is unavailable because the Spark client dependency is not installed in this environment. Showing a rules-based summary instead.",
        }
    context_text = build_result_context(results[:8])

    # Build query section with both original and rewritten queries
    query_section = f"User query: {query}"
    if rewritten_query and rewritten_query.lower() != query.lower():
        query_section += f"\nShort interpretation of the user's intent: {rewritten_query}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are the recommendation assistant for SideQuest, an Ithaca activity search app. "
                "Your job is to turn ranked search results into a short, trustworthy recommendation. "
                "Use only the retrieved results provided in the context. Do not invent facts, dates, locations, prices, or availability that are not present in the results. "
                "Treat higher-scored results as stronger matches, but do not claim certainty from score alone. "
                "Prioritize the activities that best match the semantic intent of the user's query, and mention specific titles early. "
                "Discard any retrieved results which are not semantically relevant to the user's query. "
                "When helpful, compare 2-3 strong options and explain why they fit. "
                "If important details are missing or the results are only loosely related, say that clearly. "
                "Keep the tone helpful and concise. "
                "Return response in markdown formatting with short paragraphs and bullets."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{query_section}\n\n"
                f"Retrieved results:\n{context_text}\n\n"
                "Write a recommendation grounded in these results.\n"
                "Requirements:\n"
                "- Write naturally and conversationally. Cite sources seamlessly (e.g., 'Cafe Dewitt is a great option').\n"
                "- Start with the best overall recommendation.\n"
                "- Briefly explain why it matches the query using only details from the results.\n"
                "- If multiple results refer to the exact same physical facility or building (e.g., 'Noyes' and 'Noyes Basketball Court'), aggregate them into a single recommendation.\n"
                "- If there are useful alternatives, mention at most two additional options.\n"
                "- Mention logistics like location, time, or URL only when they are present and relevant.\n"
                "- Do not mention items that are not in the retrieved results.\n"
                "- If the results are mixed or weak, say that and suggest the closest matches instead.\n"
                "- Keep the answer to one short paragraph."
            ),
        },
    ]

    try:
        response = client.chat(messages)
        answer = (response.get("content") or "").strip()
        if not answer:
            return {
                "answer": None,
                "warning": "The LLM did not return a synthesized answer.",
            }
        return {
            "answer": answer,
            "warning": None,
        }
    except Exception as exc:
        logger.exception("Failed to synthesize search answer")
        return {
            "answer": None,
            "warning": f"LLM synthesis failed: {exc}",
        }


# -----------------------------
# API routes
# -----------------------------
@app.get("/api/config")
def api_config():
    llm_available, llm_reason = get_llm_status()
    return jsonify({
        "llm_available": llm_available,
        "llm_reason": llm_reason,
    })


@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    raw_query = request.args.get("raw_q", "").strip() or query
    source = request.args.get("source", "all").strip().lower()
    mode = request.args.get("mode", "svd").strip().lower()
    top_k_raw = request.args.get("top_k", "10")
    future_only = request.args.get("future_only", "true").strip().lower() != "false"
    include_summary = request.args.get("include_summary", "").strip().lower() in {"1", "true", "yes"}
    include_reddit = request.args.get("reddit", "true").strip().lower() != "false"
    area = request.args.get("area", "any").strip().lower()

    date_from = None
    date_to = None
    date_warnings = []
    raw_from = request.args.get("date_from", "").strip()
    raw_to = request.args.get("date_to", "").strip()
    if raw_from:
        try:
            date_from = datetime.strptime(raw_from, "%Y-%m-%d")
        except ValueError:
            logger.warning("Invalid date_from param: %s", raw_from)
            date_warnings.append(f"Ignored invalid date_from value: '{raw_from}'. Expected YYYY-MM-DD.")
    if raw_to:
        try:
            date_to = datetime.strptime(raw_to, "%Y-%m-%d")
        except ValueError:
            logger.warning("Invalid date_to param: %s", raw_to)
            date_warnings.append(f"Ignored invalid date_to value: '{raw_to}'. Expected YYYY-MM-DD.")

    if mode not in {"svd", "tfidf"}:
        mode = "svd"

    try:
        top_k = max(1, min(int(top_k_raw), 50))
    except ValueError:
        top_k = 10

    if not query:
        return jsonify({
            "query": "",
            "rewritten_query": None,
            "source": source,
            "mode": mode,
            "count": 0,
            "results": [],
            "message": "Pass a query with ?q=your+query"
        }), 400

    search_query = query
    rewritten_query = None
    if include_summary:
        rewritten_query = reformulate_query_for_ir(raw_query)
        if not rewritten_query or rewritten_query.lower() == raw_query.lower():
            rewritten_query = None

    results, query_profile, effective_mode = search_documents(
        search_query,
        top_k=top_k,
        source=source,
        mode=mode,
        future_only=future_only,
        date_from=date_from,
        date_to=date_to,
        original_query=query,
        include_reddit=include_reddit,
        area=area
    )

    synthesis = {"answer": None, "warning": None}
    if include_summary:
        synthesis = synthesize_search_answer(raw_query, results, rewritten_query=rewritten_query)

    return jsonify({
        "query": query,
        "rewritten_query": rewritten_query,
        "source": source,
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "include_summary": include_summary,
        "count": len(results),
        "results": results,
        "query_latent_profile": query_profile,
        "svd_status": SVD_STATUS,
        "answer": synthesis["answer"],
        "answer_warning": synthesis["warning"],
        "warnings": date_warnings if date_warnings else None,
    })


@app.post("/api/search/reindex")
def api_reindex():
    build_search_index()
    return jsonify({
        "message": "Search index rebuilt successfully",
        "preprocessing_stats": PREPROCESSING_STATS,
        "svd_status": SVD_STATUS,
        "sample_dimensions": DIMENSION_LABELS[:5],
        "indexed_documents": len(SEARCH_DOCS),
        "campusgroups_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "campusgroups"),
        "reddit_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "reddit"),
        "osm_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "osm"),
        "recs_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "recs"),
        "trails_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "trails"),
        "dining_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "dining"),
        "libraries_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "libraries"),
        "cafes_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "cafes"),
    })


@app.get("/api/search/health")
def api_search_health():
    return jsonify({
        "indexed_documents": len(SEARCH_DOCS),
        "svd_status": SVD_STATUS,
        "sample_dimensions": DIMENSION_LABELS[:5],
        "campusgroups_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "campusgroups"),
        "reddit_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "reddit"),
        "osm_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "osm"),
        "recs_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "recs"),
        "trails_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "trails"),
        "dining_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "dining"),
        "libraries_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "libraries"),
        "cafes_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "cafes"),
        "vocab_size": len(VOCAB),
        "preprocessing": {
            "original_count": PREPROCESSING_STATS.get("original_count"),
            "empty_removed": PREPROCESSING_STATS.get("empty_removed"),
            "duplicates_removed": PREPROCESSING_STATS.get("duplicates_removed"),
            "cleaned_count": PREPROCESSING_STATS.get("cleaned_count"),
        },
        "campusgroups_json_found": os.path.exists(campusgroups_json_path),
        "reddit_conversations_found": os.path.exists(reddit_conversations_path),
        "reddit_utterances_found": os.path.exists(reddit_utterances_path),
        "reddit_corpus_found": os.path.exists(reddit_corpus_path),
        "osm_csv_found": os.path.exists(osm_csv_path),
        "recs_csv_found": os.path.exists(recs_csv_path),
        "trails_csv_found": os.path.exists(trails_csv_path),
        "dining_csv_found": os.path.exists(dining_csv_path),
        "libraries_csv_found": os.path.exists(libraries_csv_path),
        "cafes_csv_found": os.path.exists(cafes_csv_path),
    })


# -----------------------------
# Results chat
# -----------------------------
@app.post("/api/chat/results")
def api_chat_results():
    from flask import stream_with_context, Response
    import json as _json

    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    results = data.get("results") or []
    history = data.get("history") or []
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    client, api_key = get_llm_client()
    if not api_key:
        return jsonify({"error": "AI chat is unavailable right now."}), 503
    if client is None:
        return jsonify({"error": "AI chat is unavailable right now."}), 503

    context_parts = []
    for i, result in enumerate(results[:15], 1):
        lines = [f"{i}. {result.get('title', 'Unknown')}"]
        if result.get("category"):
            lines.append(f"   Category: {result['category']}")
        if result.get("location"):
            lines.append(f"   Location: {result['location']}")
        if result.get("start_time"):
            lines.append(f"   Time: {result['start_time']}")
        desc = (result.get("description") or "")[:180]
        if desc:
            lines.append(f"   About: {desc}")
        places = result.get("places_data") or {}
        bs = places.get("business_status")
        if bs == "CLOSED_PERMANENTLY":
            lines.append("   STATUS: PERMANENTLY CLOSED — do not recommend.")
        elif bs == "CLOSED_TEMPORARILY":
            lines.append("   STATUS: Temporarily closed — flag this if recommending.")
        if places.get("rating") is not None:
            rating_line = f"   Rating: {places['rating']}"
            if places.get("rating_count"):
                rating_line += f" ({places['rating_count']} reviews)"
            lines.append(rating_line)
        if places.get("price_level"):
            lines.append(f"   Price: {places['price_level']}")
        if places.get("hours"):
            lines.append(f"   Hours: {places['hours'][0]}")
        if places.get("reviews"):
            review_lines = []
            for r in places["reviews"][:3]:
                text = (r.get("text") or "")[:150]
                if text:
                    stars = f"{'★' * int(r['rating'])}" if r.get("rating") else ""
                    review_lines.append(f"     - {r.get('author', 'Reviewer')} {stars}: {text}")
            if review_lines:
                lines.append("   Reviews:\n" + "\n".join(review_lines))
        snippet = (result.get("reddit_snippet") or "")[:120]
        if snippet:
            lines.append(f"   Reddit: {snippet}")
        context_parts.append("\n".join(lines))

    context_text = "\n\n".join(context_parts) or "No results available."

    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = (
        f"You are a sharp, friendly local guide for Ithaca, NY. Today is {today}.\n\n"
        "The user is browsing SideQuest search results, listed below. Answer using the data "
        "provided — don't invent ratings, hours, or facts not present in the listing.\n\n"
        "Rules:\n"
        "• NEVER recommend a place marked PERMANENTLY CLOSED. If the user asks about one, "
        "say it's closed and suggest a similar open option from the list.\n"
        "• Mention 'Temporarily closed' if recommending one.\n"
        "• Ratings come from Google (out of 5). Price is shown in $ symbols ($ to $$$$).\n"
        "• If something isn't in the data, say 'the results don't say' rather than guessing.\n"
        "• Be concise and conversational. Skip preambles like 'Great question!'.\n"
        "• When comparing places, lead with the recommendation, then a one-line reason.\n"
        "• Use markdown formatting (bold names, bullet lists) when it makes the answer easier to scan.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Prior conversation: include up to the last 6 turns so follow-up questions
    # ("which one is cheapest?" → "is that one open Saturday?") keep their thread.
    for turn in history[-6:]:
        role = "user" if turn.get("isUser") else "assistant"
        text = (turn.get("text") or "").strip()
        if text:
            messages.append({"role": role, "content": text})

    messages.append({
        "role": "user",
        "content": f"Current search results:\n\n{context_text}\n\nQuestion: {user_message}",
    })

    def generate():
        try:
            for chunk in client.chat(messages, stream=True):
                if chunk.get("content"):
                    yield f"data: {_json.dumps({'content': chunk['content']})}\n\n"
        except Exception as exc:
            logger.error(f"Results chat streaming error: {exc}")
            yield f"data: {_json.dumps({'error': 'Streaming error occurred'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -----------------------------
# Place-specific chat
# -----------------------------
@app.post("/api/chat/place")
def api_chat_place():
    from flask import stream_with_context, Response
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    place = data.get("place") or {}
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    client, api_key = get_llm_client()
    if not api_key:
        return jsonify({"error": "AI chat is unavailable right now."}), 503
    if client is None:
        return jsonify({"error": "AI chat is unavailable right now."}), 503

    context_parts = []
    if place.get("title"):
        context_parts.append(f"Name: {place['title']}")
    if place.get("description"):
        context_parts.append(f"Description: {place['description']}")
    if place.get("category"):
        context_parts.append(f"Category: {place['category']}")
    if place.get("location"):
        context_parts.append(f"Location: {place['location']}")
    if place.get("start_time"):
        context_parts.append(f"Event time: {place['start_time']}")
    if place.get("organization"):
        context_parts.append(f"Organizer: {place['organization']}")
    if place.get("reddit_snippet"):
        context_parts.append(f"Community insight: {place['reddit_snippet']}")

    places_data = place.get("places_data") or {}
    if places_data.get("rating") is not None:
        rating_str = f"Rating: {places_data['rating']}"
        if places_data.get("rating_count"):
            rating_str += f" ({places_data['rating_count']} reviews)"
        context_parts.append(rating_str)
    if places_data.get("price_level"):
        context_parts.append(f"Price level: {places_data['price_level']}")
    if places_data.get("phone"):
        context_parts.append(f"Phone: {places_data['phone']}")
    if places_data.get("hours"):
        context_parts.append("Hours:\n" + "\n".join(places_data["hours"]))
    if places_data.get("reviews"):
        reviews_text = "\n".join(
            f"- {r['author']} ({'★' * int(r['rating']) if r.get('rating') else 'no rating'}): {r['text']}"
            for r in places_data["reviews"][:5]
            if r.get("text")
        )
        if reviews_text:
            context_parts.append(f"Reviews:\n{reviews_text}")

    context_text = "\n\n".join(context_parts) or "No detailed information available."
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful local guide for Ithaca, NY answering questions about a specific place or event. "
                "Answer only based on the provided information. If something isn't covered, say so honestly and briefly. "
                "Keep answers concise and practical."
            ),
        },
        {
            "role": "user",
            "content": f"Place information:\n\n{context_text}\n\nQuestion: {user_message}",
        },
    ]

    import json as _json

    def generate():
        try:
            for chunk in client.chat(messages, stream=True):
                if chunk.get("content"):
                    yield f"data: {_json.dumps({'content': chunk['content']})}\n\n"
        except Exception as exc:
            logger.error(f"Place chat streaming error: {exc}")
            yield f"data: {_json.dumps({'error': 'Streaming error occurred'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -----------------------------
# Frontend serving
# -----------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)

    index_file = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(app.static_folder, "index.html")

    return jsonify({
        "message": "Frontend build not found. API is running."
    })


# -----------------------------
# Startup
# -----------------------------
build_search_index()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
