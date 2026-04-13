import json
import math
import os
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing
import logging
import numpy as np
from datetime import datetime, timezone

from collections import Counter, defaultdict
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from infosci_spark_client import LLMClient
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

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


def first_nonempty(record, *keys):
    for key in keys:
        if key in record:
            value = normalize_whitespace(record.get(key))
            if value:
                return value
    return ""


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", normalize_whitespace(text).lower())


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
def normalize_campusgroup_record(record, idx):
    title = first_nonempty(record, "title", "name", "event_name", "summary")
    description = first_nonempty(record, "description", "descr", "details", "body", "content", "about")
    organization = first_nonempty(record, "organization", "org", "group", "club", "host", "organization_name")
    category = first_nonempty(record, "category", "categories", "tags", "tag", "event_type", "type")
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
            
            description_parts = []
            if db_desc: description_parts.append(db_desc)
            if category: description_parts.append(f"Category: {category}")
            if subcategory: description_parts.append(f"Type: {subcategory}")
            if address: description_parts.append(f"Address: {address}")
            description = " | ".join(description_parts)
            
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
    global SEARCH_DOCS, IDF, VOCAB, VOCAB_TERMS, TERM_INDEX, LATENT_DOC_INDEX

    docs = []
    docs.extend(load_campusgroups_documents())
    docs.extend(load_reddit_documents())
    docs.extend(load_generic_csv_documents(osm_csv_path, "osm"))
    docs.extend(load_generic_csv_documents(recs_csv_path, "recs"))
    docs.extend(load_generic_csv_documents(trails_csv_path, "trails"))
    docs.extend(load_generic_csv_documents(dining_csv_path, "dining"))

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

    for doc in deduped:
        tokens = tokenize(doc["search_text"])
        token_counts = Counter(tokens)

        for term in token_counts.keys():
            df_counter[term] += 1

        indexed_doc = dict(doc)
        indexed_doc["_tokens"] = tokens
        indexed_doc["_token_counts"] = token_counts
        indexed_docs.append(indexed_doc)

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
    print(f" - SVD Enabled: {SVD_STATUS['enabled']} ({SVD_STATUS['components']} dimensions)")


def build_query_vector(query):
    # Use non-aggressive preprocessing for queries so user intent words like
    # "campus", "event", "club" are not stripped and can match document titles.
    cleaned_query = preprocessing.normalize_text(query, aggressive=False)
    query_tokens = tokenize(cleaned_query)
    query_counts = Counter(term for term in query_tokens if term in VOCAB)
    query_tfidf = compute_tfidf_vector(query_counts, IDF)
    query_norm = vector_norm(query_tfidf)
    return query_tfidf, query_norm


def cosine_similarity(query_vec, query_norm, doc_vec, doc_norm):
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return dot_product_sparse(query_vec, doc_vec) / (query_norm * doc_norm)


def search_documents(query, top_k=10, source="all", mode="svd", future_only=True, date_from=None, date_to=None):
    query = query.strip()
    if not query or not SEARCH_DOCS:
        return [], {"positive": [], "negative": []}, "tfidf"

    query_vec, query_norm = build_query_vector(query)
    if query_norm == 0.0:
        return [], {"positive": [], "negative": []}, "tfidf"

    use_svd = mode == "svd" and SVD_STATUS["enabled"]
    query_latent = None
    query_latent_norm = 0.0
    query_profile = {"positive": [], "negative": []}

    if use_svd:
        query_latent, query_latent_norm = build_query_latent_vector(query_vec)
        if query_latent is None or query_latent_norm == 0.0:
            use_svd = False
        else:
            query_profile = summarize_query_latent_profile(query_latent)

    source_mapping = {
        "all":      {"campusgroups", "osm", "recs", "trails", "dining"},
        "events":   {"campusgroups"},
        "places":   {"osm"},
        "food":     {"dining", "osm"},
        "outdoors": {"trails", "osm"},
        "fitness":  {"recs"},
    }
    allowed_sources = source_mapping.get(source, source_mapping["all"])

    structured_results = []
    reddit_results = []

    for doc_idx, doc in enumerate(SEARCH_DOCS):
        lexical_score = cosine_similarity(query_vec, query_norm, doc["_tfidf"], doc["_norm"])

        if use_svd and doc_idx in LATENT_DOC_INDEX:
            latent_idx = LATENT_DOC_INDEX[doc_idx]
            svd_score = cosine_similarity_dense(
                query_latent,
                query_latent_norm,
                DOC_LATENT_MATRIX[latent_idx],
                float(DOC_LATENT_NORMS[latent_idx]),
            )
            matched_dimensions = explain_svd_alignment(query_latent, DOC_LATENT_MATRIX[latent_idx])
            score = (0.7 * svd_score) + (0.3 * lexical_score)
        else:
            score = lexical_score
            matched_dimensions = []

        if score <= 0:
            continue
            
        result_dict = {
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
            "search_mode": "svd" if use_svd else "tfidf",
            "matched_dimensions": matched_dimensions,
        }

        if doc["source"] == "reddit":
            reddit_results.append(result_dict)
        elif doc["source"] in allowed_sources:
            structured_results.append(result_dict)

    structured_results.sort(key=lambda x: x["score"], reverse=True)
    reddit_results.sort(key=lambda x: x["score"], reverse=True)

    if future_only:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        def keep_result(r):
            if r["source"] != "campusgroups":
                return True
            event_date = parse_event_date(r["start_time"])
            return event_date is None or event_date >= today
        structured_results = [r for r in structured_results if keep_result(r)]

    if date_from or date_to:
        def in_date_range(r):
            if r["source"] != "campusgroups":
                return True
            event_date = parse_event_date(r["start_time"])
            if event_date is None:
                return True
            if date_from and event_date < date_from:
                return False
            if date_to and event_date > date_to:
                return False
            return True
        structured_results = [r for r in structured_results if in_date_range(r)]

    top_structured = structured_results[:top_k]
    
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
            snippet = best_reddit["description"][:120] + "..." if len(best_reddit["description"]) > 120 else best_reddit["description"]
            s_res["reddit_snippet"] = f"Community mention: \"{snippet}\""
            s_res["score"] = round(s_res["score"] + 0.05, 6)
            
    top_structured.sort(key=lambda x: x["score"], reverse=True)
    return top_structured, query_profile, ("svd" if use_svd else "tfidf")


def build_result_context(results):
    context_blocks = []

    for index, result in enumerate(results, start=1):
        lines = [
            f"Result {index}",
            f"Title: {result['title']}",
            f"Source: {result['source']}",
            f"Type: {result['doc_type']}",
            f"Score: {result['score']}",
        ]

        if result["description"]:
            lines.append(f"Description: {result['description']}")
        if result["organization"]:
            lines.append(f"Organization: {result['organization']}")
        if result["category"]:
            lines.append(f"Category: {result['category']}")
        if result["location"]:
            lines.append(f"Location: {result['location']}")
        if result["start_time"]:
            lines.append(f"Start Time: {result['start_time']}")
        if result["end_time"]:
            lines.append(f"End Time: {result['end_time']}")
        if result["url"]:
            lines.append(f"URL: {result['url']}")

        context_blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(context_blocks)


def synthesize_search_answer(query, results):
    if not results:
        return {
            "answer": "I couldn't find relevant results for that query in the current dataset.",
            "warning": None,
        }

    api_key = os.getenv("API_KEY")
    if not api_key:
        return {
            "answer": None,
            "warning": "LLM synthesis is unavailable because API_KEY is not set.",
        }

    client = LLMClient(api_key=api_key)
    context_text = build_result_context(results[:8])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for SideQuest, an Ithaca activity search app. "
                "Answer the user's query using only the retrieved search results provided to you. "
                "Do not invent facts. If the results are incomplete or ambiguous, say so clearly. "
                "Prefer concise recommendations and mention specific titles when useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {query}\n\n"
                f"Retrieved results:\n{context_text}\n\n"
                "Synthesize a helpful answer grounded in these results."
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
@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    source = request.args.get("source", "all").strip().lower()
    mode = request.args.get("mode", "svd").strip().lower()
    top_k_raw = request.args.get("top_k", "10")
    future_only = request.args.get("future_only", "true").strip().lower() != "false"

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
            "source": source,
            "mode": mode,
            "count": 0,
            "results": [],
            "message": "Pass a query with ?q=your+query"
        }), 400

    results, query_profile, effective_mode = search_documents(
        query,
        top_k=top_k,
        source=source,
        mode=mode,
        future_only=future_only,
        date_from=date_from,
        date_to=date_to,
    )

    synthesis = synthesize_search_answer(query, results)

    return jsonify({
        "query": query,
        "source": source,
        "requested_mode": mode,
        "effective_mode": effective_mode,
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
    })


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
