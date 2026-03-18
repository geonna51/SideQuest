import json
import math
import os
import re
from collections import Counter, defaultdict

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

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

# -----------------------------
# Flask app
# -----------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(project_root, "frontend", "dist"),
    static_url_path=""
)
CORS(app)

# -----------------------------
# Search index globals
# -----------------------------
SEARCH_DOCS = []
IDF = {}
VOCAB = set()


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


def first_nonempty(record, *keys):
    for key in keys:
        if key in record:
            value = normalize_whitespace(record.get(key))
            if value:
                return value
    return ""


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_whitespace(text).lower())


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


def build_search_index():
    global SEARCH_DOCS, IDF, VOCAB

    docs = []
    docs.extend(load_campusgroups_documents())
    docs.extend(load_reddit_documents())

    # de-dupe by source + title + time
    deduped = []
    seen = set()
    for doc in docs:
        key = (
            doc["source"].strip().lower(),
            doc["title"].strip().lower(),
            doc["start_time"].strip().lower()
        )
        if key not in seen:
            seen.add(key)
            deduped.append(doc)

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

    print(f"Indexed {len(SEARCH_DOCS)} total docs")
    print(f" - CampusGroups: {sum(1 for d in SEARCH_DOCS if d['source'] == 'campusgroups')}")
    print(f" - Reddit: {sum(1 for d in SEARCH_DOCS if d['source'] == 'reddit')}")


def build_query_vector(query):
    query_tokens = tokenize(query)
    query_counts = Counter(term for term in query_tokens if term in VOCAB)
    query_tfidf = compute_tfidf_vector(query_counts, IDF)
    query_norm = vector_norm(query_tfidf)
    return query_tfidf, query_norm


def cosine_similarity(query_vec, query_norm, doc_vec, doc_norm):
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return dot_product_sparse(query_vec, doc_vec) / (query_norm * doc_norm)


def search_documents(query, top_k=10, source="all"):
    query = query.strip()
    if not query or not SEARCH_DOCS:
        return []

    query_vec, query_norm = build_query_vector(query)
    if query_norm == 0.0:
        return []

    allowed_sources = {"all", "campusgroups", "reddit"}
    if source not in allowed_sources:
        source = "all"

    results = []

    for doc in SEARCH_DOCS:
        if source != "all" and doc["source"] != source:
            continue

        score = cosine_similarity(query_vec, query_norm, doc["_tfidf"], doc["_norm"])
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
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# -----------------------------
# API routes
# -----------------------------
@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    source = request.args.get("source", "all").strip().lower()
    top_k_raw = request.args.get("top_k", "10")

    try:
        top_k = max(1, min(int(top_k_raw), 50))
    except ValueError:
        top_k = 10

    if not query:
        return jsonify({
            "query": "",
            "source": source,
            "count": 0,
            "results": [],
            "message": "Pass a query with ?q=your+query"
        }), 400

    results = search_documents(query, top_k=top_k, source=source)

    return jsonify({
        "query": query,
        "source": source,
        "count": len(results),
        "results": results
    })


@app.post("/api/search/reindex")
def api_reindex():
    build_search_index()
    return jsonify({
        "message": "Search index rebuilt successfully",
        "indexed_documents": len(SEARCH_DOCS),
        "campusgroups_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "campusgroups"),
        "reddit_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "reddit"),
    })


@app.get("/api/search/health")
def api_search_health():
    return jsonify({
        "indexed_documents": len(SEARCH_DOCS),
        "campusgroups_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "campusgroups"),
        "reddit_documents": sum(1 for d in SEARCH_DOCS if d["source"] == "reddit"),
        "vocab_size": len(VOCAB),
        "campusgroups_json_found": os.path.exists(campusgroups_json_path),
        "reddit_conversations_found": os.path.exists(reddit_conversations_path),
        "reddit_utterances_found": os.path.exists(reddit_utterances_path),
        "reddit_corpus_found": os.path.exists(reddit_corpus_path),
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