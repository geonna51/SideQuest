import re
import html
import datetime

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "own", "same", "so", "than", "too", "very", "s", "t", 
    "can", "will", "just", "don", "should", "now"
}

# Explicitly ensure negations and constraints are NOT in stopwords
STOPWORDS -= {"not", "no", "without", "only", "under", "over"}

FILLER_WORDS = {
    "join", "come", "attend", "event", "everyone", "anyone", "students", 
    "student", "university", "campus", "club", "organization", "hosted", 
    "located", "click", "learn", "register", "information", "details", "tbd",
    "more", "here", "available", "description", "null"
}

PHRASES = {
    "live music": "live_music",
    "free food": "free_food",
    "coffee shop": "coffee_shop",
    "study spot": "study_spot",
    "hiking trail": "hiking_trail",
    "group fitness": "group_fitness",
    "open mic": "open_mic",
    "ice cream": "ice_cream",
    "state park": "state_park",
    "farmers market": "farmers_market"
}

def basic_lemmatize(word):
    if word in {"this", "is", "as", "was", "has", "previous", "campus", "fitness", "class", "always", "plus", "status", "express", "process", "series"}:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3 and word[-3] in ['s', 'x', 'z', 'h']:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word

def strip_html_and_markdown(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'[*_`#]', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text, aggressive=False):
    text = strip_html_and_markdown(text).lower()
    
    for phrase, token in PHRASES.items():
        text = text.replace(phrase, token)
        
    words = re.findall(r"[a-z0-9_]+", text)
    
    cleaned = []
    for w in words:
        if w in STOPWORDS:
            continue
        if aggressive and w in FILLER_WORDS:
            continue
        
        lemmatized = basic_lemmatize(w)
        cleaned.append(lemmatized)
        
    return " ".join(cleaned)

def standardize_time(time_str):
    if not time_str:
        return ""
    return str(time_str).strip()

def detect_boilerplate(text):
    text_lower = str(text).lower().strip()
    boilers = [
        "tbd", "no description available", "click for details", 
        "description coming soon", "null", "none"
    ]
    if text_lower in boilers:
        return ""
    return text_lower

def clean_record(doc):
    title = str(doc.get("title", "")).strip()
    desc = strip_html_and_markdown(doc.get("description", ""))
    desc = detect_boilerplate(desc)
    
    if doc.get("source") == "campusgroups":
        desc = re.sub(r'(?i)(register here|registration link|click here|rsvp).*$', '', desc)
    elif doc.get("source") == "reddit":
        desc = re.sub(r'(?i)(edit:|deleted|\[deleted\]|\[removed\]).*', '', desc)
        
    tags = str(doc.get("category", "")).strip().lower()
    loc = str(doc.get("location", "")).strip()
    if loc.lower().startswith("private location"):
        loc = ""
        
    org = str(doc.get("organization", "")).strip()
    
    search_parts = [
        normalize_text(title, aggressive=False),
        normalize_text(desc, aggressive=True),
        normalize_text(tags, aggressive=False),
        normalize_text(loc, aggressive=False),
        normalize_text(org, aggressive=False)
    ]
    
    search_text = " ".join(p for p in search_parts if p)
    
    url = str(doc.get("url", "")).strip()
    if doc.get("source") == "campusgroups" and url.startswith("/"):
        url = "https://cornell.campusgroups.com" + url
    
    doc["title"] = title
    doc["description"] = desc
    doc["category"] = tags
    doc["location"] = loc
    doc["organization"] = org
    doc["search_text"] = search_text
    doc["url"] = url
    
    return doc

def is_meaningful(doc):
    title = doc.get("title", "").strip()
    desc = doc.get("description", "").strip()
    search_text = doc.get("search_text", "").strip()
    
    if not title and not desc:
        return False
    if len(search_text.split()) < 2:  # too short
        return False
    return True

def deduplicate_records(docs):
    seen = {}
    deduped = []
    duplicates_removed = 0
    
    for doc in docs:
        title_key = normalize_text(doc.get("title", ""))
        time_key = doc.get("start_time", "")[:10]  # rough date grouping
        loc_key = normalize_text(doc.get("location", ""))[:15]
        
        sig = f"{title_key}|{time_key}|{loc_key}"
        
        if sig in seen:
            duplicates_removed += 1
            existing = seen[sig]
            # Merge logic: keep the richer description
            if len(doc.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = doc["description"]
                existing["search_text"] = doc["search_text"]
        else:
            seen[sig] = doc
            deduped.append(doc)
            
    return deduped, duplicates_removed

def process_documents(raw_docs):
    original_count = len(raw_docs)
    
    cleaned = []
    empty_removed = 0
    
    for doc in raw_docs:
        c_doc = clean_record(doc)
        if is_meaningful(c_doc):
            cleaned.append(c_doc)
        else:
            empty_removed += 1
            
    deduped, duplicates_removed = deduplicate_records(cleaned)
    
    return {
        "docs": deduped,
        "original_count": original_count,
        "cleaned_count": len(deduped),
        "empty_removed": empty_removed,
        "duplicates_removed": duplicates_removed
    }
