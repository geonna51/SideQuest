"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that performs LLM-driven RAG.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
import requests as _requests
from flask import request, jsonify, Response, stream_with_context

try:
    from infosci_spark_client import LLMClient
except ImportError:
    class LLMClient:
        _ENDPOINT = "https://4300spark.infosci.cornell.edu/api/chat"

        def __init__(self, api_key: str):
            self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        def chat(self, messages, stream=False):
            payload = {"messages": messages, "stream": stream}
            if stream:
                return self._stream(payload)
            resp = _requests.post(self._ENDPOINT, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] if data.get("choices") else ""
            return {"content": content}

        def _stream(self, payload):
            resp = _requests.post(self._ENDPOINT, json=payload, headers=self._headers, stream=True, timeout=None)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                if line_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line_str)
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        yield {"content": delta.get("content", "")}
                        if data["choices"][0].get("finish_reason"):
                            break
                except (json.JSONDecodeError, Exception):
                    continue

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        if LLMClient is None:
            return jsonify({"error": "LLM client unavailable in this environment"}), 503
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY") or os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            episodes = json_search(search_term or "Kardashian")
            context_text = "\n\n---\n\n".join(
                f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
                for ep in episodes
            ) or "No matching episodes found."
            messages = [
                {"role": "system", "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided."},
                {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Keeping Up with the Kardashians questions."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def register_place_chat_route(app):
    """Register the /api/chat/place SSE endpoint for place-specific Q&A."""

    @app.route("/api/chat/place", methods=["POST"])
    def place_chat():
        if LLMClient is None:
            return jsonify({"error": "LLM client unavailable in this environment"}), 503
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        place = data.get("place") or {}
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY") or os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)

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

        def generate():
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Place chat streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
