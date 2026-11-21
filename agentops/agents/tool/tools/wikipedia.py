import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
TIMEOUT = 10


def wikipedia_lookup(topic: str):
    if not topic or not topic.strip():
        raise ValueError("Search topic cannot be empty.")

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": topic.strip(),
                "gsrlimit": 1,
                "prop": "extracts|info",
                "exintro": True,
                "explaintext": True,
                "inprop": "url",
                "format": "json"
            },
            headers={
                "User-Agent": "AgentOpsIntelligence/1.0 (educational project)"
            },
            timeout=TIMEOUT
        )

        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})

        if not pages:
            raise ValueError(f"No Wikipedia results found for: {topic}")

        page = next(iter(pages.values()))

        return {
            "topic": topic,
            "title": page.get("title"),
            "summary": page.get("extract", ""),
            "url": page.get("fullurl"),
            "page_id": page.get("pageid")
        }

    except requests.Timeout as exc:
        raise RuntimeError("Wikipedia API request timed out.") from exc

    except requests.RequestException as exc:
        raise RuntimeError(f"Wikipedia API request failed: {exc}") from exc