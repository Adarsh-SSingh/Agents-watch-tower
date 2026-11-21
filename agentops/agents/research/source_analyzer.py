from urllib.parse import urlparse
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MIN_CONTENT_LENGTH = 500
MIN_USABLE_CONTENT_LENGTH = 100
MAX_EVIDENCE_PER_SOURCE = 5
MAX_CONTEXT_CHARS = 1200
MIN_SENTENCE_CHARS = 35

LOW_QUALITY_DOMAINS = {
    "example.com",
    "example.org",
    "example.net"
}

# Same stop-word corpus as crawler.py, instead of a hand-written list.
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", (text or "")).strip()

    if not text:
        return []

    return [
        part.strip(" \t\r\n-•")
        for part in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])",
            text
        )
        if len(part.strip()) >= MIN_SENTENCE_CHARS
    ]


def analyze_url(url: str) -> dict:
    try:
        parsed = urlparse(url or "")
        domain = (parsed.hostname or "").lower()

        if not domain:
            raise ValueError("Missing hostname")

        return {
            "valid_url": parsed.scheme.lower() in ("http", "https"),
            "https": parsed.scheme.lower() == "https",
            "domain": domain,
            "is_placeholder": domain in LOW_QUALITY_DOMAINS
        }

    except Exception:
        return {
            "valid_url": False,
            "https": False,
            "domain": "",
            "is_placeholder": False
        }


def calculate_relevance(topic: str, content: str) -> float:
    topic_words = tokenize(topic)
    content_words = tokenize(content)

    if not topic_words or not content_words:
        return 0.0

    overlap = topic_words & content_words

    return round(
        len(overlap) / len(topic_words),
        3
    )


def build_research_terms(
    topic: str,
    research_questions: list = None
) -> set:

    text = topic or ""

    if research_questions:
        text += " " + " ".join(
            str(question)
            for question in research_questions
        )

    return tokenize(text)


def extract_evidence_snippets(
    content: str,
    topic: str,
    research_questions: list = None,
    max_snippets: int = MAX_EVIDENCE_PER_SOURCE
) -> list:

    terms = build_research_terms(
        topic,
        research_questions
    )

    sentences = split_sentences(content)

    if not terms or not sentences:
        return []

    ranked = []

    for index, sentence in enumerate(sentences):

        sentence_terms = tokenize(sentence)
        overlap = terms & sentence_terms

        if not overlap:
            continue

        coverage = len(overlap) / max(1, len(terms))
        density = len(overlap) / max(1, len(sentence_terms))

        score = coverage * 0.7 + density * 0.3

        # Preserve one sentence before and after the matching sentence.
        start = max(0, index - 1)
        end = min(len(sentences), index + 2)

        context = " ".join(
            sentences[start:end]
        )[:MAX_CONTEXT_CHARS]

        ranked.append({
            "text": sentence,
            "context": context,
            "matched_terms": sorted(overlap),
            "match_score": round(score, 4),
            "match_count": len(overlap),
            "sentence_index": index
        })

    ranked.sort(
        key=lambda item: (
            item["match_count"],
            item["match_score"]
        ),
        reverse=True
    )

    selected = []
    seen_contexts = set()

    for item in ranked:

        normalized = re.sub(
            r"\W+",
            " ",
            item["context"].lower()
        ).strip()

        if normalized in seen_contexts:
            continue

        selected.append(item)
        seen_contexts.add(normalized)

        if len(selected) >= max_snippets:
            break

    return selected


def analyze_source(
    page: dict,
    topic: str,
    research_questions: list = None
) -> dict:

    url = page.get("url", "")
    content = page.get("content", "") or ""

    url_info = analyze_url(url)

    relevance = calculate_relevance(
        topic,
        content
    )

    content_length = len(content.strip())

    snippets = extract_evidence_snippets(
        content,
        topic,
        research_questions
    )

    warnings = []

    if not url_info["valid_url"]:
        warnings.append("Invalid or unsupported URL")

    if not url_info["https"]:
        warnings.append("HTTPS is not being used")

    if url_info["is_placeholder"]:
        warnings.append("Placeholder domain detected")

    if content_length < MIN_CONTENT_LENGTH:
        warnings.append("Short webpage content")

    if relevance < 0.2:
        warnings.append("Low lexical keyword relevance")

    if not snippets:
        warnings.append(
            "No topic-matching evidence snippets extracted"
        )

    # This is a technical heuristic, not a credibility rating.
    quality_score = 100

    if not url_info["valid_url"]:
        quality_score -= 40

    if not url_info["https"]:
        quality_score -= 10

    if url_info["is_placeholder"]:
        quality_score -= 40

    if content_length < MIN_CONTENT_LENGTH:
        quality_score -= 15

    if relevance < 0.2:
        quality_score -= 20

    if not snippets:
        quality_score -= 25

    return {
        "url": url,
        "domain": url_info["domain"],
        "valid_url": url_info["valid_url"],
        "https": url_info["https"],
        "is_placeholder": url_info["is_placeholder"],
        "content_length": content_length,
        "relevance_score": relevance,
        "quality_score": max(0, min(100, quality_score)),
        "evidence_count": len(snippets),
        "warnings": warnings
    }


def remove_duplicate_sources(pages: list) -> list:

    unique_pages = []
    seen = set()

    for page in pages:

        try:
            url = page.get("url", "").strip()
            info = analyze_url(url)

            if not info["valid_url"]:
                continue

            parsed = urlparse(url)

            normalized = (
                f"{parsed.scheme.lower()}://"
                f"{parsed.netloc.lower()}"
                f"{parsed.path.rstrip('/')}"
            )

            if parsed.query:
                normalized += f"?{parsed.query}"

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_pages.append(page)

        except Exception:
            continue

    return unique_pages


def analyze_sources(
    pages: list,
    topic: str,
    research_questions: list = None
) -> dict:

    unique_pages = remove_duplicate_sources(pages)

    analyzed = []
    usable = []

    for page in unique_pages:

        content = page.get("content", "") or ""

        analysis = analyze_source(
            page,
            topic,
            research_questions
        )

        snippets = extract_evidence_snippets(
            content,
            topic,
            research_questions
        )

        enriched = {
            **page,
            "source_id": None,
            "evidence_snippets": snippets,
            "source_analysis": analysis
        }

        analyzed.append(enriched)

        if (
            analysis["valid_url"]
            and not analysis["is_placeholder"]
            and analysis["content_length"] >= MIN_USABLE_CONTENT_LENGTH
            and analysis["evidence_count"] > 0
        ):
            usable.append(enriched)

    usable.sort(
        key=lambda source: (
            source["source_analysis"]["evidence_count"],
            source["source_analysis"]["relevance_score"],
            source["source_analysis"]["quality_score"]
        ),
        reverse=True
    )

    for source_id, source in enumerate(usable, start=1):
        source["source_id"] = source_id

    ids_by_url = {
        source.get("url"): source["source_id"]
        for source in usable
    }

    for source in analyzed:
        source["source_id"] = ids_by_url.get(
            source.get("url")
        )

    return {
        "topic": topic,
        "total_sources": len(pages),
        "unique_sources": len(unique_pages),
        "usable_sources": len(usable),
        "rejected_sources": len(unique_pages) - len(usable),
        "sources": analyzed,
        "usable_pages": usable
    }