import re
import math
import html as html_lib

from collections import deque, Counter
from urllib.parse import (
    urlparse,
    unquote,
    parse_qsl,
    urlencode,
    urlunparse
)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from agentops.agents.research.tools import (
    fetch_webpage,
    extract_page_content,
    extract_page_links,
    normalize_url
)


# ============================================================
# CRAWLER CONFIGURATION
# ============================================================

MAX_PAGES = 20
MAX_DEPTH = 2
MAX_LINKS_PER_PAGE = 5
MAX_DISCOVERED_LINKS = 30

MIN_RELEVANCE_SCORE = 2.0
MIN_CONTENT_LENGTH = 300
MIN_CONTENT_MATCHES = 2
MIN_CONTENT_COVERAGE = 0.12

MAX_CONTENT_LENGTH = 12000

# Use an established stop-word corpus instead of maintaining
# our own manually curated stop-word dictionary.
STOP_WORDS = set(ENGLISH_STOP_WORDS)

TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid",
    "ref", "ref_src", "source",
    "mc_cid", "mc_eid"
}

# These are URL navigation/path categories, not topic keywords.
GENERIC_PATHS = {
    "login", "signin", "signup", "register",
    "contact", "about", "pricing", "careers",
    "privacy", "terms", "cookies", "newsletter",
    "home", "index", "search", "cart", "checkout"
}

LOW_VALUE_PATHS = {
    "authors", "author", "category", "categories",
    "tag", "tags", "events", "event",
    "webinars", "webinar", "partners", "partner",
    "customers", "customer", "jobs", "press",
    "media", "community"
}

COMMERCIAL_PATHS = {
    "product", "products", "platform",
    "solutions", "solution", "services",
    "marketplace", "pricing", "demo"
}

CONTENT_PATH_HINTS = {
    "blog", "article", "articles", "research",
    "paper", "papers", "guide", "guides",
    "documentation", "docs", "tutorial",
    "tutorials", "whitepaper", "whitepapers",
    "engineering", "technology", "insights"
}


# ============================================================
# URL NORMALIZATION
# ============================================================

def canonicalize_url(url: str) -> str:
    """Normalize URL and remove common tracking parameters."""

    normalized = normalize_url(url)
    parsed = urlparse(normalized)

    query_params = parse_qsl(
        parsed.query,
        keep_blank_values=True
    )

    filtered_params = [
        (key, value)
        for key, value in query_params
        if key.lower() not in TRACKING_PARAMS
        and not key.lower().startswith("utm_")
    ]

    filtered_params.sort()

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        "",
        urlencode(filtered_params, doseq=True),
        ""
    ))


def get_domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text: str) -> set:
    """
    Normalize text into tokens.

    Stop words are obtained from sklearn's built-in corpus.
    Two-character technical terms are retained.
    """

    if not text:
        return set()

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word for word in words
        if len(word) >= 2
        and word not in STOP_WORDS
        and not word.isdigit()
    }


def get_topic_terms(topic: str) -> set:
    return tokenize(topic)


def get_url_path_tokens(url: str) -> set:
    path = unquote(urlparse(url).path)

    return tokenize(
        path.replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )


def extract_page_title(page_html: str) -> str:
    """Extract the HTML title without another dependency."""

    if not page_html:
        return ""

    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        page_html,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return ""

    title = re.sub(r"<[^>]+>", " ", match.group(1))
    title = html_lib.unescape(title)

    return re.sub(r"\s+", " ", title).strip()


# ============================================================
# DYNAMIC KEYWORD WEIGHTING
# ============================================================

def calculate_keyword_weights(
    topic_terms: set,
    documents: list
) -> dict:
    """
    Calculate dynamic keyword importance using document frequency.

    A term appearing in many candidate documents receives lower
    specificity. A term appearing in fewer documents receives
    higher specificity.

    No manually maintained generic keyword dictionary is used.
    """

    if not topic_terms:
        return {}

    tokenized_documents = [
        tokenize(document)
        for document in documents
        if document and str(document).strip()
    ]

    if not tokenized_documents:
        return {
            term: 1.0
            for term in topic_terms
        }

    document_count = len(tokenized_documents)

    document_frequency = Counter()

    for document_tokens in tokenized_documents:
        document_frequency.update(
            document_tokens.intersection(topic_terms)
        )

    # Inverse document frequency.
    raw_weights = {}

    for term in topic_terms:
        frequency = document_frequency.get(term, 0)

        idf = math.log(
            (document_count + 1) / (frequency + 1)
        ) + 1

        raw_weights[term] = idf

    # Normalize weights between 0 and 1.
    max_weight = max(raw_weights.values(), default=1.0)
    min_weight = min(raw_weights.values(), default=1.0)

    if max_weight == min_weight:
        return {
            term: 1.0
            for term in topic_terms
        }

    return {
        term: (
            (weight - min_weight)
            / (max_weight - min_weight)
        )
        for term, weight in raw_weights.items()
    }


def calculate_weighted_match_score(
    matched_terms: set,
    keyword_weights: dict
) -> float:
    """Calculate the importance-weighted score of matched terms."""

    return sum(
        keyword_weights.get(term, 1.0)
        for term in matched_terms
    )


# ============================================================
# PATH FILTERING
# ============================================================

def get_path_parts(url: str) -> set:
    """
    Return complete path segments and individual segment tokens.
    """

    path = unquote(urlparse(url).path.lower())
    parts = set()

    for segment in path.strip("/").split("/"):
        if not segment:
            continue

        parts.add(segment)

        parts.update(
            tokenize(
                segment.replace("-", " ").replace("_", " ")
            )
        )

    return parts


def is_generic_path(url: str) -> bool:
    """Reject generic pages and archive-style paths."""

    parsed = urlparse(url)

    path_parts = [
        part.lower()
        for part in parsed.path.strip("/").split("/")
        if part
    ]

    if not path_parts:
        return True

    if any(part in GENERIC_PATHS for part in path_parts):
        return True

    if any(
        re.fullmatch(r"(page|pages|archive)-?\d+", part)
        for part in path_parts
    ):
        return True

    return False


def is_low_value_path(url: str) -> bool:
    return bool(
        get_path_parts(url).intersection(LOW_VALUE_PATHS)
    )


def is_commercial_path(url: str) -> bool:
    return bool(
        get_path_parts(url).intersection(COMMERCIAL_PATHS)
    )


# ============================================================
# LINK RELEVANCE
# ============================================================

def calculate_link_relevance(
    anchor_text: str,
    link_url: str,
    topic: str,
    keyword_weights: dict = None
) -> float:
    """
    Score links using dynamically weighted topic keywords.

    Anchor text has higher importance than URL path matches.
    """

    topic_terms = get_topic_terms(topic)

    if not topic_terms:
        return 0.0

    if keyword_weights is None:
        keyword_weights = {
            term: 1.0
            for term in topic_terms
        }

    anchor_terms = tokenize(anchor_text)
    url_terms = get_url_path_tokens(link_url)

    anchor_matches = topic_terms.intersection(anchor_terms)
    url_matches = topic_terms.intersection(url_terms)

    anchor_score = calculate_weighted_match_score(
        anchor_matches,
        keyword_weights
    )

    url_score = calculate_weighted_match_score(
        url_matches,
        keyword_weights
    )

    score = anchor_score * 3 + url_score * 2

    if anchor_matches or url_matches:
        if get_path_parts(link_url).intersection(CONTENT_PATH_HINTS):
            score += 1

    return round(score, 3)


def is_relevant_link(
    anchor_text: str,
    link_url: str,
    topic: str
) -> bool:

    if is_generic_path(link_url):
        return False

    return (
        calculate_link_relevance(
            anchor_text,
            link_url,
            topic
        ) >= MIN_RELEVANCE_SCORE
    )


# ============================================================
# LINK SELECTION
# ============================================================

def select_relevant_links(
    links: list,
    current_url: str,
    topic: str,
    visited: set,
    queued: set,
    keyword_weights: dict = None
) -> list:
    """
    Filter and rank links before adding them to BFS.

    Uses the same corpus-derived keyword weights throughout
    the complete crawl. It does not recalculate weights per page.
    """

    selected = []
    current_domain = get_domain(current_url)

    topic_terms = get_topic_terms(topic)

    candidates = []

    # First collect valid same-domain candidate links.
    for link in links:

        link_url = link.get("url")

        if not link_url:
            continue

        try:
            link_url = canonicalize_url(link_url)
        except (ValueError, TypeError):
            continue

        if link_url in visited or link_url in queued:
            continue

        if get_domain(link_url) != current_domain:
            continue

        if is_generic_path(link_url):
            continue

        anchor_text = link.get("anchor_text", "")

        candidates.append({
            "url": link_url,
            "anchor_text": anchor_text
        })

    if keyword_weights is None:
        keyword_weights = {
            term: 1.0
            for term in topic_terms
        }

    for item in candidates:

        link_url = item["url"]
        anchor_text = item["anchor_text"]

        anchor_terms = tokenize(anchor_text)
        url_terms = get_url_path_tokens(link_url)

        topic_matches = (
            topic_terms.intersection(anchor_terms)
            | topic_terms.intersection(url_terms)
        )

        if not topic_matches:
            continue

        score = calculate_link_relevance(
            anchor_text,
            link_url,
            topic,
            keyword_weights
        )

        low_value = is_low_value_path(link_url)
        commercial = is_commercial_path(link_url)

        # Low-value pages need a stronger direct match.
        if low_value and score < MIN_RELEVANCE_SCORE * 2:
            print(
                f"[CRAWLER] Link rejected: {link_url} "
                f"| Reason: Low-value path "
                f"| Score: {score}"
            )
            continue

        # Commercial pages need stronger keyword relevance.
        if commercial and score < MIN_RELEVANCE_SCORE * 2:
            print(
                f"[CRAWLER] Link rejected: {link_url} "
                f"| Reason: Weak commercial relevance "
                f"| Score: {score}"
            )
            continue

        if score < MIN_RELEVANCE_SCORE:
            continue

        selected.append({
            "url": link_url,
            "score": score,
            "matched_terms": sorted(topic_matches)
        })

    selected.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return selected[:MAX_LINKS_PER_PAGE]


def get_url_path_tokens_text(url: str) -> str:
    """Convert URL path to text for keyword analysis."""

    path = unquote(urlparse(url).path)

    return (
        path.replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )


# ============================================================
# PAGE CONTENT RELEVANCE
# ============================================================

def evaluate_page_relevance(
    title: str,
    content: str,
    topic: str,
    keyword_weights: dict = None
) -> dict:
    """
    Validate actual page content against the research topic.

    Keyword matching is deterministic and dynamically weighted.
    This is lexical relevance, not semantic understanding.
    """

    topic_terms = get_topic_terms(topic)

    if not topic_terms:
        return {
            "relevant": False,
            "score": 0.0,
            "matched_terms": [],
            "coverage": 0.0,
            "reason": "No topic keywords"
        }

    if keyword_weights is None:
        keyword_weights = {
            term: 1.0
            for term in topic_terms
        }

    title_terms = tokenize(title)
    content_terms = tokenize(content)

    title_matches = topic_terms.intersection(title_terms)
    content_matches = topic_terms.intersection(content_terms)

    matched_terms = title_matches | content_matches

    coverage = len(matched_terms) / len(topic_terms)

    title_score = calculate_weighted_match_score(
        title_matches,
        keyword_weights
    )

    content_score = calculate_weighted_match_score(
        content_matches,
        keyword_weights
    )

    score = content_score + title_score * 2

    weighted_coverage = (
        calculate_weighted_match_score(
            matched_terms,
            keyword_weights
        )
        / max(sum(keyword_weights.values()), 0.001)
    )

    required_matches = min(
        len(topic_terms),
        max(
            MIN_CONTENT_MATCHES,
            math.ceil(len(topic_terms) * MIN_CONTENT_COVERAGE)
        )
    )

    relevant = (
        len(matched_terms) >= required_matches
        and coverage >= MIN_CONTENT_COVERAGE
        and weighted_coverage >= MIN_CONTENT_COVERAGE
        and len(content.strip()) >= MIN_CONTENT_LENGTH
    )

    reason = (
        "Topic relevance passed"
        if relevant
        else "Insufficient keyword coverage or content length"
    )

    return {
        "relevant": relevant,
        "score": round(score, 3),
        "matched_terms": sorted(matched_terms),
        "title_matches": sorted(title_matches),
        "content_matches": sorted(content_matches),
        "coverage": round(coverage, 3),
        "weighted_coverage": round(weighted_coverage, 3),
        "reason": reason
    }


# ============================================================
# MULTI-HOP CRAWLER
# ============================================================

def crawl_research_sources(
    start_urls: list,
    topic: str,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    keyword_documents: list = None
) -> dict:
    """
    Breadth-first multi-hop crawler.

    Depth 0 = Starting search URLs.
    Depth 1 = Relevant links from starting URLs.
    Depth 2 = Relevant links from depth 1 pages.

    Starting URLs are preserved.
    Child pages must pass content relevance validation.
    """

    queue = deque()
    visited = set()
    queued = set()

    pages = []
    failed_urls = []
    rejected_pages = []

    # Calculate keyword weights once from the stable search-result
    # corpus supplied by the research pipeline.
    topic_terms = get_topic_terms(topic)
    keyword_documents = keyword_documents or []

    keyword_weights = calculate_keyword_weights(
        topic_terms,
        keyword_documents
    )

    print(
        f"[CRAWLER] Keyword corpus: {len(keyword_documents)} documents"
    )
    print(
        "[CRAWLER] Keyword weights: "
        f"{ {term: round(weight, 3) for term, weight in keyword_weights.items()} }"
    )

    # --------------------------------------------------------
    # Initialize Starting URLs
    # --------------------------------------------------------

    for url in start_urls:

        try:
            normalized = canonicalize_url(url)

            if normalized not in visited and normalized not in queued:
                queue.append((normalized, 0, True))
                queued.add(normalized)

        except (ValueError, TypeError):
            failed_urls.append({
                "url": url,
                "error": "Invalid starting URL"
            })

    # --------------------------------------------------------
    # BFS Crawling
    # --------------------------------------------------------

    while queue and len(pages) < max_pages:

        current_url, depth, is_start_url = queue.popleft()

        queued.discard(current_url)

        if current_url in visited:
            continue

        visited.add(current_url)

        print(
            f"\n[CRAWLER] Fetching: {current_url} "
            f"| Depth: {depth}"
        )

        page = fetch_webpage(current_url)

        if not page.get("success"):
            failed_urls.append({
                "url": current_url,
                "error": page.get("error", "Unknown error")
            })

            print(
                f"[CRAWLER] Fetch failed: {current_url} "
                f"| {page.get('error', 'Unknown error')}"
            )
            continue

        # ----------------------------------------------------
        # Normalize Final URL
        # ----------------------------------------------------

        final_url = page.get("url", current_url)

        try:
            final_url = canonicalize_url(final_url)
        except (ValueError, TypeError):
            failed_urls.append({
                "url": final_url,
                "error": "Invalid final URL"
            })
            continue

        # Avoid duplicate pages after redirects.
        if final_url in visited and final_url != current_url:
            continue

        visited.add(final_url)

        # ----------------------------------------------------
        # Extract Page Content
        # ----------------------------------------------------

        page_html = page.get("html", "")

        content = extract_page_content(page_html) or ""
        title = extract_page_title(page_html)

        content = content.strip()

        # ----------------------------------------------------
        # Validate Discovered Pages
        # ----------------------------------------------------

        relevance = evaluate_page_relevance(
            title=title,
            content=content,
            topic=topic,
            keyword_weights=keyword_weights
        )

        if not is_start_url and not relevance["relevant"]:

            rejected_pages.append({
                "url": final_url,
                "title": title,
                "depth": depth,
                "relevance_score": relevance["score"],
                "matched_terms": relevance["matched_terms"],
                "coverage": relevance["coverage"],
                "weighted_coverage": relevance["weighted_coverage"],
                "reason": relevance["reason"]
            })

            print(
                f"[CRAWLER] Rejected: {final_url} "
                f"| Score: {relevance['score']} "
                f"| Matches: {relevance['matched_terms']} "
                f"| Coverage: {relevance['coverage']} "
                f"| Weighted coverage: {relevance['weighted_coverage']} "
                f"| Reason: {relevance['reason']}"
            )

            continue

        # ----------------------------------------------------
        # Store Page
        # ----------------------------------------------------

        page_data = {
            "url": final_url,
            "domain": get_domain(final_url),
            "title": title,
            "depth": depth,
            "content": content[:MAX_CONTENT_LENGTH],
            "size_bytes": page.get("size_bytes", 0),
            "relevance": relevance
        }

        pages.append(page_data)

        print(
            f"[CRAWLER] Accepted: {final_url} "
            f"| Pages: {len(pages)}/{max_pages} "
            f"| Score: {relevance['score']} "
            f"| Matches: {relevance['matched_terms']} "
            f"| Coverage: {relevance['coverage']}"
        )

        # ----------------------------------------------------
        # Stop at Maximum Depth
        # ----------------------------------------------------

        if depth >= max_depth:
            continue

        # ----------------------------------------------------
        # Discover Candidate Links
        # ----------------------------------------------------

        links = extract_page_links(
            page_html,
            final_url,
            max_links=MAX_DISCOVERED_LINKS
        )

        relevant_links = select_relevant_links(
            links=links,
            current_url=final_url,
            topic=topic,
            visited=visited,
            queued=queued,
            keyword_weights=keyword_weights
        )

        # ----------------------------------------------------
        # Add Relevant Links to BFS
        # ----------------------------------------------------

        for item in relevant_links:

            link_url = item["url"]

            if link_url in visited or link_url in queued:
                continue

            queue.append((link_url, depth + 1, False))
            queued.add(link_url)

            print(
                f"[CRAWLER] Queued: {link_url} "
                f"| Score: {item['score']} "
                f"| Matches: {item['matched_terms']}"
            )

    # ========================================================
    # FINAL CRAWLER STATISTICS
    # ========================================================

    total_attempted = (
        len(pages)
        + len(failed_urls)
        + len(rejected_pages)
    )

    print("\n========== CRAWLER SUMMARY ==========")
    print(f"Topic: {topic}")
    print(f"Pages collected: {len(pages)}")
    print(f"Pages failed: {len(failed_urls)}")
    print(f"Pages rejected: {len(rejected_pages)}")
    print(f"Total attempted: {total_attempted}")
    print(f"Unprocessed queue: {len(queue)}")
    print("=====================================\n")

    return {
        "topic": topic,
        "total_pages": len(pages),
        "total_failed": len(failed_urls),
        "total_rejected": len(rejected_pages),
        "total_attempted": total_attempted,
        "pages": pages,
        "failed_urls": failed_urls,
        "rejected_pages": rejected_pages,
        "max_depth": max_depth,
        "max_pages": max_pages,
        "keyword_weights": keyword_weights,
        "keyword_corpus_size": len(keyword_documents)
    }