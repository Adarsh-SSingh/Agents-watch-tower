import os
import json
import socket
import ipaddress
from urllib.parse import urlparse, urljoin, urldefrag, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool


load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

REQUEST_TIMEOUT = 10
MAX_PAGE_SIZE = 1_000_000       # 1 MB
MAX_REDIRECTS = 3
MAX_LINKS = 20

USER_AGENT = (
    "Mozilla/5.0 (compatible; AgentOpsResearchBot/1.0)"
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# -----------------------------
# Tavily Client
# -----------------------------

def get_tavily_client():
    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY missing. Check your .env file."
        )

    return TavilyClient(api_key=TAVILY_API_KEY)


# -----------------------------
# URL Validation
# -----------------------------

def normalize_url(url: str) -> str:
    """Normalize URL and remove fragments."""

    url = url.strip()

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("Invalid URL: hostname missing.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are blocked.")

    hostname = parsed.hostname.lower()

    # Remove default ports
    port = parsed.port

    if port == 80 and parsed.scheme == "http":
        netloc = hostname
    elif port == 443 and parsed.scheme == "https":
        netloc = hostname
    elif port:
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    normalized = urlunparse((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "/",
        parsed.params,
        parsed.query,
        ""
    ))

    return urldefrag(normalized)[0]


def validate_url(url: str) -> str:
    """
    Reject localhost, private IPs and non-public destinations.
    Also resolves hostnames before making a request.
    """

    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    hostname = parsed.hostname

    if hostname in ("localhost", "localhost.localdomain"):
        raise ValueError("Localhost URLs are blocked.")

    try:
        ipaddress.ip_address(hostname)
        addresses = [hostname]

    except ValueError:
        try:
            records = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM
            )

            addresses = list({
                record[4][0] for record in records
            })

        except socket.gaierror:
            raise ValueError("Could not resolve hostname.")

    if not addresses:
        raise ValueError("No valid IP address found.")

    for address in addresses:
        ip = ipaddress.ip_address(address)

        if not ip.is_global:
            raise ValueError(
                f"Non-public IP destination blocked: {address}"
            )

    return normalized


# -----------------------------
# Safe Webpage Fetching
# -----------------------------

def fetch_webpage(url: str) -> dict:
    """
    Fetch a webpage safely with redirect, timeout
    and response-size limits.
    """

    current_url = url

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml"
    }

    try:
        for redirect_count in range(MAX_REDIRECTS + 1):

            current_url = validate_url(current_url)

            response = requests.get(
                current_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True
            )

            # Handle redirects manually so every target
            # gets validated before following it.
            if response.is_redirect or response.is_permanent_redirect:

                location = response.headers.get("Location")
                response.close()

                if not location:
                    return {
                        "success": False,
                        "url": current_url,
                        "error": "Redirect location missing."
                    }

                if redirect_count >= MAX_REDIRECTS:
                    return {
                        "success": False,
                        "url": current_url,
                        "error": "Maximum redirects exceeded."
                    }

                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if not (
                "text/html" in content_type
                or "application/xhtml+xml" in content_type
            ):
                response.close()
                return {
                    "success": False,
                    "url": current_url,
                    "error": "URL does not return an HTML page."
                }

            content_length = response.headers.get("Content-Length")

            if content_length:
                if int(content_length) > MAX_PAGE_SIZE:
                    response.close()
                    return {
                        "success": False,
                        "url": current_url,
                        "error": "Page exceeds maximum size limit."
                    }

            content = bytearray()

            for chunk in response.iter_content(chunk_size=8192):
                content.extend(chunk)

                if len(content) > MAX_PAGE_SIZE:
                    response.close()
                    return {
                        "success": False,
                        "url": current_url,
                        "error": "Page exceeds maximum size limit."
                    }

            response.close()

            encoding = response.encoding or "utf-8"
            html = bytes(content).decode(
                encoding,
                errors="replace"
            )

            return {
                "success": True,
                "url": current_url,
                "html": html,
                "content_type": content_type,
                "size_bytes": len(content)
            }

        return {
            "success": False,
            "url": current_url,
            "error": "Could not fetch webpage."
        }

    except Exception as error:
        return {
            "success": False,
            "url": current_url,
            "error": str(error)
        }


# -----------------------------
# HTML Content Extraction
# -----------------------------

def extract_page_content(html: str) -> str:
    """Extract readable text from HTML."""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "svg"
    ]):
        tag.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    text = main_content.get_text(
        separator=" ",
        strip=True
    )

    return " ".join(text.split())


# -----------------------------
# Link Extraction
# -----------------------------

def extract_page_links(
    html: str,
    base_url: str,
    max_links: int = MAX_LINKS
) -> list:
    """
    Extract valid HTTP/HTTPS links from a webpage.
    Keeps only same-domain links by default.
    """

    soup = BeautifulSoup(html, "html.parser")

    base_domain = urlparse(base_url).hostname
    discovered_links = []
    seen = set()

    for tag in soup.find_all("a", href=True):

        href = tag["href"].strip()

        if href.startswith((
            "mailto:",
            "tel:",
            "javascript:",
            "data:"
        )):
            continue

        absolute_url = urljoin(base_url, href)

        try:
            normalized = normalize_url(absolute_url)
        except ValueError:
            continue

        parsed = urlparse(normalized)

        # Keep same-domain links for controlled crawling.
        if parsed.hostname != base_domain:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        discovered_links.append({
            "url": normalized,
            "anchor_text": tag.get_text(
                " ",
                strip=True
            )[:150]
        })

        if len(discovered_links) >= max_links:
            break

    return discovered_links


# -----------------------------
# Structured Web Search
# -----------------------------

def search_web_results(
    query: str,
    max_results: int = 5
) -> list:
    """Return structured Tavily search results."""

    client = get_tavily_client()

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic"
    )

    results = []

    for item in response.get("results", []):

        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score")
        })

    return results


# -----------------------------
# LangChain Search Tool
# -----------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily.
    Returns titles, URLs, snippets and relevance scores.
    """

    try:
        results = search_web_results(query)

        return json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        )

    except Exception as error:
        return json.dumps({
            "success": False,
            "error": str(error)
        })


# -----------------------------
# LangChain Scraping Tool
# -----------------------------

@tool
def scrape_url(url: str) -> str:
    """
    Safely scrape a webpage and return its readable text
    along with links discovered on that page.
    """

    page = fetch_webpage(url)

    if not page["success"]:
        return json.dumps(page, indent=2)

    content = extract_page_content(page["html"])

    links = extract_page_links(
        page["html"],
        page["url"]
    )

    result = {
        "success": True,
        "url": page["url"],
        "content": content[:12000],
        "discovered_links": links,
        "size_bytes": page["size_bytes"]
    }

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )