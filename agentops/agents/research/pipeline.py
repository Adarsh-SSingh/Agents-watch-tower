import time

from agentops.agents.research.agent import (
    build_evidence_pack,
    critique_research_report,
    plan_research,
    write_research_report,
)
from agentops.agents.research.crawler import crawl_research_sources
from agentops.agents.research.source_analyzer import analyze_sources
from agentops.agents.research.tools import (
    normalize_url,
    search_web_results,
)
from agentops.monitoring import RunRecorder


# =========================================================
# CONFIGURATION
# =========================================================

MAX_SEARCH_QUERIES = 5
MAX_RESULTS_PER_QUERY = 5
MAX_START_URLS = 10
MAX_CRAWL_PAGES = 20
MAX_CRAWL_DEPTH = 2


# =========================================================
# URL HELPERS
# =========================================================

def deduplicate_urls(urls: list) -> list:
    """Remove duplicate URLs while preserving order."""

    unique_urls = []
    seen = set()

    for url in urls:

        try:
            normalized = normalize_url(url)

            if normalized not in seen:
                seen.add(normalized)
                unique_urls.append(normalized)

        except (ValueError, TypeError):
            continue

    return unique_urls


# =========================================================
# WEB SEARCH
# =========================================================

def perform_research_search(sub_questions: list) -> dict:
    """Search the web for each research question."""

    all_results = []
    failed_queries = []

    for query in sub_questions[:MAX_SEARCH_QUERIES]:

        print(f"[SEARCH] Searching: {query}")

        try:
            results = search_web_results(
                query=query,
                max_results=MAX_RESULTS_PER_QUERY
            )

            all_results.extend(results)

            print(f"[SEARCH] Found {len(results)} results")

        except Exception as error:

            failed_queries.append({
                "query": query,
                "error": str(error)
            })

            print(f"[SEARCH ERROR] {error}")

    unique_results = []
    seen_urls = set()

    for result in all_results:

        try:
            url = normalize_url(
                result.get("url", "")
            )

            if url in seen_urls:
                continue

            seen_urls.add(url)

            unique_results.append({
                **result,
                "url": url
            })

        except (ValueError, TypeError):
            continue

    return {
        "results": unique_results,

        "start_urls": [
            result["url"]
            for result in unique_results[:MAX_START_URLS]
        ],

        "total_results": len(unique_results),

        "failed_queries": failed_queries
    }

def build_keyword_corpus(search_results: list) -> list:
    """
    Build a consistent keyword corpus from Tavily results.
    No LLM calls.
    """

    documents = []

    for result in search_results:
        title = result.get("title", "")
        snippet = result.get("content", "")

        if not snippet:
            snippet = result.get("snippet", "")

        document = f"{title} {snippet}".strip()

        if document:
            documents.append(document)

    return documents


# =========================================================
# MAIN RESEARCH PIPELINE
# =========================================================

def run_research_pipeline(
    topic: str,
    mode: str = "search",
    urls: list = None
) -> dict:
    """
    Run the Research Agent and persist its execution timeline in AgentOps.

    Existing research stages, result keys, and status semantics are preserved.

    LLM stages (planner, writer, critic, critic_repair) are recorded by the
    callback handler. Non-LLM stages are recorded here through run_stage.
    """

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    topic = topic.strip()

    if mode not in ("search", "url"):
        raise ValueError("Mode must be either 'search' or 'url'.")

    recorder = RunRecorder(
        agent_name="research_agent",
        user_input=topic
    )

    def run_stage(event_type, event_name, function, *args, **kwargs):
        """Execute one non-LLM stage and persist its success/failure event."""
        stage_start = time.perf_counter()

        try:
            output = function(*args, **kwargs)
            latency = round(time.perf_counter() - stage_start, 4)

            # Keep event payloads compact; full artifacts remain in the result.
            if event_name == "web_search":
                event_output = {
                    "total_results": output.get("total_results", 0),
                    "start_url_count": len(output.get("start_urls", [])),
                    "failed_query_count": len(output.get("failed_queries", [])),
                }
            elif event_name == "crawler":
                event_output = {
                    "total_pages": output.get(
                        "total_pages", len(output.get("pages", []))
                    ),
                    "total_failed": output.get("total_failed", 0),
                }
            elif event_name == "source_analysis":
                event_output = {
                    "total_sources": output.get("total_sources"),
                    "usable_sources": output.get("usable_sources"),
                    "rejected_sources": output.get("rejected_sources"),
                }
            elif event_name == "evidence_pack":
                event_output = {
                    "evidence_sources": len(output),
                    "evidence_items": sum(
                        len(item.get("evidence", [])) for item in output
                    ),
                }
            else:
                event_output = {"completed": True}

            recorder.log_event(
                event_type=event_type,
                event_name=event_name,
                input_data={"topic": topic},
                output_data=event_output,
                latency=latency,
                status="success",
            )
            return output

        except Exception as error:
            latency = round(time.perf_counter() - stage_start, 4)
            recorder.log_event(
                event_type=event_type,
                event_name=event_name,
                input_data={"topic": topic},
                output_data=None,
                latency=latency,
                status="error",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

    def finalize(result, error_type=None, error_message=None):
        result_status = result.get("status", "failed")
        recorder.complete(
            final_output=(
                result.get("report")
                or result.get("error")
                or result_status
            ),
            status=result_status,
            error_type=error_type,
            error_message=error_message or result.get("error"),
        )
        result["trace_id"] = recorder.trace_id
        return result

    print("\n========== RESEARCH STARTED ==========")

    try:
        # STAGE 1: PLANNER
        if mode == "search":
            print("\n[PIPELINE] Creating research plan...")
            planner_result = plan_research(topic, recorder=recorder)
            research_plan = planner_result["plan"]

            questions = research_plan.get("sub_questions", [])
            if not isinstance(questions, list) or not questions:
                raise ValueError(
                    "Research planner did not generate valid questions."
                )

            questions = [
                str(question).strip()
                for question in questions
                if str(question).strip()
            ][:MAX_SEARCH_QUERIES]

            if not questions:
                raise ValueError("Research planner questions were empty.")

            research_plan["sub_questions"] = questions
            print(
                f"[PIPELINE] Generated {len(questions)} research questions."
            )
        else:
            planner_result = None
            research_plan = {
                "topic": topic,
                "research_goal": (
                    f"Analyze the supplied webpages in relation to: {topic}"
                ),
                "sub_questions": [],
                "research_focus": [topic],
            }

        # STAGE 2: SEARCH OR DIRECT URLS
        search_result = {
            "results": [],
            "start_urls": [],
            "total_results": 0,
            "failed_queries": [],
        }

        if mode == "search":
            print("\n[PIPELINE] Starting web search...")
            search_result = run_stage(
                "tool",
                "web_search",
                perform_research_search,
                research_plan["sub_questions"],
            )

            if not search_result["start_urls"]:
                return finalize({
                    "status": "failed",
                    "topic": topic,
                    "mode": mode,
                    "error": "No valid search URLs were found.",
                    "research_plan": research_plan,
                    "search": search_result,
                })
        else:
            print("\n[PIPELINE] Direct URL mode selected.")
            if not urls:
                raise ValueError("At least one URL is required in URL mode.")

            search_result["start_urls"] = deduplicate_urls(urls)
            if not search_result["start_urls"]:
                raise ValueError(
                    "No valid starting URLs were provided."
                )

            recorder.log_event(
                event_type="workflow",
                event_name="direct_urls",
                input_data={"url_count": len(urls)},
                output_data={
                    "valid_url_count": len(search_result["start_urls"])
                },
            )

        # STAGE 3: CRAWLER
        print("\n[PIPELINE] Starting multi-hop crawler...")
        keyword_documents = build_keyword_corpus(search_result["results"])
        print(
            f"[PIPELINE] Keyword corpus created: "
            f"{len(keyword_documents)} documents"
        )

        crawl_result = run_stage(
            "tool",
            "crawler",
            crawl_research_sources,
            start_urls=search_result["start_urls"],
            topic=topic,
            max_pages=MAX_CRAWL_PAGES,
            max_depth=MAX_CRAWL_DEPTH,
            keyword_documents=keyword_documents,
        )

        pages = crawl_result.get("pages", [])
        pages_collected = crawl_result.get("total_pages", len(pages))
        print(
            f"[PIPELINE] Crawling completed: "
            f"{pages_collected} pages collected."
        )

        if not pages:
            return finalize({
                "status": "failed",
                "topic": topic,
                "mode": mode,
                "error": "No webpages could be collected.",
                "research_plan": research_plan,
                "search": search_result,
                "crawl": crawl_result,
            })

        # STAGE 4: SOURCE ANALYSIS
        print(
            "\n[PIPELINE] Analyzing sources and extracting evidence..."
        )
        source_result = run_stage(
            "workflow",
            "source_analysis",
            analyze_sources,
            pages=pages,
            topic=topic,
            research_questions=research_plan.get("sub_questions", []),
        )

        usable_pages = source_result["usable_pages"]
        evidence_count = sum(
            len(page.get("evidence_snippets", []))
            for page in usable_pages
        )
        print(
            f"[PIPELINE] Usable sources: {len(usable_pages)} | "
            f"Evidence snippets: {evidence_count}"
        )

        if not usable_pages:
            return finalize({
                "status": "insufficient_sources",
                "topic": topic,
                "mode": mode,
                "error": (
                    "No sources contained enough topic-relevant evidence. "
                    "Try a more specific topic or different URLs."
                ),
                "research_plan": research_plan,
                "search": search_result,
                "crawl": crawl_result,
                "source_analysis": source_result,
            })

        # STAGE 5: EVIDENCE PACK
        print("\n[PIPELINE] Building evidence pack...")
        evidence_pack = run_stage(
            "workflow",
            "evidence_pack",
            build_evidence_pack,
            usable_pages,
        )

        total_evidence_items = sum(
            len(item.get("evidence", [])) for item in evidence_pack
        )
        print(
            f"[PIPELINE] Evidence sources: {len(evidence_pack)} | "
            f"Evidence items: {total_evidence_items}"
        )

        if not evidence_pack:
            return finalize({
                "status": "insufficient_sources",
                "topic": topic,
                "mode": mode,
                "error": "Evidence pack could not be generated.",
                "research_plan": research_plan,
                "search": search_result,
                "crawl": crawl_result,
                "source_analysis": source_result,
            })

        # STAGE 6: WRITER
        print(
            "\n[PIPELINE] Generating evidence-based research report..."
        )
        writer_result = write_research_report(
            topic=topic,
            research_plan=research_plan,
            research_pages=usable_pages,
            recorder=recorder,
        )

        report = writer_result.get("report", "").strip()
        if not report:
            raise RuntimeError(
                "Research Writer returned an empty report."
            )

        print(
            f"[PIPELINE] Report generated: {len(report)} characters."
        )

        # STAGE 7: CLAIM-LEVEL CRITIC
        print(
            "\n[PIPELINE] Verifying individual claims against evidence..."
        )
        critic_result = None
        critic_error = None
        critic_evaluation = {
            "verdict": "Critic evaluation was not completed.",
            "weaknesses": [],
            "unsupported_claims": [],
            "citation_issues": [],
            "claim_verification": [],
            "invalid_citation_ids": [],
            "citation_quality": 0,
        }

        try:
            critic_result = critique_research_report(
                topic=topic,
                report=report,
                evidence_pack=evidence_pack,
                recorder=recorder,
            )
            critic_evaluation = critic_result
            print(
                "[PIPELINE] Claim-level verification completed."
            )
        except Exception as error:
            critic_error = str(error)
            print(f"[CRITIC ERROR] {critic_error}")

        # STAGE 8: FINAL RESULT + METRICS
        print("\n========== RESEARCH COMPLETED ==========")

        critic_status = "completed" if critic_result else "failed"
        final_status = "success" if critic_result else "needs_review"

        writer_stats = recorder.llm_stats("writer")
        critic_stats = recorder.llm_stats("critic")
        repair_stats = recorder.llm_stats("critic_repair")

        metrics = {
            "pages_collected": pages_collected,
            "failed_pages": crawl_result.get("total_failed", 0),
            "total_sources": source_result["total_sources"],
            "usable_sources": source_result["usable_sources"],
            "rejected_sources": source_result["rejected_sources"],
            "evidence_snippets": evidence_count,
            "evidence_sources": len(evidence_pack),
            "evidence_items": total_evidence_items,
            "writer_latency": writer_stats.get("latency"),
            "writer_tokens": writer_stats.get("tokens"),
            "critic_latency": (
                round(
                    critic_stats.get("latency", 0)
                    + repair_stats.get("latency", 0),
                    4
                )
                if critic_stats else None
            ),
            "critic_tokens": (
                critic_stats.get("tokens", 0)
                + repair_stats.get("tokens", 0)
                if critic_stats else None
            ),
        }

        result = {
            "status": final_status,
            "topic": topic,
            "mode": mode,
            "research_plan": research_plan,
            "search": search_result,
            "crawl": crawl_result,
            "source_analysis": source_result,
            "evidence_pack": evidence_pack,
            "report": report,
            "critic": critic_evaluation,
            "critic_status": critic_status,
            "critic_error": critic_error,
            "metrics": metrics,
        }

        return finalize(result)

    except Exception as error:
        error_type = type(error).__name__
        error_message = str(error)

        recorder.log_event(
            event_type="workflow",
            event_name="pipeline_error",
            input_data={"topic": topic, "mode": mode},
            output_data=None,
            status="error",
            error_type=error_type,
            error_message=error_message,
        )

        failure_result = {
            "status": "failed",
            "topic": topic,
            "mode": mode,
            "error": error_message,
        }
        finalize(
            failure_result,
            error_type=error_type,
            error_message=error_message,
        )
        raise