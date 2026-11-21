import json
import re

from agentops.agents.research.prompts import (
    get_critic_prompt,
    planner_prompt,
    repair_prompt,
    writer_prompt,
)
from agentops.llm import get_chat_model

# =========================================================
# CONFIGURATION
# =========================================================
PLANNER_MAX_TOKENS = 1500
WRITER_MAX_TOKENS = 6000
CRITIC_MAX_TOKENS = 5000
CRITIC_REPAIR_MAX_TOKENS = 5000
MAX_SNIPPETS_PER_SOURCE = 5
MAX_CONTEXT_CHARS = 1200
MAX_CRITIC_CONTEXT_CHARS = 800
MAX_CLAIMS_TO_VERIFY = 12
MAX_JSON_REPAIR_ATTEMPTS = 1
CRITIC_REQUIRED_KEYS = {
    "overall_score",
    "topic_relevance",
    "research_completeness",
    "evidence_quality",
    "citation_quality",
    "claim_verification",
    "unsupported_claims",
    "citation_issues",
    "missing_areas",
    "strengths",
    "weaknesses",
    "verdict"
}
CRITIC_LIST_FIELDS = {
    "claim_verification",
    "unsupported_claims",
    "citation_issues",
    "missing_areas",
    "strengths",
    "weaknesses"
}
CRITIC_SCORE_FIELDS = {
    "overall_score",
    "topic_relevance",
    "research_completeness",
    "evidence_quality",
    "citation_quality"
}
VALID_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported"
}
CLAIM_REQUIRED_FIELDS = {
    "claim",
    "citation_ids",
    "supporting_evidence_ids",
    "verdict",
    "explanation"
}
# =========================================================
# LLM HELPER
# =========================================================
def call_llm(
    prompt,
    variables: dict,
    max_tokens: int,
    event_name: str,
    truncated_message: str,
    topic: str = None,
    recorder=None
) -> str:
    """
    Centralized LLM call.

    The callback handler (through `recorder.config`) records one
    execution event per call: latency, tokens, finish_reason, max_tokens.
    """
    chain = prompt | get_chat_model(max_tokens=max_tokens)

    message = chain.invoke(
        variables,
        config=(
            recorder.config(
                event_name,
                input_data={"topic": topic} if topic else None
            )
            if recorder else None
        )
    )

    finish_reason = (message.response_metadata or {}).get("finish_reason")

    if finish_reason == "length":
        raise RuntimeError(truncated_message)

    return message.content or ""
# =========================================================
# JSON EXTRACTION
# =========================================================
def extract_json(
    text: str,
    required_keys: set = None
) -> dict:
    """
    Extract a valid JSON object from an LLM response.
    Supports:
    - Normal JSON objects
    - JSON arrays containing objects
    - Markdown JSON code blocks
    - Text surrounding JSON
    Does not silently repair malformed JSON.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned an empty response.")
    cleaned = text.strip()
    # Remove Markdown code fences.
    cleaned = re.sub(
        r"^\s*`{3}(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\s*`{3}\s*$",
        "",
        cleaned
    )
    candidates = []
    # First attempt: parse the complete response.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            candidates.append(parsed)
        elif isinstance(parsed, list):
            candidates.extend(
                item for item in parsed
                if isinstance(item, dict)
            )
    except json.JSONDecodeError:
        pass
    # Second attempt: extract JSON objects from surrounding text.
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except json.JSONDecodeError:
            continue
    # Remove duplicates.
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    if not unique_candidates:
        raise ValueError(
            "LLM did not return valid JSON. "
            f"Response preview: {cleaned[:2000]}"
        )
    if required_keys:
        matching_candidates = [
            candidate
            for candidate in unique_candidates
            if required_keys.issubset(candidate.keys())
        ]
        if not matching_candidates:
            candidate_keys = [
                list(candidate.keys())
                for candidate in unique_candidates
            ]
            raise ValueError(
                "LLM JSON does not match the required schema.\n"
                f"Expected keys: {sorted(required_keys)}\n"
                f"Received keys: {candidate_keys}\n"
                f"Response preview: {cleaned[:2000]}"
            )
        return max(
            matching_candidates,
            key=len
        )
    return unique_candidates[0]
# =========================================================
# EVIDENCE PACK
# =========================================================
def build_evidence_pack(research_pages: list) -> list:
    """
    Convert analyzed research pages into a structured
    evidence pack for the Writer and Critic.
    """
    evidence_pack = []
    for index, page in enumerate(research_pages, start=1):
        source_id = page.get("source_id") or index
        snippets = page.get("evidence_snippets") or []
        if not snippets:
            content = re.sub(
                r"\s+",
                " ",
                page.get("content", "")
            ).strip()
            if content:
                snippets = [
                    {
                        "text": content[:MAX_CONTEXT_CHARS],
                        "context": content[:MAX_CONTEXT_CHARS]
                    }
                ]
        evidence_items = []
        for snippet_index, snippet in enumerate(
            snippets[:MAX_SNIPPETS_PER_SOURCE],
            start=1
        ):
            central_text = (
                snippet.get("text") or ""
            ).strip()
            context = (
                snippet.get("context") or central_text
            ).strip()
            if not central_text:
                continue
            evidence_items.append({
                "evidence_id": f"S{source_id}-E{snippet_index}",
                "key_sentence": central_text,
                "context": context[:MAX_CONTEXT_CHARS],
                "matched_terms": snippet.get(
                    "matched_terms",
                    []
                )
            })
        if evidence_items:
            evidence_pack.append({
                "source_id": source_id,
                "url": page.get("url"),
                "domain": page.get("domain"),
                "evidence": evidence_items
            })
    return evidence_pack
def build_critic_evidence_pack(
    evidence_pack: list,
    cited_ids: list
) -> list:
    """
    Keep only sources cited in the report.
    Reduce context size for the Critic.
    """
    cited_ids = {str(item) for item in cited_ids}
    filtered_pack = []
    for source in evidence_pack:
        if str(source.get("source_id")) not in cited_ids:
            continue
        compact_evidence = []
        for item in source.get("evidence", []):
            compact_evidence.append({
                "evidence_id": item.get("evidence_id"),
                "key_sentence": item.get("key_sentence", ""),
                "context": (
                    item.get("context", "")
                    [:MAX_CRITIC_CONTEXT_CHARS]
                )
            })
        if compact_evidence:
            filtered_pack.append({
                "source_id": source.get("source_id"),
                "url": source.get("url"),
                "evidence": compact_evidence
            })
    return filtered_pack
# =========================================================
# RESEARCH PLANNER
# =========================================================
def plan_research(topic: str, recorder=None) -> dict:
    output = call_llm(
        planner_prompt,
        {"topic": topic},
        max_tokens=PLANNER_MAX_TOKENS,
        event_name="planner",
        truncated_message="Research planner response was truncated.",
        topic=topic,
        recorder=recorder
    )
    result = extract_json(output)
    if not isinstance(result.get("sub_questions"), list):
        raise ValueError(
            "Invalid research plan: sub_questions missing."
        )
    result["sub_questions"] = [
        str(question).strip()
        for question in result["sub_questions"]
        if str(question).strip()
    ][:5]
    if not result["sub_questions"]:
        raise ValueError(
            "Research planner returned no research questions."
        )
    return {
        "plan": result
    }
# =========================================================
# RESEARCH WRITER
# =========================================================
def write_research_report(
    topic: str,
    research_plan: dict,
    research_pages: list,
    recorder=None
) -> dict:
    evidence_pack = build_evidence_pack(research_pages)
    if not evidence_pack:
        raise ValueError(
            "Writer received no usable evidence snippets."
        )
    output = call_llm(
        writer_prompt,
        {
            "topic": topic,
            "research_plan": json.dumps(
                research_plan, indent=2, ensure_ascii=False
            ),
            "evidence_pack": json.dumps(
                evidence_pack, indent=2, ensure_ascii=False
            ),
        },
        max_tokens=WRITER_MAX_TOKENS,
        event_name="writer",
        truncated_message="Writer response was truncated. Increase token budget.",
        topic=topic,
        recorder=recorder
    )
    writer_required_keys = {
        "report",
        "sources_used",
        "cited_source_ids"
    }
    result = extract_json(
        output,
        required_keys=writer_required_keys
    )
    report = result.get("report")
    if not isinstance(report, str) or not report.strip():
        raise ValueError(
            "Writer returned an empty or invalid report."
        )
    if not isinstance(result["sources_used"], list):
        raise ValueError("Writer sources_used must be a list.")
    if not isinstance(result["cited_source_ids"], list):
        raise ValueError("Writer cited_source_ids must be a list.")
    result["evidence_pack"] = evidence_pack
    return result
# =========================================================
# CITATION VALIDATION
# =========================================================
def extract_report_citations(report: str) -> list:
    """
    Extract numeric citations such as [1], [2], [1][3].
    """
    citations = re.findall(
        r"\[(\d+)\]",
        report or ""
    )
    return sorted({
        int(citation)
        for citation in citations
    })
def validate_citation_ids(
    cited_ids: list,
    evidence_pack: list
) -> list:
    valid_ids = {
        str(source["source_id"])
        for source in evidence_pack
    }
    invalid_ids = []
    for citation in cited_ids:
        if str(citation) not in valid_ids:
            invalid_ids.append(citation)
    return sorted(set(invalid_ids))
# =========================================================
# CRITIC JSON VALIDATION
# =========================================================
def validate_critic_result(result: dict) -> dict:
    """
    Validate the complete Critic response.
    Reject malformed or incomplete evaluations.
    """
    if not isinstance(result, dict):
        raise ValueError("Critic result must be a JSON object.")
    missing_fields = CRITIC_REQUIRED_KEYS - result.keys()
    if missing_fields:
        raise ValueError(
            f"Critic response missing fields: {sorted(missing_fields)}"
        )
    # Validate scores.
    for field in CRITIC_SCORE_FIELDS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Critic score '{field}' must be numeric."
            )
        if not 1 <= value <= 10:
            raise ValueError(
                f"Critic score '{field}' must be between 1 and 10."
            )
    # Validate arrays.
    for field in CRITIC_LIST_FIELDS:
        if not isinstance(result.get(field), list):
            raise ValueError(
                f"Critic field '{field}' must be a list."
            )
    # Validate verdict.
    if not isinstance(result.get("verdict"), str):
        raise ValueError("Critic verdict must be a string.")
    if not result["verdict"].strip():
        raise ValueError("Critic verdict cannot be empty.")
    # Validate individual claim evaluations.
    for index, claim in enumerate(result["claim_verification"]):
        if not isinstance(claim, dict):
            raise ValueError(
                f"Critic claim {index} must be an object."
            )
        missing_claim_fields = CLAIM_REQUIRED_FIELDS - claim.keys()
        if missing_claim_fields:
            raise ValueError(
                f"Critic claim {index} missing fields: "
                f"{sorted(missing_claim_fields)}"
            )
        if not isinstance(claim["claim"], str):
            raise ValueError(
                f"Critic claim {index}: claim must be a string."
            )
        if not isinstance(claim["explanation"], str):
            raise ValueError(
                f"Critic claim {index}: explanation must be a string."
            )
        if not isinstance(claim["citation_ids"], list):
            raise ValueError(
                f"Critic claim {index}: citation_ids must be a list."
            )
        if not isinstance(claim["supporting_evidence_ids"], list):
            raise ValueError(
                f"Critic claim {index}: supporting_evidence_ids must be a list."
            )
        if claim["verdict"] not in VALID_VERDICTS:
            raise ValueError(
                f"Critic claim {index}: invalid verdict "
                f"'{claim['verdict']}'."
            )
    return result
# =========================================================
# CRITIC JSON REPAIR
# =========================================================
def repair_critic_json(
    raw_response: str,
    validation_error: str,
    topic: str = None,
    recorder=None
) -> dict:
    """
    Ask the LLM to repair its own malformed JSON.
    This is a single controlled retry, not an evaluation rerun.
    """
    output = call_llm(
        repair_prompt,
        {
            "validation_error": validation_error,
            "raw_response": raw_response[:18000],
            "required_keys": json.dumps(sorted(CRITIC_REQUIRED_KEYS)),
            "claim_fields": json.dumps(sorted(CLAIM_REQUIRED_FIELDS)),
            "allowed_verdicts": json.dumps(sorted(VALID_VERDICTS)),
        },
        max_tokens=CRITIC_REPAIR_MAX_TOKENS,
        event_name="critic_repair",
        truncated_message="Critic JSON repair response was truncated.",
        topic=topic,
        recorder=recorder
    )
    repaired = extract_json(
        output,
        required_keys=CRITIC_REQUIRED_KEYS
    )
    repaired = validate_critic_result(repaired)
    return repaired
# =========================================================
# RESEARCH CRITIC
# =========================================================
def critique_report(
    topic: str,
    report: str,
    evidence_pack: list,
    recorder=None
) -> dict:
    if not report or not report.strip():
        raise ValueError(
            "Critic received an empty report."
        )
    if not evidence_pack:
        raise ValueError(
            "Critic received an empty evidence pack."
        )
    # -------------------------------------
    # Programmatic Citation Extraction
    # -------------------------------------
    cited_ids = extract_report_citations(report)
    invalid_ids = validate_citation_ids(
        cited_ids,
        evidence_pack
    )
    # -------------------------------------
    # Evidence Filtering
    # -------------------------------------
    critic_evidence = build_critic_evidence_pack(
        evidence_pack,
        cited_ids
    )
    # -------------------------------------
    # First Critic Attempt
    # -------------------------------------
    raw_output = call_llm(
        get_critic_prompt(MAX_CLAIMS_TO_VERIFY),
        {
            "topic": topic,
            "report": report,
            "critic_evidence": json.dumps(
                critic_evidence, indent=2, ensure_ascii=False
            ),
            "invalid_ids": json.dumps(invalid_ids),
        },
        max_tokens=CRITIC_MAX_TOKENS,
        event_name="critic",
        truncated_message=(
            "Critic response was truncated. "
            "Reduce claim limit or increase token budget."
        ),
        topic=topic,
        recorder=recorder
    )
    repair_used = False
    # -------------------------------------
    # Parse and Validate Critic Response
    # -------------------------------------
    try:
        result = extract_json(
            raw_output,
            required_keys=CRITIC_REQUIRED_KEYS
        )
        result = validate_critic_result(result)
    except (ValueError, TypeError) as first_error:
        if MAX_JSON_REPAIR_ATTEMPTS < 1:
            raise
        print(
            "[CRITIC] Invalid response. "
            "Attempting JSON repair..."
        )
        # ---------------------------------
        # One Controlled Repair Attempt
        # ---------------------------------
        result = repair_critic_json(
            raw_response=raw_output,
            validation_error=str(first_error),
            topic=topic,
            recorder=recorder
        )
        repair_used = True
    # -------------------------------------
    # Programmatic Evidence-ID Validation
    # -------------------------------------
    valid_evidence_ids = {
        item["evidence_id"]
        for source in critic_evidence
        for item in source.get("evidence", [])
    }
    for claim in result["claim_verification"]:
        invalid_evidence_ids = [
            evidence_id
            for evidence_id in claim["supporting_evidence_ids"]
            if evidence_id not in valid_evidence_ids
        ]
        if invalid_evidence_ids:
            claim["verdict"] = "unsupported"
            claim["explanation"] = (
                "The Critic referenced evidence IDs "
                "that are not present in the supplied evidence pack."
            )
            claim["supporting_evidence_ids"] = [
                evidence_id
                for evidence_id in claim["supporting_evidence_ids"]
                if evidence_id in valid_evidence_ids
            ]
    # -------------------------------------
    # Programmatic Citation Validation
    # -------------------------------------
    result["invalid_citation_ids"] = invalid_ids
    if invalid_ids:
        result["citation_issues"].append(
            "The report contains citation IDs that do not exist "
            "in the supplied evidence pack."
        )
    # Ensure unsupported claims are not omitted.
    unsupported_from_verification = [
        item["claim"]
        for item in result["claim_verification"]
        if item["verdict"] == "unsupported"
    ]
    existing_unsupported = result["unsupported_claims"]
    if not isinstance(existing_unsupported, list):
        existing_unsupported = []
    for claim in unsupported_from_verification:
        if claim not in existing_unsupported:
            existing_unsupported.append(claim)
    result["unsupported_claims"] = existing_unsupported
    # -------------------------------------
    # Metadata
    # -------------------------------------
    # Latency, tokens and finish_reason of every Critic call are recorded
    # as their own events ("critic", "critic_repair") by the callback.
    result["metadata"] = {"repair_used": repair_used}
    return result
# =========================================================
# PIPELINE COMPATIBILITY FUNCTION
# =========================================================
def critique_research_report(
    topic: str,
    report: str,
    evidence_pack: list,
    recorder=None
) -> dict:
    return critique_report(
        topic=topic,
        report=report,
        evidence_pack=evidence_pack,
        recorder=recorder
    )