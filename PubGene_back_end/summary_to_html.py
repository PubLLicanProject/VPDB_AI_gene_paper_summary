# summary_to_html.py (Anthropic-only; imports helpers)
# -----------------------------------------------------------------------------
# Build HTML-ready summaries for a single gene from a single PubMed paper.
# Inputs: pmid, gene_id, host_db
# Outputs: short_summary_html, extended_summary_html, evidence_bullets, paper_text
# -----------------------------------------------------------------------------

from dotenv import load_dotenv
# Load environment variables from .env file
# dotenv_path = join(dirname(__file__), 'v2.env')
load_dotenv() 

import requests
import os
import json
import html
from typing import Dict, Any, Tuple
from anthropic import Anthropic

from helpers import (
    get_pubmed_json, parse_pubmed_json,
    get_gene_synonyms_in_paper, gene_for_prompt,
    strip_fenced_json, build_extended_summary_html, ui_css, render_error_html,
    cache_get, cache_set, _ttl, CACHE_VERSION, get_vpdb_alias, _count_substrings,
    aliases_mentioned_in_paper
)

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "claude-sonnet-4-20250514")
N_QUOTES = int(os.getenv("N_QUOTES", "2"))

GET_GENE_SUMMARY_SYSTEM = (
    "ROLE: You are a scientist conducting a systematic literature review focused on ONE SPECIFIC gene of interest.\n"
    "GOAL: Analyze the provided research paper and extract ALL information relevant to the specified gene. The gene "
    "or one of its aliases is confirmed to be present in the text.\n"
    "INSTRUCTIONS: Follow this step-by-step workflow:\n"
    "1. EXPERIMENTAL ANALYSIS\n"
    "   - Identify and describe all experiments involving the specified gene\n"
    "   - Report experimental methods, results, and conclusions related to gene activity or expression\n"
    "   - Note any functional studies, assays, or measurements\n\n"
    "2. CONTEXTUAL RELEVANCE\n"
    "   - Extract any additional information relevant to understanding this gene's role\n"
    "   - Include regulatory mechanisms, interactions, pathways, or clinical significance\n"
    "   - Consider implications for gene function or therapeutic potential\n\n"
    "3. KEY FINDINGS SYNTHESIS\n"
    "   - Summarize discoveries as numbered bullet points\n"
    "   - Include one bullet point that lists all names given to the gene by the author.\n"
    "   - Prioritize findings by significance and directness of evidence\n\n"
    "4. EVIDENCE CLASSIFICATION\n"
    "   For each bullet point, determine:\n"
    "   - Is this based on DIRECT EVIDENCE from the text?\n"
    "   - Is this an INFERENCE you made from the data?\n"
    "   For direct evidence:\n"
    "   - Specify source location (e.g. Figure 1, Table 4, Results section - paragraph 4, Supplementary material, or Discussion)\n"
    f"   - Extract up to {N_QUOTES} verbatim quotes supporting each point\n\n"
    "5. STRUCTURED OUTPUT\n"
    "   Organize findings into:\n"
    "   - <Gene Summary>: Direct evidence-based bullet points with citations and quotes\n"
    "   - <Additional Inferences>: Your analytical interpretations of the data\n\n"
    "   - <Short Summary>: A concise one-sentence summary capturing and synthesising key novel findings from the "
    "Gene Summary bullet points \n\n"
    "6. MINIMAL MENTION HANDLING\n"
    "   If the gene appears only peripherally without substantial findings, provide:\n"
    "   - Single bullet point: \"Mentioned in passing\"\n"
    "   - Include context and direct quotes showing where/how it's referenced\n\n"
    "7. FORMAT COMPLIANCE\n"
    "Return ONLY valid JSON matching the schema supplied below."
)

GET_GENE_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["Aliases_in_paper", "GeneSummary", "AdditionalInferences", "ShortSummary"],
    "properties": {
        "Aliases_in_paper": {"type": "array", "items": {"type": "string"}},
        "GeneSummary": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["bullet_point", "evidence_location", "supporting_quotes"],
                "properties": {
                    "bullet_point": {"type": "string"},
                    "evidence_location": {"type": "string"},
                    "supporting_quotes": {"type": "array", "items": {"type": "string"}, "maxItems": N_QUOTES},
                },
            },
        },
        "AdditionalInferences": {"type": "array", "items": {"type": "string"}},
        "ShortSummary": {"type": "string"},
    },
    "additionalProperties": False,
}

def _schema_text(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)

def build_getGeneSummary_prompts(gene_for_llm: str, paper_text: str) -> Tuple[str, list]:
    sys_prompt = GET_GENE_SUMMARY_SYSTEM + "\n\nJSON SCHEMA:\n" + _schema_text(GET_GENE_SUMMARY_SCHEMA)
    user_prompts = [
        f"Do not respond to this message. Here is the paper text:\n{paper_text}",
        f"Please generate a summary for {gene_for_llm} based on the supplied paper text."
    ]
    return sys_prompt, user_prompts

def call_anthropic(system_prompt: str, user_messages: list, model: str = SUMMARY_MODEL) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if len(user_messages) > 1:
        content = [
            {"type": "text", "text": user_messages[0], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_messages[1]},
        ]
    else:
        content = [{"type": "text", "text": user_messages[0]}]
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=8000,
        temperature=0,
    )
    return resp.content[0].text if resp.content else ""

# add error message support do it si more informative to users
def _error_payload(msg: str, paper_text: str = "", raw: dict | None = None):
    err = ui_css() + render_error_html(msg)
    return {
        "short_summary_html": err,
        "extended_summary_html": "",
        "evidence_bullets": [],
        "paper_text": paper_text,
        "raw_json": raw or {"_error": msg},
    }

def run_summary(pmid: str, gene_id: str, host_db: str) -> Dict[str, Any]:
    
    # 0) fetch cached summary if already generated
    if os.getenv("FORCE_REFRESH") not in ("1", "true", "TRUE"):
        key = {
            "pmid": pmid,
            "gene_id": gene_id,
            "host_db": host_db,
            "model": SUMMARY_MODEL,
            "n_quotes": N_QUOTES,
        }
        cached = cache_get("summary", key, max_age_s=_ttl("SUMMARY_CACHE_MAX_AGE"))
        if cached:
            return cached   

 # 1) Fetch + parse PubMed
    try:
        pubmed_json = get_pubmed_json(pmid)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 404:
            return _error_payload(
                f"PMID {pmid} not found or not available as BioC JSON.",
                raw={"http_status": 404}
            )
        return _error_payload(
            f"PubMed service returned HTTP {code or 'error'} while fetching PMID {pmid}.",
            raw={"http_status": code}
        )
    except requests.RequestException as e:
        return _error_payload(
            f"Network error while fetching PMID {pmid}.",
            raw={"exception": str(e)}
        )
    except ValueError as e:
        return _error_payload(str(e))

    paper_text = parse_pubmed_json(pubmed_json)
    if not paper_text.strip():
        return _error_payload(
            f"PMID {pmid} fetched, but no relevant sections (FIG/TABLE/RESULTS/CONCL/DISCUSSION/SUPPL) were found."
        )

    # 2) Alias hinting
    aliases_all = get_vpdb_alias(gene_id, host_db)  # may be []
    # Count direct ID hits (also try a hyphen variant if the ID contains an underscore)
    id_hits = _count_substrings(paper_text, gene_id)
    if id_hits == 0 and "_" in gene_id:
        id_hits = _count_substrings(paper_text, gene_id.replace("_", "-"))

    if id_hits == 0 and "_" in gene_id:
        id_hits = _count_substrings(paper_text, gene_id.replace("_", ""))

    if id_hits == 0 and "-" in gene_id:
        id_hits = _count_substrings(paper_text, gene_id.replace("-", "_"))

    if id_hits == 0 and "-" in gene_id:
        id_hits = _count_substrings(paper_text, gene_id.replace("-", ""))

    synonyms_in_paper = get_gene_synonyms_in_paper(gene_id, paper_text, host_db)  # returns only aliases found in text

    if id_hits == 0 and not synonyms_in_paper:
        if not aliases_all:
            # Case A: no aliases in DB AND no text hit → likely wrong gene/database
            msg = (
                f"The paper (PMID {pmid}) does not mention the supplied gene ID "
                f"“{html.escape(gene_id)}”, and no aliases were found in {html.escape(host_db)}. "
                "Please check the Gene ID and/or selected database and try again."
            )
            return {
                "short_summary_html": ui_css() + render_error_html(msg),
                "extended_summary_html": "",
                "evidence_bullets": [],
                "paper_text": "",
                "raw_json": {
                    "_halted": True,
                    "reason": "no_aliases_and_no_text_hit",
                    "id_hits": id_hits,
                    "aliases_all": aliases_all,
                    "aliases_in_text": [],
                },
            }
        else:
            # Case B: aliases exist in DB, but none are mentioned in the paper
            msg = (
                f"The paper (PMID {pmid}) does not mention the supplied gene ID "
                f"“{html.escape(gene_id)}” or any of its aliases. "
                "Skipping AI summarisation to avoid mis-assignment."
            )
            return {
                "short_summary_html": ui_css() + render_error_html(msg),
                "extended_summary_html": "",
                "evidence_bullets": [],
                "paper_text": "",
                "raw_json": {
                    "_halted": True,
                    "reason": "aliases_exist_but_no_text_hit",
                    "id_hits": id_hits,
                    "aliases_all": aliases_all,
                    "aliases_in_text": [],
                },
            }

    mentions = aliases_mentioned_in_paper(gene_id, host_db, paper_text)  # gene_id included only if in text
    gene_for_llm = gene_for_prompt(gene_id, mentions)

    # 3) Call Anthropic
    sys_prompt, user_prompts = build_getGeneSummary_prompts(gene_for_llm, paper_text)
    try:
        raw = call_anthropic(sys_prompt, user_prompts, model=SUMMARY_MODEL)
    except requests.RequestException as e:
        return _error_payload(
            "LLM service request failed while generating the summary.",
            paper_text=paper_text,
            raw={"exception": str(e)}
        )

    content = strip_fenced_json(raw)
    try:
        parsed = json.loads(content)
    except Exception as e:
        return _error_payload(
            "Summary generation returned unparseable output.",
            paper_text=paper_text,
            raw={"_raw": raw, "_error": str(e)}
        )

    short_text = html.escape(parsed.get("ShortSummary") or "")
    short_summary_html = f"<p>{short_text}</p>" if short_text else "<p>No short summary.</p>"

    extended_html = build_extended_summary_html(parsed)
    evidence_bullets = [bp.get("bullet_point", "") for bp in (parsed.get("GeneSummary") or [])]
    evidence_bullets = [b for b in evidence_bullets if b]

    result = {
        "short_summary_html": short_summary_html,
        "extended_summary_html": extended_html,
        "evidence_bullets": evidence_bullets,
        "paper_text": paper_text,
        "raw_json": parsed,
        "aliases_mentioned": mentions,
    }

    # 4) Save to cache 
    cache_set("summary", {
        "pmid": pmid, "gene_id": gene_id, "host_db": host_db,
        "model": SUMMARY_MODEL, "n_quotes": N_QUOTES
    }, result)
    return result

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--gene_id", required=True)
    ap.add_argument("--host_db", required=True)
    args = ap.parse_args()
    out = run_summary(args.pmid, args.gene_id, args.host_db)
    print(json.dumps(out, indent=2))
