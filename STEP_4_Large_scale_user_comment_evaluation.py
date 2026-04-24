from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import pandas as pd
from collections import defaultdict

from config.global_settings_batch import *
from pipeline.prompts import *
from pipeline.pubmed_helpers import *
from pipeline.vpdb_helpers import *
from pipeline.utils import *
from pipeline.prompts import (_ensure_list, )



# -----------------------------------------------------------------------------
# Settings / constants
# -----------------------------------------------------------------------------
USER_COMMENT_EVAL_STAGE_KEY = "evaluateUserCommentAlignment"

COMMENTS_TSV = "./curated_data/all_user_comments_with_pmid_fetched.tsv"
STATUS_DIR = "./out/user_comment_evaluation"
DEBUG_RAW_DIR = Path("./out/user_comment_evaluation/debug_raw_responses")

OUTPUT_TSV = "./out/user_comment_evaluation/user_comment_eval_results.tsv"

EVAL_PROVIDER = "openrouter"
EVAL_MODEL = "openai/gpt-5.4"

ONLY_SINGLE_PMID = True
REQUIRE_PIPELINE_OUTPUTS = True
RESUME = True
SAVE_EVERY = 25
SLEEP_TIME = 0.2

FLEX_MODEL_LOOKUP = True
PREFERRED_SUMMARY_MODEL: Optional[str] = None
PREFERRED_PD_MODEL: Optional[str] = None

USE_STRUCTURED_OUTPUTS = False
OPENROUTER_JSON_PLUGINS = [] #[{"id": "response-healing"}]

LEGACY_EXTENDED_SUMMARY_STAGE = "verifyGeneSummary"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def response_format_for_schema(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def supports_structured_outputs(provider: str, model: str) -> bool:
    if provider != "openrouter":
        return False
    m = (model or "").lower()
    return ("sonnet-4-5" in m) or ("opus-4-1" in m)


def split_pmids(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in re.split(r",", text) if x.strip()]


def is_name_product_category(categories: Any) -> bool:
    if pd.isna(categories):
        return False
    return "name/product" in str(categories).strip().lower()


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(path)

def format_with_retry(content: str, schema: dict, max_attempts: int = MAX_RETRY) -> Optional[dict]:
    """Parse JSON with retry using formatter model."""

    # 1. Try direct parsing first using your utility
    data = extract_json(content)
    if isinstance(data, dict):
        return data

    print("  ! Direct parsing failed, attempting formatter model...")

    # 2. Try with formatter model
    for attempt in range(max_attempts):
        try:
            print(f"  Retry {attempt + 1}/{max_attempts} with formatter model...")

            if PROVIDER == "anthropic":
                response = anthropic_client.messages.create(
                    model=FORMATTER_MODEL,
                    max_tokens=4000,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": f"Convert this to valid JSON matching the schema. Output ONLY valid JSON:\n\n{content}\n\nSchema:\n{json.dumps(schema)}"
                    }]
                )
                result_text = response.content[0].text.strip()

            elif PROVIDER == "openrouter":

                result_text, _, _ = call_prompt(
                    provider=PROVIDER,
                    model=FORMATTER_MODEL,
                    system_prompt="You are a JSON formatter. Output ONLY valid JSON.",
                    user_prompts=[f"Fix this JSON according to schema:\n{content}\n\nSchema:\n{json.dumps(schema)}"],
                    # We don't prefill "{" here because extract_json handles the fences/text for us
                    prefill_text=""
                )
            else:
                raise ValueError(f"Unsupported PROVIDER '{PROVIDER}'")

            # 3. FIX: Use extract_json again instead of manual re.sub/json.loads
            # This handles cases where the model still includes ```json blocks
            final_data = extract_json(result_text)

            if isinstance(final_data, dict):
                return final_data

            raise ValueError("Formatter model returned non-dictionary content")

        except Exception as e:
            print(f"  Retry {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                # Debug logging remains the same
                debug_file = OUT_DIR / f"debug_failed_parse_{int(time.time())}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Original content:\n{content}\n\nSchema:\n{json.dumps(schema, indent=2)}")
                print(f"  Raw response saved to: {debug_file}")
                return None
            time.sleep(1)

    return None
def get_stage_payload(
    status_json: Dict[str, Any],
    stage: str,
    gene_id: str,
    *,
    preferred_model: Optional[str] = None,
    flex_model: bool = True,
) -> Optional[Dict[str, Any]]:
    stage_block = status_json.get(stage, {})
    gene_block = stage_block.get(gene_id)
    if not isinstance(gene_block, dict):
        return None

    candidate_keys: List[str] = []
    if preferred_model and preferred_model in gene_block:
        candidate_keys.append(preferred_model)

    if flex_model:
        for key in gene_block.keys():
            if key in {"gene_ID", "comments"} or key in candidate_keys:
                continue
            candidate_keys.append(key)

    for key in candidate_keys:
        rec = gene_block.get(key)
        if isinstance(rec, dict) and rec.get("success") is True and isinstance(rec.get("data"), dict):
            return rec

    if not flex_model and preferred_model:
        rec = gene_block.get(preferred_model)
        if isinstance(rec, dict) and rec.get("success") is True and isinstance(rec.get("data"), dict):
            return rec

    return None


def build_extended_summary_text(status_json: Dict[str, Any], gene_id: str) -> str:
    legacy_verified = get_stage_payload(
        status_json,
        LEGACY_EXTENDED_SUMMARY_STAGE,
        gene_id,
        flex_model=FLEX_MODEL_LOOKUP,
    )
    if legacy_verified:
        verified_data = legacy_verified.get("data", {})
        verified_summary = verified_data.get("VerifiedSummary", [])
        if isinstance(verified_summary, list) and verified_summary:
            bullets: List[str] = []
            for item in verified_summary:
                if isinstance(item, dict) and item.get("verification_status") != "FAIL":
                    bullet = item.get("bullet_point")
                    if bullet:
                        bullets.append(f"- {bullet}")
            if bullets:
                return "\n".join(bullets)

    summary = get_stage_payload(
        status_json,
        "getGeneSummary",
        gene_id,
        preferred_model=PREFERRED_SUMMARY_MODEL,
        flex_model=FLEX_MODEL_LOOKUP,
    )
    if not summary:
        return ""

    data = summary.get("data", {})
    parts: List[str] = []

    gene_summary = data.get("GeneSummary", [])
    if isinstance(gene_summary, list):
        for item in gene_summary:
            if isinstance(item, dict) and item.get("bullet_point"):
                parts.append(f"- {item['bullet_point']}")

    additional_inferences = data.get("AdditionalInferences", [])
    if isinstance(additional_inferences, list) and additional_inferences:
        parts.append("Additional inferences:")
        for item in additional_inferences:
            if item:
                parts.append(f"- {item}")

    return "\n".join(parts)


def extract_pipeline_outputs(status_json: Dict[str, Any], gene_id: str) -> Dict[str, Any]:
    out = {
        "short_summary": None,
        "extended_summary": None,
        "product_description": None,
        "summary_model": None,
        "pd_model": None,
        "pd_evidence_code": None,
    }

    summary = get_stage_payload(
        status_json,
        "getGeneSummary",
        gene_id,
        preferred_model=PREFERRED_SUMMARY_MODEL,
        flex_model=FLEX_MODEL_LOOKUP,
    )
    if summary:
        out["summary_model"] = summary.get("model")
        out["short_summary"] = summary.get("data", {}).get("ShortSummary")

    if not out["short_summary"]:
        generated = get_stage_payload(
            status_json,
            "generatePDs",
            gene_id,
            preferred_model=PREFERRED_PD_MODEL,
            flex_model=FLEX_MODEL_LOOKUP,
        )
        if generated:
            out["short_summary"] = generated.get("data", {}).get("short_summary")
            out["summary_model"] = out["summary_model"] or generated.get("model")

    out["extended_summary"] = build_extended_summary_text(status_json, gene_id)

    verified = get_stage_payload(
        status_json,
        "verifyPDs",
        gene_id,
        preferred_model=PREFERRED_PD_MODEL,
        flex_model=FLEX_MODEL_LOOKUP,
    )
    if verified:
        out["pd_model"] = verified.get("model")
        rpd = verified.get("data", {}).get("RPD", {})
        if isinstance(rpd, dict):
            out["product_description"] = rpd.get("description")
            out["pd_evidence_code"] = rpd.get("evidence_code")

    if not out["product_description"]:
        selected = get_stage_payload(
            status_json,
            "selectPD",
            gene_id,
            preferred_model=PREFERRED_PD_MODEL,
            flex_model=FLEX_MODEL_LOOKUP,
        )
        if selected:
            recommended_product = selected.get("data", {}).get("recommended_product", {})
            if isinstance(recommended_product, dict):
                out["product_description"] = recommended_product.get("description")
                out["pd_model"] = out["pd_model"] or selected.get("model")

    if not out["product_description"]:
        generated = get_stage_payload(
            status_json,
            "generatePDs",
            gene_id,
            preferred_model=PREFERRED_PD_MODEL,
            flex_model=FLEX_MODEL_LOOKUP,
        )
        if generated:
            pds = generated.get("data", {}).get("PDs", [])
            if isinstance(pds, list) and pds:
                out["product_description"] = pds[0].get("description")
                out["pd_evidence_code"] = pds[0].get("evidence_code")
                out["pd_model"] = out["pd_model"] or generated.get("model")

    return out



def call_evaluator(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompts: Any,
    cache: Optional[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], float]:
    call_kwargs: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompts": user_prompts,
        "prefill_text": "",  # strict mode: never fall back to brace-prefill here
        "cache": cache or {"enabled": False},
    }

    if USE_STRUCTURED_OUTPUTS:
        call_kwargs["response_format"] = response_format_for_schema(
            USER_COMMENT_EVAL_STAGE_KEY,
            schema,
        )
        call_kwargs["plugins"] = OPENROUTER_JSON_PLUGINS

    return call_prompt(**call_kwargs)


def evaluate_user_comment_alignment(
    *,
    gene_id: str,
    database: str,
    organism: str,
    comment_id: str,
    categories: str,
    pubmed_ids: str,
    is_single_pmid: bool,
    is_name_product: bool,
    headline: str,
    content: str,
    product_description: Optional[str],
    short_summary: Optional[str],
    extended_summary: Optional[str],
    provider: str = EVAL_PROVIDER,
    model: str = EVAL_MODEL,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema = global_prompts_and_schema[USER_COMMENT_EVAL_STAGE_KEY]["ValidationSchema"]

    replacements = {
        "JSON_SCHEMA": schema,
        "GENE_ID": gene_id,
        "DATABASE": database,
        "ORGANISM": organism,
        "COMMENT_ID": comment_id,
        "CATEGORIES": categories or "",
        "PUBMED_IDS": pubmed_ids or "",
        "IS_SINGLE_PMID": str(bool(is_single_pmid)).upper(),
        "IS_NAME_PRODUCT": str(bool(is_name_product)).upper(),
        "HEADLINE": headline or "",
        "CONTENT": content or "",
        "PRODUCT_DESCRIPTION": product_description or "NONE",
        "SHORT_SUMMARY": short_summary or "NONE",
        "EXTENDED_SUMMARY": extended_summary or "NONE",
    }

    system_prompt = get_prompt_and_replace(
        stage_key=USER_COMMENT_EVAL_STAGE_KEY,
        replacements=replacements,
        prompt_type="SystemPrompt",
    )
    user_prompts = get_prompt_and_replace(
        stage_key=USER_COMMENT_EVAL_STAGE_KEY,
        replacements=replacements,
        prompt_type="UserPrompts",
    )

    raw_text, usage, secs = call_evaluator(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        cache=cache,
        schema=schema,
    )

    raw_stripped = (raw_text or "").strip()

    if raw_stripped in {"", "{", "["}:
        return {
            "success": False,
            "raw_response": raw_text,
            "parsed": None,
            "usage": usage,
            "seconds": secs,
            "error": "Truncated model output",
        }

    parsed = extract_json(raw_text)
    success = isinstance(parsed, dict)

    return {
        "success": success,
        "raw_response": raw_text,
        "parsed": parsed if success else None,
        "usage": usage,
        "seconds": secs,
        "error": None if success else "JSON parsing failed",
    }




def save_failed_eval_debug(
    *,
    pmid: str,
    gene_id: str,
    comment_id: str,
    model: str,
    error: str,
    raw_response: str,
    usage: Dict[str, Any],
    seconds: float,
    headline: str,
    content: str,
    short_summary: Optional[str],
    extended_summary: Optional[str],
    product_description: Optional[str],
) -> None:
    DEBUG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = DEBUG_RAW_DIR / f"{pmid}__{gene_id}__{comment_id}.json"

    payload = {
        "pmid": pmid,
        "gene_id": gene_id,
        "comment_id": comment_id,
        "model": model,
        "error": error,
        "raw_response": raw_response,
        "raw_response_repr": repr(raw_response),
        "raw_len": len(raw_response or ""),
        "usage": usage,
        "seconds": seconds,
        "headline": headline,
        "content": content,
        "short_summary": short_summary,
        "extended_summary": extended_summary,
        "product_description": product_description,
    }

    with open(debug_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def get_saved_comment_evaluation(
    status_json: Dict[str, Any],
    gene_id: str,
    comment_id: str,
    *,
    stage_key: str = USER_COMMENT_EVAL_STAGE_KEY,
    preferred_model: Optional[str] = None,
    flex_model: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns an existing saved comment evaluation record, whether it succeeded or failed.
    """
    gene_block = (status_json.get(stage_key) or {}).get(gene_id)
    if not isinstance(gene_block, dict):
        return None

    candidate_keys: List[str] = []
    if preferred_model and preferred_model in gene_block:
        candidate_keys.append(preferred_model)

    if flex_model:
        for key in gene_block.keys():
            if key in {"gene_ID", "comments"} or key in candidate_keys:
                continue
            candidate_keys.append(key)

    for key in candidate_keys:
        rec = gene_block.get(key)
        if not isinstance(rec, dict):
            continue
        data = rec.get("data", {})
        comment_results = data.get("comment_results", {})
        comment_payload = comment_results.get(comment_id)
        if isinstance(comment_payload, dict):
            return comment_payload

    return None

def saved_comment_eval_succeeded(
    status_json: Dict[str, Any],
    gene_id: str,
    comment_id: str,
    *,
    preferred_model: Optional[str] = None,
    flex_model: bool = True,
) -> bool:
    rec = get_saved_comment_evaluation(
        status_json,
        gene_id,
        comment_id,
        preferred_model=preferred_model,
        flex_model=flex_model,
    )
    return bool(rec and rec.get("success") is True)

def collect_failed_comment_evaluations(
    comments_tsv: str,
    status_dir: str,
    *,
    model: Optional[str] = None,
    flex_model: bool = True,
    output_tsv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a table of STEP 4 evaluations that exist in status JSON and failed.
    """
    comments_df = pd.read_csv(comments_tsv, sep="\t")
    comments_df["Comment ID"] = comments_df["Comment ID"].astype(str)
    comments_df["Gene ID"] = comments_df["Gene ID"].astype(str)

    failed_rows: List[Dict[str, Any]] = []
    status_path = Path(status_dir)

    for _, row in comments_df.iterrows():
        pmid_field = row.get("PubMed IDs")
        if pd.isna(pmid_field):
            continue

        pmids = [x.strip() for x in str(pmid_field).split(",") if x.strip()]
        if len(pmids) != 1:
            continue

        pmid = pmids[0]
        gene_id = str(row["Gene ID"])
        comment_id = str(row["Comment ID"])

        status_file = status_path / f"{pmid}.json"
        if not status_file.exists():
            continue

        try:
            with open(status_file, "r", encoding="utf-8") as fh:
                status_json = json.load(fh)
        except Exception:
            continue

        rec = get_saved_comment_evaluation(
            status_json,
            gene_id,
            comment_id,
            preferred_model=model,
            flex_model=flex_model,
        )
        if not rec:
            continue

        if rec.get("success") is False:
            failed_rows.append({
                "Comment ID": comment_id,
                "Gene ID": gene_id,
                "PMID": pmid,
                "error": rec.get("error"),
                "seconds": rec.get("seconds"),
            })

    failed_df = pd.DataFrame(failed_rows)

    if output_tsv and not failed_df.empty:
        Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
        failed_df.to_csv(output_tsv, sep="\t", index=False)

    return failed_df
def save_comment_evaluation_result(
    *,
    status_path: Path,
    pmid: str,
    gene_id: str,
    comment_id: str,
    model: str,
    evaluation_result: Dict[str, Any],
    pipeline_outputs: Dict[str, Any],
    comment_metadata: Dict[str, Any],
) -> None:
    status_file = status_path / f"{pmid}.json"
    payload = load_json_file(status_file)

    stage_block = payload.setdefault(USER_COMMENT_EVAL_STAGE_KEY, {})
    gene_block = stage_block.setdefault(gene_id, {"gene_ID": gene_id})
    model_block = gene_block.setdefault(
        model,
        {
            "model": model,
            "success": True,
            "data": {"comment_results": {}},
            "usage": {
                "comment_count": 0,
                "successful_comment_count": 0,
                "failed_comment_count": 0,
            },
            "seconds": 0.0,
        },
    )

    if not isinstance(model_block.get("data"), dict):
        model_block["data"] = {"comment_results": {}}

    comment_results = model_block["data"].setdefault("comment_results", {})

    comment_results[comment_id] = {
        "comment_id": comment_id,
        "success": bool(evaluation_result.get("success")),
        "evaluation": evaluation_result.get("parsed"),
        "raw_response": evaluation_result.get("raw_response"),
        "error": evaluation_result.get("error"),
        "pipeline_outputs": pipeline_outputs,
        "comment_metadata": comment_metadata,
        "usage": evaluation_result.get("usage", {}),
        "seconds": evaluation_result.get("seconds", 0.0),
    }

    total_seconds = 0.0
    success_count = 0
    fail_count = 0

    for item in comment_results.values():
        if not isinstance(item, dict):
            continue
        total_seconds += float(item.get("seconds", 0.0) or 0.0)
        if item.get("success") is True:
            success_count += 1
        else:
            fail_count += 1

    model_block["success"] = True
    model_block["usage"] = {
        "comment_count": len(comment_results),
        "successful_comment_count": success_count,
        "failed_comment_count": fail_count,
    }
    model_block["seconds"] = round(total_seconds, 3)

    save_json_file(status_file, payload)



def flatten_evaluation_row(
    *,
    row: pd.Series,
    pmid: str,
    model: str,
    pipeline_outputs: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    flat = {
        "Comment ID": str(row["Comment ID"]),
        "Gene ID": str(row["Gene ID"]),
        "Database": row.get("Database", ""),
        "Organism": row.get("Organism", ""),
        "Headline": row.get("Headline", ""),
        "Content": row.get("Content", ""),
        "Categories": row.get("Categories", ""),
        "PubMed IDs": row.get("PubMed IDs", ""),
        "pmid": pmid,
        "single_pmid": bool(row.get("single_pmid", False)),
        "is_name_product": bool(row.get("is_name_product", False)),
        "pipeline_short_summary": pipeline_outputs.get("short_summary"),
        "pipeline_extended_summary": pipeline_outputs.get("extended_summary"),
        "pipeline_product_description": pipeline_outputs.get("product_description"),
        "summary_model": pipeline_outputs.get("summary_model"),
        "pd_model": pipeline_outputs.get("pd_model"),
        "pd_evidence_code": pipeline_outputs.get("pd_evidence_code"),
        "llm_eval_model": model,
    }

    for key, value in evaluation.items():
        flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    return flat


# -----------------------------------------------------------------------------
# Main large-scale runner
# -----------------------------------------------------------------------------
def run_user_comment_analysis(
    comments_tsv: str = COMMENTS_TSV,
    status_dir: str = STATUS_DIR,
    output_tsv: str = OUTPUT_TSV,
    only_single_pmid: bool = ONLY_SINGLE_PMID,
    require_pipeline_outputs: bool = REQUIRE_PIPELINE_OUTPUTS,
    sleep_time: float = SLEEP_TIME,
    save_every: int = SAVE_EVERY,
    provider: str = EVAL_PROVIDER,
    model: str = EVAL_MODEL,
    resume: bool = RESUME,
    rerun_failed_only: bool = False,
) -> pd.DataFrame:
    comments_path = Path(comments_tsv)
    status_path = Path(status_dir)
    output_path = Path(output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading comments from: {comments_path}")
    df = pd.read_csv(comments_path, sep="\t")
    print(f"Loaded {len(df)} comment rows")

    df["Comment ID"] = df["Comment ID"].astype(str)
    df["Gene ID"] = df["Gene ID"].astype(str)
    df["Database"] = df["Database"].astype(str)
    df["pmid_list"] = df["PubMed IDs"].apply(split_pmids)
    df["n_pmids"] = df["pmid_list"].apply(len)
    df["single_pmid"] = df["n_pmids"].eq(1)
    df["pmid"] = df["pmid_list"].apply(lambda xs: xs[0] if len(xs) == 1 else None)
    df["is_name_product"] = df["Categories"].apply(is_name_product_category)

    if only_single_pmid:
        before = len(df)
        df = df[df["single_pmid"]].copy()
        print(f"Filtered to single PMID comments: {len(df)} / {before}")

    existing_ids = set()
    if resume and output_path.exists() and not rerun_failed_only:
        try:
            prev = pd.read_csv(output_path, sep="\t")
            if "Comment ID" in prev.columns:
                existing_ids = set(prev["Comment ID"].astype(str))
            print(f"Resume mode: found {len(existing_ids)} Comment IDs already in {output_path}")
        except Exception as e:
            existing_ids = set()
            print(f"Warning: could not read existing output for resume: {e}")
    else:
        print("Resume mode: using status JSON success/failure checks")

    print(f"Looking for status JSONs in: {status_path}")
    print(f"Starting evaluation for {len(df)} candidate rows")

    results: List[Dict[str, Any]] = []
    processed = 0
    skipped_resume = 0
    skipped_no_pmid = 0
    skipped_missing_status = 0
    skipped_existing_json = 0
    skipped_no_outputs = 0
    failed = 0
    rerun_count = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        comment_id = str(row["Comment ID"])
        gene_id = str(row["Gene ID"])

        if comment_id in existing_ids:
            skipped_resume += 1
            continue

        pmid = row.get("pmid")
        if not pmid:
            skipped_no_pmid += 1
            continue

        status_file = status_path / f"{pmid}.json"
        if not status_file.exists():
            if require_pipeline_outputs:
                skipped_missing_status += 1
                print(f"Skipping Comment ID {comment_id} ({gene_id}, PMID {pmid}): missing {status_file.name}")
                continue
            status_json = {}
        else:
            try:
                with open(status_file, "r", encoding="utf-8") as fh:
                    status_json = json.load(fh)
            except Exception as e:
                failed += 1
                print(f"Failed to read {status_file}: {e}")
                continue

        existing_eval = get_saved_comment_evaluation(
            status_json,
            gene_id,
            comment_id,
            preferred_model=model,
            flex_model=FLEX_MODEL_LOOKUP,
        )

        if existing_eval is not None:
            if existing_eval.get("success") is True:
                skipped_existing_json += 1
                continue
            if rerun_failed_only:
                rerun_count += 1

        if rerun_failed_only and existing_eval is None:
            continue

        try:
            pipeline_outputs = extract_pipeline_outputs(status_json, gene_id)

            if require_pipeline_outputs and not any(
                [
                    pipeline_outputs.get("short_summary"),
                    pipeline_outputs.get("extended_summary"),
                    pipeline_outputs.get("product_description"),
                ]
            ):
                skipped_no_outputs += 1
                print(
                    f"Skipping Comment ID {comment_id} ({gene_id}, PMID {pmid}): "
                    "no pipeline outputs found in status JSON"
                )
                continue

            print(f"[{i}/{len(df)}] Evaluating Comment ID {comment_id} | Gene {gene_id} | PMID {pmid}")

            evaluation_result = evaluate_user_comment_alignment(
                gene_id=gene_id,
                database=str(row.get("Database", "")),
                organism=str(row.get("Organism", "")),
                comment_id=comment_id,
                categories=str(row.get("Categories", "") if not pd.isna(row.get("Categories", "")) else ""),
                pubmed_ids=str(row.get("PubMed IDs", "") if not pd.isna(row.get("PubMed IDs", "")) else ""),
                is_single_pmid=bool(row.get("single_pmid", False)),
                is_name_product=bool(row.get("is_name_product", False)),
                headline=str(row.get("Headline", "") if not pd.isna(row.get("Headline", "")) else ""),
                content=str(row.get("Content", "") if not pd.isna(row.get("Content", "")) else ""),
                product_description=pipeline_outputs.get("product_description"),
                short_summary=pipeline_outputs.get("short_summary"),
                extended_summary=pipeline_outputs.get("extended_summary"),
                provider=provider,
                model=model,
                cache={"enabled": False},
            )

            comment_metadata = {
                "Comment ID": comment_id,
                "Gene ID": gene_id,
                "Database": str(row.get("Database", "")),
                "Organism": str(row.get("Organism", "")),
                "Headline": str(row.get("Headline", "") if not pd.isna(row.get("Headline", "")) else ""),
                "Content": str(row.get("Content", "") if not pd.isna(row.get("Content", "")) else ""),
                "Categories": str(row.get("Categories", "") if not pd.isna(row.get("Categories", "")) else ""),
                "PubMed IDs": str(row.get("PubMed IDs", "") if not pd.isna(row.get("PubMed IDs", "")) else ""),
                "single_pmid": bool(row.get("single_pmid", False)),
                "is_name_product": bool(row.get("is_name_product", False)),
                "pmid": pmid,
            }

            save_comment_evaluation_result(
                status_path=status_path,
                pmid=pmid,
                gene_id=gene_id,
                comment_id=comment_id,
                model=model,
                evaluation_result=evaluation_result,
                pipeline_outputs=pipeline_outputs,
                comment_metadata=comment_metadata,
            )

            if evaluation_result["success"]:
                flattened_eval = evaluation_result["parsed"]
                processed += 1
            else:
                flattened_eval = {
                    "success": False,
                    "error": evaluation_result["error"],
                    "raw_response": evaluation_result["raw_response"][:1000],
                }
                failed += 1
                print(f"  ✗ Comment ID {comment_id}: {evaluation_result['error']}")

                save_failed_eval_debug(
                    pmid=pmid,
                    gene_id=gene_id,
                    comment_id=comment_id,
                    model=model,
                    error=evaluation_result["error"],
                    raw_response=evaluation_result.get("raw_response", ""),
                    usage=evaluation_result.get("usage", {}),
                    seconds=evaluation_result.get("seconds", 0.0),
                    headline=str(row.get("Headline", "") if not pd.isna(row.get("Headline", "")) else ""),
                    content=str(row.get("Content", "") if not pd.isna(row.get("Content", "")) else ""),
                    short_summary=pipeline_outputs.get("short_summary"),
                    extended_summary=pipeline_outputs.get("extended_summary"),
                    product_description=pipeline_outputs.get("product_description"),
                )

            results.append(
                flatten_evaluation_row(
                    row=row,
                    pmid=pmid,
                    model=model,
                    pipeline_outputs=pipeline_outputs,
                    evaluation=flattened_eval,
                )
            )

            if len(results) >= save_every:
                batch_df = pd.DataFrame(results)
                mode = "a" if output_path.exists() else "w"
                header = not output_path.exists()
                batch_df.to_csv(output_path, sep="\t", index=False, mode=mode, header=header)
                print(f"Saved {len(batch_df)} rows to {output_path}")
                results = []

            if sleep_time:
                time.sleep(sleep_time)

        except Exception as e:
            failed += 1
            print(f"Error on Comment ID {comment_id} ({gene_id}, PMID {pmid}): {e}")
            continue

    if results:
        batch_df = pd.DataFrame(results)
        mode = "a" if output_path.exists() else "w"
        header = not output_path.exists()
        batch_df.to_csv(output_path, sep="\t", index=False, mode=mode, header=header)
        print(f"Saved final {len(batch_df)} rows to {output_path}")

    print("Run complete")
    print(
        f"processed={processed} | rerun_failed={rerun_count} | skipped_resume={skipped_resume} | "
        f"skipped_existing_json={skipped_existing_json} | skipped_no_pmid={skipped_no_pmid} | "
        f"skipped_missing_status={skipped_missing_status} | skipped_no_outputs={skipped_no_outputs} | "
        f"failed={failed}"
    )

    if output_path.exists():
        return pd.read_csv(output_path, sep="\t")
    return pd.DataFrame()


if __name__ == "__main__":
    df_results = run_user_comment_analysis()
    print(df_results.head())