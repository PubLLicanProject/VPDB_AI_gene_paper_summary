"""
Utility functions for file I/O, JSON handling, and status management.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from config.models import MAX_RETRY, FORMATTER_LLM

def extract_json(text: str):
    """
    Strip ```json fences if present and json-load the remainder.
    Returns a dict/list on success, or the raw text on failure.
    Use on LLM outputs to get string as json
    """
    text = text.strip()
    # remove ```json … ``` or plain ``` … ``` fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ Could not parse JSON – saving raw text. ({e})")
        return text


def save_status(pubmed_id, out_dir, step_key, gene_id, model_name, data, success=True, usage=None, seconds=None):
    """
    Save or merge LLM results for a PubMed ID + gene + workflow stage into a JSON file.

    Args:
        pubmed_id (str): The PubMed ID.
        out_dir (Path): Pathlib.Path object for the output directory.
        step_key (str): The workflow step, e.g. "getGeneSummary".
        gene_id (str): The gene locus tag.
        model_name (str): The model used for this step.
        data (any): The response data from the LLM.
        success (bool): Whether the LLM call succeeded.
        usage (dict): breakdown of input/output/total tokens used
        seconds (): time it took to complete pipeline step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    filepath = out_dir / f"{pubmed_id}.json"

    # Load existing data if present
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError: # NB this shouldn't happen if formatted correctly as per next if block upon saving.
            print(f"Warning: {filepath} is invalid JSON, overwriting.")

            existing_data = {}
    else:
        existing_data = {}


    retry_count = 1
    error_message = "Initial attempt failed to produce valid JSON." # default
    while not isinstance(data, (dict, list)) and retry_count <= max_retry:
        try:
            # Call LLM
            print(f"⚙️  Retrying with formatter model, attempt {retry_count}/{max_retry}")

            data_formatted, formatter_usage, formatter_secs = call_prompt(provider=formatter_llm[0],
                                                   model=formatter_llm[1],
                                                   system_prompt="Convert the supplied string to parsable JSON. respong with the corrected JSON ONLY and nothing else.",
                                                   user_prompts=[f"I tried parsing this JSON: <JSON> {data} </JSON> \n "
                                                                f"The following error message popped up: {error_message}"],
                                                   prefill_text="{")
            # print(f"Formatter answer: {data_formatted}")
            # make sure not a string for saving as json
            data = extract_json(data_formatted)
            # print(f"formatter_parsed data: {data}")

        except Exception as e:
            # print the error or log it
            error_message = f"parsing JSON failed: {e}"
            print(f"LLM call failed: {e}")
        finally:
            retry_count +=1

    if not isinstance(data, (dict, list)):
        print(f"❌ Failed to recover valid JSON after {max_retry} attempts.")
        return {"error": "Max retries exceeded"}, {}, None

    #
    # Merge new data into the existing file structure

    if step_key not in existing_data:
        existing_data[step_key] = {}

    # Create or update the gene entry
    if gene_id not in existing_data[step_key]:
        existing_data[step_key][gene_id] = {"gene_ID": gene_id}

    # Add or overwrite the model-specific block; success is True if we get this far
    existing_data[step_key][gene_id][model_name] = {
        "model": model_name,
        "success": True,
        "data": data,
        "usage": usage or {},
        "seconds": seconds
    }

    # ─────────────────────────────────────────
    # Save merged JSON
    # ─────────────────────────────────────────
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)

    # Return the final parsed data + token usage + time (for logging/debug)
    return data, usage or {}, seconds


def load_status(pubmed_id, out_dir, step_key, gene_id, llm):
    filepath = out_dir / f"{pubmed_id}.json"

    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"⚠️ Could not read {filepath} – invalid or unreadable JSON.")
        return None

        # Safely walk down the nesting
    step_data = data.get(step_key, {})
    gene_data = step_data.get(gene_id, {})
    model_entry = gene_data.get(llm)

    return model_entry  # will be None if any level missing

def check_if_ok(status_data):
    ok = (
            isinstance(status_data, dict)  # right type
            and status_data.get("success") is True  # flagged OK
            and isinstance(status_data.get("data"), dict)  # data exists
            and "error" not in status_data["data"]  # no nested error
    )
    if ok:
        return True
    return False


def get_prompt_and_replace(stage_key, replacements, prompt_type = "SystemPrompt"):
    """
    Retrieves a specific prompt text from the global_prompts dictionary and replace [] placeholders with provided values.

    Args:
        stage_key (str): The key for the prompt in the global_prompts dictionary.
        replacements (dict): Keys are placeholder strings in the prompt and values are the text to replace them with.
        prompt_type (str): Whether replacements are made in "UserPrompt" or "SystemPrompt".

    Returns:
        str: The prompt text with placeholders replaced by the corresponding values from replacements.
    """
    # two sub-helpers to deal with numerics and when a json needs to be a string for replacemtn
    def _stringify(value):
        """Convert non-strings (dicts, ints, etc.) to pretty JSON strings."""
        return value if isinstance(value, str) else json.dumps(value, indent=2)

    def _replace_in_text(text):
        for ph, val in replacements.items():
            text = text.replace(f"[{ph}]", _stringify(val))
        return text

    raw = global_prompts_and_schema[stage_key][prompt_type]

    # If we have a list of prompts as in the case of user prompts, process each element; otherwise process the single string.
    if isinstance(raw, list):
        return [_replace_in_text(txt) for txt in raw]
    else:
        return _replace_in_text(raw)



def _verified_to_select_candidates(verified: dict) -> dict:
    """
    Convert verifyPDs output into {"PDs": [{"description","evidence_code","code_reason"}, ...]}
    so it can be fed into selectPD(). Returns an empty "PDs" list if nothing usable.
    """
    try:
        if not isinstance(verified, dict):
            return {"PDs": []}

        def _extract_triplet(d):
            if not isinstance(d, dict):
                return None
            desc = d.get("description") or d.get("Description")
            ev = d.get("evidence_code") or d.get("evidenceCode") or d.get("evidence")
            reason = d.get("code_reason") or d.get("reason") or d.get("evidence_reason")
            if not isinstance(desc, str) or not desc.strip():
                return None
            return {"description": desc, "evidence_code": (ev or ""), "code_reason": (reason or "")}

        candidates = []

        # Include RPD if present
        rpd = verified.get("RPD")
        rpd_triplet = _extract_triplet(rpd)
        if rpd_triplet:
            candidates.append(rpd_triplet)

        # Include APDs (could be list or single)
        apds = verified.get("APDs")
        if isinstance(apds, list):
            for item in apds:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)
        else:
            t = _extract_triplet(apds)
            if t:
                candidates.append(t)

        # Fallback: sometimes verify step may echo a "PDs" list
        if not candidates and isinstance(verified.get("PDs"), list):
            for item in verified["PDs"]:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)

        # De-duplicate by description
        seen = set()
        unique = []
        for c in candidates:
            key = c["description"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return {"PDs": unique}
    except Exception:
        return {"PDs": []}

def _safe_get_pubmed_json(pmid: str, cache: Dict[str, Tuple[bool, Optional[dict], Optional[str]]]):
    """
    Why: avoid re-fetching and tolerate non-open access or missing papers without crashing.
    Cache shape: pmid -> (success, json_or_none, error_message_or_none)
    """
    if pmid in cache:
        return cache[pmid]
    try:
        j = get_pubmed_json(pmid)
        cache[pmid] = (True, j, None)
    except Exception as e:
        cache[pmid] = (False, None, str(e))
    return cache[pmid]


def _collect_per_paper_summary_for_model(gene_id: str, pmid: str, step_key: str, model_name: str) -> Tuple[List[str], List[str]]:
    """Return (bullets, aliases) from a specific step+model for PMID→gene.
    Aliases are only read from raw getGeneSummary when present."""
    entry = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=step_key, gene_id=gene_id, llm=model_name)
    if not entry or not isinstance(entry, dict):
        return [], []

    data = entry.get("data") if isinstance(entry.get("data"), (dict, list)) else entry

    bullets: List[str] = []
    try:
        bullets = collect_bullets(data)
    except Exception:
        bullets = []

    aliases: List[str] = []
    if step_key == "getGeneSummary" and isinstance(data, dict) and isinstance(data.get("Aliases_in_paper"), list):
        aliases = [a for a in data.get("Aliases_in_paper", []) if isinstance(a, str)]

    return bullets, aliases



def _scan_pmids_for_gene_by_model(gene_id: str, step_key: str, model_name: str) -> List[str]:
    """Find PMIDs in OUT_DIR that contain the given step_key for the gene under model_name."""
    pmids: List[str] = []
    for p in OUT_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        step = data.get(step_key, {})
        gene_block = step.get(gene_id, {})
        if model_name in gene_block:
            pmids.append(p.stem)
    return sorted(set(pmids))



