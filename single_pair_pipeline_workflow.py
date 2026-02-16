"""
Core workflow functions for the gene PD pipeline.

Implements the multi-stage pipeline:
1. getGeneSummary: Extract gene-specific information
2. generatePDs: Brainstorm product descriptions
3. verifyPDs: Verify and audit PDs
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any

from config.models import N_PDS
from pd_pipeline.utils import get_prompt_and_replace, extract_json
from pd_pipeline.llm_clients import call_prompt

def getGeneSummary(pubmed_text, gene_text, key = "getGeneSummary"):
    """
        Converts a gene ID and a list of synonyms into a readable string for prompts.

        Args:
            pubmed_text (str): paper to process
            gene_text (str): The database identifer for the gene and its aliases found in the text
            key (str): the dictionary key name for the specified stage

        Returns:
            dict: a JSON object with summary bullet points, each supported by quotes and a specified evidence location
        """

    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # Set up replacements inside user and system prompts.
    replacements = {"N_QUOTES": N_QUOTES,
                      "GENE": gene_text,
                      "JSON_SCHEMA":  JSON_SCHEMA,
                      "PAPER_TEXT": pubmed_text}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{get_summary_system}")
    user= get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{get_summary_user}")
    # print(f"Generating summary...")
    try:
        # Call LLM
        LLM_summary, usage, secs = call_prompt(provider=summary_llm[0],
                                  model=summary_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        # print(f"LLM_summary: {LLM_summary}")
        parsed_summary = extract_json(LLM_summary)
        # print(f"Parsed summary: {parsed_summary}")
        return parsed_summary, usage, secs

    except Exception as e:
         # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None


def verifyGeneSummary(gene_text, pubmed_text, summary_json, key="verifyGeneSummary"):
    """
        Double-checks the initial gene summary for completenes and correctness.

        Args:
            gene_text (str): PMID and its aliases found in the paper text
            pubmed_text (str): paper to refer back to
            summary_json (str): The original summary json to correct.
            key (str): the dictionary key name for the specified stage

        Returns:
            dict: a JSON object with summary bullet points, each supported by quotes and a specified evidence location
            plus verification status and reason for it.
        """

    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # Set up replacements inside user and system prompts.
    replacements = {"SUMMARY": summary_json,
                    "GENE": gene_text,
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "PAPER_TEXT": pubmed_text}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements,
                                                prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{get_summary_system}")
    user = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{get_summary_user}")
    # print(f"Verifying summary...")
    try:
        # Call LLM
        LLM_summary, usage, secs = call_prompt(provider=summary_QC_llm[0],
                                  model=summary_QC_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        # print(f"LLM_summary: {LLM_summary}")
        parsed_summary = extract_json(LLM_summary)
        # print(f"Parsed summary: {parsed_summary}")

        return parsed_summary, usage, secs

    except Exception as e:
        # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None


def collect_bullets(summary_json, wanted_status={"PASS", "WARN", "NEW"}):
    """
    Return a list of bullet_point strings from either a raw GeneSummary
    or a verified summary.  For verified summaries, keep only the
    bullets whose verification_status is in wanted_status.
    """
    if not summary_json:
        return []
    # ── 0) if it's already a list of bullet dicts ──────────────────
    if isinstance(summary_json, list):
        return [bp.get("bullet_point", "") for bp in summary_json]
    # 1) raw summary → simple list of dicts under "GeneSummary"
    if "GeneSummary" in summary_json:
        return [bp["bullet_point"] for bp in summary_json["GeneSummary"]]

    # 2) verified summary – sometimes under "VerifiedSummary"
    if "VerifiedSummary" in summary_json:
        return [
            bp["bullet_point"]
            for bp in summary_json["VerifiedSummary"]
            if bp.get("verification_status") in wanted_status
        ]

    # 3) verified summary per the schema
    #    (top object has {"type": "array", "items":[...]} )
    if summary_json.get("type") == "array" and "items" in summary_json:
        return [
            bp["bullet_point"]
            for bp in summary_json["items"]
            if bp.get("verification_status") in wanted_status
        ]

    # 4) last resort – nothing matched
    return []


def generatePDs(summary_json: Dict[Any, Any], gene_text: str, n_pds: int = N_PDs) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Generate product descriptions from gene summary.

    Args:
        summary_json: Output from getGeneSummary
        gene_text: Gene ID with aliases (e.g., "PF3D7_1234 (also known as ABC, DEF)")
        n_pds: Maximum number of PDs to generate

    Returns:
        Tuple of (parsed_result, usage_dict, elapsed_seconds)
        Returns (None, {}, None) on failure
    """
    # Validate input
    if not isinstance(summary_json, dict):
        print("  ERROR: generatePDs requires dict input from summary")
        return None, {}, None

    # Get schema
    schema = global_prompts_and_schema["generatePDs"]["ValidationSchema"]

    # Extract bullet points
    bullets = collect_bullets(summary_json)
    if not bullets:
        print("  WARNING: No bullet points found in summary")
        return {"error": "No bullets in summary"}, {}, None

    summary_text = "\n".join(bullets)

    # Build prompts
    replacements = {
        "N_PDs": n_pds,
        "GENE": gene_text,
        "JSON_SCHEMA": schema,
        "SUMMARY": summary_text
    }

    system_prompt = get_prompt_and_replace("generatePDs", replacements, "SystemPrompt")
    user_prompt = get_prompt_and_replace("generatePDs", replacements, "UserPrompts")

    # Call API and track timing
    start = time.time()

    try:
        response = anthropic_client.messages.create(
            model=PD_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=MODEL_TEMP,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        elapsed = time.time() - start

        # Track usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Parse with retry
        result = format_with_retry(response.content[0].text, schema)

        if not result:
            print("  ERROR: generatePDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: generatePDs API call failed: {e}")
        return None, {}, elapsed


def verifyPDs(brainstormed_pds: Dict[Any, Any], paper_text: str, gene_text: str,
              use_caching: bool = True) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Verify and select product descriptions against paper evidence.

    IMPORTANT: This function can reuse cached paper text from summary generation!
    When use_caching=True, the paper text will be cached for efficient reuse.

    Args:
        brainstormed_pds: Output from generatePDs
        paper_text: Full paper text (will be cached if use_caching=True)
        gene_text: Gene ID with aliases
        use_caching: Whether to cache paper text (should be True in batch mode)

    Returns:
        Tuple of (parsed_result, usage_dict, elapsed_seconds)
        Returns (None, {}, None) on failure
    """
    # Validate input
    if not isinstance(brainstormed_pds, dict):
        print("  ERROR: verifyPDs requires dict input from generatePDs")
        return None, {}, None

    # Get schema
    schema = global_prompts_and_schema["verifyPDs"]["ValidationSchema"]

    # Extract PDs into formatted list
    pds = brainstormed_pds.get("PDs", [])
    if not pds:
        print("  WARNING: No PDs found to verify")
        return {"error": "No PDs to verify"}, {}, None

    pd_lines = [
        f"{i + 1}. {pd['description']}  "
        f"[evidence_code: {pd.get('evidence_code', 'N/A')}]"
        for i, pd in enumerate(pds)
    ]
    pd_text = "\n".join(pd_lines)

    # Build prompts
    replacements = {
        "GENE": gene_text,
        "JSON_SCHEMA": schema,
        "PAPER_TEXT": paper_text,
        "PDs": pd_text
    }

    system_prompt = get_prompt_and_replace("verifyPDs", replacements, "SystemPrompt")
    user_prompt = get_prompt_and_replace("verifyPDs", replacements, "UserPrompts")

    # Call API with optional caching
    start = time.time()

    try:
        if use_caching:
            # Use caching structure - paper text will be cached
            # This reuses the cache from summary generation if within cache TTL!
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do not respond. Here is the paper text:"},
                        {"type": "text", "text": paper_text, "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "assistant", "content": "I have received the paper text."},
                {"role": "user", "content": user_prompt}
            ]

            response = anthropic_client.messages.create(
                model=PD_VERIFIER_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages
            )

            # Track cache usage
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
            }
        else:
            # No caching - simple call
            response = anthropic_client.messages.create(
                model=PD_VERIFIER_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        elapsed = time.time() - start

        # Parse with retry
        result = format_with_retry(response.content[0].text, schema)

        if not result:
            print("  ERROR: verifyPDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: verifyPDs API call failed: {e}")
        return None, {}, elapsed


def bullets_to_paragraph(merged_json: dict) -> str:
    """
    Build a plain-text paragraph from the merged literature bullets.
    - Uses only 'bullet_point' text.
    - Ensures each bullet ends with a full stop.
    Why: deterministic, non-LLM "literature_review" field.
    """
    if not isinstance(merged_json, dict):
        return ""
    bullets = []
    for item in (merged_json.get("LiteratureSummary") or []):
        bp = (item or {}).get("bullet_point", "")
        if not isinstance(bp, str) or not bp.strip():
            continue
        s = bp.strip()
        if s[-1:] not in ".!?":
            s += "."
        bullets.append(s)
    return " ".join(bullets)

def process_paper_gene_pair(pubmed_id = "31795916", gene_id = "Tb927.10.4200", save = True, verify_summary = False, generate_PD = True, verifyPDs = True, selectPDs = False):
    """
    Uses the optimal settings determined through pipeline variaiton testing by default:
    Generate gummary -> Brainstorm PDs -> verify against paper and set of rules and select best PD

    """
    print(f"Fetching paper text from PMID: {pubmed_id}...")
    # Get the PubMed JSON for the given ID.
    pubmed_json = get_pubmed_json(pubmed_id)
    # Parse the PubMed JSON to get the text of the required sections.
    pubmed_text = parse_pubmed_json(pubmed_json)

    print(f"Checking {gene_id} or its aliases are mentioned in the available text...")
    # Get the synonyms for the gene (e.g. from PlasmoDB).
    synonyms = get_gene_synonyms(gene_id, pubmed_text)
    gene_text = gene_to_prompt(gene_id, synonyms)
    #### STAGE 1 - summary gen
    key = "getGeneSummary"
    # check if summary already exists; if so load from status file and skip LLM call
    summary_data = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id, llm=summary_llm[1])
    # print(f"SUMMARY DATA LOADED:{summary_data}")

    if check_if_ok(summary_data):
        print(f"✔️ Found existing summary for {gene_id} in {pubmed_id}, skipping generation.")
        parsed_summary = summary_data["data"]
    else:
        print(f"Generating gene summary for {gene_id} from {pubmed_id}...")
        parsed_summary, usage, secs = getGeneSummary(pubmed_text, gene_text)
        # else get the data for the next step
        if save:
            # Save result
            parsed_summary, _, _ = save_status(pubmed_id=pubmed_id, ## update with formatter here for next steps
                                    out_dir=OUT_DIR,
                                    step_key=key,
                                    gene_id=gene_id,
                                    model_name=summary_llm[1],
                                    data=parsed_summary,
                                    success=isinstance(parsed_summary, (dict, list)), # only success if saved correctly
                                    usage=usage,
                                    seconds=secs)
            # print(f"Summary successfuly generated and saved in {OUT_DIR}")

    summary_for_PD = parsed_summary

    #### STAGE1.1 - optional summary verification.tends to make summary longer/more detailed which could be good, but hinders PD performance
    if verify_summary:
        key = "verifyGeneSummary"
        # check if summary already verified; if so load from status file and skip LLM call
        summary_data = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id, llm=summary_QC_llm[1])
        # check if it is a dict, not just if it exists here:

        if check_if_ok(summary_data):
            print(f"✔️ Found verified summary for {gene_id} in {pubmed_id}, skipping verification.")
            verified_summary = summary_data["data"]

        else:
            print(f"Verifying gene summary for {gene_id} from {pubmed_id}...")
            verified_summary, usage, secs = verifyGeneSummary(summary_json= parsed_summary, pubmed_text=pubmed_text, gene_text=gene_text)
            # else get the data for the next step
            if save:
                # Save result
                verified_summary, _,_ = save_status(pubmed_id=pubmed_id,
                                                    out_dir=OUT_DIR,
                                                    step_key=key,
                                                    gene_id=gene_id,
                                                    model_name=summary_QC_llm[1],
                                                    data=verified_summary,
                                                    success=isinstance(verified_summary, (dict, list)), # only success if saved correctly
                                                    usage=usage,
                                                    seconds=secs)
                # print(f"Summary successfuly verified and saved in {OUT_DIR}")
        summary_for_PD = verified_summary

    # terminate function if we are only using it to generate summary and not for PDs
    if not generate_PD:
        return

    #### STAGE 2: process only summary bullet points without quotes for the PD brainstorming
    key = "generatePDs"
    suggested_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                                  llm=PD_generator_llm[1])

    if check_if_ok(suggested_entry):
        print(f"✔️ Found suggested PDs for {gene_id} in {pubmed_id}, skipping generation.")
        suggested_PDs = suggested_entry["data"]
    else:
        print(f"Generating suggested PDs for {gene_id} from {pubmed_id}...")
        suggested_PDs, usage, secs = generatePDs(summary_json=summary_for_PD, gene_text=gene_text)
        if save:
            suggested_PDs, _, _ = save_status(pubmed_id=pubmed_id,
                                              out_dir=OUT_DIR,
                                              step_key=key,
                                              gene_id=gene_id,
                                              model_name=PD_generator_llm[1],
                                              data=suggested_PDs,
                                              success=isinstance(suggested_PDs, (dict, list)),
                                              usage=usage,
                                              seconds=secs)

    #### STAGE 3: verify PDs against paper text  and set of rules
    if not verify_PDs:
        return
    key = "verifyPDs"
    verified_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                                 llm=PD_QC_llm[1])

    if check_if_ok(verified_entry):
        print(f"✔️ Found verified PDs for {gene_id} in {pubmed_id}, skipping verification.")
        verified_PDs = verified_entry["data"]
    else:
        print(f"Verifying suggested PDs for {gene_id} from {pubmed_id}...")
        verified_PDs, usage, secs = verifyPDs(paper_text=pubmed_text, gene_text=gene_text,
                                              brainstormed_pds=suggested_PDs)
        if save:
            save_status(pubmed_id=pubmed_id,
                        out_dir=OUT_DIR,
                        step_key=key,
                        gene_id=gene_id,
                        model_name=PD_QC_llm[1],
                        data=verified_PDs,
                        usage=usage,
                        seconds=secs)

    #### STAGE 5: optional: select best PD FROM VERIFIED CANDIDATES and format based on set of examples
    if not selectPDs:
        return

    key = "selectPD"
    select_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                               llm=PD_picker_llm[1])

    if check_if_ok(select_entry):
        print(f"✔️ Found selected PD for {gene_id} in {pubmed_id}, skipping selection.")
        top_PD = select_entry["data"]
    else:
        print(f"Selecting best among VERIFIED PDs for {gene_id} from {pubmed_id}...")
        verified_candidates = _verified_to_select_candidates(verified_PDs)
        # Fallback: if verification produced nothing usable, fall back to suggested_PDs
        if not verified_candidates.get("PDs"):
            verified_candidates = {"PDs": suggested_PDs.get("PDs", []) if isinstance(suggested_PDs, dict) else []}
        top_PD, usage, secs = selectPD(summary_for_pd=summary_for_PD, suggestedPDs=verified_candidates)
        if save:
            top_PD, _, _ = save_status(pubmed_id=pubmed_id,
                                       out_dir=OUT_DIR,
                                       step_key=key,
                                       gene_id=gene_id,
                                       model_name=PD_picker_llm[1],
                                       data=top_PD,
                                       success=isinstance(top_PD, (dict, list)),
                                       usage=usage,
                                       seconds=secs)


