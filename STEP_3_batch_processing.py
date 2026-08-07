"""
additional helper functions specifically for the batch processing script
"""
import pandas as pd
from collections import defaultdict

from config.global_settings_batch import *
from pipeline.prompts import *
from pipeline.pubmed_helpers import *
from pipeline.supplementary_helpers import get_supplementary_text, count_supplementary_mentions
from pipeline.vpdb_helpers import *
from pipeline.utils import *
from pipeline.prompts import (_ensure_list, )

# Extract commonly used variables from global_prompts_and_schema
SUMMARY_SYSTEM_PROMPT = global_prompts_and_schema["getGeneSummary"]["SystemPrompt"]
VALIDATION_SCHEMA = global_prompts_and_schema["getGeneSummary"]["ValidationSchema"]
USER_PROMPTS = global_prompts_and_schema["getGeneSummary"]["UserPrompts"]
# NB can change default step checking behaviour in config
def is_gene_already_processed(
    pubmed_id: str,
    gene_id: str,
    check_all_steps: bool = True,
    flex_model: bool = True,
) -> bool:
    """
    Check if gene-paper pair has been fully processed.

    Args:
        pubmed_id: Paper ID
        gene_id: Gene ID
        check_all_steps: If True, checks summary + PDs + verify all complete
                        If False, only checks summary (backward compatible)
        flex_model:
            - True (default): each step may be completed by ANY model key (best resumability).
            - False: all required steps must be completed under the SAME single model key
                     (strict mode; useful if you expect a single model to do all steps).

    Returns:
        True if processing is complete
    """
    filename = OUT_DIR / f"{pubmed_id}.json"
    if not filename.exists():
        return False

    def _success_models(step_node: dict) -> set[str]:
        models = set()
        for k, v in (step_node or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("success") and v.get("data"):
                models.add(k)
        return models

    try:
        with open(filename, "r") as f:
            paper_data = json.load(f)

        # Old behavior - just check summary
        if not check_all_steps:
            step_node = paper_data.get("getGeneSummary", {}).get(gene_id, {})
            if not isinstance(step_node, dict):
                return False

            if flex_model:
                return len(_success_models(step_node)) > 0

            model_result = step_node.get(SUMMARY_MODEL, {})
            return bool(model_result.get("success") and model_result.get("data"))

        # New behavior - check all three steps
        required_steps = ["getGeneSummary", "generatePDs", "verifyPDs"]

        # Flexible: each step can be satisfied by any successful model key
        if flex_model:
            for step_key in required_steps:
                step_node = paper_data.get(step_key, {}).get(gene_id, {})
                if not isinstance(step_node, dict):
                    return False
                if len(_success_models(step_node)) == 0:
                    return False
            return True  # All steps complete (by any models)

        # Strict: all steps must share the same successful model key
        shared: set[str] | None = None
        for step_key in required_steps:
            step_node = paper_data.get(step_key, {}).get(gene_id, {})
            if not isinstance(step_node, dict):
                return False
            step_models = _success_models(step_node)
            if not step_models:
                return False
            shared = step_models if shared is None else (shared & step_models)
            if not shared:
                return False
        return True  # All steps complete under at least one shared model key

    except Exception:
        return False

POSSIBLE_SUMMARY_MODEL_KEYS = [
    SUMMARY_MODEL,
    EXISTING_SUMMARY_MODEL,
    "anthropic/claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
]
def load_existing_summary(pubmed_id: str, gene_id: str):
    filename = OUT_DIR / f"{pubmed_id}.json"
    if not filename.exists():
        return None

    try:
        with open(filename, "r") as f:
            data = json.load(f)

        # NEW: check modern nested format first
        node = data.get("getGeneSummary", {}).get(gene_id, {})
        for model_key in POSSIBLE_SUMMARY_MODEL_KEYS:
            model_blob = node.get(model_key)
            if not isinstance(model_blob, dict):
                continue
            if model_blob.get("success") is True and isinstance(model_blob.get("data"), dict) and "error" not in model_blob["data"]:
                return model_blob["data"]

            # Formatter-only repair path if raw_response exists
            raw = (model_blob.get("data") or {}).get("raw_response")
            if model_blob.get("success") is False and isinstance(raw, str) and raw.strip():
                repaired = format_with_retry(raw, VALIDATION_SCHEMA)
                if isinstance(repaired, dict):
                    save_result(pubmed_id, gene_id, repaired, True, "getGeneSummary", model=model_key)
                    return repaired

        # fallback: old formats you already had
        if gene_id in data and isinstance(data[gene_id], dict):
            return data[gene_id]

        return None
    except Exception:
        return None
def save_result(pubmed_id: str, gene_id: str, data: dict, success: bool,
                step_key: str = "getGeneSummary", usage: dict = None,
                seconds: float = None, model: str = None):
    """
    Save result to JSON file (one file per paper with nested genes and pipeline steps).

    Args:
        pubmed_id: Paper ID
        gene_id: Gene ID
        data: Result data (or error dict)
        success: Whether step succeeded
        step_key: Pipeline step name ("getGeneSummary", "generatePDs", "verifyPDs")
        usage: Token usage dict
        seconds: Elapsed time
        model: Model name (defaults to appropriate model for step_key)
    """
    filename = OUT_DIR / f"{pubmed_id}.json"

    # Load existing file if it exists
    if filename.exists():
        with open(filename, 'r') as f:
            paper_data = json.load(f)
    else:
        paper_data = {}

    # Initialize step if needed
    if step_key not in paper_data:
        paper_data[step_key] = {}

    # Add or update this gene's data
    if gene_id not in paper_data[step_key]:
        paper_data[step_key][gene_id] = {"gene_ID": gene_id}

    # Determine model name if not provided
    if model is None:
        if step_key == "getGeneSummary":
            model = SUMMARY_MODEL
        elif step_key == "generatePDs":
            model = PD_GENERATOR_MODEL
        elif step_key == "verifyPDs":
            model = PD_VERIFIER_MODEL
        else:
            model = SUMMARY_MODEL  # fallback

    # Store model results
    paper_data[step_key][gene_id][model] = {
        "model": model,
        "success": success,
        "data": data
    }

    if usage:
        paper_data[step_key][gene_id][model]["usage"] = usage
    if seconds:
        paper_data[step_key][gene_id][model]["seconds"] = seconds

    # Save back to file
    with open(filename, 'w') as f:
        json.dump(paper_data, f, indent=2)

def format_with_retry(content: str, schema: dict, max_attempts: int = MAX_RETRY, label: str = "") -> Optional[dict]:
    """Parse JSON with retry using formatter model."""

    # 0. Never fabricate from nothing: an empty/whitespace response means the model returned no answer
    #    (e.g. a safety-classifier refusal). Handing it to the formatter would invent a schema-shaped
    #    placeholder, so fail honestly instead.
    if not content or not content.strip():
        print("  ! Empty response — treating as failure (not fabricating).")
        return None

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
                _safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(label))[:80]
                debug_file = OUT_DIR / f"debug_failed_parse_{_safe or 'unknown'}_{int(time.time())}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Original content:\n{content}\n\nSchema:\n{json.dumps(schema, indent=2)}")
                print(f"  Raw response saved to: {debug_file}")
                return None
            time.sleep(1)

    return None

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
        if PROVIDER == "anthropic":
            response = anthropic_client.messages.create(
                model=PD_GENERATOR_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            elapsed = time.time() - start
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            raw = response.content[0].text
        elif PROVIDER == "openrouter":
            # OpenRouter: structured outputs + response-healing (massively reduces formatter use)
            rf = response_format_for_schema("generatePDs", schema)
            plugins = [{"id": "response-healing"}]

            raw, usage, elapsed = call_prompt(
                provider=PROVIDER,
                model=PD_GENERATOR_MODEL,
                system_prompt=system_prompt,
                user_prompts=user_prompt,
                prefill_text="{",
                cache={"enabled": False}, # we do not use paper text for this step so no caching
                response_format=rf, # call_prompt suto-disables this if model doen't support structured outputs
                plugins=plugins,
            )

            # OpenRouter/Claude often returns ONLY the completion after the prefill.
            # raw = raw.strip()
            # if not raw.startswith("{"):
            #     print("  ! Prepending missing prefill brace to response")
            #     raw = "{" + raw
        else:
            raise ValueError(f"Unsupported PROVIDER '{PROVIDER}' in generatePDs. Use 'anthropic' or 'openrouter'.")


        # Parse with retry
        result = format_with_retry(raw, schema)

        if not result:
            print("  ERROR: generatePDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: generatePDs API call failed: {e}")
        return None, {}, elapsed


def verifyPDs(suggested_pds: Dict[Any, Any], paper_text: str, gene_text: str,
              use_caching: bool = True) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Verify and select product descriptions against paper evidence.

    IMPORTANT: This function can reuse cached paper text from summary generation!
    When use_caching=True, the first user prompt (paper text)
    is sent as a multipart content block with cache_control so subsequent calls for the same
    paper can hit the cache. OpenRouter sticky routing keys off the first system message and
    first non-system message, so keep the opening messages consistent.

    Args:
        suggested_pds: Output from generatePDs
        paper_text: Full paper text (will be cached if use_caching=True)
        gene_text: Gene ID with aliases
        use_caching: Whether to cache paper text (should be True in batch mode)

    Returns:
        Tuple of (parsed_result, usage_dict, elapsed_seconds)
        Returns (None, {}, None) on failure
    """
    # Validate input
    if not isinstance(suggested_pds, dict):
        print("  ERROR: verifyPDs requires dict input from generatePDs")
        return None, {}, None

    # Get schema
    schema = global_prompts_and_schema["verifyPDs"]["ValidationSchema"]

    # Extract PDs into formatted list
    pds = suggested_pds.get("PDs", [])
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
    user_prompts = get_prompt_and_replace("verifyPDs", replacements, "UserPrompts")

    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]
    # Call API with optional caching
    start = time.time()

    rf = response_format_for_schema("verifyPDs", schema) if PROVIDER == "openrouter" else None
    plugins = [{"id": "response-healing"}] if PROVIDER == "openrouter" else None
    cache_cfg = {"enabled": True, "ttl": "1h"} if (PROVIDER == "openrouter" and use_caching) else {"enabled": False}

    try:
        if PROVIDER == "anthropic":
            if use_caching:
                # Optimized for Anthropic Native Caching
                messages = [
                    {
                        "role": "user",
                        "content": [
                           {"type": "text", "text": user_prompts[0], "cache_control": {"type": "ephemeral"}}
                        ]
                    },
                    {"role": "assistant", "content": "I have received the paper text."},
                    {"role": "user", "content": user_prompts[1]}
                ]
                response = anthropic_client.messages.create(
                    model=PD_VERIFIER_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=MODEL_TEMP,
                    system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                    messages=messages
                )
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                    "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
                }
            else:
                # Standard Anthropic call
                response = anthropic_client.messages.create(
                    model=PD_VERIFIER_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=MODEL_TEMP,
                    system=system_prompt,
                    messages= [{"role": "user", "content": p} for p in user_prompts]
                )
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            elapsed = time.time() - start
            raw = response.content[0].text

        elif PROVIDER == "openrouter":

           raw, usage, elapsed = call_prompt(
                provider=PROVIDER,
                model=PD_VERIFIER_MODEL,
                system_prompt=system_prompt,
                user_prompts=user_prompts,  # Pass as single list item
                prefill_text="{",
                cache=cache_cfg,
                response_format=rf,
                plugins=plugins,
            )

        else:
            raise ValueError(f"Unsupported PROVIDER '{PROVIDER}' in verifyPDs.")

        # Parse with retry (Now uses the updated format_with_retry we fixed in Step 2)
        result = format_with_retry(raw, schema)

        if not result:
            print("  ERROR: verifyPDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start  # Use perf_counter for accuracy
        print(f"  ERROR: verifyPDs API call failed: {e}")
        return None, {}, elapsed

#
def _step_schema(step_key: str) -> Dict[str, Any]:
    return global_prompts_and_schema[step_key]["ValidationSchema"]


def _step_system(step_key: str, replacements: Dict[str, Any]) -> str:
    return get_prompt_and_replace(step_key, replacements, "SystemPrompt")


def _step_user(step_key: str, replacements: Dict[str, Any]) -> List[str]:
    return _ensure_list(get_prompt_and_replace(step_key, replacements, "UserPrompts"))


def _format_openrouter_cache_info(usage: Dict[str, Any]) -> str:
    ptd = usage.get("prompt_tokens_details") or {}
    cached = int(ptd.get("cached_tokens", 0) or 0)
    wrote = int(ptd.get("cache_write_tokens", 0) or 0)
    cost = usage.get("cost", None)
    bits = [f"cached={cached:,}", f"write={wrote:,}"]
    if cost is not None:
        bits.append(f"cost={cost}")
    return " ".join(bits)

# pubmed_id = "12057017"
# gene_list = [('PF3D7_1414400', 'plasmodb')]
def process_paper_with_caching(pubmed_id: str, gene_list: List[Tuple[str, str]],
                               save: bool = True) -> List[dict]:
    """Process multiple genes from same paper with prompt caching for the paper text."""
    print(f"\n{'=' * 80}")
    print(f"Paper {pubmed_id}: {len(gene_list)} genes (USING CACHING)")
    print(f"{'=' * 80}")

    try:
        print("Fetching paper text...")
        paper_text = get_paper_text(pubmed_id)
    except PaperNotInOA as e:
        # permanent: genuinely not in the OA subset
        print(f"  ✗ Paper not in PMC OA subset: {e}")
        return [
            {
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": f"Paper not available in PMC OA subset: {e}",
                "paper_available": False,
                "alias_in_text": False,
                "mentions": 0,
            }
            for gene_id, _host_db in gene_list
        ]
    except Exception as e:
        # transient fetch failure (timeout/5xx/429) that survived retries -> retryable on re-run.
        # paper_available left as None (unknown) to avoid mislabelling an available paper.
        print(f"  ✗ Paper fetch failed (transient/retryable): {e}")
        return [
            {
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": f"Paper fetch error (transient/retryable): {e}",
                "paper_available": None,
                "alias_in_text": False,
                "mentions": 0,
            }
            for gene_id, _host_db in gene_list
        ]

    results = []

    for i, (gene_id, host_db) in enumerate(gene_list, 1):
        print(f"  [{i}/{len(gene_list)}] {gene_id}...", end=" ")

        alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, host_db)
        aliases = None  # computed lazily below (early only if the supplement gate needs it)
        gene_source = "main_text"
        if not alias_in_text:
            suppl_hit = 0
            if FETCH_SUPPLEMENTARY:
                aliases = get_gene_synonyms(gene_id, paper_text, host_db)
                _sm = count_supplementary_mentions(pubmed_id, gene_id, aliases, host_db,
                                                   caps=(SUPPLEMENTARY_CAPS or None))
                suppl_hit = _sm.get("mentions", 0)
            if suppl_hit > 0:
                # gene absent from the main text but present in the supplement -> unlock it
                gene_source = "supplement_only"
                mentions = suppl_hit
                print(f"(supplement x{suppl_hit}) ", end="")
            else:
                print("✗ Gene/aliases not found in text or supplement (skipped)")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "Gene/aliases not found in paper text or supplement",
                    "paper_available": True,
                    "alias_in_text": False,
                    "mentions": 0,
                })
                continue
        else:
            print(f"({mentions} mentions) ", end="")

        if not OVERWRITE_EXISTING and is_gene_already_processed(pubmed_id, gene_id):
            print("✓ Already processed (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "skipped": True,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions,
            })
            continue

        existing_summary = None if OVERWRITE_EXISTING else load_existing_summary(pubmed_id, gene_id)


        try:
            if aliases is None:
                aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            raw = None
            usage = {}
            elapsed = 0.0

            if existing_summary is not None:
                result = existing_summary
                print("✓ Summary loaded from cache", end=" | ")
            else:
                # Build user prompts from global schema templates
                # (Prompt 0 contains PAPER_TEXT and should be cached; prompt 1 is gene-specific)
                step_key = "getGeneSummary"
                user_prompts = get_prompt_and_replace(
                    stage_key=step_key,
                    replacements={"PAPER_TEXT": paper_text, "GENE": gene_display},
                    prompt_type="UserPrompts",
                )
                JSON_SCHEMA = global_prompts_and_schema.get(step_key).get("ValidationSchema")
                system_prompt = get_prompt_and_replace(
                    stage_key=step_key,
                    replacements={"N_QUOTES": N_QUOTES, "JSON_SCHEMA": JSON_SCHEMA},
                    prompt_type="SystemPrompt",
                )

                # Optionally augment with gene-filtered supplementary materials.
                # Appended to the gene-specific prompt (last message) so prompt[0]
                # (the paper text) stays identical across genes and remains cacheable.
                if FETCH_SUPPLEMENTARY:
                    _suppl = get_supplementary_text(pubmed_id, gene_id, aliases, host_db,
                                                    caps=(SUPPLEMENTARY_CAPS or None))
                    if _suppl:
                        user_prompts = list(user_prompts)
                        user_prompts[-1] = f"{user_prompts[-1]}\n\n{_suppl}"
                        print("📎+suppl", end=" ")

                if PROVIDER == "anthropic":
                    start = time.time()
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompts[0], "cache_control": {"type": "ephemeral"}},
                            ],
                        },
                        {"role": "assistant", "content": "I have received the paper text."},
                        {"role": "user", "content": user_prompts[1]},
                    ]
                    response = anthropic_client.messages.create(
                        model=SUMMARY_MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=MODEL_TEMP,
                        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                        messages=messages,
                    )
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                    }
                    elapsed = time.time() - start
                    raw = response.content[0].text

                elif PROVIDER == "openrouter":
                    cache_cfg = {"enabled": True, "ttl": "1h"}  # upped to 1h as caching wasnt working as intended at default 5 m
                    rf = response_format_for_schema("getGeneSummary", VALIDATION_SCHEMA)
                    plugins = [{"id": "response-healing"}]

                    raw, usage, elapsed = call_prompt(
                        provider=PROVIDER,
                        model=SUMMARY_MODEL,
                        system_prompt=system_prompt,
                        user_prompts=user_prompts,
                        prefill_text="{",  # ignored when response_format is set
                        cache=cache_cfg,  # caching ON explicitly
                        response_format=rf,  # strict JSON schema
                        plugins=plugins,
                    )

                else:
                    raise ValueError(
                        f"Unsupported PROVIDER '{PROVIDER}' in process_paper_with_caching. "
                        f"Use 'anthropic' or 'openrouter'."
                    )

                result = format_with_retry(raw, VALIDATION_SCHEMA, label=f"{pubmed_id}_{gene_id}")

                if not result:
                    print("✗ Failed to parse response - saving raw output for debugging")
                    debug_data = {
                        "error": "JSON parsing failed after retries",
                        "raw_response": (raw or "")[:1000],
                        "model": SUMMARY_MODEL,
                    }
                    if save:
                        save_result(pubmed_id, gene_id, debug_data, False, "getGeneSummary", usage, elapsed)
                    results.append({
                        "pubmed_id": pubmed_id,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "JSON parsing failed",
                        "paper_available": True,
                        "alias_in_text": True,
                        "mentions": mentions,
                    })
                    continue

                if save:
                    save_result(pubmed_id, gene_id, result, True, "getGeneSummary", usage, elapsed)

                print(f"✓ Summary ({elapsed:.1f}s)", end=" | ")

            # ✅ IMPORTANT: this must run for BOTH cached and newly-generated summaries
            if check_if_in_passing(result):
                print("⚠️ Mentioned in passing only - skipping PD generation")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": True,
                    "data": result,
                    "usage": usage,
                    "summary_time": elapsed,
                    "in_passing": True,
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions,
                })
                continue

            # STEP 2: Generate PDs
            pd_result, pd_usage, pd_time = generatePDs(
                summary_json=result,
                gene_text=gene_display,
                n_pds=N_PDs,
            )

            if not pd_result:
                print("✗ PD gen failed")
                if save:
                    save_result(pubmed_id, gene_id, {"error": "PD generation failed"},
                               False, "generatePDs", pd_usage, pd_time)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "PD generation failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions,
                })
                continue

            if save:
                save_result(pubmed_id, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)

            print(f"✓ PDs ({(pd_time or 0):.1f}s)", end=" | ")
            print(f"Usage: {_format_openrouter_cache_info(pd_usage)}")

            # STEP 3: Verify PDs
            print("verifying...", end=" ", flush=True)
            verify_result, verify_usage, verify_time = verifyPDs(
                suggested_pds=pd_result,
                paper_text=paper_text,
                gene_text=gene_display,
                use_caching=True,
            )

            if not verify_result:
                print("✗ Verify failed")
                if save:
                    save_result(pubmed_id, gene_id, {"error": "PD verification failed"},
                               False, "verifyPDs", verify_usage, verify_time)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "PD verification failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions,
                })
                continue

            if save:
                save_result(pubmed_id, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)

            total_time = (elapsed or 0) + (pd_time or 0) + (verify_time or 0)
            if PROVIDER == "anthropic":
                cache_info = (
                    f"cache={verify_usage.get('cache_read_input_tokens', 0):,}"
                    if i > 1 else f"created={verify_usage.get('cache_creation_input_tokens', 0):,}"
                )
                print(f"✓ Verified ({verify_time:.1f}s) | Total: {total_time:.1f}s ({cache_info})")
            else:
                print(f"✓ Verified ({(verify_time or 0):.1f}s) | Total: {total_time:.1f}s")
                if verify_usage:
                    print(f"Usage: {_format_openrouter_cache_info(verify_usage)}")

            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "data": result,
                "usage": usage,
                "summary_time": elapsed,
                "pd_time": pd_time,
                "verify_time": verify_time,
                "total_time": total_time,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions,
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": str(e),
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions,
            })

    return results

def has_successful_step(
    data: Dict[str, Any],
    step_key: str,
    gene_id: str,
    *,
    model: Optional[str] = None,
    any_model: bool = False,
) -> bool:
    """Return True if a saved status file indicates the step is complete."""
    step_node = (data.get(step_key) or {}).get(gene_id)
    if not isinstance(step_node, dict):
        return False

    if any_model:
        for v in step_node.values():
            if isinstance(v, dict) and v.get("success") and v.get("data"):
                return True
        return False

    if model is None:
        return False

    model_node = step_node.get(model)
    return isinstance(model_node, dict) and bool(model_node.get("success")) and bool(model_node.get("data"))

def process_batch_fallback(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
    """Fallback to standard processing when batch API is unavailable."""
    print(f"Processing {len(pairs)} pairs with standard API (sequential)...")

    results = []
    
    for i, (pubmed_id, gene_id, host_db) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {pubmed_id}_{gene_id}...", end=" ")

        # Check if already processed
        if not OVERWRITE_EXISTING and is_gene_already_processed(pubmed_id, gene_id):
            print("✓ Already processed (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "skipped": True
            })
            continue

        start = time.time()

        # Initialize diagnostic flags
        paper_available = False
        alias_in_text = False
        mentions = 0

        try:
            # Fetch paper - check if available
            try:
                paper_text = get_paper_text(pubmed_id)
                paper_available = True
            except Exception as e:
                print(f"✗ Paper not available ({time.time() - start:.1f}s)")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": f"Paper not available: {str(e)}",
                    "paper_available": False,
                    "alias_in_text": False,
                    "mentions": 0
                })
                continue

            # Check if gene/aliases are in text
            alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, host_db)

            if not alias_in_text:
                print(f"✗ Gene not in text ({time.time() - start:.1f}s)")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "Gene/aliases not found in paper text",
                    "paper_available": True,
                    "alias_in_text": False,
                    "mentions": 0
                })
                continue

            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            existing_summary = None if OVERWRITE_EXISTING else load_existing_summary(pubmed_id, gene_id)

            if existing_summary is not None:
                result = existing_summary
                usage = {}
                elapsed = 0.0
                print(f"✓ Summary loaded from cache", end=" | ")
            else:
                step_key = "getGeneSummary"
                schema = global_prompts_and_schema[step_key]["ValidationSchema"]

                system_prompt = get_prompt_and_replace(
                    step_key,
                    {"N_QUOTES": N_QUOTES, "JSON_SCHEMA": schema},
                    "SystemPrompt",
                )
                user_prompts = get_prompt_and_replace(
                    step_key,
                    {"PAPER_TEXT": paper_text, "GENE": gene_display},
                    "UserPrompts",
                )
                # Optionally augment with gene-filtered supplementary materials
                # (appended to the gene-specific prompt to keep prompt[0] cacheable).
                if FETCH_SUPPLEMENTARY:
                    _suppl = get_supplementary_text(pubmed_id, gene_id, aliases, host_db,
                                                    caps=(SUPPLEMENTARY_CAPS or None))
                    if _suppl:
                        user_prompts = list(user_prompts)
                        user_prompts[-1] = f"{user_prompts[-1]}\n\n{_suppl}"
                        print("📎+suppl", end=" ")
                # Call API
                if PROVIDER == "anthropic":
                    start = time.time()
                    response = anthropic_client.messages.create(
                        model=SUMMARY_MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=MODEL_TEMP,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompts[0]},
                            {"role": "assistant", "content": "I have received the paper text."},
                            {"role": "user", "content": user_prompts[1]},
                        ]
                    )
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                    elapsed = time.time() - start
                    raw = response.content[0].text
                elif PROVIDER == "openrouter":
                    rf = response_format_for_schema(step_key, schema)
                    plugins = [{"id": "response-healing"}]
                    raw, usage, elapsed = call_prompt(
                        provider=PROVIDER,
                        model=SUMMARY_MODEL,
                        system_prompt=system_prompt,
                        user_prompts=user_prompts,
                        prefill_text="{",
                        response_format=rf,
                        plugins=plugins,
                    )
                else:
                    raise ValueError(
                        f"Unsupported PROVIDER '{PROVIDER}' in process_batch_fallback. Use 'anthropic' or 'openrouter'.")

                # Parse result
                result = format_with_retry(raw, VALIDATION_SCHEMA)
                # elapsed = time.time() - start # dont overwrite call_prompt tiome

                if not result:
                    print(f"✗ Parse failed ({elapsed:.1f}s)")
                    error_data = {"error": "JSON parsing failed", "raw_response": raw[:1000]}
                    if save:
                        save_result(pubmed_id, gene_id, error_data, False, "getGeneSummary", usage, elapsed)
                    results.append({
                        "pubmed_id": pubmed_id,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "JSON parsing failed",
                        "paper_available": True,
                        "alias_in_text": True,
                        "mentions": mentions
                    })
                else:
                    # Summary success
                    if save:
                        save_result(pubmed_id, gene_id, result, True, "getGeneSummary", usage, elapsed)

                    print(f"✓ Summary ({elapsed:.1f}s)", end=" | ")
            # Check if gene is only mentioned in passing
            if check_if_in_passing(result):
                print(f"⚠️ Mentioned in passing only - skipping PD generation")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": True,
                    "data": result,
                    "usage": usage,
                    "summary_time": elapsed,
                    "in_passing": True,
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue  # Skip to next gene


            # STEP 2: Generate PDs
            try:
                print("generating PDs...", end=" ", flush=True)
                pd_result, pd_usage, pd_time = generatePDs(
                    summary_json=result,
                    gene_text=gene_display,
                    n_pds=N_PDs
                )

                if not pd_result:
                    print("✗ PD gen failed")
                    if save:
                        save_result(
                            pubmed_id, gene_id,
                            {"error": "PD generation failed"},
                            False, "generatePDs", pd_usage, pd_time
                        )
                    results.append({
                        "pubmed_id": pubmed_id,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "PD generation failed",
                        "paper_available": True,
                        "alias_in_text": True,
                        "mentions": mentions
                    })
                    continue

                if save:
                    save_result(pubmed_id, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)

                print(f"✓ PDs ({(pd_time or 0):.1f}s)", end=" | ")

            except Exception as e:
                print(f"✗ PD error: {e}")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": f"PD generation error: {str(e)}",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue

            # STEP 3: Verify PDs
            try:
                print("verifying...", end=" ", flush=True)
                verify_result, verify_usage, verify_time = verifyPDs(
                    suggested_pds=pd_result,
                    paper_text=paper_text,
                    gene_text=gene_display,
                    use_caching=False
                )

                if not verify_result:
                    print("✗ Verify failed")
                    if save:
                        save_result(
                            pubmed_id, gene_id,
                            {"error": "PD verification failed"},
                            False, "verifyPDs", verify_usage, verify_time
                        )
                    results.append({
                        "pubmed_id": pubmed_id,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "PD verification failed",
                        "paper_available": True,
                        "alias_in_text": True,
                        "mentions": mentions
                    })
                    continue

                if save:
                    save_result(pubmed_id, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)

                total_time = (elapsed or 0) + (pd_time or 0) + (verify_time or 0)
                print(f"✓ Verified ({(verify_time or 0):.1f}s) | Total: {total_time:.1f}s")

            except Exception as e:
                print(f"✗ verify PD error: {e}")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": f"PD verification error: {str(e)}",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue

            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "data": result,
                "usage": usage,
                "summary_time": elapsed,
                "pd_time": pd_time,
                "verify_time": verify_time,
                "total_time": total_time,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ Error: {e} ({elapsed:.1f}s)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": str(e),
                "paper_available": paper_available,
                "alias_in_text": alias_in_text,
                "mentions": mentions
            })

    return results

def create_batch_requests(pairs: List[Tuple[str, str, str]]) -> List[dict]:
    """Create batch requests for low-density papers."""
    batch_requests = []
    step_key = "getGeneSummary"
    schema = global_prompts_and_schema[step_key]["ValidationSchema"]

    for pubmed_id, gene_id, host_db in pairs:
        try:
            paper_text = get_paper_text(pubmed_id)
            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            system_prompt = get_prompt_and_replace(
                step_key,
                {"N_QUOTES": N_QUOTES, "JSON_SCHEMA": schema},
                "SystemPrompt",
            )
            user_prompts = get_prompt_and_replace(
                step_key,
                {"PAPER_TEXT": paper_text, "GENE": gene_display},
                "UserPrompts",
            )

            batch_requests.append({
                "custom_id": f"{pubmed_id}_{gene_id}",
                "params": {
                    "model": SUMMARY_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "temperature": MODEL_TEMP,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompts[0]},
                        {"role": "assistant", "content": "I have received the paper text."},
                        {"role": "user", "content": user_prompts[1]},
                    ]
                }
            })
        except Exception as e:
            print(f"  ✗ Failed to prepare {pubmed_id}_{gene_id}: {e}")

    return batch_requests

# def process_batch(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
#     """Process pairs using batch API (with fallback to standard if unavailable)."""
#     print(f"\n{'=' * 80}")
#     print(f"BATCH PROCESSING: {len(pairs)} pairs")
#     print(f"{'=' * 80}")
#
#     # Check if batch API is available
#     if not hasattr(anthropic_client.messages, 'batches'):
#         print("Warning: Batch API not available in your Anthropic SDK version.")
#         print("Falling back to standard processing...")
#         print("To enable batch API, upgrade: pip install --upgrade anthropic")
#         print()
#         return process_batch_fallback(pairs, save)
#
#     batch_requests = create_batch_requests(pairs)
#     if not batch_requests:
#         return []
#
#     try:
#         batch = anthropic_client.messages.batches.create(requests=batch_requests)
#         batch_id = batch.id
#         print(f"Batch submitted: {batch_id}")
#     except Exception as e:
#         print(f"Batch submission failed: {e}")
#         print("Falling back to standard processing...")
#         return process_batch_fallback(pairs, save)
#
#     # Wait for completion
#     while True:
#         batch = anthropic_client.messages.batches.retrieve(batch_id)
#         counts = batch.request_counts
#         total = counts.processing + counts.succeeded + counts.errored
#         print(f"  Status: {batch.processing_status} | Succeeded: {counts.succeeded}/{total}", end="\r")
#
#         if batch.processing_status in ["ended", "canceled", "expired"]:
#             break
#         time.sleep(60)
#
#     print(f"\n✓ Batch completed")
#
#     # Retrieve results
#     results = []
#     for result in anthropic_client.messages.batches.results(batch_id):
#         pmid, gene_id = result.custom_id.split("_", 1)
#         mention_count = 0
#         if result.result.type == "succeeded":
#             try:
#                 content = result.result.message.content[0].text
#                 parsed = format_with_retry(content, VALIDATION_SCHEMA)
#
#                 if not parsed:
#                     # Parsing failed
#                     error_data = {
#                         "error": "JSON parsing failed",
#                         "raw_response": content[:1000]
#                     }
#                     if save:
#                         save_result(pmid, gene_id, error_data, False, "getGeneSummary")
#                     results.append({
#                         "pubmed_id": pmid,
#                         "gene_id": gene_id,
#                         "success": False,
#                         "error": "JSON parsing failed",
#                         "paper_available": True,
#                         "alias_in_text": True,
#                         "mentions": 0,
#                     })
#                     continue  # ✅ don't fall through
#
#                if check_if_in_passing(parsed):
#                      print(f"  ⚠️ {gene_id} in passing - PDs skipped")
#                      results.append({
#                          "pubmed_id": pmid,
#                          "gene_id": gene_id,
#                          "success": False,
#                          "error": "JSON parsing failed",
#                          "paper_available": True,  # Paper was available (batch succeeded)
#                          "alias_in_text": True,    # Assumed true if batch API succeeded
#                          "mentions": 0             # Unknown for batch
#                         })
#                         continue # skip to next result without generating PD
#                 else:
#                     # Summary success - now process PDs
#                     if save:
#                         save_result(pmid, gene_id, parsed, True, "getGeneSummary")
#
#                     # For PD steps, we need paper text and gene display
#                     # Extract host_db from pairs
#                     host_db = next((hdb for p, g, hdb in pairs if p == pmid and g == gene_id), None)
#
#                     paper_available = True
#                     alias_in_text = True  # Assumed if summary was generated
#
#                     if host_db:
#                         try:
#                             # Fetch paper and prepare gene display
#                             paper_text = get_paper_text(pmid)
#
#                             # Check gene in text
#                             alias_in_text, mention_count = check_gene_in_text(gene_id, paper_text, host_db)
#
#                             aliases = get_gene_synonyms(gene_id, paper_text, host_db)
#                             gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id
#
#                             # Generate PDs
#                             print(f"generating PDs...", end=" ", flush=True)
#                             pd_result, pd_usage, pd_time = generatePDs(
#                                 summary_json=parsed,
#                                 gene_text=gene_display,
#                                 n_pds=N_PDs
#                             )
#
#                             if pd_result and save:
#                                 save_result(pmid, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)
#
#                             # Verify PDs (use caching even though batch papers are low-density)
#                             if pd_result:
#                                 verify_result, verify_usage, verify_time = verifyPDs(
#                                     suggested_pds=pd_result,
#                                     paper_text=paper_text,
#                                     gene_text=gene_display,
#                                     use_caching=False  # Don't use caching for batch (single genes)
#                                 )
#
#                                 if verify_result and save:
#                                     save_result(pmid, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)
#
#                         except Exception as e:
#                             print(f"  Warning: PD processing failed for {gene_id}: {e}")
#
#                     results.append({
#                         "pubmed_id": pmid,
#                         "gene_id": gene_id,
#                         "success": True,
#                         "data": parsed,
#                         "paper_available": paper_available,
#                         "alias_in_text": alias_in_text,
#                         "mentions": mention_count if alias_in_text else 0
#                     })
#             except Exception as e:
#                 results.append({
#                     "pubmed_id": pmid,
#                     "gene_id": gene_id,
#                     "success": False,
#                     "error": str(e),
#                     "paper_available": True,  # Batch succeeded so paper was available
#                     "alias_in_text": None,    # Unknown if exception occurred
#                     "mentions": 0
#                 })
#         else:
#             results.append({
#                 "pubmed_id": pmid,
#                 "gene_id": gene_id,
#                 "success": False,
#                 "error": result.result.error.message,
#                 "paper_available": None,  # Unknown if batch failed
#                 "alias_in_text": None,
#                 "mentions": 0
#             })
#
#     return results
#
def process_batch(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
    """Process pairs using batch API (with fallback to standard if unavailable)."""
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING: {len(pairs)} pairs")
    print(f"{'=' * 80}")

    # Check if batch API is available
    if not hasattr(anthropic_client.messages, "batches"):
        print("Warning: Batch API not available in your Anthropic SDK version.")
        print("Falling back to standard processing...")
        print("To enable batch API, upgrade: pip install --upgrade anthropic")
        print()
        return process_batch_fallback(pairs, save)

    batch_requests = create_batch_requests(pairs)
    if not batch_requests:
        return []

    try:
        batch = anthropic_client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        print(f"Batch submitted: {batch_id}")
    except Exception as e:
        print(f"Batch submission failed: {e}")
        print("Falling back to standard processing...")
        return process_batch_fallback(pairs, save)

    # Wait for completion
    while True:
        batch = anthropic_client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored
        print(f"  Status: {batch.processing_status} | Succeeded: {counts.succeeded}/{total}", end="\r")

        if batch.processing_status in ["ended", "canceled", "expired"]:
            break
        time.sleep(60)

    print("\n✓ Batch completed")

    # Retrieve results
    results: List[dict] = []
    for result in anthropic_client.messages.batches.results(batch_id):
        pmid, gene_id = result.custom_id.split("_", 1)

        # Defaults (batch mode doesn't compute mentions unless we fetch paper later)
        mention_count = 0
        paper_available = None
        alias_in_text = None

        if result.result.type == "succeeded":
            try:
                content = result.result.message.content[0].text
                parsed = format_with_retry(content, VALIDATION_SCHEMA)

                if not parsed:
                    # Parsing failed
                    error_data = {
                        "error": "JSON parsing failed",
                        "raw_response": content[:1000],
                    }
                    if save:
                        save_result(pmid, gene_id, error_data, False, "getGeneSummary")

                    results.append({
                        "pubmed_id": pmid,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "JSON parsing failed",
                        "paper_available": True,   # Batch succeeded so paper was available to Anthropic
                        "alias_in_text": True,     # Assumed true if summary generation succeeded upstream
                        "mentions": 0,             # Unknown for batch
                    })
                    continue  # don't fall through

                # Summary success - now process PDs
                if save:
                    save_result(pmid, gene_id, parsed, True, "getGeneSummary")

                # If gene is only mentioned in passing, skip PD generation
                if check_if_in_passing(parsed):
                    print(f"  ⚠️ {gene_id} in passing - PDs skipped")
                    results.append({
                        "pubmed_id": pmid,
                        "gene_id": gene_id,
                        "success": True,
                        "data": parsed,
                        "in_passing": True,
                        "paper_available": True,  # Batch succeeded so paper was available
                        "alias_in_text": True,    # Assumed true if summary was generated
                        "mentions": 0,            # Unknown for batch
                    })
                    continue  # skip to next result without generating PD

                # For PD steps, we need paper text and gene display
                # Extract host_db from pairs
                host_db = next((hdb for p, g, hdb in pairs if p == pmid and g == gene_id), None)

                paper_available = True
                alias_in_text = True  # Assumed if summary was generated

                if host_db:
                    try:
                        # Fetch paper and prepare gene display
                        paper_text = get_paper_text(pmid)

                        # Check gene in text
                        alias_in_text, mention_count = check_gene_in_text(gene_id, paper_text, host_db)

                        aliases = get_gene_synonyms(gene_id, paper_text, host_db)
                        gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

                        # Generate PDs
                        print("generating PDs...", end=" ", flush=True)
                        pd_result, pd_usage, pd_time = generatePDs(
                            summary_json=parsed,
                            gene_text=gene_display,
                            n_pds=N_PDs,
                        )

                        if pd_result and save:
                            save_result(pmid, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)

                        # Verify PDs (use caching even though batch papers are low-density)
                        if pd_result:
                            verify_result, verify_usage, verify_time = verifyPDs(
                                suggested_pds=pd_result,
                                paper_text=paper_text,
                                gene_text=gene_display,
                                use_caching=False,  # Don't use caching for batch (single genes)
                            )

                            if verify_result and save:
                                save_result(pmid, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)

                    except Exception as e:
                        print(f"  Warning: PD processing failed for {gene_id}: {e}")

                results.append({
                    "pubmed_id": pmid,
                    "gene_id": gene_id,
                    "success": True,
                    "data": parsed,
                    "paper_available": paper_available,
                    "alias_in_text": alias_in_text,
                    "mentions": mention_count if alias_in_text else 0,
                })

            except Exception as e:
                results.append({
                    "pubmed_id": pmid,
                    "gene_id": gene_id,
                    "success": False,
                    "error": str(e),
                    "paper_available": True,  # Batch succeeded so paper was available
                    "alias_in_text": None,    # Unknown if exception occurred
                    "mentions": 0,
                })
        else:
            results.append({
                "pubmed_id": pmid,
                "gene_id": gene_id,
                "success": False,
                "error": result.result.error.message,
                "paper_available": None,  # Unknown if batch failed
                "alias_in_text": None,
                "mentions": 0,
            })

    return results

def _step_succeeded_any_model(data: Dict[str, Any], step_key: str, gene_id: str) -> bool:
    step_node = (data.get(step_key) or {}).get(gene_id)
    if not isinstance(step_node, dict):
        return False
    # any model key under this gene
    for v in step_node.values():
        if isinstance(v, dict) and v.get("success") and v.get("data"):
            return True
    return False

def _step_succeeded_for_model(data: Dict[str, Any], step_key: str, gene_id: str, model: str) -> bool:
    step_node = (data.get(step_key) or {}).get(gene_id)
    if not isinstance(step_node, dict):
        return False
    model_node = step_node.get(model)
    return isinstance(model_node, dict) and bool(model_node.get("success")) and bool(model_node.get("data"))

def process_from_csv(csv_path: str, save: bool = True) -> pd.DataFrame:
    """
    Process gene-paper pairs from CSV file.

    CSV must have columns: Gene ID, PMID_clean, Database, paper_available, alias_in_text
    Only processes rows where paper_available==TRUE and alias_in_text==TRUE

    Saves results as {pmid}.json with structure:
    {
      "getGeneSummary": {
        "gene_id": {
          "gene_ID": "gene_id",
          "model_name": {
            "model": "model_name",
            "success": true/false,
            "data": {...},
            "usage": {...},
            "seconds": float
          }
        }
      }
    }
    """
    print(f"\n{'=' * 80}")
    print(f"LOADING DATA FROM: {csv_path}")
    print(f"{'=' * 80}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    # Show available columns for debugging
    print(f"CSV columns: {', '.join(df.columns.tolist())}")

    # Check for required columns with flexible naming
    gene_col = CSV_COLUMNS['gene_id']
    pmid_col = CSV_COLUMNS['pmid']
    db_col = CSV_COLUMNS['database']

    required_columns = [gene_col, pmid_col, db_col]
    missing_required = [col for col in required_columns if col not in df.columns]

    if missing_required:
        print(f"\n❌ ERROR: Missing required columns: {missing_required}")
        print(f"\nExpected column names (adjust in CSV_COLUMNS at top of script):")
        print(f"  - Gene ID column: '{gene_col}'")
        print(f"  - PMID column: '{pmid_col}'")
        print(f"  - Database column: '{db_col}'")
        print(f"\nYour CSV has: {', '.join(df.columns.tolist())}")
        print(f"\nEither:")
        print(f"  1. Rename your CSV columns to match expected names, OR")
        print(f"  2. Update CSV_COLUMNS dictionary at line ~50 in the config script")
        return pd.DataFrame()

    # Optional filter columns (if they exist, use them; if not, skip filtering)
    filter_applied = False
    paper_avail_col = CSV_COLUMNS.get('paper_available')
    alias_col = CSV_COLUMNS.get('alias_in_text')

    if paper_avail_col in df.columns and alias_col in df.columns:
        valid_df = df[(df[paper_avail_col] == True) & (df[alias_col] == True)].copy()
        print(f"Valid pairs ({paper_avail_col}=TRUE, {alias_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    elif paper_avail_col in df.columns:
        valid_df = df[df[paper_avail_col] == True].copy()
        print(f"Valid pairs ({paper_avail_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    elif alias_col in df.columns:
        valid_df = df[df[alias_col] == True].copy()
        print(f"Valid pairs ({alias_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    else:
        valid_df = df.copy()
        print(f"⚠ No filtering applied (optional filter columns not found)")
        print(f"Processing all {len(valid_df)} rows")

    if len(valid_df) == 0:
        print("No valid pairs to process after filtering!")
        return pd.DataFrame()

    # Group by paper
    grouped = defaultdict(list)
    for _, row in valid_df.iterrows():
        pmid = str(row[pmid_col])
        gene_id = str(row[gene_col])
        database = str(row[db_col]).lower()
        grouped[pmid].append((gene_id, database))

    # Deduplicate genes within each paper
    total_pairs = sum(len(genes) for genes in grouped.values())
    for pmid in grouped:
        grouped[pmid] = list(set(grouped[pmid]))  # Remove duplicates
    unique_pairs = sum(len(genes) for genes in grouped.values())

    print(f"Unique papers: {len(grouped)}")
    if total_pairs > unique_pairs:
        print(f"  Removed {total_pairs - unique_pairs} duplicate gene entries")

    # Analyze distribution
    genes_per_paper = [len(genes) for genes in grouped.values()]
    avg_genes = sum(genes_per_paper) / len(genes_per_paper)
    high_density = sum(1 for x in genes_per_paper if x >= MIN_GENES_FOR_CACHING)
    low_density = len(genes_per_paper) - high_density

    print(f"\nDistribution:")
    print(f"  Average genes per paper: {avg_genes:.2f}")
    print(f"  Papers with {MIN_GENES_FOR_CACHING}+ genes (caching): {high_density}")
    print(f"  Papers with <{MIN_GENES_FOR_CACHING} genes (batch): {low_density}")

    # Process
    all_results = []

    # High-density papers with caching
    high_density_papers = {pmid: genes for pmid, genes in grouped.items()
                           if len(genes) >= MIN_GENES_FOR_CACHING}

    if high_density_papers:
        print(f"\n{'=' * 80}")
        print(f"PHASE 1: CACHING ({len(high_density_papers)} papers)")
        print(f"{'=' * 80}")
        # pmid, genes = next(iter(high_density_papers.items()))
        for pmid, genes in high_density_papers.items():
            print(genes)
            results = process_paper_with_caching(pubmed_id=pmid, gene_list = genes, save=save)
            all_results.extend(results)

            # Print summary for this paper
            processed = sum(1 for r in results if not r.get("skipped", False))
            skipped = sum(1 for r in results if r.get("skipped", False))
            if skipped > 0:
                print(f"  Paper {pmid}: {processed} processed, {skipped} skipped (already done)")

    # Low-density papers with batch
    if USE_BATCH_FOR_LOW_DENSITY:
        low_density_pairs = [
            (pmid, g[0], g[1])
            for pmid, genes in grouped.items()
            if len(genes) < MIN_GENES_FOR_CACHING
            for g in genes
        ]

        # Filter out already-processed pairs if not overwriting
        if not OVERWRITE_EXISTING:
            fully_done = []
            needs_resume = []  # summary exists; PD and/or verify missing
            needs_summary = []  # no usable summary yet

            for pmid, gene_id, host_db in low_density_pairs:
                if is_gene_already_processed(
                        pmid,
                        gene_id,
                        check_all_steps=True,
                        flex_model=True,
                ):
                    fully_done.append((pmid, gene_id, host_db))
                    continue

                existing_summary = load_existing_summary(pmid, gene_id)

                if existing_summary is not None:
                    needs_resume.append((pmid, gene_id, host_db))
                else:
                    needs_summary.append((pmid, gene_id, host_db))

            if fully_done:
                print(f"  Skipping {len(fully_done)} already-processed pairs")
            if needs_resume:
                print(f"  Resuming {len(needs_resume)} pairs with existing summaries")
            if needs_summary:
                print(f"  Batch-processing {len(needs_summary)} pairs needing summaries")

            low_density_pairs_resume = needs_resume
            low_density_pairs_batch = needs_summary
        else:
            low_density_pairs_resume = []
            low_density_pairs_batch = low_density_pairs

        if low_density_pairs_resume:
            print(f"\n{'=' * 80}")
            print(f"PHASE 2A: RESUME PARTIAL LOW-DENSITY PAIRS ({len(low_density_pairs_resume)} pairs)")
            print(f"{'=' * 80}")
            results = process_batch_fallback(list(set(low_density_pairs_resume)), save=save)
            all_results.extend(results)

        if low_density_pairs_batch:
            print(f"\n{'=' * 80}")
            print(f"PHASE 2B: BATCH PROCESSING ({len(low_density_pairs_batch)} pairs)")
            print(f"{'=' * 80}")
            results = process_batch(list(set(low_density_pairs_batch)), save=save)
            all_results.extend(results)
        # # Deduplicate pairs (in case same pair appears multiple times in CSV)
        # original_count = len(low_density_pairs)
        # low_density_pairs = list(set(low_density_pairs))  # Remove duplicates
        # duplicate_count = original_count - len(low_density_pairs)
        # if duplicate_count > 0:
        #     print(f"  Removed {duplicate_count} duplicate pairs from batch")
        #
        # if low_density_pairs:
        #     print(f"\n{'=' * 80}")
        #     print(f"PHASE 2: BATCH PROCESSING")
        #     print(f"{'=' * 80}")
        #     results = process_batch(low_density_pairs, save=save)
        #     all_results.extend(results)

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'pmid': r['pubmed_id'],
            'gene_id': r['gene_id'],
            'success': r['success'],
            'paper_available': r.get('paper_available', None),
            'alias_in_text': r.get('alias_in_text', None),
            'mentions': r.get('mentions', 0),
            'error': r.get('error', '')
        }
        for r in all_results
    ])

    # Generate detailed step completion log
    step_log_data = []
    for _, row in results_df.iterrows():
        pmid = row['pmid']
        gene_id = row['gene_id']

        # Initialize status for each step
        step_status = {
            'pmid': pmid,
            'gene_id': gene_id,
            'paper_available': row.get('paper_available'),
            'alias_in_text': row.get('alias_in_text'),
            'Summary': False,
            'generatePDs': False,
            'verifyPDs': False
        }

        # Check completion status from saved files
        filepath = OUT_DIR / f"{pmid}.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Check each step
                for step_key, model in [
                    ("getGeneSummary", SUMMARY_MODEL),
                    ("generatePDs", PD_GENERATOR_MODEL),
                    ("verifyPDs", PD_VERIFIER_MODEL),
                ]:
                    if step_key == "getGeneSummary":
                        # ✅ summary complete if ANY model has a successful payload
                        step_status["Summary"] = _step_succeeded_any_model(data, step_key, gene_id)
                    else:
                        # PD steps: keep current behavior (only count current model) unless you want "any model" here too
                        step_status[step_key] = _step_succeeded_for_model(data, step_key, gene_id, model)


            except:
                pass

        step_log_data.append(step_status)

    step_log_df = pd.DataFrame(step_log_data)

    print(f"\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total gene-paper pairs: {len(results_df)}")

    if not step_log_df.empty:
        # Diagnostic statistics
        paper_avail_count = step_log_df['paper_available'].sum() if 'paper_available' in step_log_df.columns else 0
        alias_count = step_log_df['alias_in_text'].sum() if 'alias_in_text' in step_log_df.columns else 0

        print(f"\nDiagnostic checks:")
        print(f"  ✓ Paper available:    {paper_avail_count}/{len(step_log_df)}")
        print(f"  ✓ Gene in text:       {alias_count}/{len(step_log_df)}")

        print(f"\nPipeline completion by step:")
        print(f"  ✓ Summary generated:  {step_log_df['Summary'].sum()}/{len(step_log_df)}")
        print(f"  ✓ PDs generated:      {step_log_df['generatePDs'].sum()}/{len(step_log_df)}")
        print(f"  ✓ PDs verified:       {step_log_df['verifyPDs'].sum()}/{len(step_log_df)}")
        print(f"  ✗ Failed:             {(~results_df['success']).sum()}/{len(step_log_df)}")

        # Calculate average times if available
        successful = [r for r in all_results if r.get('total_time')]
        if successful:
            avg_time = sum(r['total_time'] for r in successful) / len(successful)
            print(f"\nAverage time per gene: {avg_time:.1f}s")

    # Save step completion log
    if not step_log_df.empty:
        log_path = csv_path.replace('.csv', '_step_log.csv')
        step_log_df.to_csv(log_path, index=False)
        print(f"\nStep completion log saved to: {log_path}")

    return results_df


# Entry point is main.py (run_batch -> process_from_csv). Importing this module must NOT
# auto-run a batch, so the previous unguarded module-level call has been removed.
# if __name__ == "__main__":
#     import sys
#
#     if len(sys.argv) < 2:
#         print("Usage: python pipeline.py <csv_file_path>")
#         print("\nExample: python pipeline.py gene_paper_pairs.csv")
#         sys.exit(1)
#
#     csv_path = sys.argv[1]
#
#     if not Path(csv_path).exists():
#         print(f"Error: File not found: {csv_path}")
#         sys.exit(1)
#
#     # Process
#     results_df = process_from_csv(csv_path, save=True)
#
#     # Save results summary
#     if not results_df.empty:
#         output_path = csv_path.replace('.csv', '_results.csv')
#         results_df.to_csv(output_path, index=False)
#         print(f"\nResults saved to: {output_path}")