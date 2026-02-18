"""
additional helper functions specifically for the batch processing script
"""
import pandas as pd
from collections import defaultdict

from config.global_settings_batch import *
from pipeline.prompts import *
from pipeline.pubmed_helpers import *
from pipeline.vpdb_helpers import *
from pipeline.utils import *

# Extract commonly used variables from global_prompts_and_schema
SUMMARY_SYSTEM_PROMPT = global_prompts_and_schema["getGeneSummary"]["SystemPrompt"]
VALIDATION_SCHEMA = global_prompts_and_schema["getGeneSummary"]["ValidationSchema"]

def is_gene_already_processed(pubmed_id: str, gene_id: str, check_all_steps: bool = True) -> bool:
    """
    Check if gene-paper pair has been fully processed.

    Args:
        pubmed_id: Paper ID
        gene_id: Gene ID
        check_all_steps: If True, checks summary + PDs + verify all complete
                        If False, only checks summary (backward compatible)

    Returns:
        True if processing is complete
    """
    filename = OUT_DIR / f"{pubmed_id}.json"

    if not filename.exists():
        return False

    try:
        with open(filename, 'r') as f:
            paper_data = json.load(f)

        if not check_all_steps:
            # Old behavior - just check summary
            if gene_id in paper_data.get("getGeneSummary", {}):
                gene_data = paper_data["getGeneSummary"][gene_id]
                if SUMMARY_MODEL in gene_data:
                    model_result = gene_data[SUMMARY_MODEL]
                    # Consider it processed if success=True and data is not None
                    if model_result.get("success") and model_result.get("data") is not None:
                        return True
            return False

        # New behavior - check all three steps
        required_steps = [
            ("getGeneSummary", SUMMARY_MODEL),
            ("generatePDs", PD_GENERATOR_MODEL),
            ("verifyPDs", PD_VERIFIER_MODEL)
        ]

        for step_key, model in required_steps:
            step_data = paper_data.get(step_key, {}).get(gene_id, {})
            model_data = step_data.get(model, {})

            if not (model_data.get("success") and model_data.get("data")):
                return False  # This step not complete

        return True  # All steps complete!

    except Exception:
        return False


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


def format_with_retry(content: str, schema: dict, max_attempts: int = MAX_RETRY) -> Optional[dict]:
    """Parse JSON with retry using formatter model."""
    # Try direct parsing first, after stripping markdown
    try:
        # Strip markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
            cleaned = re.sub(r'\n```\s*$', '', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed: {e}")
        print(f"  First 200 chars: {content[:200]}")

    # Try with formatter model
    for attempt in range(max_attempts):
        try:
            print(f"  Retry {attempt + 1}/{max_attempts} with formatter model...")
            response = anthropic_client.messages.create(
                model=FORMATTER_MODEL,
                max_tokens=4000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"Convert this to valid JSON matching the schema. Output ONLY valid JSON, no markdown:\n\n{content}\n\nSchema:\n{json.dumps(schema)}"
                }]
            )
            result_text = response.content[0].text.strip()
            # Strip markdown again
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\s*\n', '', result_text)
                result_text = re.sub(r'\n```\s*$', '', result_text)
            return json.loads(result_text)
        except Exception as e:
            print(f"  Retry {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                print(f"  ERROR: All retries failed. Saving raw response for debugging.")
                # Save raw response for debugging
                debug_file = OUT_DIR / "debug_failed_parse.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Original content:\n{content}\n\n")
                    f.write(f"Schema:\n{json.dumps(schema, indent=2)}\n")
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


def verifyPDs(suggested_pds: Dict[Any, Any], paper_text: str, gene_text: str,
              use_caching: bool = True) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Verify and select product descriptions against paper evidence.

    IMPORTANT: This function can reuse cached paper text from summary generation!
    When use_caching=True, the paper text will be cached for efficient reuse.

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



def process_paper_with_caching(pubmed_id: str, gene_list: List[Tuple[str, str]],
                               save: bool = True) -> List[dict]:
    """Process multiple genes from same paper with prompt caching."""
    print(f"\n{'=' * 80}")
    print(f"Paper {pubmed_id}: {len(gene_list)} genes (USING CACHING)")
    print(f"{'=' * 80}")

    # Try to fetch paper - check if available
    paper_available = False
    paper_text = None
    try:
        paper_text = get_paper_text(pubmed_id)
        paper_available = True
    except Exception as e:
        print(f"  ✗ Paper not available in PMC: {e}")
        # Return all genes as failed with paper_available=False
        results = []
        for gene_id, host_db in gene_list:
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": "Paper not available in PMC",
                "paper_available": False,
                "alias_in_text": False,
                "mentions": 0
            })
        return results

    system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
    system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

    results = []

    for i, (gene_id, host_db) in enumerate(gene_list, 1):
        print(f"  [{i}/{len(gene_list)}] {gene_id}...", end=" ")

        # Check if gene/aliases are in paper text
        alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, host_db)

        if not alias_in_text:
            print(f"✗ Gene/aliases not found in text (skipped)")
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

        print(f"({mentions} mentions) ", end="")

        # Check if already processed
        if not OVERWRITE_EXISTING and is_gene_already_processed(pubmed_id, gene_id):
            print(f"✓ Already processed (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "skipped": True,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })
            continue

        start = time.time()

        try:
            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            # Messages with caching
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do not respond. Here is the paper text:"},
                        {"type": "text", "text": paper_text, "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "assistant", "content": "I have received the paper text."},
                {"role": "user", "content": f"Generate summary for {gene_display}."}
            ]

            response = anthropic_client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages
            )

            # Track usage first (needed even if parsing fails)
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
            }

            result = format_with_retry(response.content[0].text, VALIDATION_SCHEMA)

            elapsed = time.time() - start

            if not result:
                # Save raw response for debugging
                print(f"  ✗ Failed to parse response - saving raw output for debugging")
                debug_data = {
                    "error": "JSON parsing failed after retries",
                    "raw_response": response.content[0].text[:1000],  # First 1000 chars
                    "model": SUMMARY_MODEL
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
                    "mentions": mentions
                })
                continue

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

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Generate PDs
            # ═══════════════════════════════════════════════════════════════
            pd_result, pd_usage, pd_time = generatePDs(
                summary_json=result,
                gene_text=gene_display,
                n_pds=N_PDs
            )

            if not pd_result:
                print(f"✗ PD gen failed")
                if save:
                    save_result(pubmed_id, gene_id,
                               {"error": "PD generation failed"},
                               False, "generatePDs", pd_usage, pd_time)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "PD generation failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue  # Skip verify if PD gen failed

            # Save PDs
            if save:
                save_result(pubmed_id, gene_id, pd_result, True,
                           "generatePDs", pd_usage, pd_time)

            print(f"✓ PDs ({pd_time:.1f}s)", end=" | ")

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Verify PDs
            # ═══════════════════════════════════════════════════════════════
            verify_result, verify_usage, verify_time = verifyPDs(
                suggested_pds=pd_result,
                paper_text=paper_text,  # Reuses cached paper!
                gene_text=gene_display,
                use_caching=True  # CRITICAL for cache reuse!
            )

            if not verify_result:
                print(f"✗ Verify failed")
                if save:
                    save_result(pubmed_id, gene_id,
                               {"error": "PD verification failed"},
                               False, "verifyPDs", verify_usage, verify_time)
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

            # Save verified PDs
            if save:
                save_result(pubmed_id, gene_id, verify_result, True,
                           "verifyPDs", verify_usage, verify_time)

            total_time = elapsed + pd_time + verify_time
            cache_info = f"cache={verify_usage['cache_read_input_tokens']:,}" if i > 1 else f"created={verify_usage.get('cache_creation_input_tokens', 0):,}"
            print(f"✓ Verified ({verify_time:.1f}s) | Total: {total_time:.1f}s ({cache_info})")

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
            print(f"✗ Error: {e}")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": str(e),
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })

    return results


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

            # Prepare prompts
            system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
            system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

            # Call API
            response = anthropic_client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Do not respond. Here is the paper text:\n\n{paper_text}"},
                    {"role": "assistant", "content": "I have received the paper text."},
                    {"role": "user", "content": f"Generate summary for {gene_display}."}
                ]
            )

            # Track usage
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            # Parse result
            result = format_with_retry(response.content[0].text, VALIDATION_SCHEMA)
            elapsed = time.time() - start

            if not result:
                print(f"✗ Parse failed ({elapsed:.1f}s)")
                error_data = {"error": "JSON parsing failed", "raw_response": response.content[0].text[:1000]}
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

                # Generate PDs
                try:
                    pd_result, pd_usage, pd_time = generatePDs(
                        summary_json=result,
                        gene_text=gene_display,
                        n_pds=N_PDs
                    )

                    if pd_result and save:
                        save_result(pubmed_id, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)
                        print(f"✓ PDs ({pd_time:.1f}s)", end=" | ")

                        # Verify PDs
                        verify_result, verify_usage, verify_time = verifyPDs(
                            suggested_pds=pd_result,
                            paper_text=paper_text,
                            gene_text=gene_display,
                            use_caching=False
                        )

                        if verify_result and save:
                            save_result(pubmed_id, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)
                            total_time = elapsed + pd_time + verify_time
                            print(f"✓ Verified ({verify_time:.1f}s) | Total: {total_time:.1f}s")
                        else:
                            print(f"✗ Verify failed")
                    else:
                        print(f"✗ PD gen failed")
                except Exception as e:
                    print(f"✗ PD error: {e}")

                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": True,
                    "data": result,
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

    for pubmed_id, gene_id, host_db in pairs:
        try:
            paper_text = get_paper_text(pubmed_id)
            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
            system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

            batch_requests.append({
                "custom_id": f"{pubmed_id}_{gene_id}",
                "params": {
                    "model": SUMMARY_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "temperature": MODEL_TEMP,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": f"Do not respond. Here is the paper text:\n\n{paper_text}"},
                        {"role": "assistant", "content": "I have received the paper text."},
                        {"role": "user", "content": f"Generate summary for {gene_display}."}
                    ]
                }
            })
        except Exception as e:
            print(f"  ✗ Failed to prepare {pubmed_id}_{gene_id}: {e}")

    return batch_requests

def process_batch(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
    """Process pairs using batch API (with fallback to standard if unavailable)."""
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING: {len(pairs)} pairs")
    print(f"{'=' * 80}")

    # Check if batch API is available
    if not hasattr(anthropic_client.messages, 'batches'):
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

    print(f"\n✓ Batch completed")

    # Retrieve results
    results = []
    for result in anthropic_client.messages.batches.results(batch_id):
        pmid, gene_id = result.custom_id.split("_", 1)

        if result.result.type == "succeeded":
            try:
                content = result.result.message.content[0].text
                parsed = format_with_retry(content, VALIDATION_SCHEMA)

                if not parsed:
                    # Parsing failed
                    error_data = {
                        "error": "JSON parsing failed",
                        "raw_response": content[:1000]
                    }
                    if save:
                        save_result(pmid, gene_id, error_data, False, "getGeneSummary")

                    if check_if_in_passing(parsed):
                        print(f"  ⚠️ {gene_id} in passing - PDs skipped")
                        results.append({
                            "pubmed_id": pmid,
                            "gene_id": gene_id,
                            "success": False,
                            "error": "JSON parsing failed",
                            "paper_available": True,  # Paper was available (batch succeeded)
                            "alias_in_text": True,    # Assumed true if batch API succeeded
                            "mentions": 0             # Unknown for batch
                        })
                        continue # skip to next result without generating PD
                else:
                    # Summary success - now process PDs
                    if save:
                        save_result(pmid, gene_id, parsed, True, "getGeneSummary")

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
                            pd_result, pd_usage, pd_time = generatePDs(
                                summary_json=parsed,
                                gene_text=gene_display,
                                n_pds=N_PDs
                            )

                            if pd_result and save:
                                save_result(pmid, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)

                            # Verify PDs (use caching even though batch papers are low-density)
                            if pd_result:
                                verify_result, verify_usage, verify_time = verifyPDs(
                                    suggested_pds=pd_result,
                                    paper_text=paper_text,
                                    gene_text=gene_display,
                                    use_caching=False  # Don't use caching for batch (single genes)
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
                        "mentions": mention_count if alias_in_text else 0
                    })
            except Exception as e:
                results.append({
                    "pubmed_id": pmid,
                    "gene_id": gene_id,
                    "success": False,
                    "error": str(e),
                    "paper_available": True,  # Batch succeeded so paper was available
                    "alias_in_text": None,    # Unknown if exception occurred
                    "mentions": 0
                })
        else:
            results.append({
                "pubmed_id": pmid,
                "gene_id": gene_id,
                "success": False,
                "error": result.result.error.message,
                "paper_available": None,  # Unknown if batch failed
                "alias_in_text": None,
                "mentions": 0
            })

    return results



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
        print(f"  2. Update CSV_COLUMNS dictionary at line ~50 in the script")
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
        for pmid, genes in high_density_papers.items():
            results = process_paper_with_caching(pmid, genes, save=save)
            all_results.extend(results)

            # Print summary for this paper
            processed = sum(1 for r in results if not r.get("skipped", False))
            skipped = sum(1 for r in results if r.get("skipped", False))
            if skipped > 0:
                print(f"  Paper {pmid}: {processed} processed, {skipped} skipped (already done)")

    # Low-density papers with batch
    if USE_BATCH_FOR_LOW_DENSITY:
        low_density_pairs = [(pmid, g[0], g[1]) for pmid, genes in grouped.items()
                             if len(genes) < MIN_GENES_FOR_CACHING for g in genes]

        # Filter out already-processed pairs if not overwriting
        if not OVERWRITE_EXISTING:
            original_count = len(low_density_pairs)
            low_density_pairs = [(pmid, gene_id, host_db) for pmid, gene_id, host_db in low_density_pairs
                                 if not is_gene_already_processed(pmid, gene_id)]
            skipped_count = original_count - len(low_density_pairs)
            if skipped_count > 0:
                print(f"  Skipping {skipped_count} already-processed pairs")

        # Deduplicate pairs (in case same pair appears multiple times in CSV)
        original_count = len(low_density_pairs)
        low_density_pairs = list(set(low_density_pairs))  # Remove duplicates
        duplicate_count = original_count - len(low_density_pairs)
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate pairs from batch")

        if low_density_pairs:
            print(f"\n{'=' * 80}")
            print(f"PHASE 2: BATCH PROCESSING")
            print(f"{'=' * 80}")
            results = process_batch(low_density_pairs, save=save)
            all_results.extend(results)

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
                    ("verifyPDs", PD_VERIFIER_MODEL)
                ]:
                    step_data = data.get(step_key, {}).get(gene_id, {})
                    model_data = step_data.get(model, {})

                    # Map step_key to column name
                    if step_key == "getGeneSummary":
                        col_name = "Summary"
                    else:
                        col_name = step_key

                    if model_data.get("success") and model_data.get("data"):
                        step_status[col_name] = True
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



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <csv_file_path>")
        print("\nExample: python pipeline.py gene_paper_pairs.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Process
    results_df = process_from_csv(csv_path, save=True)

    # Save results summary
    if not results_df.empty:
        output_path = csv_path.replace('.csv', '_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")