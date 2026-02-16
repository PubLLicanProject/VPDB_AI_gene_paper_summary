"""
Testing and evaluation frameworks for the pipeline.

Includes:
- Model comparison testing
- Pipeline variation testing
- Multi-model benchmarking
"""

import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from config.models import MODELS_TO_TEST, TEST_SET_PATH, OUT_DIR
from single_pair_pipeline_workflow import getGeneSummary, generatePDs, verifyPDs
from pd_pipeline.pubmed import get_pubmed_json, parse_pubmed_json
from pd_pipeline.gene_data import get_gene_synonyms, gene_to_prompt
from pd_pipeline.utils import load_status, save_status, check_if_ok

def load_test_set(subset_file=TEST_SET_PATH):
    """Load the 30 selected representative paper-gene pairs"""
    df = pd.read_csv(subset_file)

    test_pairs = []
    for _, row in df.iterrows():
        test_pairs.append({
            "pmid": str(row["pmid"]),
            "gene_ID": str(row["gene_ID"]),
            "pair_type": row.get("pair_type", "Unknown"),
            "mean_performance": row.get("mean_performance", 0)
        })

    return test_pairs



def run_model_comparison_on_test_set():
    """
    Run all models on the selected 30 paper-gene pairs
    Pipeline: getGeneSummary -> generatePDs -> verifyPDs
    """
    print("=" * 80)
    print("MULTI-MODEL COMPARISON ON SELECTED TEST SET")
    print("=" * 80)

    # Load test set
    test_pairs = load_test_set()
    print(f"\nLoaded {len(test_pairs)} paper-gene pairs for testing:\n")
    print(test_pairs)

    # Count by type
    from collections import Counter
    type_counts = Counter(p["pair_type"] for p in test_pairs)
    print("Test set composition:")
    for pair_type, count in type_counts.items():
        print(f"  {pair_type}: {count}")
    print()

    total_runs = len(test_pairs) * len(MODELS_TO_TEST)
    completed = 0

    # Test each model
    for provider, model_name, display_name in MODELS_TO_TEST:
        print(f"\n{'=' * 80}")
        print(f"TESTING MODEL: {display_name} ({model_name})")
        print(f"{'=' * 80}\n")

        # Override LLM settings for this model
        global summary_llm, PD_generator_llm, PD_QC_llm
        summary_llm = [provider, model_name]
        PD_generator_llm = [provider, model_name]
        PD_QC_llm = [provider, model_name]

        model_completed = 0

        for pair in test_pairs:
            pmid = pair["pmid"]
            gene_id = pair["gene_ID"]
            pair_type = pair["pair_type"]

            print(f"[{completed + 1}/{total_runs}] Processing {gene_id} (PMID {pmid}) - {pair_type}")

            try:
                process_paper_gene_pair(
                    pubmed_id=pmid,
                    gene_id=gene_id,
                    save=True,
                    verify_summary=False,  # Skip summary verification
                    generate_PD=True  # Do generate and verify PDs
                )
                print(f"  ✓ Completed successfully")

            except Exception as e:
                print(f"  ✗ Error: {e}")

            completed += 1
            model_completed += 1

            # Progress for this model
            model_pct = (model_completed / len(test_pairs)) * 100
            total_pct = (completed / total_runs) * 100
            print(f"  Model progress: {model_completed}/{len(test_pairs)} ({model_pct:.1f}%)")
            print(f"  Overall progress: {completed}/{total_runs} ({total_pct:.1f}%)")

            time.sleep(1)  # Rate limiting

        print(f"\n✓ Completed {display_name}: {model_completed}/{len(test_pairs)} pairs\n")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"Total runs: {completed}/{total_runs}")



################
# QUICK comparison - with vs without summary
################


def run_without_summary_comparison():
    """
    Run Claude Sonnet 4.5 on test set with direct PD generation (no summary step).
    Uses same test set as run_model_comparison_on_test_set().
    """
    print("=" * 80)
    print("TESTING: PD GENERATION WITHOUT SUMMARY STEP")
    print("Model: Claude Sonnet 4.5 via OpenRouter")
    print("=" * 80)

    # Load same test set
    test_pairs = load_test_set()
    print(f"\nLoaded {len(test_pairs)} paper-gene pairs for testing\n")

    completed = 0
    total = len(test_pairs)

    for pair in test_pairs:
        pmid = pair["pmid"]
        gene_id = pair["gene_ID"]
        pair_type = pair["pair_type"]

        print(f"[{completed + 1}/{total}] Processing {gene_id} (PMID {pmid}) - {pair_type}")

        try:
            data, usage, secs = generate_PD_without_summary(
                pubmed_id=pmid,
                gene_id=gene_id,
                save=True
            )

            if "error" not in data:
                print(f"  ✓ Completed in {secs:.1f}s")
            else:
                print(f"  ✗ Error: {data['error']}")

        except Exception as e:
            print(f"  ✗ Exception: {e}")

        completed += 1
        progress_pct = (completed / total) * 100
        print(f"  Progress: {completed}/{total} ({progress_pct:.1f}%)\n")

        time.sleep(1)  # Rate limiting

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print(f"Completed: {completed}/{total}")
    print("=" * 80)



############# OTHER PIPELIE STEPS COMPARISON WITH NEWEST MODEL ##########


def run_pipeline_variation_comparison():
    """
    Test different pipeline configurations to determine optimal workflow.
    Uses original step keys for prompts, but variation-specific keys for saving results.
    """
    print("=" * 80)
    print("PIPELINE VARIATION COMPARISON")
    print("=" * 80)

    # Load test set
    test_pairs = load_test_set()
    print(f"\nLoaded {len(test_pairs)} paper-gene pairs for testing\n")

    # Define pipeline variations
    variations = [
        # {
        #     "name": "Option_1_baseline",
        #     "desc": "summary -> PDs",
        #     "use_verified_summary": False,
        #     "verify_pds": False,
        #     "select_pd": False
        # },
        # {
        #     "name": "Option_2_verify_summary",
        #     "desc": "summary -> verify_summary -> PDs",
        #     "use_verified_summary": True,
        #     "verify_pds": False,
        #     "select_pd": False
        # },
        {
            "name": "Option_3_verify_pds",
            "desc": "summary -> PDs -> verify_PDs",
            "use_verified_summary": False,
            "verify_pds": True,
            "select_pd": False
        },
        # {
        #     "name": "Option_4_select_pd",
        #     "desc": "summary -> PDs -> select_PD",
        #     "use_verified_summary": False,
        #     "verify_pds": False,
        #     "select_pd": True
        # },
        # {
        #     "name": "Option_2+3_combined",
        #     "desc": "verify_summary -> PDs -> verify_PDs",
        #     "use_verified_summary": True,
        #     "verify_pds": True,
        #     "select_pd": False
        # },
        # {
        #     "name": "Option_2+4_combined",
        #     "desc": "verify_summary -> PDs -> select_PD",
        #     "use_verified_summary": True,
        #     "verify_pds": False,
        #     "select_pd": True
        # }
    ]

    completed = 0
    total = len(test_pairs) * len(variations)

    for pair in test_pairs:
        pmid = pair["pmid"]
        gene_id = pair["gene_ID"]
        pair_type = pair["pair_type"]

        print(f"\n{'=' * 80}")
        print(f"Processing {gene_id} (PMID {pmid}) - {pair_type}")
        print(f"{'=' * 80}")

        # ── STAGE 0: Get paper text and gene info (shared across all variations) ──
        try:
            pubmed_json = get_pubmed_json(pmid)
            pubmed_text = parse_pubmed_json(pubmed_json)
            synonyms = get_gene_synonyms(gene_id, pubmed_text)
            gene_text = gene_to_prompt(gene_id, synonyms)
        except Exception as e:
            print(f"  ✗ Failed to load paper/gene: {e}")
            completed += len(variations)
            continue

        # ── STAGE 1: Generate base summary (if not exists) ──
        key = "getGeneSummary"
        summary_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                   gene_id=gene_id, llm=summary_llm[1])

        if check_if_ok(summary_data):
            print(f"  ✔ Found existing summary")
            base_summary = summary_data["data"]
        else:
            print(f"  → Generating base summary...")
            try:
                base_summary, usage, secs = getGeneSummary(pubmed_text, gene_text)
                if isinstance(base_summary, (dict, list)):
                    save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                gene_id=gene_id, model_name=summary_llm[1],
                                data=base_summary, success=True, usage=usage, seconds=secs)
                    print(f"    ✔ Generated in {secs:.1f}s")
                else:
                    print(f"    ✗ Failed to generate valid summary")
                    completed += len(variations)
                    continue
            except Exception as e:
                print(f"    ✗ Error: {e}")
                completed += len(variations)
                continue

        # ── STAGE 2: Generate verified summary (if not exists) ──
        key = "verifyGeneSummary"
        verified_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                    gene_id=gene_id, llm=summary_QC_llm[1])

        if check_if_ok(verified_data):
            print(f"  ✔ Found existing verified summary")
            verified_summary = verified_data["data"]
        else:
            print(f"  → Verifying summary...")
            try:
                verified_summary, usage, secs = verifyGeneSummary(
                    summary_json=base_summary, pubmed_text=pubmed_text, gene_text=gene_text
                )
                if isinstance(verified_summary, (dict, list)):
                    save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                gene_id=gene_id, model_name=summary_QC_llm[1],
                                data=verified_summary, success=True, usage=usage, seconds=secs)
                    print(f"    ✔ Verified in {secs:.1f}s")
                else:
                    print(f"    ✗ Failed to verify summary")
                    verified_summary = None
            except Exception as e:
                print(f"    ✗ Error: {e}")
                verified_summary = None

        # ── STAGE 3: Test each pipeline variation ──
        for variation in variations:
            completed += 1
            var_name = variation["name"]
            var_desc = variation["desc"]

            print(f"\n  [{completed}/{total}] Testing: {var_desc}")

            # Select which summary to use
            summary_for_pds = verified_summary if variation["use_verified_summary"] else base_summary

            if summary_for_pds is None:
                print(f"    ✗ Skipping - required summary not available")
                continue

            # Generate PDs with variation-specific save key
            pd_save_key = f"generatePDs_{var_name}"
            pd_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=pd_save_key,
                                  gene_id=gene_id, llm=PD_generator_llm[1])

            if check_if_ok(pd_data):
                print(f"    ✔ Found existing PDs for {var_name}")
                generated_pds = pd_data["data"]
            else:
                print(f"    → Generating PDs...")
                try:
                    # Use default key "generatePDs" for prompt lookup
                    generated_pds, usage, secs = generatePDs(
                        summary_json=summary_for_pds,
                        gene_text=gene_text
                    )
                    if isinstance(generated_pds, (dict, list)):
                        # Save with variation-specific key
                        save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=pd_save_key,
                                    gene_id=gene_id, model_name=PD_generator_llm[1],
                                    data=generated_pds, success=True, usage=usage, seconds=secs)
                        print(f"      ✔ Generated in {secs:.1f}s")
                    else:
                        print(f"      ✗ Failed to generate PDs")
                        continue
                except Exception as e:
                    print(f"      ✗ Error: {e}")
                    continue

            # Apply post-processing if needed
            if variation["verify_pds"]:
                verify_save_key = f"verifyPDs_{var_name}"
                verify_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=verify_save_key,
                                          gene_id=gene_id, llm=PD_QC_llm[1])

                if check_if_ok(verify_data):
                    print(f"    ✔ Found existing verified PDs")
                else:
                    print(f"    → Verifying PDs...")
                    try:
                        # Use default key "verifyPDs" for prompt lookup
                        verified_pds, usage, secs = verifyPDs(
                            suggestedPDs=generated_pds,
                            paper_text=pubmed_text,
                            gene_text=gene_text
                        )
                        if isinstance(verified_pds, (dict, list)):
                            # Save with variation-specific key
                            save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=verify_save_key,
                                        gene_id=gene_id, model_name=PD_QC_llm[1],
                                        data=verified_pds, success=True, usage=usage, seconds=secs)
                            print(f"      ✔ Verified in {secs:.1f}s")
                        else:
                            print(f"      ✗ Failed to verify PDs")
                    except Exception as e:
                        print(f"      ✗ Error: {e}")

            elif variation["select_pd"]:
                select_save_key = f"selectPD_{var_name}"
                select_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=select_save_key,
                                          gene_id=gene_id, llm=PD_picker_llm[1])

                if check_if_ok(select_data):
                    print(f"    ✔ Found existing selected PD")
                else:
                    print(f"    → Selecting PD (examples-based)...")
                    try:
                        # Use default key "selectPD" for prompt lookup
                        selected_pd, usage, secs = selectPD(
                            suggestedPDs=generated_pds,
                            summary_for_pd=summary_for_pds
                        )
                        if isinstance(selected_pd, (dict, list)):
                            # Save with variation-specific key
                            save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=select_save_key,
                                        gene_id=gene_id, model_name=PD_picker_llm[1],
                                        data=selected_pd, success=True, usage=usage, seconds=secs)
                            print(f"      ✔ Selected in {secs:.1f}s")
                        else:
                            print(f"      ✗ Failed to select PD")
                    except Exception as e:
                        print(f"      ✗ Error: {e}")

            # Progress tracking
            progress_pct = (completed / total) * 100
            print(f"    Overall progress: {completed}/{total} ({progress_pct:.1f}%)")

            time.sleep(0.5)  # Brief rate limiting between variations

        time.sleep(1)  # Rate limiting between papers

    print("\n" + "=" * 80)
    print("PIPELINE VARIATION TESTING COMPLETE")
    print("=" * 80)
    print(f"Completed: {completed}/{total}")
    print("\nVariations tested:")
    for v in variations:
        print(f"  - {v['name']}: {v['desc']}")



def run_option3_model_comparison():
    """
    Test multiple models on Option 3 pipeline: summary → PDs → verify_PDs
    No summary verification, no PD selection - just the winning pipeline!
    """
    print("=" * 80)
    print("OPTION 3 PIPELINE - MULTI-MODEL COMPARISON")
    print("Pipeline: getGeneSummary → generatePDs → verifyPDs")
    print("=" * 80)

    # Load test set
    test_pairs = load_test_set()
    print(f"\nLoaded {len(test_pairs)} paper-gene pairs for testing:\n")

    total_runs = len(test_pairs) * len(MODELS_TO_TEST)
    completed = 0

    # Test each model
    for provider, model_name, display_name in MODELS_TO_TEST:
        print(f"\n{'=' * 80}")
        print(f"TESTING MODEL: {display_name} ({model_name})")
        print(f"{'=' * 80}\n")

        # Override LLM settings for this model
        global summary_llm, PD_generator_llm, PD_QC_llm
        summary_llm = [provider, model_name]
        PD_generator_llm = [provider, model_name]
        PD_QC_llm = [provider, model_name]

        for pair in test_pairs:
            pmid = pair["pmid"]
            gene_id = pair["gene_ID"]
            pair_type = pair["pair_type"]

            print(f"[{completed + 1}/{total_runs}] Processing {gene_id} (PMID {pmid})")

            try:
                # Get paper text
                pubmed_json = get_pubmed_json(pmid)
                pubmed_text = parse_pubmed_json(pubmed_json)
                synonyms = get_gene_synonyms(gene_id, pubmed_text)
                gene_text = gene_to_prompt(gene_id, synonyms)

                # STEP 1: Generate summary
                key = "getGeneSummary"
                summary_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                           gene_id=gene_id, llm=summary_llm[1])

                if check_if_ok(summary_data):
                    print(f"  ✔ Found existing summary")
                    summary = summary_data["data"]
                else:
                    print(f"  → Generating summary...")
                    summary, usage, secs = getGeneSummary(pubmed_text, gene_text)
                    if isinstance(summary, (dict, list)):
                        save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                    gene_id=gene_id, model_name=summary_llm[1],
                                    data=summary, success=True, usage=usage, seconds=secs)
                    else:
                        print(f"  ✗ Summary generation failed")
                        completed += 1
                        continue

                # STEP 2: Generate PDs
                key = "generatePDs"
                pd_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                      gene_id=gene_id, llm=PD_generator_llm[1])

                if check_if_ok(pd_data):
                    print(f"  ✔ Found existing PDs")
                    pds = pd_data["data"]
                else:
                    print(f"  → Generating PDs...")
                    pds, usage, secs = generatePDs(summary_json=summary, gene_text=gene_text)
                    if isinstance(pds, (dict, list)):
                        save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                    gene_id=gene_id, model_name=PD_generator_llm[1],
                                    data=pds, success=True, usage=usage, seconds=secs)
                    else:
                        print(f"  ✗ PD generation failed")
                        completed += 1
                        continue

                # STEP 3: Verify PDs (FINAL STEP for Option 3)
                key = "verifyPDs"
                verify_data = load_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                          gene_id=gene_id, llm=PD_QC_llm[1])

                if check_if_ok(verify_data):
                    print(f"  ✔ Found existing verified PDs")
                else:
                    print(f"  → Verifying PDs...")
                    verified, usage, secs = verifyPDs(
                        suggestedPDs=pds,
                        paper_text=pubmed_text,
                        gene_text=gene_text
                    )
                    if isinstance(verified, (dict, list)):
                        save_status(pubmed_id=pmid, out_dir=OUT_DIR, step_key=key,
                                    gene_id=gene_id, model_name=PD_QC_llm[1],
                                    data=verified, success=True, usage=usage, seconds=secs)
                        print(f"  ✔ Complete!")
                    else:
                        print(f"  ✗ PD verification failed")

            except Exception as e:
                print(f"  ✗ Error: {e}")

            completed += 1
            time.sleep(1)

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)



