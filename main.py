"""
main.py – Top-level entry point for the gene curation pipeline.

Public API
----------
>>> from main import run_pipeline
>>> run_pipeline(pmid="24877144", gene_id="PF3D7_1329700",  host_db = "PlasmoDB")
# Run with summary verification
>>> run_pipeline(pmid="24877144", gene_id="PF3D7_1329700",  host_db = "PlasmoDB", verify_summary = True)

>>> from main import run_tests, run_batch
>>> run_tests(test="model_comparison", yes=True)
>>> run_batch(csv_path="gene_paper_pairs.csv") # needs to gene_ID, pmid, and host_DB columns

CLI
---
# Single pair (Step 1)
python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db PlasmoDB

# Model / variation testing (Step 2)
python main.py test --test model_comparison --yes

# Batch processing (Step 3)
python main.py batch --csv gene_paper_pairs.csv
"""

import sys
import argparse


# ── Lazy imports (only load the step that's needed) ───────────────────────────

def _import_step1():
    from STEP_1_single_pair_processing import process_paper_gene_pair
    return process_paper_gene_pair

def _import_step2():
    from STEP_2_running_tests import (
        run_model_comparison_on_test_set,
        run_without_summary_comparison,
        run_pipeline_variation_comparison,
    )
    return {
        "model_comparison":     run_model_comparison_on_test_set,
        "no_summary":           run_without_summary_comparison,
        "variation_comparison": run_pipeline_variation_comparison,
    }

def _import_step3():
    from STEP_3_batch_processing import process_from_csv
    return process_from_csv


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    pmid: str,
    gene_id: str,
    host_db: str,
    save: bool = True,
    verify_summary: bool = False,
    generate_pd: bool = True,
    verify_pds: bool = True,      # NEW
    select_pd: bool = True         # NEW
):
    """
    Run the full curation pipeline for a single paper–gene pair.

    Parameters
    ----------
    ...
    verify_pds : bool
        Run PD verification stage (default True).
    select_pd : bool
        Run PD selection stage (default True).
    """
    process_paper_gene_pair = _import_step1()

    print(f"\n{'=' * 60}")
    print(f"  Pipeline: Single Pair")
    print(f"  PubMed ID      : {pmid}")
    print(f"  Gene ID        : {gene_id}")
    print(f"  Host DB        : {host_db}")
    print(f"  Save           : {save}")
    print(f"  Verify summary : {verify_summary}")
    print(f"  Generate PDs   : {generate_pd}")
    print(f"  Verify PDs     : {verify_pds}")    # NEW
    print(f"  Select PD      : {select_pd}")     # NEW
    print(f"{'=' * 60}\n")

    try:
        process_paper_gene_pair(
            pubmed_id=pmid,
            gene_id=gene_id,
            host_db=host_db,
            save=save,
            verify_summary=verify_summary,
            generate_PD=generate_pd,
            verify_PDs=verify_pds,    # NEW
            select_PD=select_pd       # NEW
        )
        print(f"\n✅ Done: {gene_id} / PMID {pmid}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

def run_tests(
    test: str = None,
    yes: bool = False,
):
    """
    Run model comparison or pipeline variation tests (Step 2).

    Parameters
    ----------
    test : str or None
        Which test to run. One of:
          - "model_comparison"     – multi-model comparison on the test set
          - "no_summary"           – PD generation without the summary step
          - "variation_comparison" – compare different pipeline configurations
        If None, an interactive menu is shown.
    yes : bool
        Skip the confirmation prompt (default False).

    Example
    -------
    >>> from main import run_tests
    >>> run_tests(test="model_comparison", yes=True)
    """
    fns = _import_step2()

    options = {
        "1": ("Multi-model comparison (original)",       fns["model_comparison"]),
        "2": ("Without summary step (Sonnet 4.5 only)",  fns["no_summary"]),
        "3": ("Pipeline variation comparison",           fns["variation_comparison"]),
    }
    cli_map = {
        "model_comparison":     "1",
        "no_summary":           "2",
        "variation_comparison": "3",
    }

    if test:
        if test not in cli_map:
            raise ValueError(f"Unknown test '{test}'. Choose from: {list(cli_map)}")
        choice = cli_map[test]
    else:
        print("\nSelect test to run:")
        for key, (label, _) in options.items():
            print(f"  {key}. {label}")
        choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice not in options:
        print(f"Invalid choice '{choice}'.")
        return

    label, run_fn = options[choice]

    if not yes:
        response = input(f"\nProceed with '{label}'? (yes/no): ").strip().lower()
        if response not in ("yes", "y"):
            print("Cancelled.")
            return

    run_fn()


def run_batch(csv_path: str, save: bool = True):
    """
    Run batch processing on a CSV of paper–gene pairs (Step 3).

    Parameters
    ----------
    csv_path : str
        Path to CSV file with at least 'pmid', 'gene_ID', and 'host_DB' columns.
    save : bool
        Persist results to disk (default True).

    Returns
    -------
    pd.DataFrame
        Results summary with one row per gene–paper pair.

    Example
    -------
    >>> from main import run_batch
    >>> results = run_batch("gene_paper_pairs.csv")
    """
    from pathlib import Path

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    process_from_csv = _import_step3()

    print(f"\n{'=' * 60}")
    print(f"  Pipeline: Batch Processing")
    print(f"  Input CSV : {csv_path}")
    print(f"  Save      : {save}")
    print(f"{'=' * 60}\n")

    results_df = process_from_csv(csv_path, save=save)

    if not results_df.empty:
        output_path = csv_path.replace(".csv", "_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults summary saved to: {output_path}")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Gene curation pipeline – single pair, testing, or batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single pair - default pipeline (summary → PDs → verify)
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb

  # Single pair - full pipeline with selection (summary → PDs → verify → select)
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb --select-pd

  # Single pair - skip verification (summary → PDs → select)
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb --no-verify-pds --select-pd

  # Single pair - baseline only (summary → PDs)
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb --no-verify-pds

  # Single pair - summary only, no PD generation
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb --no-generate-pd

  # Single pair - with summary verification (verify_summary → PDs → verify)
  python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db plasmodb --verify-summary

  # Model comparison testing
  python main.py test --test model_comparison --yes

  # Pipeline variation testing
  python main.py test --test variation_comparison

  # Batch processing
  python main.py batch --csv gene_paper_pairs.csv
""",
    )

    subparsers = parser.add_subparsers(dest="step", metavar="STEP")
    subparsers.required = True

    # ── Step 1: single ──────────────────────────────────────────────────────────────────────
    p1 = subparsers.add_parser("single", help="Process one paper–gene pair (Step 1).")
    p1.add_argument("--pmid", required=True, help="PubMed ID")
    p1.add_argument("--gene_id", required=True, help="Gene ID on VPDB")
    p1.add_argument("--host_db", required=True, help="Daughter VPDB Project hosting the Gene ID")
    p1.add_argument("--no-save", dest="save", action="store_false",
                    help="Skip saving intermediate results")
    p1.add_argument("--verify-summary", dest="verify_summary", action="store_true",
                    help="Enable the summary verification stage (Stage 2)")
    p1.add_argument("--no-generate-pd", dest="generate_pd", action="store_false",
                    help="Stop after summary; skip PD generation")
    p1.add_argument("--no-verify-pds", dest="verify_pds", action="store_false",
                    help="Skip PD verification stage (Stage 4)")
    p1.add_argument("--select-pd", dest="select_pd", action="store_true",
                    help="Enable PD selection stage (Stage 5)")
    p1.set_defaults(save=True, verify_summary=False, generate_pd=True,
                    verify_pds=True, select_pd=False)

    # ── Step 2: test ──────────────────────────────────────────────────────────
    p2 = subparsers.add_parser("test", help="Run model / pipeline variation tests (Step 2).")
    p2.add_argument(
        "--test",
        choices=["model_comparison", "no_summary", "variation_comparison"],
        default=None,
        help="Which test to run (interactive menu if omitted)",
    )
    p2.add_argument("--yes", "-y", action="store_true",
                    help="Skip the confirmation prompt")

    # ── Step 3: batch ─────────────────────────────────────────────────────────
    p3 = subparsers.add_parser("batch", help="Batch-process a CSV of pairs (Step 3).")
    p3.add_argument("--csv",     required=True, help="Path to input CSV file")
    p3.add_argument("--no-save", dest="save", action="store_false",
                    help="Skip saving results to disk")
    p3.set_defaults(save=True)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.step == "single":
        run_pipeline(
            pmid=args.pmid,
            gene_id=args.gene_id,
            host_db=args.host_db,
            save=args.save,
            verify_summary=args.verify_summary,
            generate_pd=args.generate_pd,
            verify_pds=args.verify_pds,
            select_pd=args.select_pd,
        )

    elif args.step == "test":
        run_tests(test=args.test, yes=args.yes)

    elif args.step == "batch":
        run_batch(csv_path=args.csv, save=args.save)


if __name__ == "__main__":
    main()