#!/usr/bin/env python3
"""
Example usage of the gene PD pipeline.

This script demonstrates:
1. How to retrieve paper data from PubMed
2. How to get gene synonyms
3. How to run the complete pipeline
4. How to access and save results
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from pd_pipeline.pubmed import get_pubmed_json, parse_pubmed_json
from pd_pipeline.gene_data import get_gene_synonyms, gene_to_prompt
from single_pair_pipeline_workflow import getGeneSummary, generatePDs, verifyPDs
from pd_pipeline.utils import save_status, load_status, check_if_ok
from config.models import SUMMARY_LLM, PD_GENERATOR_LLM, PD_QC_LLM, OUT_DIR


def run_pipeline_example(pmid: str, gene_id: str):
    """
    Run the complete gene PD pipeline for a paper-gene pair.
    
    Args:
        pmid: PubMed ID of the paper
        gene_id: Gene identifier (e.g., "PF3D7_0810800")
    """
    print(f"\n{'=' * 80}")
    print(f"Running pipeline for Gene {gene_id} in PMID {pmid}")
    print(f"{'=' * 80}\n")
    
    # Step 1: Get paper text from PubMed
    print("Step 1: Retrieving paper from PubMed...")
    pubmed_json = get_pubmed_json(pmid)
    pubmed_text = parse_pubmed_json(pubmed_json)
    print(f"  ✓ Retrieved paper ({len(pubmed_text)} characters)")
    
    # Step 2: Get gene synonyms
    print("\nStep 2: Retrieving gene synonyms...")
    synonyms = get_gene_synonyms(gene_id, pubmed_text)
    gene_text = gene_to_prompt(gene_id, synonyms)
    print(f"  ✓ Found synonyms: {gene_text}")
    
    # Step 3: Generate gene summary
    print("\nStep 3: Generating gene-specific summary...")
    summary_key = "getGeneSummary"
    
    # Check if already exists
    summary_data = load_status(
        pubmed_id=pmid,
        out_dir=OUT_DIR,
        step_key=summary_key,
        gene_id=gene_id,
        llm=SUMMARY_LLM[1]
    )
    
    if check_if_ok(summary_data):
        print("  ✓ Found existing summary")
        summary = summary_data["data"]
    else:
        summary, usage, secs = getGeneSummary(pubmed_text, gene_text)
        if isinstance(summary, (dict, list)):
            save_status(
                pubmed_id=pmid,
                out_dir=OUT_DIR,
                step_key=summary_key,
                gene_id=gene_id,
                model_name=SUMMARY_LLM[1],
                data=summary,
                success=True,
                usage=usage,
                seconds=secs
            )
            print(f"  ✓ Generated summary in {secs:.1f}s")
            print(f"    Tokens: {usage['input']} in, {usage['output']} out")
        else:
            print("  ✗ Summary generation failed")
            return
    
    # Step 4: Generate product descriptions
    print("\nStep 4: Generating product descriptions...")
    pd_key = "generatePDs"
    
    pd_data = load_status(
        pubmed_id=pmid,
        out_dir=OUT_DIR,
        step_key=pd_key,
        gene_id=gene_id,
        llm=PD_GENERATOR_LLM[1]
    )
    
    if check_if_ok(pd_data):
        print("  ✓ Found existing PDs")
        pds = pd_data["data"]
    else:
        pds, usage, secs = generatePDs(summary_json=summary, gene_text=gene_text)
        if isinstance(pds, (dict, list)):
            save_status(
                pubmed_id=pmid,
                out_dir=OUT_DIR,
                step_key=pd_key,
                gene_id=gene_id,
                model_name=PD_GENERATOR_LLM[1],
                data=pds,
                success=True,
                usage=usage,
                seconds=secs
            )
            print(f"  ✓ Generated PDs in {secs:.1f}s")
            print(f"    Tokens: {usage['input']} in, {usage['output']} out")
        else:
            print("  ✗ PD generation failed")
            return
    
    # Step 5: verify and recommend PD
    print("\nStep 5: Verifying product descriptions...")
    verify_key = "verifyPDs"
    
    verify_data = load_status(
        pubmed_id=pmid,
        out_dir=OUT_DIR,
        step_key=verify_key,
        gene_id=gene_id,
        llm=PD_QC_LLM[1]
    )
    
    if check_if_ok(verify_data):
        print("  ✓ Found existing verified PDs")
        verified = verify_data["data"]
    else:
        verified, usage, secs = verifyPDs(
            suggestedPDs=pds,
            paper_text=pubmed_text,
            gene_text=gene_text
        )
        if isinstance(verified, (dict, list)):
            save_status(
                pubmed_id=pmid,
                out_dir=OUT_DIR,
                step_key=verify_key,
                gene_id=gene_id,
                model_name=PD_QC_LLM[1],
                data=verified,
                success=True,
                usage=usage,
                seconds=secs
            )
            print(f"  ✓ Verified PDs in {secs:.1f}s")
            print(f"    Tokens: {usage['input']} in, {usage['output']} out")
        else:
            print("  ✗ PD verification failed")
            return
    
    # Step 6: Display results
    print(f"\n{'=' * 80}")
    print("Pipeline Complete!")
    print(f"{'=' * 80}\n")
    
    if isinstance(verified, dict) and "RPD" in verified:
        print("Recommended Product Description (RPD):")
        rpd = verified["RPD"]
        print(f"  Description: {rpd.get('description', 'N/A')}")
        print(f"  Evidence Code: {rpd.get('evidence_code', 'N/A')}")
        print(f"  Verification: {rpd.get('verification_status', 'N/A')}")
        
        if "APDs" in verified and verified["APDs"]:
            print("\nAlternative Product Descriptions (APDs):")
            for i, apd in enumerate(verified["APDs"], 1):
                print(f"  {i}. {apd.get('description', 'N/A')}")
                print(f"     Evidence: {apd.get('evidence_code', 'N/A')}")
    
    print(f"\nResults saved to: {OUT_DIR}/")
    print(f"  - PMID_{pmid}_GENE_{gene_id}_*.json")


def main():
    """Run example pipeline."""
    # Example gene and paper
    pmid = "31795916"  # Example PubMed ID
    gene_id = "PF3D7_0810800"  # Example gene ID
    
    # You can also take these from command line
    if len(sys.argv) > 2:
        pmid = sys.argv[1]
        gene_id = sys.argv[2]
    
    run_pipeline_example(pmid, gene_id)


if __name__ == "__main__":
    main()
