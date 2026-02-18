"""
Preprocessing Script: Validate Gene-Paper Pairs
Checks which papers are available in PMC and which genes are mentioned in those papers.

Output: CSV with paper_available, alias_in_text, and mentions columns
Usage: python preprocess_validate_pairs.py input.csv output_validated.csv
"""

import os
import re
import requests
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

#######################################################################################################################
#                                           CONFIGURATION                                                             #
#######################################################################################################################

HTTP_TIMEOUT = 30

# CSV Column Mapping (adjust if your CSV has different names)
CSV_COLUMNS = {
    'gene_id': 'Gene ID',
    'pmid': 'PMID_Final',
    'database': 'Database',
}

#######################################################################################################################
#                                    PAPER FETCHING                                                                   #
#######################################################################################################################

PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
PUBMED_SECTIONS = ['RESULTS', 'FIG', 'DISCUSSION', 'DISCUSS', 'CONCLUSION', 'FIGURE', 'CONCL', 'TABLE', 'SUPPL']


def parse_pubmed_json(pubmed_json):
    """Parse PubMed JSON to extract relevant sections."""
    document_text = ""
    for doc in pubmed_json:
        for document in doc.get("documents", []):
            for passage in document.get("passages", []):
                section_type = passage.get("infons", {}).get("section_type", "")
                if section_type.upper() in {s.upper() for s in PUBMED_SECTIONS}:
                    if "text" in passage:
                        document_text += passage["text"] + "\n"
    return document_text


def get_pubmed_json(pubmed_id):
    """Fetch PubMed JSON with improved error handling."""
    url = PUBMED_BASE_URL + str(pubmed_id) + "/unicode"
    response = requests.get(url, timeout=HTTP_TIMEOUT)

    if response.status_code == 200:
        if not response.headers.get('Content-Type', '').startswith('application/json'):
            raise ValueError(f"Paper {pubmed_id} not available in PMC Open Access subset")
        return response.json()

    raise ValueError(f"Paper fetch failed for {pubmed_id}, status: {response.status_code}")


def get_paper_text(pubmed_id: str) -> str:
    """Fetch and parse paper text."""
    pubmed_json = get_pubmed_json(pubmed_id)
    return parse_pubmed_json(pubmed_json)


#######################################################################################################################
#                                    GENE ALIAS RETRIEVAL                                                             #
#######################################################################################################################

def get_vpdb_alias(gene_id: str, host_db: str) -> List[str]:
    """Fetch aliases from VEuPathDB."""
    if not host_db or not gene_id:
        return []

    host_db_lc = host_db.strip().lower()

    db_mappings = {
        "microbiomedb": ("mbio", "MicrobiomeDB"),
        "clinepidb": ("ce", "ClinEpiDB"),
        "amoebadb": ("amoeba", "AmoebaDB"),
        "cryptodb": ("cryptodb", "CryptoDB"),
        "fungidb": ("fungidb", "FungiDB"),
        "giardiadb": ("giardiadb", "GiardiaDB"),
        "hostdb": ("hostdb", "HostDB"),
        "microsporidiadb": ("micro", "MicrosporidiaDB"),
        "piroplasmadb": ("piro", "PiroplasmaDB"),
        "plasmodb": ("plasmo", "PlasmoDB"),
        "toxodb": ("toxo", "ToxoDB"),
        "trichdb": ("trichdb", "TrichDB"),
        "tritrypdb": ("tritrypdb", "TriTrypDB"),
        "vectorbase": ("vectorbase", "VectorBase"),
        "orthomcl": ("orthomcl", "OrthoMCL"),
    }

    if host_db_lc not in db_mappings:
        return []

    path_prefix, project = db_mappings[host_db_lc]
    url = f"https://{host_db_lc}.org/{path_prefix}/service/record-types/gene/records"

    data = {
        "attributes": [],
        "primaryKey": [
            {"name": "source_id", "value": gene_id},
            {"name": "project_id", "value": project},
        ],
        "tables": ["AllProducts", "Alias"],
    }

    try:
        resp = requests.post(url, headers={"content-type": "application/json"},
                             json=data, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return []
        payload = resp.json() or {}
    except Exception:
        return []

    alias_set = set()
    for row in (payload.get("tables", {}).get("Alias", []) or []):
        val = row.get("alias")
        if val and val != gene_id:
            alias_set.add(val)

    return sorted(alias_set)


def _count_substrings(paper: str, alias: str) -> int:
    """Count occurrences of alias in paper with flexible matching."""
    if not paper or not alias:
        return 0

    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", alias)
    if m:
        letters, digits = m.groups()
        inner = f"{re.escape(letters)}-?{re.escape(digits)}"
    else:
        parts = []
        i = 0
        while i < len(alias):
            if alias[i] in "_- ":
                while i < len(alias) and alias[i] in "_- ":
                    i += 1
                parts.append(r"[-_\s]+")
                continue
            parts.append(re.escape(alias[i]))
            i += 1
        inner = "".join(parts)

    pattern = rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])"
    return len(re.findall(pattern, paper, flags=re.IGNORECASE))


def check_gene_in_text(gene_id: str, paper: str, host_db: str) -> Tuple[bool, int]:
    """
    Check if gene ID or any of its aliases appear in paper text.

    Returns:
        Tuple of (found: bool, total_mentions: int)
    """
    if not paper:
        return False, 0

    # Check gene ID itself
    gene_count = _count_substrings(paper, gene_id)

    # Check ALL aliases
    aliases = get_vpdb_alias(gene_id, host_db)

    # Count each alias
    total_mentions = gene_count
    for alias in aliases:
        alias_count = _count_substrings(paper, alias)
        total_mentions += alias_count

    found = total_mentions > 0

    return found, total_mentions


#######################################################################################################################
#                                    VALIDATION PROCESSING                                                            #
#######################################################################################################################

def validate_pairs(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Validate gene-paper pairs for paper availability and gene mentions.

    Args:
        csv_path: Path to input CSV with gene-paper pairs
        output_path: Path to save validated CSV (optional)

    Returns:
        DataFrame with validation results
    """
    print(f"\n{'=' * 80}")
    print(f"PREPROCESSING: Validating Gene-Paper Pairs")
    print(f"{'=' * 80}")
    print(f"Input: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    # Get column names
    gene_col = CSV_COLUMNS['gene_id']
    pmid_col = CSV_COLUMNS['pmid']
    db_col = CSV_COLUMNS['database']

    # Check required columns exist
    required_cols = [gene_col, pmid_col, db_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing columns: {missing}")
        print(f"Available columns: {', '.join(df.columns)}")
        return pd.DataFrame()

    # Group by paper to minimize redundant fetches
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        pmid = str(row[pmid_col])
        gene_id = str(row[gene_col])
        database = str(row[db_col])
        grouped[pmid].append((gene_id, database))

    print(f"Unique papers: {len(grouped)}")
    print(f"Unique gene-paper pairs: {len(df)}")

    # Process papers
    results = []
    paper_cache = {}  # Cache paper text to avoid re-fetching

    for i, (pmid, genes) in enumerate(grouped.items(), 1):
        print(f"\n[{i}/{len(grouped)}] Paper {pmid}: {len(genes)} genes")

        # Check if paper is available
        paper_available = False
        paper_text = None

        try:
            paper_text = get_paper_text(pmid)
            paper_available = True
            paper_cache[pmid] = paper_text
            print(f"  ✓ Paper available ({len(paper_text):,} chars)")
        except Exception as e:
            print(f"  ✗ Paper not available: {e}")
            # Add all genes from this paper as unavailable
            for gene_id, database in genes:
                results.append({
                    gene_col: gene_id,
                    pmid_col: pmid,
                    db_col: database,
                    'paper_available': False,
                    'alias_in_text': False,
                    'mentions': 0,
                    'error': str(e)
                })
            continue

        # Check each gene
        for gene_id, database in genes:
            alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, database)

            status = "✓" if alias_in_text else "✗"
            print(f"    {status} {gene_id}: {mentions} mentions")

            results.append({
                gene_col: gene_id,
                pmid_col: pmid,
                db_col: database,
                'paper_available': True,
                'alias_in_text': alias_in_text,
                'mentions': mentions,
                'error': '' if alias_in_text else 'Gene not found in text'
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Summary statistics
    print(f"\n{'=' * 80}")
    print(f"VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total pairs: {len(results_df)}")
    print(f"Paper available: {results_df['paper_available'].sum()}/{len(results_df)}")
    print(f"Gene in text: {results_df['alias_in_text'].sum()}/{len(results_df)}")
    print(
        f"Both checks passed: {((results_df['paper_available']) & (results_df['alias_in_text'])).sum()}/{len(results_df)}")

    # Mention statistics for valid pairs
    valid_pairs = results_df[results_df['alias_in_text'] == True]
    if len(valid_pairs) > 0:
        print(f"\nMention statistics (for genes found in text):")
        print(f"  Mean: {valid_pairs['mentions'].mean():.1f}")
        print(f"  Median: {valid_pairs['mentions'].median():.0f}")
        print(f"  Min: {valid_pairs['mentions'].min()}")
        print(f"  Max: {valid_pairs['mentions'].max()}")

    # Save output
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Validated CSV saved to: {output_path}")
        print(f"\nNext step: Use this CSV with the main pipeline")
        print(f"  The pipeline will filter to paper_available=TRUE and alias_in_text=TRUE")

    return results_df


#######################################################################################################################
#                                              MAIN                                                                   #
#######################################################################################################################

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_validate_pairs.py <input.csv> [output.csv]")
        print("\nInput CSV must have columns:")
        print(f"  - {CSV_COLUMNS['gene_id']}")
        print(f"  - {CSV_COLUMNS['pmid']}")
        print(f"  - {CSV_COLUMNS['database']}")
        print("\nOutput CSV will add columns:")
        print("  - paper_available (True/False)")
        print("  - alias_in_text (True/False)")
        print("  - mentions (integer count)")
        print("  - error (error message if failed)")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv.replace('.csv', '_validated.csv')

    if not Path(input_csv).exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Run validation
    validate_pairs(input_csv, output_csv)