"""
Helper functions for fetching paper text from PMID and converting DOI and PMCID to PMID
"""

import requests
from typing import List, Dict, Tuple, Optional, Any

# TODO: Add auto-conversion of DOI and PMCID to PMID for our pipeline; consider using metapub?

# change to configure which paper sections are analysied for the summary
PUBMED_SECTIONS = ['RESULTS', 'FIG', 'DISCUSSION', 'DISCUSS', 'CONCLUSION', 'FIGURE', 'CONCL', 'TABLE', 'SUPPL']
# optional - change if you want
optional_tool = "" # e,g,"PD_generator"
optional_email = "" # your_email@example.com
# constant
PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
HTTP_TIMEOUT = 180

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


def check_paper_available(pubmed_id: str) -> bool:
    """
    Check if paper is available via PMC API.

    Args:
        pubmed_id: PubMed ID

    Returns:
        True if paper can be fetched, False otherwise
    """
    try:
        get_paper_text(pubmed_id)
        return True
    except Exception:
        return False

# # example test run
# pubmed_id = 27128092
# # Get the PubMed JSON for the given ID.
# pubmed_json = get_pubmed_json(pubmed_id)
# # Parse the PubMed JSON to get the text of the required sections.
# pubmed_text = parse_pubmed_json(pubmed_json)
#
# # or directly using convenience wrapper:
# pubmed_text = get_paper_text(pubmed_id)
# print(pubmed_text)
# # you can pre-check availability with convenicence function too.
# check_paper_available(pubmed_id)