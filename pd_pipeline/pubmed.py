"""
PubMed integration for retrieving and parsing scientific papers.
"""

import requests
import re
from typing import Dict, Optional

pubmed_base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
#  The sections of PubMed documents that are relevant for gene curation.
pubmed_sections = ['FIG', 'TABLE', 'RESULTS', 'CONCL', 'DISCUSSION', 'SUPPL']


def get_pubmed_json(pubmed_id):
    """
    Fetches the PubMed JSON for a given PubMed ID.
    Args:
        pubmed_id (str): The PubMed ID to fetch.
    Raises:
        ValueError: If the PubMed ID is not found or if the response is not in JSON format.
    Returns:
        dict: The JSON response from the PubMed API.
    Notes:
        The PubMed ID should be a valid identifier, and the function constructs the URL
        to fetch the JSON data using the global pubmed_base_url
    """
    url = pubmed_base_url + str(pubmed_id)
    response = requests.get(url)

    if response.status_code == 200:
        # Checks if there is a json response - the API returns html if the paper is not found.
        if not response.headers.get('Content-Type', '').startswith('application/json'):
            raise ValueError("Paper not found")

        return response.json()

    raise ValueError(f"Paper fetch status code: {response.status_code}")


# # example test run
# pubmed_id = 27128092
# # Get the PubMed JSON for the given ID.
# pubmed_json = get_pubmed_json(pubmed_id)
# # Parse the PubMed JSON to get the text of the required sections.
# pubmed_text = parse_pubmed_json(pubmed_json)
# print(pubmed_text)

################################################################################################
# STAGE 2: Get gene aliases
################################################################################################
# Preparation -  Fetching all gene aliases

def parse_pubmed_json(pubmed_json):
    """
    Parses a PubMed JSON response to extract and concatenate text from relevant sections.
    Args:
        pubmed_json (list of dict): The JSON response from the PubMed API,
                                    typically a list of (usually 1) documents.
    Returns:
        str: A single string containing the concatenated text from required sections, preserving the original ordering.
    Notes:
        Uses the global variable `pubmed_sections` to determine which sections to include.
    """
    document_text = ""

    for doc in pubmed_json:
        for document in doc.get("documents", []):
            for passage in document.get("passages", []):
                infons = passage.get("infons", {})
                section_type = infons.get("section_type", "")
                if section_type.upper() in {s.upper() for s in pubmed_sections}:
                    if "text" in passage:
                        document_text += passage["text"] + "\n"

    return document_text


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
