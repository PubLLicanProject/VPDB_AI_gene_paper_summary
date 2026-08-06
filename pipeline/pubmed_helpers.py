"""
Helper functions for fetching paper text from PMID and converting DOI and PMCID to PMID
"""

import time
import random
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

# --- Fetch robustness (retry/backoff) ---------------------------------------
# The BioC endpoint occasionally returns transient errors (timeouts, 5xx, or
# throttling under bursty load). Without retries a single hiccup was being
# recorded as "paper not available" (a false negative). Retry those, but treat a
# genuine "not in the OA subset" (HTTP 200 + non-JSON body) as permanent.
FETCH_MAX_RETRIES = 4          # retries AFTER the first attempt (up to 5 tries)
FETCH_BACKOFF_BASE = 1.5       # seconds; grows exponentially with jitter
FETCH_BACKOFF_CAP = 30.0       # per-sleep ceiling
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PaperNotInOA(Exception):
    """Paper is not in the PMC Open Access subset (permanent — do not retry)."""


class PaperFetchError(Exception):
    """Transient/other fetch failure that persisted after retries (retryable later)."""

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


def get_pubmed_json(pubmed_id, *, max_retries: int = None, timeout: int = None):
    """
    Fetch BioC JSON for a PMID, retrying transient failures with exponential backoff.

    Distinguishes a genuine "not in the OA subset" (permanent) from transient
    network/server errors (timeouts, connection resets, 429/5xx) so the latter are
    retried instead of being silently recorded as "paper unavailable".

    Raises:
        PaperNotInOA:    HTTP 200 with a non-JSON body -> not in the OA subset (permanent).
        PaperFetchError: transient errors that persisted after all retries, or a
                         non-retryable HTTP status (e.g. 404).
    """
    max_retries = FETCH_MAX_RETRIES if max_retries is None else max_retries
    timeout = HTTP_TIMEOUT if timeout is None else timeout
    url = PUBMED_BASE_URL + str(pubmed_id) + "/unicode"

    last_err = "unknown error"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
        else:
            status = response.status_code
            if status == 200:
                if not response.headers.get("Content-Type", "").startswith("application/json"):
                    # 200 but not JSON -> genuinely not in the OA subset (permanent)
                    raise PaperNotInOA(
                        f"Paper {pubmed_id} not available in PMC Open Access subset"
                    )
                try:
                    return response.json()
                except ValueError as e:
                    # declared JSON but unparseable -> treat as transient and retry
                    last_err = f"JSON decode error: {e}"
            elif status in RETRYABLE_STATUS:
                last_err = f"HTTP {status}"
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), FETCH_BACKOFF_CAP))
                        continue  # server told us when to retry; skip generic backoff
                    except ValueError:
                        pass
            else:
                # non-retryable HTTP status (e.g. 404)
                raise PaperFetchError(
                    f"Paper fetch failed for {pubmed_id}, status: {status}"
                )

        # exponential backoff with jitter before the next attempt (if any remain)
        if attempt < max_retries:
            delay = min(FETCH_BACKOFF_CAP, FETCH_BACKOFF_BASE * (2 ** attempt))
            time.sleep(delay * (0.5 + random.random()))

    raise PaperFetchError(
        f"Paper fetch failed for {pubmed_id} after {max_retries + 1} attempts: {last_err}"
    )


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