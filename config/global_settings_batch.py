import sys
import os
import time
import json
import requests
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI  # if processing using GPT
from anthropic import Anthropic  # if processing suing Claude
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# API Setup
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



# Models
# PROVIDER = "anthropic"
# SUMMARY_MODEL = "claude-sonnet-4-20250514"
# PD_GENERATOR_MODEL = "claude-sonnet-4-20250514"
# PD_VERIFIER_MODEL = "claude-sonnet-4-20250514"
# FORMATTER_MODEL = "claude-sonnet-4-20250514"

EXISTING_SUMMARY_MODEL = "claude-sonnet-4-5-20250929"    # key pre-existing saved summaries are stored under

PROVIDER = "openrouter"
OPENROUTER_JSON_PLUGINS = [{"id": "response-healing"}]
OPENROUTER_CACHE_CFG = {"enabled": True, "ttl": "1h"}

# update to openrouter naming
SUMMARY_MODEL = "anthropic/claude-sonnet-4.5"    # for any fresh summary generation
PD_GENERATOR_MODEL = "anthropic/claude-sonnet-4.5"
PD_VERIFIER_MODEL = "anthropic/claude-sonnet-4.5"
FORMATTER_MODEL = "anthropic/claude-sonnet-4.5"


# Processing Parameters
MAX_TOKENS = 20000
MODEL_TEMP = 0
N_QUOTES = 2
MAX_RETRY = 3
HTTP_TIMEOUT = 30
N_PDs = 3  # Maximum number of PDs to generate

# Output
OUT_DIR = Path("./out/VPDB_PD_set_expansion/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing Strategy
MIN_GENES_FOR_CACHING = 1  # TEST: route every paper through the OpenRouter caching path
# USE_BATCH_FOR_LOW_DENSITY = True  # Use batch API for papers with <3 genes (falls back to standard if unavailable)
USE_BATCH_FOR_LOW_DENSITY = True  # Anthropic Batch API not available via OpenRouter
OVERWRITE_EXISTING = False  # Set to True to reprocess already-completed pairs

# --- Supplementary materials (opt-in) ---------------------------------------
# When True, fetch + gene-filter an OA paper's supplementary files (PDF / tables /
# spreadsheets) via pipeline.supplementary_helpers and append the gene-relevant block
# to the summary input. Off by default. NOTE: summaries already cached without
# supplements are reused as-is unless OVERWRITE_EXISTING=True.
FETCH_SUPPLEMENTARY = False
SUPPLEMENTARY_CAPS = {}  # optional overrides for supplementary_helpers.DEFAULT_CAPS

# CSV Column Mapping (adjust these if your CSV has different column names)
CSV_COLUMNS = {
    'gene_id': 'Gene ID',        # Column containing gene IDs (e.g., "PF3D7_1234")
    'pmid': 'pmid_CLEAN',        # Column containing PubMed IDs (e.g., "12345678")
    'database': 'Database',       # Column containing database names (e.g., "plasmodb", "toxodb")
    # Optional columns for filtering (will be used if present):
    'paper_available': 'paper_available',  # Boolean: whether paper is available in PMC
    'alias_in_text': 'alias_in_text'       # Boolean: whether gene alias found in paper
}