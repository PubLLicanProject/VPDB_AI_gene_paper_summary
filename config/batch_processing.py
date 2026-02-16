"""
Configuration for batch processing settings
"""

from dotenv import load_dotenv

load_dotenv()
# API Setup
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Models
SUMMARY_MODEL = "claude-sonnet-4-20250514"
PD_GENERATOR_MODEL = "claude-sonnet-4-20250514"
PD_VERIFIER_MODEL = "claude-sonnet-4-20250514"
FORMATTER_MODEL = "claude-sonnet-4-20250514"


# Processing Parameters
MAX_TOKENS = 20000
MODEL_TEMP = 0
N_QUOTES = 2
MAX_RETRY = 3
HTTP_TIMEOUT = 30
N_PDs = 3  # Maximum number of PDs to generate

# Output
OUT_DIR = Path("./out/VPDB_UserComments_batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing Strategy
MIN_GENES_FOR_CACHING = 3  # Use caching if paper has 3+ genes
USE_BATCH_FOR_LOW_DENSITY = True  # Use batch API for papers with <3 genes (falls back to standard if unavailable)
OVERWRITE_EXISTING = False  # Set to True to reprocess already-completed pairs

# CSV Column Mapping (adjust these if your CSV has different column names)
CSV_COLUMNS = {
    'gene_id': 'Gene ID',        # Column containing gene IDs (e.g., "PF3D7_1234")
    'pmid': 'PMID_Final',        # Column containing PubMed IDs (e.g., "12345678")
    'database': 'Database',       # Column containing database names (e.g., "plasmodb", "toxodb")
    # Optional columns for filtering (will be used if present):
    'paper_available': 'paper_available',  # Boolean: whether paper is available in PMC
    'alias_in_text': 'alias_in_text'       # Boolean: whether gene alias found in paper
}