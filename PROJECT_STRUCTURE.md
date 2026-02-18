# Project Structure Guide

## TL;DR

**What it does:** LLM-powered pipeline that reads scientific papers, extracts gene-specific information, and generates standardized product descriptions (PDs) for VEuPathDB gene annotations.

**Three modes:**
1. **Single pair** - Process one gene-paper pair (development/testing)
2. **Model testing** - Compare different LLMs and pipeline configurations on a test set
3. **Batch processing** - Production runs on hundreds/thousands of gene-paper pairs with caching and batch API optimization

**Key tech:** Python + Anthropic/OpenAI/OpenRouter APIs + PubMed/VEuPathDB integration

**Entry points:** `main.py` (API + CLI) or direct imports from `STEP_1/2/3_*.py`

#### NB: This doc was generated using Claude Sonnet 4.5 and has been checked over by Jordan

---

## Directory Structure

```
VPDB_AI_gene_paper_summary/
├── config/                          
│   ├── global_settings_single.py    # Single-pair processing config
│   ├── global_settings_testing.py   # Model and pipeline variation comparison config
│   └── global_settings_batch.py     # Batch processing config (e.g. for user comments)
│
├── pipeline/                        # Core pipeline sub-modules
│   ├── pipeline_steps.py            # Main workflow stages (summary, PDs, verification)
│   ├── prompts.py                   # Prompt templates + LLM client wrapper
│   ├── utils.py                     # Helper functions (JSON parsing, data extraction)
│   ├── pubmed_helpers.py            # Paper fetching from PubMed/PMC
│   └── vpdb_helpers.py              # Gene alias fetching from VEuPathDB
│
├── curated_data/                    
│   ├── example_PDs_2025_07_10.txt   # Example product descriptions for few-shot learning
│   └── 2026_01_12_subset_balanced_by_species.csv  # Test set for model and pipeline comparison
│
├── out/                             
│   ├── cache/                       # Single-pair results (one JSON per paper)
│   ├── tests/                       # Model comparison results (ignored)
│   └── VPDB_UserComments_batch/     # Batch processing results (ignored)
│
├── main.py                          # Main entry point (API + CLI supported)
├── STEP_1_single_pair_processing.py # Single gene-paper pair processing functions
├── STEP_2_running_tests.py          # Model comparison and pipeline variation testing funcitons
├── STEP_3_batch_processing.py       # Batch processing functions
│
├── .env                             # API keys (not in repo, add to gitignore)
├── .env.example                     # Template for API keys
├── requirements.txt                 # Python dependencies
└── README.md                        
```

---

## Core Pipeline Flow

### Stage 0: Data Fetching
```
PubMed ID → fetch paper text (PMC API)
Gene ID + host_db → fetch aliases (VEuPathDB API)
Check if gene/aliases appear in paper text
```

### Stage 1: Summary Generation (`getGeneSummary`)
```
Paper text + Gene info → LLM
↓
{
  "GeneSummary": [bullet points with evidence],
  "AdditionalInferences": [...],
  "ShortSummary": str, # single sentence that can be used as Header for User comments,
  "only_in_passing": bool  # Skip PD generation if true
}
```

### Stage 2: Summary Verification (Optional) (`verifyGeneSummary`)
```
Original summary + Paper text → LLM
↓
Verified/corrected summary
```

### Stage 3: PD Generation (`generatePDs`)
```
Summary bullets → LLM
↓
{
  "PDs": [
    {"description": "...", "evidence_code": "IMP", ...},
    ...
  ]
}
```

### Stage 4: PD Verification (`verifyPDs`)
```
Suggested PDs + Paper text → LLM
↓
{
  "RPD": {recommended PD}, # this is the one to display
  "APDs": [alternative PDs] # These can be added on if we think users might want multiple suggestions
}
```

### Stage 5: PD Selection (Alternative to verification) (`selectPD`)
```
Suggested PDs + Example PDs → LLM
↓
Best PD selected based on examples
```

---

## Key Components

### `main.py` - Unified Entry Point

**API:**
```python
from main import run_pipeline, run_tests, run_batch

# Single pair
run_pipeline(pmid="12345678", gene_id="PF3D7_1234", host_db="plasmodb")

# Model testing
run_tests(test="model_comparison", yes=True) # uses best pipeline variation; configure models in config/global_settings_testing

# Batch processing
results = run_batch("gene_paper_pairs.csv") # good idea to pre-check paper availability and alias presence
```

**CLI:**
```bash
python main.py single --pmid 12345678 --gene_id PF3D7_1234 --host_db plasmodb
python main.py test --test model_comparison --yes
python main.py batch --csv pairs.csv
```

### `config/` - Configuration Files

Each mode has its own config file:

**`global_settings_single.py`** - Development/testing single pairs
- Model selection per pipeline stage
- Prompt parameters (N_QUOTES, N_PDs)
- Output directory: `./out/cache`

**`global_settings_testing.py`** - Model comparison experiments
- List of models to test (`MODELS_TO_TEST`)
- Test set path
- Output directory: `./out/tests`

**`global_settings_batch.py`** - Production batch runs
- Caching strategy (use caching for papers with 3+ genes)
- Batch API settings
- CSV column mappings
- Output directory: `./out/VPDB_UserComments_batch`

### `pipeline/` - Core Modules

#### `pipeline_steps.py` - Main Workflow Functions

Five primary stage functions:
```python
getGeneSummary(pubmed_text, gene_text) → (summary_json, usage, seconds)
verifyGeneSummary(gene_text, pubmed_text, summary_json) → (verified_summary, usage, seconds)
generatePDs(summary_json, gene_text, n_pds=3) → (pds_json, usage, seconds)
verifyPDs(suggestedPDs, paper_text, gene_text) → (verified_pds, usage, seconds)
selectPD(suggestedPDs, summary_for_pd) → (selected_pd, usage, seconds)
```

All functions:
- Return tuple: `(data_dict, usage_dict, elapsed_seconds)`
- Handle errors (return `{"error": "..."}` on failure)
- Use global config for model selection

#### `prompts.py` - LLM Interface

**Core function:**
```python
call_prompt(provider, model, system_prompt, user_prompts, prefill_text="")
```

Supports:
- `provider="anthropic"` - Direct Anthropic API
- `provider="openai"` - Direct OpenAI API  
- `provider="openai-chat"` - OpenAI Chat Completions
- `provider="openai-responses"` - OpenAI Responses API (reasoning models)
- `provider="openrouter"` - OpenRouter (multi-provider)

**Prompt management:**
```python
get_prompt_and_replace(stage_key, replacements, prompt_type)
```
- Loads prompts from `global_prompts_and_schema` dict
- Replaces `[PLACEHOLDERS]` with actual values
- Preserves list structure for multi-turn prompts

**Key data structure:**
```python
global_prompts_and_schema = {
    "getGeneSummary": {
        "SystemPrompt": "...",
        "UserPrompts": ["...", "..."],  # Can be list for multi-turn
        "ValidationSchema": {...}
    },
    # ... other stages
}
```

#### `utils.py` - Helper Functions

**JSON handling:**
- `extract_json(text)` - Strip markdown fences and parse JSON
- `collect_bullets(summary_json)` - Extract bullet points from summary

**Data validation:**
- `check_if_ok(status_data)` - Verify saved results are valid
- `check_if_in_passing(summary_data)` - Check if gene only mentioned in passing

**Gene formatting:**
- `gene_to_prompt(gene, genes)` - Format "GENE_ID (also known as ALIAS1, ALIAS2)"

**Private functions (imported explicitly in prompts.py):**
- `_extract_usage(provider, response)` - Parse token usage from API responses
- `_to_responses_input(messages)` - Convert messages for OpenAI Responses API
- `_responses_supports_temperature(model)` - Check if model supports temperature param

#### `pubmed_helpers.py` - Paper Fetching

**Main functions:**
```python
get_paper_text(pubmed_id) → str  # One-liner: fetch + parse
check_paper_available(pubmed_id) → bool  # Pre-check availability
```

**Configuration:**
- `PUBMED_SECTIONS` - Which sections to extract (RESULTS, DISCUSSION, etc.)
- Fetches from PMC Open Access subset only

#### `vpdb_helpers.py` - Gene Data

**Main functions:**
```python
get_vpdb_alias(gene_id, host_db) → List[str]  # All aliases for gene
get_gene_synonyms(gene_id, paper, host_db) → List[str]  # Top 3 aliases in paper
check_gene_in_text(gene_id, paper, host_db) → (bool, int)  # Found + mention count
```

**Supported VPDB daughter databases:**
- plasmodb, toxodb, tritrypdb, cryptodb, fungidb, giardiadb, piroplasmadb, amoebadb, microsporidiadb, trichdb, hostdb, vectorbase, microbiomedb, clinepidb, orthomcl

---

## Execution Modes

### STEP_1: Single Pair Processing

**Purpose:** Process one gene-paper pair (likely what we want for on-the-fly user comment implementation)

**Key function:**
```python
process_paper_gene_pair(
    pubmed_id, 
    gene_id, 
    host_db,           # Required - which VEuPathDB database
    save=True,         # Save results to JSON
    verify_summary=False,  # Optional summary verification stage
    generate_PD=True,   # Run PD generation stages
    verify_PDs=True, # Verify against set of rules and the original paper text
    selectt_PDFalse # Optional - you can use a selection step to re-format PDs; tends to work well even if verify_PD = False
)
```

**File structure output:**
```json
{
  "getGeneSummary": {
    "VPDB_GENE_ID": {
      "gene_ID": "VPDB_GENE_ID",
      "claude-sonnet-4-20250514": {
        "model": "claude-sonnet-4-20250514",
        "success": true, # ignore this, a bit buggy
        "data": {...},
        "usage": {"input": 123, "output": 456, "total": 579},
        "seconds": 15.2
      }
    }
  },
  "generatePDs": { ... },
  "verifyPDs": { ... }
}
```

**Key features:**
- Result caching (skips stages if already completed)
- Handles errors gracefully (saves error states)
- JSON formatter with retry logic (auto-fixes malformed JSON)

### STEP_2: Model Testing

**Purpose:** Compare LLMs and test pipeline variations on curated test set

**Three test modes:**

1. **`run_model_comparison_on_test_set()`**
   - Runs all models in `MODELS_TO_TEST` on test set
   - Saves results per model per gene-paper pair
   - Tracks progress and timing

2. **`run_without_summary_comparison()`**
   - Tests direct PD generation (skip summary stage)
   - Compares against standard pipeline

3. **`run_pipeline_variation_comparison()`**
   - Tests 6 pipeline configurations:
     - Option 1: summary → PDs (baseline)
     - Option 2: summary → verify_summary → PDs
     - Option 3: summary → PDs → verify_PDs
     - Option 4: summary → PDs → select_PD
     - Option 2+3: verify_summary → PDs → verify_PDs
     - Option 2+4: verify_summary → PDs → select_PD

**Test set requirements:**
- CSV with columns: `pmid`, `gene_ID`, `host_DB`, `pair_type`, `mean_performance`
- Balanced across different species/scenarios

### STEP_3: Batch Processing

**Purpose:** Production runs on hundreds/thousands of pairs

**Key optimization strategies:**

1. **Prompt caching** (papers with 3+ genes)
   - Paper text cached after first gene
   - Subsequent genes reuse cache (~90% cost reduction)
   - Cache TTL: 5 minutes

2. **Batch API** (papers with <3 genes)
   - Submit batch of requests
   - 50% cost reduction vs standard API
   - Async processing (polls for completion)

3. **Smart filtering:**
   - Pre-checks paper availability
   - Verifies gene/aliases in text before processing
   - Skips already-completed pairs (unless `OVERWRITE_EXISTING=True`)

**CSV requirements:**
```csv
Gene ID,PMID_Final,Database,paper_available,alias_in_text
PF3D7_1234,12345678,plasmodb,TRUE,TRUE
```

**Configuration in `global_settings_batch.py`:**
```python
MIN_GENES_FOR_CACHING = 3  # Threshold for caching strategy
USE_BATCH_FOR_LOW_DENSITY = True  # Use batch API for <3 genes
OVERWRITE_EXISTING = False  # Skip already-processed pairs
CSV_COLUMNS = {...}  # Map your CSV column names
```

**Output structure:**
- One JSON file per paper: `{pmid}.json`
- Nested by stage → gene → model
- Comprehensive step completion log CSV (for batches)

---

## Data Flow

### File Organization

**Single pair mode:**
```
out/cache/{pmid}.json  # One file per paper
```

**Testing mode:**
```
out/tests/{pmid}.json  # Same structure, different directory
```

**Batch mode:**
```
out/VPDB_UserComments_batch/{pmid}.json
out/VPDB_UserComments_batch/{input}_results.csv  # Summary
out/VPDB_UserComments_batch/{input}_step_log.csv  # Detailed completion status
```

### JSON Structure (Common Across All Modes)

```json
{
  "getGeneSummary": {
    "GENE_ID_1": {
      "gene_ID": "GENE_ID_1",
      "MODEL_NAME": {
        "model": "MODEL_NAME",
        "success": true,
        "data": {
          "GeneSummary": [...],
          "only_in_passing": false
        },
        "usage": {"input": 100, "output": 200, "total": 300},
        "seconds": 12.3
      }
    }
  },
  "generatePDs": { ... },
  "verifyPDs": { ... },
  "selectPD": { ... }
}
```

**Key fields:**
- `success` - Whether stage completed successfully
- `data` - Parsed JSON response from LLM
- `usage` - Token counts (for cost tracking)
- `seconds` - Execution time
- `only_in_passing` - Flag to skip PD generation (new in latest version)

---

## Configuration Deep Dive

### Model Selection

Each config file specifies models per stage:

```python
# Format: [provider, model_name]
summary_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_generator_llm = ["anthropic", "claude-sonnet-4-20250514"]
# ... etc
```

**Provider options:**
- `"anthropic"` - Direct Anthropic API
- `"openai"` - Direct OpenAI API
- `"openrouter"` - OpenRouter (access to msny models)

### Prompt Parameters

**Global constants:**
```python
N_QUOTES = 2  # Quotes per summary bullet point
N_PDs = 3     # Number of PD candidates to generate
model_temp = 0  # Temperature (0 for consistency)
max_tokens = 20000  # Max output tokens
```

**Batch-specific:**
```python
MIN_GENES_FOR_CACHING = 3  # Caching threshold
USE_BATCH_FOR_LOW_DENSITY = True  # Use batch API
OVERWRITE_EXISTING = False  # Reprocess completed pairs
```

### Example PDs

For `selectPD` stage (few-shot learning):
```
curated_data/example_PDs_2025_07_10.txt
```
Contains 200 hand-curated product descriptions (10-50 chars) to guide style.

---

## Common Development Tasks

### Add a New Pipeline Stage

1. **Add prompt to `prompts.py`:**
```python
global_prompts_and_schema["myNewStage"] = {
    "SystemPrompt": "...",
    "UserPrompts": ["..."],
    "ValidationSchema": {...}
}
```

2. **Create function in `pipeline_steps.py`:**
```python
def myNewStage(input_data, key="myNewStage"):
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    replacements = {"INPUT": input_data, "JSON_SCHEMA": JSON_SCHEMA}
    
    system = get_prompt_and_replace(key, replacements, "SystemPrompt")
    user = get_prompt_and_replace(key, replacements, "UserPrompts")
    
    result, usage, secs = call_prompt(
        provider=my_stage_llm[0],
        model=my_stage_llm[1],
        system_prompt=system,
        user_prompts=user,
        prefill_text="{"
    )
    
    parsed = extract_json(result)
    return parsed, usage, secs
```

3. **Integrate into workflow in `STEP_1_single_pair_processing.py`:**
```python
# After existing stage...
key = "myNewStage"
result_data = load_status(pubmed_id, OUT_DIR, key, gene_id, my_stage_llm[1])

if check_if_ok(result_data):
    print(f"✔️ Found existing result")
    my_result = result_data["data"]
else:
    print(f"Running myNewStage...")
    my_result, usage, secs = myNewStage(previous_stage_output)
    if save:
        save_status(pubmed_id, OUT_DIR, key, gene_id, 
                   my_stage_llm[1], my_result, True, usage, secs)
```

### Test a New Model

1. **Add to `MODELS_TO_TEST` in `global_settings_testing.py`:**
```python
MODELS_TO_TEST = [
    ["openrouter", "anthropic/claude-3.5-sonnet", "Sonnet3.5"],
    ["openrouter", "your/new-model", "YourModel"],  # Add here
]
```

2. **Run comparison:**
```bash
python main.py test --test model_comparison --yes
```

3. **Analyse results:**
```
out/tests/{pmid}.json  # Check each model's outputs
# NB - analysis complete, integration of the analysis workflows is under dev for this repo
```

### Modify Prompt Templates

Edit `prompts.py`:
```python
global_prompts_and_schema["getGeneSummary"]["SystemPrompt"] = """
You are an expert biocurator...
[Your new instructions]
Use this JSON schema: [JSON_SCHEMA]
"""
```

**Placeholder syntax:**
- `[PLACEHOLDER]` - Replaced by `get_prompt_and_replace()`
- Common placeholders: `[GENE]`, `[PAPER_TEXT]`, `[JSON_SCHEMA]`, `[N_QUOTES]`, `[SUMMARY]`, `[PDs]`

### Change Output Directories

In respective config file:
```python
OUT_DIR = Path("./out/my_custom_dir")
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Error Handling

### Common Issues

**1. Paper not available:**
```
Error: Paper {pmid} not available in PMC Open Access subset
```
→ Only PMC open-access papers supported. Check paper availability first.

**2. Gene not in text:**
```
⚠️ Gene/aliases not found in text (skipped)
```
→ Pre-check with `check_gene_in_text()` before processing.

**3. JSON parsing failures:**
```
⚠️ Could not parse JSON – saving raw text.
```
→ Auto-retry with formatter model (up to `max_retry` times)

**4. API rate limits:**
```
Error: Rate limit exceeded
```
→ Built-in retry logic with exponential backoff

### Debugging Tips

1. **Enable verbose output:**
```python
# Uncomment print statements in pipeline_steps.py
# print(f"system prompt fetched: \n{system}")
# print(f"LLM_summary: {LLM_summary}")
```

2. **Check saved status:**
```python
from STEP_1_single_pair_processing import load_status

data = load_status(pmid, OUT_DIR, "getGeneSummary", gene_id, model)
if data:
    print(data["data"])  # Inspect actual LLM output
    print(data["usage"])  # Check token counts
```

3. **Test single stage in isolation:**
```python
from pipeline.pipeline_steps import getGeneSummary
from pipeline.pubmed_helpers import get_paper_text

paper = get_paper_text("12345678")
result, usage, secs = getGeneSummary(paper, "GENE_ID (aka ALIAS)")
print(result)
```

---


## Cost Reduction Strategies

1. **Prompt caching (Anthropic only):**
   - Enabled automatically in batch mode for 3+ genes per paper
   - Cache TTL: 5 minutes
   - Savings: ~90% for cached paper text

2. **Batch API (Anthropic only):**
   - Enabled for low-density papers (<3 genes)
   - 50% cost reduction
   - Async processing (waits for completion)

3. **Result caching:**
   - All modes cache completed stages
   - Set `OVERWRITE_EXISTING=False` to skip reprocessing

4. **Early exit for "in passing" genes:**
   - Check `only_in_passing` flag after summary
   - Skip expensive PD generation stages

5. **Pipeline variations tested and minimal optimum performing pipeline selected**
   - Consider setting certain pipeline steps to False, but keep in mind summary generation is essential and biggest expense.
---


## Environment Variables

Required in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

Optional:
```bash
HTTP_TIMEOUT=180  # Seconds for API requests
```

---

## Dependencies

Core:
- `anthropic` - Anthropic API client
- `requests` - HTTP requests (PubMed, VEuPathDB)
- `pandas` - CSV processing
- `python-dotenv` - Environment variable management

Full list: `requirements.txt`

---

## FAQ

**Q: How do I process only the summary stage?**
A: Set `generate_pd=False`:
```python
run_pipeline(pmid, gene_id, host_db, generate_pd=False)
```

**Q: What if my CSV has different column names?**
A: Update `CSV_COLUMNS` in `global_settings_batch.py`

**Q: How do I increase the number of PDs brainstormed?**
A: Change `N_PDs` in the config file for your mode

---


