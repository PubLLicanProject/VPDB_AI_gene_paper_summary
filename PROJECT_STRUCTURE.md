# Project Structure Overview

## Repository Organization

This document provides a comprehensive overview of the project structure

## Directory Tree

```
gene-pd-pipeline/
│
├── README.md                           # Project overview and quick start
├── .gitignore                          # Git ignore rules
├── .env.example                        # API key template
├── requirements.txt                    # Python dependencies
├── MIGRATION_GUIDE.md                  # Migration from original script
│
├── config/                             # Configuration module
│   ├── __init__.py                     # Config exports
│   ├── models.py                       # Model configurations and parameters
│   └── prompts.py                      # Prompt templates (~1000 lines)
│
├── gene_pd_pipeline/                   # Main package
│   ├── __init__.py                     # Package initialization
│   ├── llm_clients.py                  # LLM API client functions
│   ├── pubmed.py                       # PubMed retrieval and parsing
│   ├── gene_data.py                    # VEuPathDB gene synonym retrieval
│   ├── workflow.py                     # Pipeline workflow functions
│   ├── utils.py                        # Helper utilities and I/O
│   └── testing.py                      # Testing frameworks
│
├── docs/                               # Documentation
│   ├── USAGE.md                        # Comprehensive usage guide
│   └── SETUP.md                        # Setup and deployment guide
│
├── examples/                           # Example scripts
│   └── run_pipeline.py                 # Example pipeline execution
│
├── tests/                              # Testing scripts
│   └── run_tests.py                    # Interactive test runner
│
├── scripts/                            # Utility scripts
│   └── complete_extraction.py          # Code extraction utility
│
├── curated_data/                       # Data files
│   ├── 2026_01_12_subset_balanced_by_species.csv  # Test set
│   └── example_PDs_2025_07_10.txt      # Example product descriptions
│
├── out/                                # Output directory
│   └── debug/                          # Pipeline results (JSON files)
│
└── original_script.py                  # Original monolithic script (reference)
```

## Module Breakdown

### config/

Configuration and constants used throughout the pipeline.

**models.py** 
- Model configurations (SUMMARY_LLM, PD_GENERATOR_LLM, etc.)
- Pipeline parameters (N_PDS, N_QUOTES, MAX_RETRY)
- Model parameters (MAX_TOKENS, MODEL_TEMP, REASONING_EFFORT)
- File paths (OUT_DIR, TEST_SET_PATH, EXAMPLE_PDS_PATH)
- Helper function: `load_example_pds()`

**prompts.py** (~1000 lines)
- `global_prompts_and_schema` dictionary containing:
  - System prompts for each pipeline stage
  - User prompt templates
  - JSON validation schemas

**__init__.py**
- Exports all config variables for easy import

### summary_pd_pipeline/

Core pipeline functionality.

**llm_clients.py** 
- `call_prompt()` - Main function to call any LLM provider
- `_extract_usage()` - Parse token usage from responses
- `_to_responses_input()` - Convert messages to Responses API format
- `_responses_supports_temperature()` - Check temperature support
- `_get_http_timeout()` - Get HTTP timeout setting
- `_is_openai_responses_model()` - Detect Responses API models
- `_openai_responses_create()` - Call OpenAI Responses API
- Supports: Anthropic (Claude), OpenAI (GPT + Responses), OpenRouter

**pubmed.py** 
- `get_pubmed_json()` - Retrieve paper from PubMed by PMID
- `parse_pubmed_json()` - Extract text from PubMed JSON sections 

**gene_data.py** 
- `get_vpdb_alias()` - Query VEuPathDB for gene aliases
- `get_gene_synonyms()` - Find all synonyms for a gene
- `count_substrings()` - Count gene mentions in paper
- `gene_to_prompt()` - Format gene info for prompts

**workflow.py** 
- `getGeneSummary()` - Stage 1: Extract gene-specific findings
- `verifyGeneSummary()` - Optional: Verify summary quality
- `collect_bullets()` - Merge verified bullet points
- `generatePDs()` - Stage 2: Brainstorm product descriptions
- `verifyPDs()` - Stage 3: Verify and select PD based on set of rules
- `selectPD()` - Optional stage 3 replacement: Select single best PD base don set of examples
- `bullets_to_paragraph()` - Convert bullets to paragraph
- `process_paper_gene_pair()` - Complete pipeline execution

**utils.py** 
- `extract_json()` - Extract JSON from LLM response
- `save_status()` - Save pipeline stage results
- `load_status()` - Load existing results
- `check_if_ok()` - Validate loaded results comply with schema
- `get_prompt_and_replace()` - Build prompts with replacements
- `_verified_to_select_candidates()` - Format verified PDs
- `_safe_get_pubmed_json()` - Safe PubMed retrieval with caching
Also beta functions for literature-wide summary generation:
- `_collect_per_paper_summary_for_model()` - Collect model summaries
- `_scan_pmids_for_gene_by_model()` - Find PMIDs for gene

**testing.py** 
- `load_test_set()` - Load paper-gene pairs from CSV
- `run_model_comparison_on_test_set()` - Compare multiple models
- `run_without_summary_comparison()` - Test without summary stage
- `run_pipeline_variation_comparison()` - Test different pipelines
- `run_option3_model_comparison()` - Run winning pipeline variation on multiple models

**__init__.py**
- Package initialisation
- Version info

### examples/

Practical usage examples.

**run_pipeline.py**
- Complete working example
- Demonstrates all pipeline stages
- Shows result handling
- Can be run standalone or imported

### tests/

Testing infrastructure.

**run_tests.py**
- Interactive test menu
- Calls different testing functions
- User-friendly CLI interface

### scripts/

Utility scripts for development, ignored

### Adding a New Model

1. Add model to `config/models.py`:
   ```python
   MODELS_TO_TEST.append(["provider", "model-name", "DisplayName"])
   ```

2. Test with:
   ```bash
   python tests/run_tests.py
   # Select option multi-model comparison
   ```

### Adding a New Pipeline Stage

1. Create function in `gene_pd_pipeline/workflow.py`
2. Add corresponding prompt to `config/prompts.py`
3. Update example in `examples/run_pipeline.py`
4. Add to test suite in `gene_pd_pipeline/testing.py`

### Modifying Prompts

1. Edit `config/prompts.py`
2. Find the appropriate key in `global_prompts_and_schema`
3. Modify `SystemPrompt`, `UserPrompts`, and/or `ValidationSchema`
4. Test with small dataset first!

## Note to Self

1. **Configuration**: Keep all constants in `config/`
2. **Documentation**: Update docs when adding features
3. **API Keys**: Never commit `.env` file :D 
4. **Results**: Back up `out/` directory regularly on DropBox

## Future Additions
- Batch processing script for large datasets - cost-optimised caching. Script is ready but needs merge to this repo
- Scoring pipeline using NLP metrics - needs merge with repo