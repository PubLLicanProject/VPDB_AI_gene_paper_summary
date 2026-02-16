"""
Model configurations and parameters for the PD pipeline.
"""

from pathlib import Path

# Directory Configuration
OUT_DIR = Path("./out/debug")  # Status files for each paper in .json format


# Pipeline parameters
N_QUOTES = 2  # How many quotes per claim to be extracted
N_PDS = 3  # Up to how many suggested product descriptions to brainstorm before selection

# Models selected for each workflow stage
# Sonnet 4 performed best
SUMMARY_LLM = ["anthropic", "claude-sonnet-4-20250514"]
SUMMARY_QC_LLM = ["anthropic", "claude-sonnet-4-20250514"]
PD_GENERATOR_LLM = ["anthropic", "claude-sonnet-4-20250514"]
PD_PICKER_LLM = ["anthropic", "claude-sonnet-4-20250514"]
PD_QC_LLM = ["anthropic", "claude-sonnet-4-20250514"]

# Additional model to force correct JSON schema if original LLM fails
# Attempts this up to MAX_RETRY times
FORMATTER_LLM = ["anthropic", "claude-sonnet-4-20250514"]
MAX_RETRY = 3

# Model parameters
REASONING_EFFORT = "medium"  # options: "minimal", "low", "medium", "high"
VERBOSITY = "low"  # options: "low", "medium", "high"
RESPONSES_MAX_OUTPUT_TOKENS = 8000
MAX_TOKENS = 20000  # Default max tokens for Claude
MODEL_TEMP = 0  # Temperature: 0 to maximise consistency
# Note: Temperature can only be set if thinking is not enabled

# Example gene (for testing)
EXAMPLE_GENE = "PF3D7_0810800"
# Gene ID format: aliases will be added as a string in prompts
# Example: f"{GENE}, also known as {ALIASES}" # ALIASES = "Alias1 or Alias 2 or Alias3..."

# Data paths
TEST_SET_PATH = "./curated_data/2026_01_12_subset_balanced_by_species.csv"

# NB: example PDs for pipeline variant that selects PD based on 200 PDs. Not needed for final rule-based pipeline.
EXAMPLE_PDS_PATH = "./curated_data/example_PDs_2025_07_10.txt"

def load_example_pds(path: str = EXAMPLE_PDS_PATH) -> str:
    """
    Load example product descriptions from file.

    """
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


# Models to test in pipeline comparisons
MODELS_TO_TEST = [
    ["openrouter", "anthropic/claude-sonnet-4.5", "Sonnet_4.5"],
    ["openrouter", "google/gemini-3-pro-preview", "Gemini_3_Pro"],
    ["openrouter", "anthropic/claude-opus-4.5", "Opus_4.5"],
    ["openrouter", "deepseek/deepseek-v3.2", "DeepSeek_3.2"],
    ["openrouter", "meta-llama/llama-4-maverick", "Llama4_Maverick"],
    ["openrouter", "qwen/qwen3-max", "QWEN_3Max"],
    ["openrouter", "z-ai/glm-4.7", "GLM_4.7"],

    # Additional models (commented out - uncomment to test)
    # ["openrouter", "google/gemini-2.5-pro", "Gemini2.5Pro"],
    # ["openrouter", "x-ai/grok-4", "Grok4"],
    # ["openrouter", "qwen/qwen3-max", "Qwen3_Max"],
    # ["openrouter", "z-ai/glm-4.6", "GLM4.6"],
    # ["openrouter", "deepseek/deepseek-v3.2-exp", "DeepSeek_v3.2_Exp"],
]
