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


# Load environment variables from .env file
load_dotenv() #- include API KEys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


#### DIRS ####
OUT_DIR = Path("./out/tests/") # status flies for each paper in .json that store LLM outputs of each processing step
TEST_SET_PATH = "./curated_data/2026_01_12_subset_balanced_by_species.csv"

#### LLMs ####

# Updated model configurations - these will be tested on the pipleine we picked for the tool.
MODELS_TO_TEST = [
    # PAID MODELS (run first )
    ["openrouter", "anthropic/claude-sonnet-4.5", "Sonnet4.5"],
    # ["openrouter", "google/gemini-2.5-pro", "Gemini2.5Pro"],
    # ["openrouter", "x-ai/grok-4", "Grok4"],
    # ["openrouter", "qwen/qwen3-max", "Qwen3_Max"],
    # ["openrouter", "z-ai/glm-4.6", "GLM4.6"],
    # ["openrouter", "deepseek/deepseek-v3.2-exp", "DeepSeek_v3.2_Exp"],

    # FREE MODELS (may have rate limits/queue times)
    # ["openrouter", "meta-llama/llama-4-scout:free", "Llama4_Scout"],
    # ["openrouter", "meta-llama/llama-4-maverick:free", "Llama4_Maverick"],
    # ["openrouter", "deepseek/deepseek-chat-v3.1:free", "DeepSeek_v3.1"],
]


# additional model to force correct JSON schema if original LLM fails; attempt this up to max_retry times
formatter_llm = ["anthropic", "claude-sonnet-4-5"]
max_retry = 5


REASONING_EFFORT = "medium"   # options: "minimal", "low", "medium", "high"
VERBOSITY = "low"              # options: "low", "medium", "high"
RESPONSES_MAX_OUTPUT_TOKENS = 6000  # hard cap; adjusted to keep outputs shorter


#### PARAMETERS ####
# model params - keep as standard accross all models
max_tokens = 20000  # TODO: default in Claude tester online, do we want to change?
model_temp = 0 # NB: can only be set if thinking is not enabled. we use 0 to maximise consistency; # TODO: EXPERIMENT on small subset for creative tasks -  what happens if we enable thinking; e.g. PD gen and picker might need some creativity?
# params for use in prompts
GENE = "PF3D7_0810800" # example
# the gene ID; its aliases will be added as a string to supply in the prompts;
# here is an example, we retrieve these from VEuPathDB: f"{GENE}, also known as {ALIASES}" # ALIASES =  PPPK-DHPS, or PF08_0095
N_QUOTES = 2 # how many quotes per claim to be extracted
N_PDs = 5 # up to how many suggested product descriptions to be assigned before selection
# initiate schema as empty string, each workflow stage has different schema
JSON_SCHEMA = "" # schema as string to append to the system prompt; stored in the prompts and schema dict and will be updated for each workflow step as needed.

# list of 200 example product descriptions picked at random; all >10 characters and < 50
with open('./curated_data/example_PDs_2025_07_10.txt', 'r', encoding='utf-8') as file:
    PD_EXAMPLES = file.read()