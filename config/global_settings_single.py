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

# models selected for each workflow stage
# Testing Sonnet 4 as same price like Sonnet 3.7, our prev best performer
summary_llm = ["anthropic", "claude-sonnet-4-20250514"]
summary_QC_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_generator_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_picker_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_QC_llm = ["anthropic", "claude-sonnet-4-20250514"]
# additional model to force correct JSON schema if original LLM fails; attempt this up to max_retry times
formatter_llm = ["anthropic", "claude-sonnet-4-20250514"]
max_retry = 3
# claude-3-haiku-20240307 for slightly cheaper per retry attempt
# however, sometimes fails completely - cost of reliability + speed is worth it for 3.5


REASONING_EFFORT = "medium"   # options: "minimal", "low", "medium", "high"
VERBOSITY = "low"              # options: "low", "medium", "high"
RESPONSES_MAX_OUTPUT_TOKENS = 8000  # hard cap; adjusted to keep outputs shorter

#### PARAMETERS ####
# model params - keep as standard accross all models
max_tokens = 20000  # TODO: default in Claude tester online, do we want to change?
model_temp = 0 # NB: can only be set if thinking is not enabled. we use 0 to maximise consistency; # TODO: EXPERIMENT on small subset for creative tasks -  what happens if we enable thinking; e.g. PD gen and picker might need some creativity?
# params for use in prompts
GENE = "PF3D7_0810800" # example
# the gene ID; its aliases will be added as a string to supply in the prompts;
# here is an example, we retrieve these from VEuPathDB: f"{GENE}, also known as {ALIASES}" # ALIASES =  PPPK-DHPS, or PF08_0095
N_QUOTES = 2 # how many quotes per claim to be extracted
N_PDs = 3 # up to how many suggested product descriptions to be assigned before selection
# initiate schema as empty string, each workflow stage has different schema
JSON_SCHEMA = "" # schema as string to append to the system prompt; stored in the prompts and schema dict and will be updated for each workflow step as needed.

# list of 200 example product descriptions picked at random; all >10 characters and < 50
with open('./curated_data/example_PDs_2025_07_10.txt', 'r', encoding='utf-8') as file:
    PD_EXAMPLES = file.read()