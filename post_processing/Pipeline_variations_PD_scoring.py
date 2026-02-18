#!/usr/bin/env python
"""
Gene / Product description similarity scoring pipeline
------------------------------------------------------
* Removes gene symbols & synonyms via lookup table before scoring.
* Short descriptions (≈3–6words) → overlap + embedding scores suffice.
* Stop‑words removed, SciSpaCy lemmatisation applied.
* Python ≥3.9, scikit‑learn ≥1.3
"""
print("Setting up library and package imports ...")
import gc
import re
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from gene_lookup import find

# --------------------------------------------------------------------------- #
#                             CONFIGURATION                                   #
# --------------------------------------------------------------------------- #
print("Configuring ...")
INPUT_FILE = "./out/with_vs_withou_summary_verification_species_balanced/extracted/PD_rows_all_variations.csv" #
OUTPUT_FILE = "./out/with_vs_withou_summary_verification_species_balanced/scored/PDs_all_variations_scored_new.csv"
## for getting descriptions
VEUPATH_CSV = "./veupath_df_from_training.csv"      #
GAF_CSV     = "./curated_data/GAF_species_of_interest_withPMIDs.csv"  #

GENE_ID_FIELD = "gene_ID"  # column to use for getting synonym
# FIELDS_TO_SCORE: List[str] = ["RPD", "SPD", "UPD"] # MOVED to dynamic update
TARGET_FIELD: str = "Official_Description"           # "Official_Description" or Product

DB_PATH = r"C:/Users/jtzve/Desktop/PubLLican/LLM_testbed_JT/gene_lookup/gene_lookup.db"

# parallelism – for <100 k lines, “threading” backend is fastest
N_JOBS = 8
BACKEND = "threading"

HF_MODELS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "pritamdeka/S-BioBert-snli-multinli-stsb",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------- #
#                       NLTK STOP‑WORDS  &  SciSpaCy                          #
# --------------------------------------------------------------------------- #
print("Downloading NLTK resources and building stop‑word list ...")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english")).union(
    {
        "using",
        "based",
        "via",
        "data",
        # "gene",
        # "protein",
        # "rna",
        # "dna",
    }
)

print("Loading large SciSpaCy model (en_core_sci_lg) ...")
try:
    nlp = spacy.load("en_core_sci_lg")
except OSError:
    raise RuntimeError(
        "SciSpaCy model not installed. Install with:\n"
        "  pip install scispacy==0.5.5\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
        "v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
    )
print("SciSpaCy model loaded ✔")

# --------------------------------------------------------------------------- #
#                         GENE‑SYMBOL LOOK‑UP                               #
# --------------------------------------------------------------------------- #
print("Loading gene symbol dictionary from SQLite ...")

def connect_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def fetch_synonyms_for_ids(gene_ids: List[str]) -> Dict[str, Set[str]]:
    """
    Fetch just the entries where column = 4 ("Gene Name or Symbol")
    for every row whose column = 1 ("Gene ID") matches one of the
    supplied gene_ids.

    Returns
    -------
    dict : {gene_id : {symbol1, symbol2, ...}}
    """
    if not gene_ids:
        return {}

    placeholders = ",".join("?" * len(gene_ids))

    query = f"""
        WITH target_rows AS (
            SELECT id, content AS gene_id
            FROM records
            WHERE column = 1                    -- 1 = "Gene ID"
              AND content IN ({placeholders})
        )
        SELECT r2.content  AS symbol,
               tr.gene_id  AS gene_id
        FROM target_rows tr
        JOIN records r2 ON r2.id = tr.id
        WHERE r2.column = 4                     -- 4 = "Gene Name or Symbol"
    """

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, gene_ids)
    rows = cur.fetchall()
    conn.close()

    out: Dict[str, Set[str]] = {gid: set() for gid in gene_ids}
    for symbol, gid in rows:
        out[gid].add(symbol)
    return out


# --------------------------------------------------------------------------- #
#                         TEXT PRE‑PROCESSING                                 #
# --------------------------------------------------------------------------- #
GENE_SYMBOLS: Dict[str, Set[str]] = {}   # populated after reading CSV

def strip_gene_symbols(text: str, gene_id: str) -> str:
    """
    Remove any whole‑word occurrence of a synonym belonging to gene_id.
    Case‑insensitive, keeps surrounding punctuation/spacing intact.
    """
    if not text or gene_id not in GENE_SYMBOLS:
        return text
    symbols = sorted(GENE_SYMBOLS[gene_id], key=len, reverse=True)
    if not symbols:
        return text
    # build regex:  \b(sym1|sym2|sym‑3)\b  , ignore‑case
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, symbols)) + r")\b", re.I)
    return pattern.sub("", text)

# old version missed some cases
# def preprocess_tokens(text: str) -> List[str]:
#     """lemmatise → stop‑word filter → return list of lemmas (lower‑case)."""
#     if not isinstance(text, str):
#         return []
#     doc = nlp(text.lower())
#     return [
#         tok.lemma_
#         for tok in doc
#         if tok.is_alpha and tok.text not in STOP_WORDS
#     ]
# --------------------------------------------------------------------- #
#  Greek letter normaliser ## might need expanding                      #
# --------------------------------------------------------------------- #
GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota",  "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu",    "ξ": "xi",    "ο": "omicron", "π": "pi",
    "ρ": "rho",   "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi",   "χ": "chi",   "ψ": "psi", "ω": "omega"
}

def _greek_to_latin(text: str) -> str:
    """Replace single‑character Greek symbols with their Latin names."""
    return "".join(GREEK_MAP.get(ch, ch) for ch in text)

def preprocess_tokens(text: str) -> List[str]:
    """

    1. lower‑case
    2. SciSpaCy tokenise / lemmatise
    3. split lemmas on various separators (-,.:\/...)
    4. keep fragments that contain ≥1 letter and are not stop‑words # UPDATE - some added stopwords excluded
    5. convert Greek symbols to Latin strings (β → beta)
    """
    if not isinstance(text, str):
        return []

    doc = nlp(text.lower())
    out: List[str] = []

    for tok in doc:
        if tok.text in STOP_WORDS:
            continue

        lemma = _greek_to_latin(tok.lemma_)
        # split on hyphen or slash, e.g. "inositol-4-phosphate"
        for frag in re.split(r"[-_.(),/\\]", lemma):
            frag = frag.strip()
            if frag and re.search(r"[a-z]", frag) and frag not in STOP_WORDS:
                out.append(frag)

    return out
def clean_text(text: str, gene_id: str) -> str:
    """
    1) strip gene symbols   2) lemmatise & stop‑word filter
    Returns a single space‑separated string ("" if input is NaN/None).
    """
    if not isinstance(text, str):
        return ""
    stripped = strip_gene_symbols(text, gene_id)
    return " ".join(preprocess_tokens(stripped))

# --------------------------------------------------------------------------- #
#                         SIMILARITY FUNCTIONS                                #
# --------------------------------------------------------------------------- #
# exact match
def exact_match(a: str, b: str) -> float:
    return float(a == b) if a and b else np.nan

# one word off
# def one_word_off(a: str, b: str) -> float:
#     if not a or not b:
#         return np.nan
#
#     toks_a, toks_b = a.split(), b.split()
#
#     # If length difference is more than 1, it's not a one-word-off
#     if abs(len(toks_a) - len(toks_b)) > 1:
#         return 0.0
#
#     # Try removing each word from the longer string and check for equality
#     short, long = (toks_a, toks_b) if len(toks_a) <= len(toks_b) else (toks_b, toks_a)
#     for i in range(len(long)):
#         long_minus_one = long[:i] + long[i + 1:]
#         if long_minus_one == short:
#             return 1.0
#
#     return 0.0

def one_word_off(a: str, b: str) -> float:
    if not a or not b:
        return np.nan

    toks_a = a.split()
    toks_b = b.split()

    # If length difference is more than 1, it's not a one-word-off
    if abs(len(toks_a) - len(toks_b)) > 1:
        return 0.0

    # Convert to sets for order-independent comparison
    set_a = set(toks_a)
    set_b = set(toks_b)

    # Exclude exact matches (same tokens, regardless of order)
    if set_a == set_b:
        return 0.0

    # Case 1: Same length - should differ by exactly 1 word
    if len(toks_a) == len(toks_b):
        symmetric_diff = set_a.symmetric_difference(set_b)
        # Exactly 2 words different (1 in each set) OR exact match of sets
        if len(symmetric_diff) == 2:
            return 1.0
        elif len(symmetric_diff) == 0:  # Same words, different order
            return 1.0  # or 1.0 if you want to count reorderings as one-word-off

    # Case 2: Length differs by 1 - one set should be subset of the other
    else:
        smaller_set = set_a if len(toks_a) < len(toks_b) else set_b
        larger_set = set_b if len(toks_a) < len(toks_b) else set_a

        # The smaller set should be a subset of the larger
        # AND they should differ by exactly 1 word
        if smaller_set.issubset(larger_set) and len(larger_set - smaller_set) == 1:
            return 1.0

    return 0.0


# toks1 and toks2 is the same as
# toks_cand (candidate from LLMs) and toks_gold (Gold standard from curators)

def jaccard_similarity(toks1: List[str], toks2: List[str]) -> float:
    set1, set2 = set(toks1), set(toks2)
    return len(set1 & set2) / len(set1 | set2) if set1 and set2 else np.nan

def soft_jaccard(toks1: List[str], toks2: List[str]) -> float:
    if not toks1 or not toks2:
        return np.nan
    return np.mean([max(fuzz.partial_ratio(t1, t2) for t2 in toks2) / 100 for t1 in toks1])

def precision_jaccard(toks_cand, toks_gold) -> float:
    return len(set(toks_cand) & set(toks_gold)) / len(set(toks_cand)) if toks_cand else np.nan

def levenshtein_normalized(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return np.nan
    return fuzz.ratio(str1, str2) / 100.0

def tfidf_cosine(str1: str, str2: str, vect: TfidfVectorizer) -> float:
    mat = vect.transform([str1, str2])
    return cos_sim(mat[0], mat[1])[0, 0]

def embed_cosine(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(cos_sim([e1], [e2])[0, 0])

# --------------------------------------------------------------------------- #
#                    SENTENCE‑TRANSFORMER HELPER                              #
# --------------------------------------------------------------------------- #
def build_st_model(hf_id: str) -> SentenceTransformer:
    word_emb = Transformer(hf_id, max_seq_length=128)
    pooling = Pooling(word_emb.get_word_embedding_dimension(), "cls")
    return SentenceTransformer(modules=[word_emb, pooling])

print("Loading embedding models ...")
st_models = [build_st_model(m) for m in HF_MODELS]
print(f"  → {len(st_models)} models loaded")

# --------------------------------------------------------------------------- #
#                               DATA LOADING                                  #
# --------------------------------------------------------------------------- #
print("Reading input descriptions ...")

###################### HANDLE NEW CSV FORMAT ##############################
raw_df = pd.read_csv(INPUT_FILE)


# Check if variation column exists and create combined field names
if 'variation' in raw_df.columns:
    print("📊 Variation column detected - creating combined field identifiers...")

    # Create combined field: variation_PD_type
    raw_df['scoring_field'] = raw_df['variation'] + '_' + raw_df['PD_type']

    # Get unique combinations that will become columns after pivot
    FIELDS_TO_SCORE = sorted(raw_df['scoring_field'].dropna().unique().tolist())

    print(f"   Found {len(FIELDS_TO_SCORE)} unique variation-PD combinations:")
    for field in FIELDS_TO_SCORE:
        count = (raw_df['scoring_field'] == field).sum()
        print(f"     - {field}: {count} entries")

    # Update the PD_type column to use the combined field for pivoting
    raw_df['PD_type'] = raw_df['scoring_field']

else:
    print("📊 No variation column - using standard PD_type scoring...")
    # Keep original behavior
    FIELDS_TO_SCORE = ["RPD", "APD", "selected_PD", "raw_PD"]

print(f"\n🎯 Will score the following fields: {FIELDS_TO_SCORE}\n")
#####################
# No pivot needed - work with long format directly
df = raw_df.query("PD_type in @FIELDS_TO_SCORE").copy()

print(f"\n✅ Working with long format. Shape: {df.shape}")
print(f"   Each row is one PD to be scored")

# Add Official_Description for each row
official_lookup = {}

# 1️⃣  VEuPath training set – highest priority
if os.path.exists(VEUPATH_CSV):
    vdf = pd.read_csv(VEUPATH_CSV, dtype=str, usecols=["reference", "gene_ID", "product"])
    for _, r in vdf.iterrows():
        key = (str(r["reference"]).lstrip("PMID:").strip(), r["gene_ID"])
        official_lookup[key] = r["product"]

# 2️⃣  GAF fallback – only fill if key still missing
if os.path.exists(GAF_CSV):
    gdf = pd.read_csv(GAF_CSV, dtype=str, usecols=["pmid", "Gene_ID", "Name"])
    for _, r in gdf.iterrows():
        pmid = str(r["pmid"]).lstrip("PMID:").strip()
        key  = (pmid, r["Gene_ID"])
        official_lookup.setdefault(key, r["Name"])

# Add Official_Description column
df["Official_Description"] = df.apply(
    lambda row: official_lookup.get((str(row["pmid"]), row["gene_ID"]), ""),
    axis=1
)

# For scoring, we'll compare "description" column against "Official_Description"
# Update field references
TARGET_FIELD = "Official_Description"
CANDIDATE_FIELD = "description"

# Fill NaN values
df[[CANDIDATE_FIELD, TARGET_FIELD]] = df[[CANDIDATE_FIELD, TARGET_FIELD]].fillna("")


# Ensure Gene_ID is str for dict lookup
df[GENE_ID_FIELD] = df[GENE_ID_FIELD].astype(str)

# build gene‑symbol dictionary once
unique_gene_ids = df[GENE_ID_FIELD].unique().tolist()
GENE_SYMBOLS = fetch_synonyms_for_ids(unique_gene_ids)
print(f"Loaded synonyms for {len(GENE_SYMBOLS)} unique Gene_IDs")

# --------------------------------------------------------------------------- #
#                         CLEAN DESCRIPTIONS COLUMN                            #
# --------------------------------------------------------------------------- #
print("Cleaning & lemmatising descriptions ...")
df[f"Clean_{CANDIDATE_FIELD}"] = df.apply(lambda r: clean_text(r[CANDIDATE_FIELD], r[GENE_ID_FIELD]), axis=1)
df[f"Clean_{TARGET_FIELD}"] = df.apply(lambda r: clean_text(r[TARGET_FIELD], r[GENE_ID_FIELD]), axis=1)

# -------------------------TF‑IDF VECTORIZER -------------------------------- #
print("Fitting TF‑IDF vectorizer (character 3–5‑grams) ...")
tfidf_corpus = df[[f"Clean_{TARGET_FIELD}", f"Clean_{CANDIDATE_FIELD}"]].values.ravel()

tfidf_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 5),
    sublinear_tf=True, lowercase=True
).fit(tfidf_corpus)
print("TF‑IDF vectorizer fitted")

# --------------------------------------------------------------------------- #
#                                   SCORING                                   #
# --------------------------------------------------------------------------- #
def score_row(row: pd.Series) -> Dict[str, Any]:
    """Score a single candidate description against the official description"""
    out: Dict[str, Any] = {}

    # ---------- build target representations --------------------------
    raw_target  = str(row[TARGET_FIELD]).lower()
    nosym_target = strip_gene_symbols(raw_target, str(row[GENE_ID_FIELD]))
    target_clean = row[f"Clean_{TARGET_FIELD}"]
    target_tokens = target_clean.split()

    target_embs = [m.encode(nosym_target) for m in st_models]   # BERT on nosym

    # ---------- build candidate representations --------------------------
    raw_cand   = str(row[CANDIDATE_FIELD]).lower()
    nosym_cand = strip_gene_symbols(raw_cand, str(row[GENE_ID_FIELD]))
    cand_clean = row[f"Clean_{CANDIDATE_FIELD}"]
    cand_tokens = cand_clean.split()

    # handle empty strings
    if not cand_tokens:
        for m in ["jaccard","soft_jaccard","prec_jaccard",
                  "levenshtein","tfidf_cosine",
                  "embed1_cosine","embed2_cosine",
                  "Exact_raw","OneWord_raw",
                  "Exact_nosym","OneWord_nosym",
                  "Exact_clean","OneWord_clean"]:
            out[f"SemSim_{m}"] = np.nan
        return out

    # token / string metrics
    out["SemSim_jaccard"]      = jaccard_similarity(cand_tokens, target_tokens)
    out["SemSim_soft_jaccard"] = soft_jaccard(cand_tokens, target_tokens)
    out["SemSim_prec_jaccard"] = precision_jaccard(cand_tokens, target_tokens)
    out["SemSim_levenshtein"]  = levenshtein_normalized(cand_clean, target_clean)
    out["SemSim_tfidf_cosine"] = tfidf_cosine(raw_cand, raw_target, tfidf_vectorizer)

    # BERT embeddings on symbol‑free lowercase
    cand_embs = [m.encode(nosym_cand) for m in st_models]
    for idx, (ce, te) in enumerate(zip(cand_embs, target_embs), start=1):
        out[f"SemSim_embed{idx}_cosine"] = embed_cosine(ce, te)

    # simple exact / one‑word‑off flags
    out["Exact_raw"]    = exact_match(raw_cand, raw_target)
    out["OneWord_raw"]  = one_word_off(raw_cand,  raw_target)

    out["Exact_nosym"]  = exact_match(nosym_cand, nosym_target)
    out["OneWord_nosym"]= one_word_off(nosym_cand, nosym_target)

    out["Exact_clean"]  = exact_match(cand_clean, target_clean)
    out["OneWord_clean"]= one_word_off(cand_clean, target_clean)

    return out

print("Scoring descriptions row‑by‑row ...")
with Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=1) as pool:
    row_scores = pool(delayed(score_row)(row) for _, row in df.iterrows())

df_scores = pd.DataFrame(row_scores, index=df.index)
df = pd.concat([df, df_scores], axis=1)
gc.collect()

# --------------------------------------------------------------------------- #
#                                   OUTPUT                                    #
# --------------------------------------------------------------------------- #
print(f"✅  Scoring complete. Final shape: {df.shape}")
print(f"   Columns include: variation, PD_type, scoring_field, and similarity metrics")

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅  Results saved to {OUTPUT_FILE}")