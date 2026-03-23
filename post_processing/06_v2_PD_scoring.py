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
INPUT_FILE = "./out/large_scale_PD_results/Tianhui_results_final.csv" #
OUTPUT_FILE = "./out/large_scale_PD_results/Tianhui_results_final_scored.csv"
## for getting descriptions
VEUPATH_CSV = "./curated_data/VPDB_PDs_withPMID.csv"      #
GAF_CSV     = "./curated_data/GAF_species_of_interest_withPMIDs.csv"  #

GENE_ID_FIELD = "gene_ID"  # column to use for getting synonym
FIELDS_TO_SCORE: List[str] =  ["RPD"] # ["RPD", "SPD", "UPD"]
TARGET_FIELD: str = "Official_Description"  # "Official_Description" or Product

DB_PATH = r"C:/Users/jtzve/Desktop/PubLLican/LLM_testbed_JT/gene_lookup/gene_lookup.db"

# parallelism – for <100 k lines, “threading” backend is fastest
N_JOBS = 8
BACKEND = "threading"

HF_MODELS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    # "pritamdeka/S-BioBert-snli-multinli-stsb",
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
def one_word_off(a: str, b: str) -> float:
    if not a or not b:
        return np.nan

    # If it's an exact match, return 0 (not one-word-off)
    if a == b:
        return 0.0

    toks_a = a.split()
    toks_b = b.split()

    # If length difference is more than 1, it's not a one-word-off
    if abs(len(toks_a) - len(toks_b)) > 1:
        return 0.0

    # Convert to sets for order-independent comparison
    set_a = set(toks_a)
    set_b = set(toks_b)

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
def _pivot_to_wide(group):
    """Pivot PD_type rows into separate columns for scoring pipeline"""
    apds = group.loc[group["PD_type"] == "APD", "description"].tolist()
    sel  = group.loc[group["PD_type"] == "selected_PD", "description"]
    raw  = group.loc[group["PD_type"] == "raw_PD", "description"]
    rpd  = group.loc[group["PD_type"] == "RPD", "description"]

    return pd.Series({
        "pmid"   : group["pmid"].iloc[0],
        "gene_ID": group["gene_ID"].iloc[0],
        "model"  : group["model"].iloc[0],
        "RPD"    : rpd.iloc[0] if not rpd.empty else "",
        "APD1"   : apds[0] if len(apds) > 0 else "",
        "APD2"   : apds[1] if len(apds) > 1 else "",
        "SPD"    : sel.iloc[0] if not sel.empty else "",
        "UPD"    : raw.iloc[0] if not raw.empty else "",
    })

raw_df = pd.read_csv(INPUT_FILE)

raw_df = raw_df.rename(columns={
    "gene_ID": "gene_ID",
    "PD_type": "PD_Type",
    "LLM_Description": "description"
})
#####################
# Then ensure the groupby uses all three keys:
df = (
    raw_df
      .query("PD_Type in ['RPD', 'APD', 'selected_PD', 'raw_PD']")
      .pivot_table(
          index=["pmid", "gene_ID", "model"],
          columns="PD_Type",
          values="description",
          aggfunc="first"
      )
      .reset_index()
)
# --- NEW robust two-tier mapping of original descriptions
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

# add column
df["Official_Description"] = df.apply(
    lambda row: official_lookup.get((str(row["pmid"]), row["gene_ID"]), ""),
    axis=1
)


#
# df = pd.read_csv(INPUT_FILE)
# handle NaN values
all_text_cols = FIELDS_TO_SCORE + [TARGET_FIELD]
df[all_text_cols] = df[all_text_cols].fillna("")

df.columns
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
for col in FIELDS_TO_SCORE + [TARGET_FIELD]:
    df[f"Clean_{col}"] = df.apply(lambda r: clean_text(r[col], r[GENE_ID_FIELD]), axis=1)

# -------------------------TF‑IDF VECTORIZER -------------------------------- #
print("Fitting TF‑IDF vectorizer (character 3–5‑grams) ...")
tfidf_corpus = df[[f"Clean_{TARGET_FIELD}"] +
                  [f"Clean_{c}" for c in FIELDS_TO_SCORE]].values.ravel()

tfidf_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 5),
    sublinear_tf=True, lowercase=True
).fit(tfidf_corpus)
print("TF‑IDF vectorizer fitted")


# --------------------------------------------------------------------------- #
#                    PRECOMPUTE SYMBOL-FREE TEXT + EMBEDDINGS                 #
# --------------------------------------------------------------------------- #
# NEW: stop bug when scoring lots of descriptions in parallel
print("Preparing symbol-free text for embedding lookups ...")

df[f"Raw_{TARGET_FIELD}"] = df[TARGET_FIELD].astype(str).str.lower()
df[f"NoSym_{TARGET_FIELD}"] = df.apply(
    lambda r: strip_gene_symbols(df.at[r.name, TARGET_FIELD].lower(), str(r[GENE_ID_FIELD])),
    axis=1
)

for field in FIELDS_TO_SCORE:
    df[f"Raw_{field}"] = df[field].astype(str).str.lower()
    df[f"NoSym_{field}"] = df.apply(
        lambda r: strip_gene_symbols(df.at[r.name, field].lower(), str(r[GENE_ID_FIELD])),
        axis=1
    )

# Unique texts to encode once
texts_to_encode = df[f"NoSym_{TARGET_FIELD}"].fillna("").tolist()
for field in FIELDS_TO_SCORE:
    texts_to_encode.extend(df[f"NoSym_{field}"].fillna("").tolist())

# De-duplicate while preserving order
texts_to_encode = list(dict.fromkeys(texts_to_encode))

# Optional: skip empty string from model encoding
texts_for_model = [t for t in texts_to_encode if t.strip()]

print(f"Precomputing embeddings for {len(texts_for_model):,} unique texts ...")

embedding_maps = []
for model_idx, model in enumerate(st_models, start=1):
    print(f"  Encoding with model {model_idx}/{len(st_models)} ...")
    embs = model.encode(
        texts_for_model,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    emb_map = {txt: emb for txt, emb in zip(texts_for_model, embs)}
    emb_map[""] = None
    embedding_maps.append(emb_map)

print("Embedding cache ready")

# --------------------------------------------------------------------------- #
#                                   SCORING                                   #
# --------------------------------------------------------------------------- #
def score_row(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # ---------- build target representations --------------------------
    raw_target  = str(row[TARGET_FIELD]).lower()
    nosym_target = strip_gene_symbols(raw_target, str(row[GENE_ID_FIELD]))
    target_clean = row[f"Clean_{TARGET_FIELD}"]           # lemma string
    target_tokens = target_clean.split()

    # target_embs = [m.encode(nosym_target) for m in st_models]   # BERT on nosym
    target_embs = [
        emb_map.get(nosym_target, None)
        for emb_map in embedding_maps
    ]
    # ---------- iterate over candidate fields ------------------------
    for field in FIELDS_TO_SCORE:
        raw_cand   = str(row[field]).lower()
        nosym_cand = strip_gene_symbols(raw_cand, str(row[GENE_ID_FIELD]))
        cand_clean = row[f"Clean_{field}"]
        cand_tokens = cand_clean.split()

        # handle empty strings
        if not cand_tokens:
            for m in ["jaccard","soft_jaccard","prec_jaccard",
                      "levenshtein","tfidf_cosine",
                      "embed1_cosine","embed2_cosine",
                      "Exact_raw","OneWord_raw",
                      "Exact_nosym","OneWord_nosym",
                      "Exact_clean","OneWord_clean"]:
                out[f"SemSim_{field}_{m}"] = np.nan
            continue

        # token / string metrics
        out[f"SemSim_{field}_jaccard"]      = jaccard_similarity(cand_tokens, target_tokens)
        out[f"SemSim_{field}_soft_jaccard"] = soft_jaccard(cand_tokens, target_tokens)
        out[f"SemSim_{field}_prec_jaccard"] = precision_jaccard(cand_tokens, target_tokens)
        out[f"SemSim_{field}_levenshtein"]  = levenshtein_normalized(cand_clean, target_clean)
        out[f"SemSim_{field}_tfidf_cosine"] = tfidf_cosine(raw_cand, raw_target, tfidf_vectorizer)

        # BERT embeddings on symbol‑free lowercase
        # cand_embs = [m.encode(nosym_cand) for m in st_models]
        cand_embs = [
            emb_map.get(nosym_cand, None)
            for emb_map in embedding_maps
        ]
        for idx, (ce, te) in enumerate(zip(cand_embs, target_embs), start=1):
            out[f"SemSim_{field}_embed{idx}_cosine"] = embed_cosine(ce, te)

        # simple exact / one‑word‑off flags
        out[f"{field}_Exact_raw"]    = exact_match(raw_cand, raw_target)
        out[f"{field}_OneWord_raw"]  = one_word_off(raw_cand,  raw_target)

        out[f"{field}_Exact_nosym"]  = exact_match(nosym_cand, nosym_target)
        out[f"{field}_OneWord_nosym"]= one_word_off(nosym_cand, nosym_target)

        out[f"{field}_Exact_clean"]  = exact_match(cand_clean, target_clean)
        out[f"{field}_OneWord_clean"]= one_word_off(cand_clean, target_clean)

    return out

# syns = fetch_synonyms_for_ids("PF3D7_1129600")
# print(syns)

print("Scoring descriptions row‑by‑row ...")
with Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=1) as pool:
    row_scores = pool(delayed(score_row)(row) for _, row in df.iterrows())

df_scores = pd.DataFrame(row_scores, index=df.index)
df = pd.concat([df, df_scores], axis=1)
gc.collect()

# --------------------------------------------------------------------------- #
#                            RANK AGGREGATION                                 #
# --------------------------------------------------------------------------- #
metrics = [
    "embed1_cosine",
    # "embed2_cosine",
    # "soft_jaccard",
    "jaccard",
    # "prec_jaccard",
    # "tfidf_cosine",
    "levenshtein",
]
WEIGHTS = {m: 1 for m in metrics}
# WEIGHTS["prec_jaccard"] = 2       # extra penalty for verbosity

for field in FIELDS_TO_SCORE:
    rank_cols = []
    for m in metrics:
        col = f"SemSim_{field}_{m}"
        df[f"Rank_{field}_{m}"] = df[col].rank(ascending=False, na_option="bottom")
        rank_cols.append(f"Rank_{field}_{m}")

    total_w = sum(WEIGHTS.values())
    df[f"RankAgg_{field}"] = (
        df[rank_cols]
        .apply(lambda r: sum(r[col] * WEIGHTS[m] for col, m in zip(rank_cols, metrics)) / total_w, axis=1)
    )

df["RankAgg_Overall"] = df[[f"RankAgg_{f}" for f in FIELDS_TO_SCORE]].mean(axis=1)

# --------------------------------------------------------------------------- #
#                             VISUAL SUMMARY                                  #
# --------------------------------------------------------------------------- #
# print("Generating visual summary ...")
# plt.figure(figsize=(8, 5))
# plot_df = (
#     df[[f"RankAgg_{f}" for f in FIELDS_TO_SCORE]]
#     .melt(value_name="Rank", var_name="Source")
#     .assign(Source=lambda d: d["Source"].str.replace("RankAgg_", "", regex=False))
# )
# sns.boxplot(data=plot_df, x="Source", y="Rank")
# plt.title("Average rank per input source (lower is better)")
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------------- #
#                                   OUTPUT                                    #
# --------------------------------------------------------------------------- #
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅  Pipeline complete – results saved to {OUTPUT_FILE}")