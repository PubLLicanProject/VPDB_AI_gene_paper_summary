# Scripts behind the manuscript

This file maps the scripts in this repository to the results they produced.
Figure and table numbers refer to the manuscript and its supplement. Paths are
relative to the repository root.

Scripts not listed here were used during development and are documented in
`README.md` and `PROJECT_STRUCTURE.md`.

---

## The pipeline

`pipeline/` holds the stages every entry point calls. The three stages the paper
describes are `getGeneSummary` (Stage 1), `generatePDs` (Stage 2) and
`verifyPDs` (Stage 3). Prompts and output schemas are in `pipeline/prompts.py`;
the eleven-level specificity rubric used throughout the analysis is the
`<HIERARCHY>` block in that file.

`config/global_settings_*.py` select the model, provider and output directory per
entry point. API keys are read from `.env`, which is **not distributed**; see
`.env.example` for the variable names.

Every run writes one status JSON per publication, `out/<run>/<PMID>.json`, keyed
by stage, then gene, then model, each leaf carrying the parsed output, token
usage and timing. These files are the unit of record: the analysis repository
reads them rather than any console output.

---

## Entry points

### `STEP_1_single_pair_processing.py`

One gene-paper pair through the pipeline, and the place to start when reading
the code. Also available as:

```
python main.py single --pmid 24877144 --gene_id PF3D7_1329700 --host_db PlasmoDB
```

### `STEP_2_running_tests.py`

Development-set comparisons on the 96 curated gene-paper pairs and their
balanced subsets in `curated_data/`. Three modes, selected with `--test`:

| Mode | Compares | Produces |
|---|---|---|
| `variation_comparison` | the seven pipeline configurations | **Figure 2A-C** |
| `no_summary` | paper straight to PD, with no summary stage | the no-summary arm of the same comparison |
| `model_comparison` | commercial and locally hosted models | **Supplementary Figure 1** |

### `STEP_3_batch_processing.py`

The batch engine, used for the large-scale validation against curator-written
descriptions. Resumable: it checks whether each gene-paper pair already has
outputs and skips it, so a run can be stopped and restarted. Also available as
`python main.py batch --csv <pairs.csv>`.

Produced the **10,196 scorable descriptions** behind **Figure 3** and **Table 3**.

### `run_user_comment_pd_batch.py`

A wrapper around `STEP_3` for the VEuPathDB user-comment application. It runs
`getGeneSummary` → `generatePDs` → `verifyPDs` over the gene-paper pairs in
`curated_data/user_comments_precheck.csv`, threaded over publications and
resumable, writing to `out/user_comment_evaluation/`.

Supplementary processing is **off by default** and enabled with `--supplements`.
Without it, pairs unlocked only by supplementary material produce nothing,
because the gene is absent from the main text.

```
python run_user_comment_pd_batch.py [precheck.csv] [workers] [--supplements]
```

Produced the **5,722 gene-paper pairs** behind **Figure 4A** and **Table S10**.

---

## Supplementary-material processing

Behind **Supplementary Methods S11** and **Supplementary Figure 3**, including
the 1,425 additional analysable pairs and the change in the fungal Gold rate
from 22.3% to 49.4% reported in the Results.

| Script | Role |
|---|---|
| `pipeline/supplementary_helpers.py` | fetch and parse supplementary files, count gene mentions |
| `preprocess_with_supplementary.py` | re-run the pre-check with supplementary text included |
| `build_supplement_rerun.py` | assemble the re-run input set |
| `run_supplement_rerun.py` | execute the re-run |
| `reverify_with_supplement.py` | re-run `verifyPDs` with supplementary evidence |
| `report_supplementary_impact.py` | tabulate the with-minus-without comparison |
| `rescan_legends_ocr.py` | OCR pass over figure legends in supplementary PDFs |

## Pre-processing

`preprocess_pairs.py`, `preprocess_user_comments.py` and
`pre_processing/get_user_comments_by_id.py` build the candidate pair sets and run
the gene-alias pre-check that **Figure 1D** describes as prefiltered gene-paper
pairs. `check_which_need_processing.R` reports what remains to be run.

---

## Related repositories

| Repository | Role |
|---|---|
| `PubLLicanProject/NLP_scoring` | scores generated descriptions against curated ones (Levenshtein, BioMedBERT, Jaccard) and the random-pairing null |
| `JordanTzvetkov/SummaryAndPDtool_manuscript` | analysis, figures and tables for the manuscript |

DOIs for all three repositories are listed in the manuscript's Data Availability statement.
