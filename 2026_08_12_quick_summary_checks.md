# Quick summary checks — 2026-08-12

Reference numbers from the user-comment PD-generation work and related quick checks run this session.
(Branch `JT`. All PD generation uses Sonnet 4.5 via OpenRouter. Pipeline API key is on a separate account.)

---

## 1. Priority PD-generation batch (this session)

Input: `curated_data/priority_precheck.csv` candidates → `run_user_comment_pd_batch.py ... --supplements`
into `out/user_comment_evaluation/`. Priority target file:
`C:\Users\jtzve\Desktop\PD_PIPELINE\curated_data\batch_needs_description_outstanding.tsv`.

- Candidates run: **2,297 pairs across 98 papers** (supplements ON — 2,202 are supplement-unlock-only).
- **BioC fetch failures: 0** (service was stable for the whole run).
- **Fully processed (verified PD produced): 1,629 / 2,297 = 70.9%.**
- **"Mentioned in passing" QC skips: 435** (valid — no PD warranted).
- Remaining ~233 = guardrail refusals / other failures.

## 2. "Mentioned in passing" — old vs new user-comment batch

`check_if_in_passing` reads a `only_in_passing` flag in the summary; older summary schemas lack it.

| batch | model | summaries | carry `only_in_passing` | flagged in-passing |
|---|---|---|---|---|
| old user-comment batch | `claude-sonnet-4-20250514` (Sonnet 4) | 3,693 | 1 (≈none) | **0** |
| this priority batch | `anthropic/claude-sonnet-4.5` | 2,177 | 2,176 | **435 (20.0%)** |

- This batch (Sonnet 4.5): **20.0%** of successful summaries blocked as mentioned-in-passing (≈18.9% of
  the 2,297 attempted candidates).
- **The old batch did NOT have the filter** — it ran on Sonnet 4 with an older schema; those genes all
  proceeded to PD generation regardless. Not directly comparable (different genes; metric never computed).

## 3. Guardrail-blocked cases (all Sonnet-4.5 runs)

Compiled by `collect_guardrail_examples.py` → `out/guardrail_examples/` (CSV + `raw/` full outputs +
`summary.txt`). A block = a stage saved success=False with a refusal signature (empty / content_filter /
≤2 output tokens), deduped per (pmid, gene, stage).

- **482 cases / 481 unique gene-paper pairs.**
- By stage: getGeneSummary 253, generatePDs 228, verifyPDs 1.
- By signature: empty_response 240, zero_output_tokens 229, parse_fail_after_retries 13.
- **FungiDB is the hotspot: 238 / 482** (ToxoDB 69, TriTrypDB 66, PlasmoDB 65, …). 250 fall in the
  priority papers. Ready to share with collaborators for testing alternative models/settings.

## 4. Comments+related pre-check (broader set, NOT yet batched)

`preprocess_user_comments.py --include-related` → `curated_data/comments_rel_precheck.csv`.

- 34,240 pairs → **11,681 pass** the mention filter (4,079 main-text + **7,602 supplement-unlock-only**).
- Already processed 3,814; **not-yet-processed batch candidates: 7,867** (TriTrypDB dominates: 5,753).

## 5. Supplement archive size check (60 MB cap validation)

`measure_suppl_sizes.py` over 1,242 mined supplement archives (Europe PMC), git-ignored.

- Compressed size MB: median **3.4**, mean 7.3, p95 29.4, p99 48.7, **max 59.7**.
- **Archives exceeding the 60 MB download cap: 0** — nothing truncated, though the max sits right at the
  cap (essentially no headroom). Optional: nudge `DEFAULT_CAPS['max_zip_bytes']` to ~80–100 MB for margin.
