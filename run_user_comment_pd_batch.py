"""
run_user_comment_pd_batch.py  [precheck.csv] [workers] [--supplements]

STEP_3 PD-generation batch (getGeneSummary -> generatePDs -> verifyPDs; NO user-comment judge) for the
user-comment gene-paper pairs that PASS the pre-check and are NOT yet processed. Writes into
out/user_comment_evaluation/ so STEP_4's judge can later consume the outputs. Threaded over papers,
resumable (OVERWRITE_EXISTING=False -> already-completed genes are skipped by process_paper_with_caching).

!! PAID LLM RUN — do NOT launch until the pre-check counts + funds are reviewed. !!
Supplement behaviour is a DEFERRED decision: default OFF. Pass --supplements to enable unlock +
supplement-aware summary/verify (the fixed behaviour). If run WITHOUT --supplements, supplement-unlocked
pairs (gene absent from main text) will refuse/produce nothing, so only enable it if unlocked pairs are
in the batch.
"""
import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
import pandas as pd
import STEP_3_batch_processing as S3

OUT = 'out/user_comment_evaluation'
PRECHECK = 'curated_data/user_comments_precheck.csv'
_lock = threading.Lock()


def _b(s):
    return str(s).strip().upper() in ('TRUE', '1')


def main():
    args = [a for a in sys.argv[1:]]
    supplements = '--supplements' in args
    args = [a for a in args if a != '--supplements']
    precheck = next((a for a in args if a.endswith('.csv')), PRECHECK)
    workers = next((int(a) for a in args if a.isdigit()), 8)

    S3.OUT_DIR = Path(OUT)
    S3.OUT_DIR.mkdir(parents=True, exist_ok=True)
    S3.OVERWRITE_EXISTING = False
    S3.FETCH_SUPPLEMENTARY = supplements

    df = pd.read_csv(precheck, dtype=str)
    passing = df[df['passes_precheck'].apply(_b) & ~df['already_processed'].apply(_b)]
    passing = passing.drop_duplicates(['pmid_CLEAN', 'Gene ID'])
    papers = {}
    for _, r in passing.iterrows():
        papers.setdefault(str(r['pmid_CLEAN']), []).append((str(r['Gene ID']), str(r['Database'])))
    items = sorted(papers.items(), key=lambda kv: -len(kv[1]))  # largest papers first
    print(f"{len(passing)} pairs / {len(items)} papers -> {workers} threads | "
          f"FETCH_SUPPLEMENTARY={supplements}", flush=True)

    done = [0]
    all_res = []

    def work(pmid, gene_list):
        try:
            res = S3.process_paper_with_caching(pmid, gene_list, save=True)
        except Exception as e:
            res = [{'pubmed_id': pmid, 'gene_id': g, 'success': False,
                    'error': f'paper task error: {e}'} for g, _ in gene_list]
        with _lock:
            done[0] += 1
            all_res.extend(res)
            ok = sum(1 for x in res if x.get('success'))
            print(f"[{done[0]}/{len(items)}] {pmid}: {len(gene_list)} genes, {ok} ok", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, p, gl) for p, gl in items]):
            f.result()

    pd.DataFrame(all_res).to_csv(os.path.join(OUT, 'batch_results.csv'), index=False)
    print("BATCH DONE ->", OUT, flush=True)


if __name__ == '__main__':
    main()
