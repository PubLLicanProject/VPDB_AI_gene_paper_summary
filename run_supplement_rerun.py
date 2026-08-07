"""
run_supplement_rerun.py  <input.csv> [workers]

Supplement-aware batch re-run, ONE process with a thread pool over papers. LLM/HTTP calls are
I/O-bound so threads give real concurrency at low memory and without spawning many subprocesses
(which the harness killed). Each paper's genes are processed together by one thread, so per-paper
prompt caching + a single supplement fetch per paper are preserved. Resumable (OVERWRITE_EXISTING
is False -> already-completed genes are skipped). Writes {pmid}.json to a clean new cache folder.
"""
import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
import pandas as pd
import STEP_3_batch_processing as S3

CACHE = 'out/VPDB_PD_supplementary_rerun/cache'
_lock = threading.Lock()


def main():
    inp = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    S3.FETCH_SUPPLEMENTARY = True
    S3.OVERWRITE_EXISTING = False
    S3.OUT_DIR = Path(CACHE)
    S3.OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = S3.CSV_COLUMNS
    gcol, pcol, dcol = cols['gene_id'], cols['pmid'], cols['database']
    df = pd.read_csv(inp, dtype=str)
    papers = {}
    for _, r in df.iterrows():
        papers.setdefault(str(r[pcol]), []).append((str(r[gcol]), str(r[dcol])))
    items = sorted(papers.items(), key=lambda kv: -len(kv[1]))  # largest papers first
    print(f"{len(df)} pairs / {len(items)} papers -> {workers} threads (single process)", flush=True)

    done = [0]

    def work(pmid, gene_list):
        try:
            res = S3.process_paper_with_caching(pmid, gene_list, save=True)
        except Exception as e:
            res = [{'pubmed_id': pmid, 'gene_id': g, 'success': False,
                    'error': f'paper task error: {e}'} for g, _ in gene_list]
        with _lock:
            done[0] += 1
            ok = sum(1 for x in res if x.get('success'))
            print(f"[{done[0]}/{len(items)}] {pmid}: {len(gene_list)} genes, {ok} ok", flush=True)
        return res

    all_res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p, gl) for p, gl in items]
        for f in as_completed(futs):
            all_res.extend(f.result())

    pd.DataFrame(all_res).to_csv('out/VPDB_PD_supplementary_rerun/results.csv', index=False)
    print("BATCH DONE", flush=True)


if __name__ == '__main__':
    main()
