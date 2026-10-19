"""
run_model_test.py MODEL OUT_DIR [workers]

Feasibility test of a new model on the 52-pair set used for the Sonnet 4.5 vs Sonnet 5 tests
(out/test/test_pairs.csv): full STEP_3 chain (getGeneSummary -> generatePDs -> verifyPDs), ALL stages
on MODEL, fresh summaries, prompts unchanged, no supplements. Resumable.

Pipeline code is NOT modified. At runtime this script:
  - adds `Authorization: Bearer $VEUPATHDB_API_KEY` to VEuPathDB requests (records API needs a key
    since the 20 Aug 2026 release; without it get_vpdb_alias silently returns []);
  - falls back to the alias lists saved by the July runs (out/summary_test/) for any gene whose live
    lookup comes back empty; the source used per gene is logged to OUT_DIR/alias_source.csv.

e.g. run_model_test.py anthropic/claude-opus-5.5 out/opus55_test 6
"""
import csv
import glob
import json
import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
import pandas as pd
import requests
import STEP_3_batch_processing as S3
import pipeline.vpdb_helpers as V

PAIRS = 'out/test/test_pairs.csv'
ALIAS_SRC = 'out/summary_test/sonnet45/*.json'
_lock = threading.Lock()


def _patch_aliases(out_dir):
    key = os.getenv('VEUPATHDB_API_KEY', '').strip()
    _post = requests.post

    class _Req:  # vpdb_helpers' own `requests`, with the key added for VEuPathDB hosts only
        def __getattr__(self, name):
            return getattr(requests, name)

        def post(self, url, *a, headers=None, **kw):
            headers = dict(headers or {})
            if key and '.org/' in url and 'service/record-types' in url:
                headers['Authorization'] = f'Bearer {key}'
            return _post(url, *a, headers=headers, **kw)

    V.requests = _Req()

    saved = {}
    for f in glob.glob(ALIAS_SRC):
        d = json.load(open(f, encoding='utf-8'))
        _, _, aka = d.get('gene_display', '').partition(', also known as ')
        saved[d['gene']] = [a.strip() for a in aka.split(', ') if a.strip()]

    live_fn, seen = V.get_vpdb_alias, set()
    log = open(os.path.join(out_dir, 'alias_source.csv'), 'a', newline='', encoding='utf-8')
    w = csv.writer(log)

    def get_alias(gene_id, host_db):
        live = live_fn(gene_id, host_db)
        src, res = ('live', live) if live else ('saved_july', saved.get(gene_id, []))
        with _lock:
            if gene_id not in seen:
                seen.add(gene_id)
                w.writerow([gene_id, host_db, src, len(res)])
                log.flush()
        return res

    V.get_vpdb_alias = S3.get_vpdb_alias = get_alias
    print(f"VEuPathDB key {'set' if key else 'MISSING'}; July fallback aliases for {len(saved)} genes",
          flush=True)


def main():
    model, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    S3.SUMMARY_MODEL = S3.PD_GENERATOR_MODEL = S3.PD_VERIFIER_MODEL = S3.FORMATTER_MODEL = model
    S3.POSSIBLE_SUMMARY_MODEL_KEYS = [model]  # never reuse another model's summary
    S3.OUT_DIR = Path(out)
    S3.OUT_DIR.mkdir(parents=True, exist_ok=True)
    S3.OVERWRITE_EXISTING = False
    S3.FETCH_SUPPLEMENTARY = False  # match the original 52-pair test conditions
    _patch_aliases(out)

    df = pd.read_csv(PAIRS, dtype=str).drop_duplicates(['pmid', 'gene_ID'])
    papers = {}
    for _, r in df.iterrows():
        papers.setdefault(r['pmid'], []).append((r['gene_ID'], r['host_db']))
    items = sorted(papers.items(), key=lambda kv: -len(kv[1]))
    print(f"{len(df)} pairs / {len(items)} papers -> {workers} threads | model={model}", flush=True)

    done, all_res = [0], []

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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, p, gl) for p, gl in items]):
            f.result()

    pd.DataFrame(all_res).to_csv(os.path.join(out, 'batch_results.csv'), index=False)
    print("BATCH DONE ->", out, flush=True)


if __name__ == '__main__':
    main()
