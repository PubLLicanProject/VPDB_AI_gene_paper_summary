"""
rescan_legends_ocr.py  [workers=8]

Re-scans the "supplement present, 0 mentions" UNLOCK candidates with the enhanced supplement parser
(figure/table LEGENDS from fullTextXML + image OCR) to find genes that NOW become findable. Uses the
same count_supplementary_mentions the pipeline uses (so results match production). Threaded, resumable
(per-paper checkpoint), incremental save. Output: which candidates newly unlock, and via which source.
"""
import os
import sys
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '.')
from pipeline import supplementary_helpers as S
S.OCR_IMAGES = True  # legends are always on; enable OCR for this rescan

UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
OUTDIR = 'out/supplementary_eval/rescan_legends_ocr'
CKPT = os.path.join(OUTDIR, 'rescan_results.csv')
IMG_EXT = ('.gif', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
_lock = threading.Lock()


def _source(files):
    if any('legend' in str(n).lower() for n, _ in files):
        return 'legend'
    if any(str(n).lower().endswith(IMG_EXT) for n, _ in files):
        return 'ocr_image'
    return 'file' if files else ''


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUTDIR, exist_ok=True)
    u = pd.read_csv(UNLOCK, dtype=str)
    _b = lambda s: s.astype(str).str.upper() == 'TRUE'
    cand = u[_b(u.paper_available) & ~_b(u.alias_in_text) & _b(u.suppl_available)
             & (pd.to_numeric(u.suppl_mentions, errors='coerce').fillna(0) == 0)][
        ['Gene ID', 'pmid_CLEAN', 'Database']].drop_duplicates()
    papers = {pmid: g.to_dict('records') for pmid, g in cand.groupby('pmid_CLEAN')}

    rows, done = [], set()
    if os.path.exists(CKPT):
        prev = pd.read_csv(CKPT, dtype=str)
        rows = prev.to_dict('records')
        done = set(prev['pmid_CLEAN'].astype(str))
    todo = [p for p in papers if str(p) not in done]
    print(f"{len(cand)} candidates / {len(papers)} papers; {len(done)} done, {len(todo)} todo, {workers} threads", flush=True)

    cnt = [0]

    def work(pmid, genes):
        local = []
        for r in genes:
            g, db = r['Gene ID'], r['Database']
            try:
                res = S.count_supplementary_mentions(pmid, g, host_db=db)
            except Exception:
                res = {'mentions': 0, 'files': []}
            m = int(res.get('mentions', 0) or 0)
            local.append({'Gene ID': g, 'pmid_CLEAN': pmid, 'Database': db,
                          'new_mentions': m, 'unlocked': m > 0, 'source': _source(res.get('files', []))})
        with _lock:
            rows.extend(local); cnt[0] += 1
            if cnt[0] % 10 == 0:
                pd.DataFrame(rows).to_csv(CKPT, index=False)
                nu = sum(1 for x in rows if x['unlocked'])
                print(f"  {cnt[0]}/{len(todo)} papers | newly unlocked so far: {nu}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(work, p, papers[p]) for p in todo]:
            try:
                f.result()
            except Exception as e:
                print("paper error:", e, flush=True)

    df = pd.DataFrame(rows); df.to_csv(CKPT, index=False)
    df['unlocked'] = df['unlocked'].astype(str).str.upper().eq('TRUE')
    nu = df[df['unlocked']]
    print("\n" + "=" * 60)
    print(f"LEGENDS+OCR RESCAN: {len(df)} no-mention candidates")
    print(f"  newly unlocked: {len(nu)} ({100*len(nu)/max(len(df),1):.1f}%)")
    if len(nu):
        print("  by source:"); print(nu['source'].value_counts().to_string())
        print("  by database:"); print(nu['Database'].value_counts().to_string())
    print("BATCH DONE ->", OUTDIR, flush=True)


if __name__ == '__main__':
    main()
