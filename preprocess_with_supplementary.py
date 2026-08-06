"""
preprocess_with_supplementary.py

Scans a paper's SUPPLEMENTARY files for gene mentions (software-only, no LLM) to quantify what the
main-text-only preprocessing misses. Two modes:

  --candidates false  (default) : UNLOCK scan. Pairs absent from the main text
        (paper_available==TRUE & alias_in_text==FALSE). Measures how many become findable via
        supplements. -> all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv, comparison.txt
  --candidates true             : MENTION-BOOST scan. Pairs already in the main text
        (paper_available==TRUE & alias_in_text==TRUE). Measures how many EXTRA mentions the
        supplement adds. -> all_PDs_with_PMID_2026_supplementary_mention_boost.csv, comparison_boost.txt

Reuses the existing main-text results in all_PDs_with_PMID_2026_preprocessed.csv, so the only variable
is the supplement. Each paper's supplement zip is fetched/parsed once and reused across its genes.
Resumable via a per-mode checkpoint. Usage: python preprocess_with_supplementary.py [--candidates ...]
                                            [--limit N] [--workers 8]
"""
import os
import io
import sys
import zipfile
import argparse
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
from pipeline import supplementary_helpers as S
from pipeline.vpdb_helpers import get_gene_synonyms

RAW = 'curated_data/all_PDs_with_PMID_2026_to_preprocess.csv'
BASE = 'curated_data/all_PDs_with_PMID_2026_preprocessed.csv'
OUT_UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
OUT_BOOST = 'curated_data/all_PDs_with_PMID_2026_supplementary_mention_boost.csv'
CKPT_DIR = 'out/supplementary_eval/fullscan'
CAPS = dict(S.DEFAULT_CAPS)

_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def paper_records(pmid):
    """Fetch + parse a paper's supplement into a flat list of text records (thread-safe: bypasses
    the module's shared small cache). Returns (pmcid, records|None). None = no supplement."""
    pmcid = S._normalize_pmcid(pmid)
    if not pmcid:
        return None, None
    content = S._download_capped(f"{S.EPMC_BASE}/{pmcid}/supplementaryFiles", CAPS['max_zip_bytes'])
    if not content or content[:2] != b'PK':
        return pmcid, None
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return pmcid, None
    state = S._new_state(); state['recs'] = []
    def collect(name, parsed, st):
        st['recs'].extend(parsed.get('always', []))
        st['recs'].extend(parsed.get('candidates', []))
        return False
    try:
        S._walk_zip(zf, CAPS, state, 0, collect)
    except Exception:
        pass
    return pmcid, state['recs']


def scan_paper(pmid, rows):
    """rows = [{Gene ID, Database}, ...]. Returns per-gene supplement scan results."""
    pmcid, recs = paper_records(pmid)
    available = recs is not None
    out = []
    for r in rows:
        gene, db = r['Gene ID'], r['Database']
        mentions = 0
        if available:
            try:
                aliases = get_gene_synonyms(gene, "", db)
            except Exception:
                aliases = []
            rx = S._build_gene_regex(gene, aliases)
            if rx is not None:
                for rec in recs:
                    mentions += len(rx.findall(rec))
        out.append({'Gene ID': gene, 'pmid_CLEAN': pmid, 'Database': db,
                    'pmcid': pmcid or '', 'suppl_available': available, 'suppl_mentions': mentions})
    return out


def run_scan(cand, ckpt, workers, limit):
    papers = {pmid: g.to_dict('records') for pmid, g in cand.groupby('pmid_CLEAN')}
    all_pmids = list(papers)
    if limit:
        all_pmids = all_pmids[:limit]
    _log(f"candidates: {len(cand)} pairs across {len(papers)} papers (scanning {len(all_pmids)})")

    done_rows, done_pmids = [], set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt, dtype=str)
        done_rows = prev.to_dict('records')
        done_pmids = set(prev['pmid_CLEAN'].astype(str).unique())
        _log(f"resuming: {len(done_pmids)} papers already scanned")
    todo = [p for p in all_pmids if str(p) not in done_pmids]

    results = list(done_rows)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_paper, p, papers[p]): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                _log(f"  paper {p} ERROR {e}")
                rows = [{'Gene ID': r['Gene ID'], 'pmid_CLEAN': p, 'Database': r['Database'],
                         'pmcid': '', 'suppl_available': False, 'suppl_mentions': 0}
                        for r in papers[p]]
            results.extend(rows)
            completed += 1
            hits = sum(1 for x in rows if int(x['suppl_mentions']) > 0)
            _log(f"[{completed}/{len(todo)}] {p}: {len(rows)} genes, {hits} with suppl mentions")
            if completed % 20 == 0:
                pd.DataFrame(results).to_csv(ckpt, index=False)
    pd.DataFrame(results).to_csv(ckpt, index=False)

    scan = pd.DataFrame(results)
    scan['suppl_mentions'] = pd.to_numeric(scan['suppl_mentions'], errors='coerce').fillna(0).astype(int)
    scan['suppl_available'] = scan['suppl_available'].astype(str).str.upper() == 'TRUE'
    return scan, set(str(p) for p in all_pmids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', choices=['false', 'true'], default='false',
                    help="false = unlock scan (alias_in_text FALSE); true = mention-boost scan (TRUE)")
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    mode = args.candidates

    ckpt = os.path.join(CKPT_DIR, f'paper_scan_{mode}_checkpoint.csv')
    if mode == 'false':  # honour the original checkpoint name if present
        legacy = os.path.join(CKPT_DIR, 'paper_scan_checkpoint.csv')
        if os.path.exists(legacy) and not os.path.exists(ckpt):
            ckpt = legacy
    out_csv = OUT_UNLOCK if mode == 'false' else OUT_BOOST
    cmp_txt = os.path.join(CKPT_DIR, 'comparison.txt' if mode == 'false' else 'comparison_boost.txt')

    base = pd.read_csv(BASE, dtype=str)
    base['_avail'] = base['paper_available'].astype(str).str.upper() == 'TRUE'
    base['_ait'] = base['alias_in_text'].astype(str).str.upper() == 'TRUE'
    base['_mentions'] = pd.to_numeric(base['mentions'], errors='coerce').fillna(0).astype(int)

    cand_mask = base['_avail'] & (~base['_ait'] if mode == 'false' else base['_ait'])
    cand = base[cand_mask][['Gene ID', 'pmid_CLEAN', 'Database']].drop_duplicates()

    scan, scanned_pmids = run_scan(cand, ckpt, args.workers, args.limit)

    merged = base.merge(scan[['Gene ID', 'pmid_CLEAN', 'pmcid', 'suppl_available', 'suppl_mentions']],
                        on=['Gene ID', 'pmid_CLEAN'], how='left')
    merged['suppl_available'] = merged['suppl_available'].fillna(False)
    merged['suppl_mentions'] = merged['suppl_mentions'].fillna(0).astype(int)
    merged['mentions_combined'] = merged['_mentions'] + merged['suppl_mentions']

    scanned = merged['pmid_CLEAN'].astype(str).isin(scanned_pmids) & cand_mask

    if mode == 'false':
        merged['unlocked'] = (~merged['_ait']) & (merged['suppl_mentions'] > 0)
        merged['alias_in_text_combined'] = merged['_ait'] | merged['unlocked']
        out_cols = ['Gene ID', 'pmid_CLEAN', 'Database', 'paper_available', 'alias_in_text', 'mentions',
                    'suppl_available', 'suppl_mentions', 'unlocked', 'alias_in_text_combined',
                    'mentions_combined', 'pmcid', 'error']
    else:
        merged['boosted'] = merged['_ait'] & (merged['suppl_mentions'] > 0)
        merged['pct_increase'] = merged.apply(
            lambda r: round(100 * r['suppl_mentions'] / r['_mentions'], 1) if r['_mentions'] else '', axis=1)
        out_cols = ['Gene ID', 'pmid_CLEAN', 'Database', 'paper_available', 'alias_in_text', 'mentions',
                    'suppl_available', 'suppl_mentions', 'mentions_combined', 'boosted', 'pct_increase',
                    'pmcid', 'error']

    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_csv, index=False)
    _log(f"\nsaved -> {out_csv}")

    # ---------- comparison ----------
    cs = merged[scanned]
    L = ['=' * 72]
    if mode == 'false':
        L.append('UNLOCK: supplementary vs main-text-only (alias_in_text==FALSE pairs)')
        L.append('=' * 72)
        base_ok = int(merged['_ait'].sum())
        unlocked = int(merged['unlocked'].sum())
        L.append(f"baseline processable pairs (alias_in_text==TRUE): {base_ok}")
        L.append(f"candidates scanned (FALSE & available)          : {len(cs)}")
        L.append(f"UNLOCKED by supplements                         : {unlocked}")
        if len(cs):
            L.append(f"  unlock rate among scanned candidates          : {100*unlocked/len(cs):.1f}%")
        if base_ok:
            L.append(f"  uplift in processable pairs                   : +{100*unlocked/base_ok:.1f}%")
        L.append("\nBy database (candidates scanned):")
        by = cs.groupby('Database').agg(candidates=('Gene ID', 'size'),
                                        suppl_available=('suppl_available', 'sum'),
                                        unlocked=('unlocked', 'sum'))
        L.append(by.to_string())
    else:
        L.append('MENTION BOOST: supplements added to genes already in the main text (TRUE pairs)')
        L.append('=' * 72)
        n = len(cs)
        boosted = int(cs['boosted'].sum())
        avail = int(cs['suppl_available'].sum())
        main_tot = int(cs['_mentions'].sum())
        add_tot = int(cs['suppl_mentions'].sum())
        L.append(f"TRUE & available pairs scanned          : {n}")
        L.append(f"  supplements available                 : {avail} ({100*avail/n:.0f}%)" if n else "")
        L.append(f"  BOOSTED (gene also in supplement)     : {boosted} ({100*boosted/n:.0f}%)" if n else "")
        L.append(f"total main-text mentions                : {main_tot}")
        L.append(f"total supplement mentions added         : {add_tot}  (+{100*add_tot/main_tot:.0f}% overall)"
                 if main_tot else "")
        bo = cs[cs['boosted']]
        if len(bo):
            L.append(f"among boosted pairs — added mentions    : "
                     f"mean {bo['suppl_mentions'].mean():.1f}, median {bo['suppl_mentions'].median():.0f}, "
                     f"max {int(bo['suppl_mentions'].max())}")
            ge = int((bo['suppl_mentions'] >= bo['_mentions']).sum())
            L.append(f"  pairs where suppl >= main-text mentions: {ge} ({100*ge/len(bo):.0f}% of boosted)")
        L.append("\nBy database (TRUE pairs scanned):")
        by = cs.groupby('Database').agg(pairs=('Gene ID', 'size'),
                                        boosted=('boosted', 'sum'),
                                        main_mentions=('_mentions', 'sum'),
                                        suppl_mentions=('suppl_mentions', 'sum'))
        L.append(by.to_string())

    txt = "\n".join(str(x) for x in L if x != "")
    _log("\n" + txt)
    open(cmp_txt, 'w', encoding='utf-8').write(txt)
    _log(f"\nsaved -> {cmp_txt}")


if __name__ == "__main__":
    main()
