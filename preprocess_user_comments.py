"""
preprocess_user_comments.py  [--priority PATH | --include-related] [--limit N] [--workers 8]

Mention pre-check for gene-paper pairs referenced by VEuPathDB user comments. Mirrors STEP_3's own
gate so the counts predict exactly what the batch would process, WITHOUT running any LLM:

  paper text (hardened fetch) -> check_gene_in_text -> paper_available / alias_in_text / mentions
  if the gene is NOT in the main text -> count_supplementary_mentions -> suppl_available / suppl_mentions
  unlocked = (gene absent from main text) AND (>=1 supplement mention)      [supplement-only recovery]

Also flags pairs ALREADY processed (getGeneSummary+generatePDs+verifyPDs all succeeded) in
out/user_comment_evaluation, so we can see the not-yet-processed batch candidates.

Pair sources (a pair is one (gene, pmid, database)):
  default (comments)   : all_user_comments_with_pmid_fetched.tsv, primary `Gene ID` x each PubMed ID.
  --include-related    : ALSO expand the comma-separated `Related Stable IDs` column (a comment often
                         relates a paper to ~100 genes). Superset of the default.
  --priority PATH      : pre-check a curated priority TSV directly (columns gene_id/pmid/database), e.g.
                         batch_needs_description_outstanding.tsv. PMIDs are normalized (strip trailing
                         '.0'; non-numeric DOIs/PMCIDs dropped). Small + exact for the real target set.

Threaded over papers, gene-aware resume (a paper is re-scanned if any requested gene is not yet done,
so transient BioC failures AND newly-added genes are picked up). No LLM calls, no batch launch.
"""
import os
import re
import sys
import json
import argparse
import threading
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
from pipeline.pubmed_helpers import get_paper_text, PaperNotInOA, PaperFetchError
from pipeline.vpdb_helpers import check_gene_in_text
from pipeline import supplementary_helpers as S

TSV = 'curated_data/all_user_comments_with_pmid_fetched.tsv'
STATUS_DIR = Path('out/user_comment_evaluation')          # where STEP_3 outputs live / would be written
REQUIRED_STEPS = ('getGeneSummary', 'generatePDs', 'verifyPDs')

OUT_COLS = ['Gene ID', 'pmid_CLEAN', 'Database', 'paper_available', 'alias_in_text', 'mentions',
            'suppl_available', 'suppl_mentions', 'unlocked', 'mentions_combined', 'pmcid',
            'already_processed', 'passes_precheck', 'comment_ids', 'error']
_lock = threading.Lock()


def _split_pmids(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r',', str(value)) if x.strip()]


def _clean_pmid(p):
    """Normalize a PMID cell: strip whitespace and a trailing '.0' (float artifact); keep only
    numeric PMIDs. Non-numeric values (DOIs, PMCIDs, junk) return None and are dropped."""
    p = str(p).strip()
    if p.endswith('.0'):
        p = p[:-2]
    return p if p.isdigit() else None


def _status_success(status_json, gene_id):
    """True if all 3 STEP_3 stages succeeded for this gene (any model key). Mirrors
    STEP_3.is_gene_already_processed(flex_model=True)."""
    for step in REQUIRED_STEPS:
        node = status_json.get(step, {}).get(gene_id, {})
        if not isinstance(node, dict):
            return False
        if not any(isinstance(v, dict) and v.get('success') and v.get('data')
                   for k, v in node.items() if k != 'gene_ID'):
            return False
    return True


def _load_status(pmid):
    fp = STATUS_DIR / f"{pmid}.json"
    if not fp.exists():
        return {}
    try:
        return json.load(open(fp, encoding='utf-8'))
    except Exception:
        return {}


def scan_paper(pmid, genes):
    """genes = [(gene_id, database, comment_ids_str), ...]. Fetch paper once, gate each gene."""
    paper_text, paper_available, err = '', False, ''
    try:
        paper_text = get_paper_text(pmid)
        paper_available = True
    except PaperNotInOA as e:
        err = f'not_in_OA: {e}'
    except PaperFetchError as e:
        err = f'fetch_error: {e}'
    except Exception as e:
        err = f'error: {type(e).__name__}: {e}'

    status_json = _load_status(pmid)
    rows = []
    for gene_id, db, comment_ids in genes:
        alias_in_text, mentions = (False, 0)
        suppl_available, suppl_mentions, pmcid = '', 0, ''
        if paper_available:
            try:
                alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, db)
            except Exception:
                alias_in_text, mentions = False, 0
            if not alias_in_text:
                # only scan supplements for main-text-negative genes (STEP_3's unlock gate)
                sm = S.count_supplementary_mentions(pmid, gene_id, None, db, caps=None)
                suppl_available = bool(sm.get('available'))
                suppl_mentions = int(sm.get('mentions', 0) or 0)
                pmcid = sm.get('pmcid') or ''
        unlocked = (paper_available and not alias_in_text and suppl_mentions > 0)
        rows.append({
            'Gene ID': gene_id, 'pmid_CLEAN': pmid, 'Database': db,
            'paper_available': paper_available, 'alias_in_text': bool(alias_in_text),
            'mentions': int(mentions or 0),
            'suppl_available': suppl_available, 'suppl_mentions': suppl_mentions,
            'unlocked': unlocked, 'mentions_combined': int(mentions or 0) + suppl_mentions,
            'pmcid': pmcid,
            'already_processed': _status_success(status_json, gene_id),
            'passes_precheck': bool(alias_in_text) or unlocked,
            'comment_ids': comment_ids, 'error': err,
        })
    return rows


def _group(cids, dbs):
    papers = defaultdict(list)
    for (gene, p), cid_set in cids.items():
        clean = ';'.join(sorted(x for x in cid_set if x and x != 'nan'))
        papers[p].append((gene, dbs[(gene, p)], clean))
    return papers


def build_pairs_comments(include_related):
    df = pd.read_csv(TSV, sep='\t', dtype=str)
    cids, dbs = defaultdict(set), {}
    for _, r in df.iterrows():
        db = str(r.get('Database', '')); cid = str(r.get('Comment ID', ''))
        genes = [str(r['Gene ID']).strip()]
        if include_related and pd.notna(r.get('Related Stable IDs')):
            genes += [x.strip() for x in re.split(r',', str(r['Related Stable IDs'])) if x.strip()]
        for p in _split_pmids(r.get('PubMed IDs')):
            cp = _clean_pmid(p)
            if not cp:
                continue
            for g in genes:
                if not g or g == 'nan':
                    continue
                key = (g, cp); cids[key].add(cid); dbs.setdefault(key, db)
    return _group(cids, dbs)


def build_pairs_priority(path):
    df = pd.read_csv(path, sep='\t', dtype=str)
    cids, dbs = defaultdict(set), {}
    dropped = 0
    for _, r in df.iterrows():
        g = str(r['gene_id']).strip(); db = str(r.get('database', ''))
        cid = str(r.get('comment_id', ''))
        cp = _clean_pmid(r['pmid'])
        if not cp or not g or g == 'nan':
            dropped += 1
            continue
        key = (g, cp); cids[key].add(cid); dbs.setdefault(key, db)
    print(f"priority file: {len(df)} rows, {dropped} dropped (non-numeric/blank PMID)", flush=True)
    return _group(cids, dbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--priority', metavar='PATH', default=None,
                    help='pre-check this priority TSV (gene_id/pmid/database) instead of the comments TSV')
    ap.add_argument('--include-related', action='store_true',
                    help='(comments mode) also expand the Related Stable IDs column into pairs')
    ap.add_argument('--limit', type=int, default=0, help='limit number of papers (sanity runs)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    if args.priority:
        tag = 'priority'
        papers = build_pairs_priority(args.priority)
    elif args.include_related:
        tag = 'comments_rel'
        papers = build_pairs_comments(True)
    else:
        tag = 'comments'
        papers = build_pairs_comments(False)

    # default comments mode keeps the original filenames; other modes get their own
    if tag == 'comments':
        out_csv, summary_txt = 'curated_data/user_comments_precheck.csv', 'curated_data/user_comments_precheck_summary.txt'
    else:
        out_csv, summary_txt = f'curated_data/{tag}_precheck.csv', f'curated_data/{tag}_precheck_summary.txt'
    ckpt_dir = f'out/user_comment_evaluation/precheck_ckpt_{tag}'
    ckpt = os.path.join(ckpt_dir, 'checkpoint.csv')
    os.makedirs(ckpt_dir, exist_ok=True)

    all_pmids = sorted(papers)
    n_pairs = sum(len(v) for v in papers.values())
    if args.limit:
        all_pmids = all_pmids[:args.limit]
    print(f"[{tag}] {n_pairs} pairs across {len(papers)} papers (scanning {len(all_pmids)}), "
          f"{args.workers} workers", flush=True)

    # ---- gene-aware resume ----
    # A pair is "done" only if it has a non-transient outcome. A paper is (re)scanned if any of its
    # requested genes is not yet done -> picks up transient BioC failures AND newly-added genes.
    def _transient(e):
        e = str(e or '')
        return e.startswith('fetch_error') or e.startswith('paper_error') or e.startswith('error:')

    requested = {(p, g) for p in all_pmids for g, _, _ in papers[p]}
    prev_rows, done_pairs = [], set()
    if os.path.exists(ckpt):
        prev_rows = [r for r in pd.read_csv(ckpt, dtype=str).to_dict('records')
                     if (str(r['pmid_CLEAN']), str(r['Gene ID'])) in requested]
        done_pairs = {(str(r['pmid_CLEAN']), str(r['Gene ID'])) for r in prev_rows if not _transient(r.get('error'))}

    todo = [p for p in all_pmids if any((p, g) not in done_pairs for g, _, _ in papers[p])]
    todo_set = set(todo)
    rows = [r for r in prev_rows if str(r['pmid_CLEAN']) not in todo_set]  # keep fully-done papers only
    if os.path.exists(ckpt):
        print(f"resuming: {len(done_pairs)} pairs done; {len(todo)} papers to (re)scan", flush=True)

    done = [0]

    def work(pmid):
        try:
            r = scan_paper(pmid, papers[pmid])
        except Exception as e:
            r = [{'Gene ID': g, 'pmid_CLEAN': pmid, 'Database': db, 'paper_available': False,
                  'alias_in_text': False, 'mentions': 0, 'suppl_available': '', 'suppl_mentions': 0,
                  'unlocked': False, 'mentions_combined': 0, 'pmcid': '', 'already_processed': False,
                  'passes_precheck': False, 'comment_ids': c, 'error': f'paper_error: {e}'}
                 for g, db, c in papers[pmid]]
        with _lock:
            rows.extend(r); done[0] += 1
            if done[0] % 25 == 0:
                pd.DataFrame(rows).to_csv(ckpt, index=False)
                npass = sum(1 for x in rows if x.get('passes_precheck'))
                nunl = sum(1 for x in rows if x.get('unlocked'))
                print(f"  [{done[0]}/{len(todo)}] papers | passing={npass} unlocked={nunl}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in todo]):
            f.result()

    df = pd.DataFrame(rows)
    df.to_csv(ckpt, index=False)
    df.to_csv(out_csv, index=False, columns=[c for c in OUT_COLS if c in df.columns])
    report(df, summary_txt, tag)
    print(f"\nsaved -> {out_csv}", flush=True)


def report(df, summary_txt, tag):
    def B(col):
        return df[col].astype(str).str.upper().isin(('TRUE', '1'))
    for c in ('paper_available', 'alias_in_text', 'unlocked', 'already_processed', 'passes_precheck'):
        df[c] = B(c)
    df['mentions'] = pd.to_numeric(df['mentions'], errors='coerce').fillna(0).astype(int)
    df['suppl_mentions'] = pd.to_numeric(df['suppl_mentions'], errors='coerce').fillna(0).astype(int)

    n = len(df)
    avail = int(df['paper_available'].sum())
    main_pass = int(df['alias_in_text'].sum())
    unlocked = int(df['unlocked'].sum())
    passing = int(df['passes_precheck'].sum())
    proc = int((df['passes_precheck'] & df['already_processed']).sum())
    todo = int((df['passes_precheck'] & ~df['already_processed']).sum())
    unl = df[df['unlocked']]
    L = ['=' * 72, f'PRE-CHECK [{tag}]', '=' * 72,
         f"total (gene,pmid,db) pairs           : {n:,}",
         f"paper available (OA full text)       : {avail:,} ({100*avail/max(n,1):.1f}%)",
         f"gene in MAIN TEXT (alias_in_text)    : {main_pass:,} ({100*main_pass/max(n,1):.1f}%)",
         f"UNLOCKED via supplement only         : {unlocked:,}",
         (f"  supplement mentions on unlocked    : total {int(unl['suppl_mentions'].sum()):,}, "
          f"mean {unl['suppl_mentions'].mean():.1f}, median {unl['suppl_mentions'].median():.0f}, "
          f"max {int(unl['suppl_mentions'].max())}") if len(unl) else
         "  supplement mentions on unlocked    : (none)",
         f"PASSES pre-check (main-text OR unlock): {passing:,} ({100*passing/max(n,1):.1f}%)",
         f"  already processed (3 stages ok)    : {proc:,}",
         f"  NOT yet processed (batch candidates) : {todo:,}",
         '', 'passing pairs, by Database:']
    by = df[df['passes_precheck']].groupby('Database').agg(
        passing=('Gene ID', 'size'),
        main_text=('alias_in_text', 'sum'),
        unlocked=('unlocked', 'sum'),
        already_done=('already_processed', 'sum'))
    by['to_process'] = by['passing'] - by['already_done']
    L.append(by.to_string())
    txt = "\n".join(str(x) for x in L)
    print("\n" + txt, flush=True)
    open(summary_txt, 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
