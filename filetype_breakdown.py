"""
filetype_breakdown.py  [unlock|boost|both]

Attributes supplement gene-mentions to the SOURCE FILE TYPE (.xlsx/.xls/.csv-.tsv/.pdf/.docx/.doc/.xml).
Software-only, ID-based matching (no VEuPathDB alias calls -> light on any running batch). Each paper's
supplement is fetched/parsed once (records kept per file) and every gene of that paper is matched against
each file. Writes out/supplementary_eval/report/filetype_<set>.csv and a figure.
"""
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import supplementary_helpers as S

OUTDIR = 'out/supplementary_eval/report'
UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
BOOST = 'curated_data/all_PDs_with_PMID_2026_supplementary_mention_boost.csv'
CAPS = dict(S.DEFAULT_CAPS)
EXT_LABEL = {'.xlsx': 'xlsx', '.xls': 'xls', '.csv': 'csv', '.tsv': 'tsv', '.tab': 'tsv',
             '.txt': 'txt', '.pdf': 'pdf', '.docx': 'docx', '.doc': 'doc',
             '.xml': 'xml', '.html': 'html', '.htm': 'html'}
_lock = threading.Lock()


def paper_byfile(pmid):
    pmcid = S._normalize_pmcid(pmid)
    if not pmcid:
        return None
    content = S._download_capped(f"{S.EPMC_BASE}/{pmcid}/supplementaryFiles", CAPS['max_zip_bytes'])
    if not content or content[:2] != b'PK':
        return {}
    import io, zipfile
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return {}
    state = S._new_state(); state['byfile'] = {}
    def collect(name, parsed, st):
        recs = list(parsed.get('always', [])) + list(parsed.get('candidates', []))
        st['byfile'].setdefault(name, []).extend(recs)
        return False
    try:
        S._walk_zip(zf, CAPS, state, 0, collect)
    except Exception:
        pass
    return state['byfile']


def scan(pairs_by_paper, workers=6):
    agg = defaultdict(int)          # ext_label -> total mentions
    genes_hit = defaultdict(int)    # ext_label -> gene-pairs with >=1 mention in that type
    done = [0]

    def work(pmid, genes):
        byfile = paper_byfile(pmid)
        local = defaultdict(int); local_g = defaultdict(int)
        if byfile:
            compiled = [(g, S._build_gene_regex(g, [])) for g in genes]
            for name, recs in byfile.items():
                ext = EXT_LABEL.get(os.path.splitext(name)[1].lower())
                if not ext:
                    continue
                blob = '\n'.join(recs)
                for g, rx in compiled:
                    if rx is None:
                        continue
                    c = len(rx.findall(blob))
                    if c:
                        local[ext] += c
                        local_g[ext] += 1
        with _lock:
            for k, v in local.items():
                agg[k] += v
            for k, v in local_g.items():
                genes_hit[k] += v
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{len(pairs_by_paper)} papers", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p, gs) for p, gs in pairs_by_paper.items()]
        for f in as_completed(futs):
            f.result()
    return agg, genes_hit


def load_pairs(which):
    if which == 'unlock':
        df = pd.read_csv(UNLOCK)
        df = df[df['unlocked'].astype(str).str.upper() == 'TRUE']
    else:
        df = pd.read_csv(BOOST)
        df = df[df['boosted'].astype(str).str.upper() == 'TRUE']
    by = {}
    for _, r in df.iterrows():
        by.setdefault(str(r['pmid_CLEAN']), []).append(str(r['Gene ID']))
    return by


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'unlock'
    sets = ['unlock', 'boost'] if which == 'both' else [which]
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}
    for s in sets:
        by = load_pairs(s)
        print(f"[{s}] {sum(len(v) for v in by.values())} pairs / {len(by)} papers", flush=True)
        agg, genes_hit = scan(by)
        t = pd.DataFrame({'mentions': agg, 'gene_pairs_hit': genes_hit}).fillna(0).astype(int)
        t = t.sort_values('mentions', ascending=False)
        t['mentions_%'] = (100 * t['mentions'] / max(t['mentions'].sum(), 1)).round(1)
        t.to_csv(os.path.join(OUTDIR, f'filetype_{s}.csv'))
        results[s] = t
        print(f"\n[{s}] mentions by file type:")
        print(t.to_string())

    # figure
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.2), squeeze=False)
    for ax, (s, t) in zip(axes[0], results.items()):
        ax.bar(t.index, t['mentions'], color='#2a9d8f')
        for i, (m, p) in enumerate(zip(t['mentions'], t['mentions_%'])):
            ax.text(i, m, f'{p:.0f}%', ha='center', va='bottom', fontsize=9)
        ax.set_title(f'{s}: supplement mentions by file type')
        ax.set_ylabel('mentions'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig_filetype.png'), dpi=150)
    print("\nsaved -> filetype_*.csv, fig_filetype.png")


if __name__ == '__main__':
    main()
