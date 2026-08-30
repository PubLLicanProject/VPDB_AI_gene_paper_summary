"""
report_supplementary_impact.py

Comprehensive collaborator report on the supplementary-materials work. Sections:
  1. Original-pipeline limitations: not-OA / OA-no-mention / OA-with-mention, by database, genus,
     and paper size (genes/paper: single, 2-5, 6-10, >10).
  2. Supplement impact: UNLOCK (recovered no-mention pairs) + BOOST (extra evidence for in-text genes),
     by database and paper size, plus mentions-by-file-type (from filetype_breakdown.py, if present).
  3. Insights: condensed findings for a paper's "pipeline development" results subsection.

Outputs to out/supplementary_eval/report/: *.csv tables, *.png figures, report.html (self-contained).
Re-runs safely; picks up filetype_*.csv when the file-type scan has finished.
"""
import os
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = 'out/supplementary_eval/report'
PRE = 'curated_data/all_PDs_with_PMID_2026_preprocessed.csv'
RAW = 'curated_data/all_PDs_with_PMID_2026_to_preprocess.csv'
UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
BOOST = 'curated_data/all_PDs_with_PMID_2026_supplementary_mention_boost.csv'
BUCKETS = ['1 (single)', '2-5', '6-10', '11-20', '21-50', '51-100', '>100']
CATS = ['1_not_open_access', '2_OA_no_mention', '3_OA_with_mention']
CAT_LBL = {'1_not_open_access': 'not open-access', '2_OA_no_mention': 'OA, gene not in text',
           '3_OA_with_mention': 'OA, gene in text'}
CAT_COL = {'1_not_open_access': '#c0392b', '2_OA_no_mention': '#e0a458', '3_OA_with_mention': '#2a9d8f'}


def _b(s):
    return s.astype(str).str.upper() == 'TRUE'


def _bucket(n):
    if n <= 1:
        return '1 (single)'
    if n <= 5:
        return '2-5'
    if n <= 10:
        return '6-10'
    if n <= 20:
        return '11-20'
    if n <= 50:
        return '21-50'
    if n <= 100:
        return '51-100'
    return '>100'


def _emb(path):
    return 'data:image/png;base64,' + base64.b64encode(open(path, 'rb').read()).decode() if os.path.exists(path) else ''


def load():
    pre = pd.read_csv(PRE, dtype=str)
    raw = pd.read_csv(RAW, dtype=str)
    raw['genus'] = raw['Organism'].astype(str).str.split().str[0]
    pre = pre.merge(raw[['Gene ID', 'pmid_CLEAN', 'genus']].drop_duplicates(['Gene ID', 'pmid_CLEAN']),
                    on=['Gene ID', 'pmid_CLEAN'], how='left')
    pre['avail'] = _b(pre['paper_available'])
    pre['ait'] = _b(pre['alias_in_text'])
    pre['category'] = pre.apply(lambda r: CATS[0] if not r['avail'] else (CATS[2] if r['ait'] else CATS[1]), axis=1)
    npp = pre.groupby('pmid_CLEAN')['Gene ID'].nunique()
    pre['n_genes'] = pre['pmid_CLEAN'].map(npp)
    pre['bucket'] = pre['n_genes'].apply(_bucket)

    u = pd.read_csv(UNLOCK, dtype=str)
    u['unlocked'] = _b(u['unlocked'])
    u['avail'] = _b(u['paper_available'])
    u['ait'] = _b(u['alias_in_text'])
    u = u.merge(pre[['Gene ID', 'pmid_CLEAN', 'bucket', 'genus']], on=['Gene ID', 'pmid_CLEAN'], how='left')
    b = pd.read_csv(BOOST, dtype=str)
    b['boosted'] = _b(b['boosted'])
    b['ait'] = _b(b['alias_in_text'])
    b['suppl_mentions'] = pd.to_numeric(b['suppl_mentions'], errors='coerce').fillna(0).astype(int)
    b['mentions'] = pd.to_numeric(b['mentions'], errors='coerce').fillna(0).astype(int)
    b = b.merge(pre[['Gene ID', 'pmid_CLEAN', 'bucket', 'genus']], on=['Gene ID', 'pmid_CLEAN'], how='left')
    return pre, u, b


def stacked(ct, path, title, pct=False):
    ct = ct.reindex(columns=[c for c in CATS if c in ct.columns]).fillna(0)
    data = ct.div(ct.sum(1), axis=0) * 100 if pct else ct
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bottom = [0] * len(data)
    for c in data.columns:
        ax.bar(range(len(data)), data[c], bottom=bottom, label=CAT_LBL[c], color=CAT_COL[c])
        bottom = [x + y for x, y in zip(bottom, data[c])]
    ax.set_xticks(range(len(data))); ax.set_xticklabels(data.index, rotation=0)
    ax.set_ylabel('% of pairs' if pct else 'pairs'); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def impact_bucket_fig(u, b, path):
    ub = u[u['avail'] if 'avail' in u else (~u['ait'] if 'ait' in u else u['unlocked'].notna())]
    unl = u[u['unlocked']].groupby('bucket').size().reindex(BUCKETS).fillna(0)
    bo = b[b['boosted']].groupby('bucket').size().reindex(BUCKETS).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(len(BUCKETS)); w = 0.38
    ax.bar([i - w / 2 for i in x], unl, width=w, color='#2a9d8f', label='unlocked pairs')
    ax.bar([i + w / 2 for i in x], bo, width=w, color='#8ab0c4', label='boosted pairs')
    ax.set_xticks(list(x)); ax.set_xticklabels(BUCKETS)
    ax.set_xlabel('genes per paper'); ax.set_ylabel('pairs'); ax.set_title('Supplement impact by paper size')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def original_vs_unlocked_fig(pre, u, path):
    """Side-by-side bars: what the ORIGINAL pipeline could use (OA + gene in main text) vs the pairs
    supplements UNLOCK (gene absent from main text, recovered via a supplement mention). Left panel =
    overall totals; right panel = the same comparison grouped by database."""
    orig_db = pre[pre['category'] == CATS[2]].groupby('Database').size()
    cand = u[u['avail'] & ~u['ait']]
    unl_db = cand[cand['unlocked']].groupby('Database').size()
    dbs = orig_db.sort_values(ascending=False).index.tolist()
    orig = orig_db.reindex(dbs).fillna(0).astype(int)
    unl = unl_db.reindex(dbs).fillna(0).astype(int)
    orig_tot, unl_tot = int(orig.sum()), int(unl.sum())
    C_ORIG, C_UNL = '#2a9d8f', '#e76f51'

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1, max(2.4, len(dbs) * 0.75)]})
    # Panel A: overall
    axA.bar([0], [orig_tot], width=0.62, color=C_ORIG)
    axA.bar([1], [unl_tot], width=0.62, color=C_UNL)
    for xi, v in ((0, orig_tot), (1, unl_tot)):
        axA.text(xi, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    axA.set_xticks([0, 1]); axA.set_xticklabels(['Original', 'Unlocked'])
    axA.set_title(f'Overall  (+{100 * unl_tot / max(orig_tot, 1):.1f}%)')
    axA.set_ylabel('gene-paper pairs'); axA.margins(y=0.15)
    axA.spines[['top', 'right']].set_visible(False)
    # Panel B: grouped by database
    x = range(len(dbs)); w = 0.4
    axB.bar([i - w / 2 for i in x], orig, width=w, color=C_ORIG, label='Original (OA + gene in text)')
    axB.bar([i + w / 2 for i in x], unl, width=w, color=C_UNL, label='Unlocked (supplement-only)')
    axB.set_xticks(list(x)); axB.set_xticklabels(dbs, rotation=30, ha='right')
    axB.set_title('By database'); axB.set_ylabel('gene-paper pairs'); axB.margins(y=0.12)
    axB.legend(frameon=False, fontsize=9); axB.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Original usable vs supplement-unlocked pairs', y=1.02, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return orig_tot, unl_tot


def filetype_fig(path):
    sets = [s for s in ('unlock', 'boost') if os.path.exists(os.path.join(OUTDIR, f'filetype_{s}.csv'))]
    if not sets:
        return None, {}
    tabs = {s: pd.read_csv(os.path.join(OUTDIR, f'filetype_{s}.csv'), index_col=0) for s in sets}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4.2), squeeze=False)
    for ax, s in zip(axes[0], sets):
        t = tabs[s].sort_values('mentions', ascending=False)
        ax.bar(t.index, t['mentions'], color='#2a9d8f')
        for i, p in enumerate(t['mentions_%']):
            ax.text(i, t['mentions'].iloc[i], f'{p:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{s}: supplement mentions by file type'); ax.set_ylabel('mentions')
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path, tabs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pre, u, b = load()
    n = len(pre)

    # ---- Section 1: baseline ----
    by_cat = pre['category'].value_counts().reindex(CATS)
    baseline_db = pd.crosstab(pre['Database'], pre['category']).reindex(columns=CATS).fillna(0).astype(int)
    genus_top = pre['genus'].value_counts().head(12).index
    baseline_genus = pd.crosstab(pre['genus'], pre['category']).reindex(genus_top).reindex(columns=CATS).fillna(0).astype(int)
    baseline_bucket = pd.crosstab(pre['bucket'], pre['category']).reindex(BUCKETS).reindex(columns=CATS).fillna(0).astype(int)
    papers_per_bucket = pre.drop_duplicates('pmid_CLEAN').groupby('bucket').size().reindex(BUCKETS).fillna(0).astype(int)
    baseline_db.to_csv(os.path.join(OUTDIR, 'baseline_by_database.csv'))
    baseline_genus.to_csv(os.path.join(OUTDIR, 'baseline_by_genus.csv'))
    baseline_bucket.assign(papers=papers_per_bucket).to_csv(os.path.join(OUTDIR, 'baseline_by_papersize.csv'))

    stacked(baseline_db, os.path.join(OUTDIR, 'fig_coverage_by_db.png'),
            'Original pipeline coverage by database', pct=True)
    stacked(baseline_bucket, os.path.join(OUTDIR, 'fig_coverage_by_papersize.png'),
            'Original pipeline coverage by paper size (genes/paper)', pct=True)

    # ---- Section 2: unlock + boost ----
    cand = u[u['avail'] & ~u['ait']]
    impact_db = pd.DataFrame({
        'unlock_candidates': cand.groupby('Database').size(),
        'unlocked': cand[cand['unlocked']].groupby('Database').size(),
        'boost_pairs': b[b['ait']].groupby('Database').size(),
        'boosted': b[b['boosted']].groupby('Database').size(),
        'added_mentions': b[b['boosted']].groupby('Database')['suppl_mentions'].sum(),
    }).fillna(0).astype(int)
    impact_db['unlock_%'] = (100 * impact_db.unlocked / impact_db.unlock_candidates).round(1)
    impact_bucket = pd.DataFrame({
        'unlock_candidates': cand.groupby('bucket').size(),
        'unlocked': cand[cand['unlocked']].groupby('bucket').size(),
        'boost_pairs': b[b['ait']].groupby('bucket').size(),
        'boosted': b[b['boosted']].groupby('bucket').size(),
        'added_mentions': b[b['boosted']].groupby('bucket')['suppl_mentions'].sum(),
    }).reindex(BUCKETS).fillna(0).astype(int)
    impact_bucket['unlock_%'] = (100 * impact_bucket.unlocked / impact_bucket.unlock_candidates.replace(0, 1)).round(1)
    impact_db.to_csv(os.path.join(OUTDIR, 'impact_by_database.csv'))
    impact_bucket.to_csv(os.path.join(OUTDIR, 'impact_by_papersize.csv'))
    impact_bucket_fig(u, b, os.path.join(OUTDIR, 'fig_impact_by_papersize.png'))
    original_vs_unlocked_fig(pre, u, os.path.join(OUTDIR, 'fig_original_vs_unlocked.png'))
    ftpath, fttabs = filetype_fig(os.path.join(OUTDIR, 'fig_filetype.png'))

    # ---- headline numbers ----
    usable = int(by_cat[CATS[2]]); no_mention = int(by_cat[CATS[1]]); not_oa = int(by_cat[CATS[0]])
    unlocked = int(cand['unlocked'].sum())
    boosted = int(b['boosted'].sum()); boost_pairs = int(b['ait'].sum())
    add_tot = int(b[b['boosted']]['suppl_mentions'].sum()); main_tot = int(b[b['ait']]['mentions'].sum())

    # ---- Section 3: insights ----
    insights = [
        f"Original pipeline could use only <b>{usable:,}/{n:,} ({100*usable/n:.1f}%)</b> of curated pairs "
        f"(open-access AND gene in the main text). <b>{not_oa:,} ({100*not_oa/n:.1f}%)</b> are not in the "
        f"PMC OA subset; <b>{no_mention:,} ({100*no_mention/n:.1f}%)</b> are OA but the gene is absent from "
        f"the main text.",
        f"<b>Unlock:</b> supplements recover <b>{unlocked:,}</b> of the {no_mention:,} no-mention pairs "
        f"(+{100*unlocked/usable:.1f}% over the usable baseline).",
        f"<b>Boost:</b> {boosted:,}/{boost_pairs:,} ({100*boosted/boost_pairs:.0f}%) of in-text genes gain "
        f"supplement evidence, adding {add_tot:,} mentions on top of {main_tot:,} (+{100*add_tot/main_tot:.0f}%).",
        "Recovery concentrates in <b>multi-gene papers</b> (see paper-size tables) — i.e. high-throughput "
        "genome-annotation / screen papers whose data lives in supplementary tables.",
        "The 'not open-access' count is a soft upper bound: a sampled audit found the availability check "
        "produced transient false negatives (fixed with fetch retries), so true OA coverage is somewhat higher.",
        "Pilot end-to-end run of unlocked genes produced correct product descriptions (e.g. COX IX, DRC4, "
        "sah-4, AP2XII-1); the pipeline's own 'mentioned in passing' QC filters weak single-hit unlocks.",
    ]
    if fttabs:
        top = next(iter(fttabs.values())).sort_values('mentions', ascending=False)
        insights.insert(3, f"<b>Where the evidence lives:</b> supplement mentions come mainly from "
                           f"<b>{top.index[0]}</b> files ({top['mentions_%'].iloc[0]:.0f}%); "
                           f"full breakdown in the file-type table/figure.")

    def _html_tbl(df):
        return df.to_html(border=0, classes='t')

    figs = [('fig_coverage_by_db.png', 'Coverage by database'),
            ('fig_coverage_by_papersize.png', 'Coverage by paper size'),
            ('fig_original_vs_unlocked.png', 'Original usable vs supplement-unlocked (overall + by database)'),
            ('fig_impact_by_papersize.png', 'Unlock/boost by paper size'),
            ('fig_filetype.png', 'Mentions by file type')]
    imgs = "".join(f'<figure><img src="{_emb(os.path.join(OUTDIR, f))}" style="max-width:100%"><figcaption>{c}</figcaption></figure>'
                   for f, c in figs if os.path.exists(os.path.join(OUTDIR, f)))

    html = f"""<style>body{{font-family:system-ui,Arial;max-width:1000px;margin:auto;padding:1em}}
table.t{{border-collapse:collapse;font-size:13px}}.t td,.t th{{border:1px solid #ddd;padding:3px 8px}}
figure{{margin:1em 0}}figcaption{{color:#666;font-size:12px}}</style>
<h1>Supplementary-materials scanning: pipeline limitations & recovery</h1>
<p>Curated set: {n:,} gene-paper pairs across {pre['pmid_CLEAN'].nunique():,} papers.</p>
<h2>1. Original-pipeline limitations</h2>
<ul>{''.join(f'<li>{CAT_LBL[c]}: <b>{int(by_cat[c]):,}</b> ({100*by_cat[c]/n:.1f}%)</li>' for c in CATS)}</ul>
<h3>By database</h3>{_html_tbl(baseline_db)}
<h3>By paper size (genes/paper)</h3>{_html_tbl(baseline_bucket.assign(papers=papers_per_bucket))}
<h3>By genus (top 12)</h3>{_html_tbl(baseline_genus)}
<h2>2. Supplement impact</h2>
<h3>By database</h3>{_html_tbl(impact_db)}
<h3>By paper size</h3>{_html_tbl(impact_bucket)}
{('<h3>Mentions by file type</h3>' + ''.join(_html_tbl(t) for t in fttabs.values())) if fttabs else '<p><i>file-type breakdown pending scan</i></p>'}
<h2>3. Insights (pipeline-development notes)</h2>
<ul>{''.join(f'<li>{x}</li>' for x in insights)}</ul>
<h2>Figures</h2>{imgs}
"""
    open(os.path.join(OUTDIR, 'report.html'), 'w', encoding='utf-8').write(html)

    print("BASELINE:", dict(by_cat.astype(int)))
    print("\nby paper size:\n", baseline_bucket.assign(papers=papers_per_bucket).to_string())
    print("\nIMPACT by paper size:\n", impact_bucket.to_string())
    print("\nfiletype ready:", bool(fttabs))
    print("saved ->", OUTDIR)


if __name__ == '__main__':
    main()
