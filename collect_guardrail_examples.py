"""
collect_guardrail_examples.py   [out_dir]

Compiles every gene-paper case where a Sonnet-4.5 pipeline stage was blocked by the model's safety
guardrail (empty / content_filter response), across ALL of our Sonnet-4.5 batch caches. Intended to be
shared with collaborators so they can test alternative models / settings on the same inputs.

A "guardrail block" = a stage (getGeneSummary / generatePDs / verifyPDs) saved with success=False whose
signature looks like a refusal rather than a normal parse error:
  - empty raw_response, OR completion/output tokens <= 2, OR 'content_filter'/'refus' in the error.
(Reuses the heuristic from summarize_rerun._looks_guardrail.) Only records saved under a Sonnet-4.5
model key are counted; other models in the same file are ignored. De-duplicated on (pmid, gene, stage)
so the verify_reverify copies of shared stages are not double-counted.

Outputs to out/guardrail_examples/:
  guardrail_blocked_sonnet45.csv   one row per blocked (pmid, gene, stage)
  raw/<pmid>__<gene>__<stage>.txt  the raw model output (when non-empty), for reproduction
  summary.txt                      counts by stage / signature / source / database
No LLM calls; safe to re-run.
"""
import os
import sys
import csv
import json
import glob
import pandas as pd

CACHE_DIRS = [
    'out/VPDB_PD_set_expansion/cache',
    'out/VPDB_PD_supplementary_rerun/cache',
    'out/VPDB_PD_verify_reverify/cache',
    'out/user_comment_evaluation',
    'out/cache',
]
STAGES = ('getGeneSummary', 'generatePDs', 'verifyPDs')
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else 'out/guardrail_examples'


def is_sonnet45(model_key):
    m = str(model_key).lower()
    return 'sonnet-4.5' in m or 'sonnet-4-5' in m


def looks_guardrail(res):
    data = res.get('data') if isinstance(res.get('data'), dict) else {}
    usage = res.get('usage') if isinstance(res.get('usage'), dict) else {}
    raw = data.get('raw_response', None)
    ct = usage.get('completion_tokens', usage.get('output_tokens', usage.get('output', None)))
    err = (str(data.get('error', '')) + ' ' + str(res.get('error', ''))).lower()
    return (raw == '' or (isinstance(ct, int) and ct <= 2)
            or 'content_filter' in err or 'refus' in err
            or 'json parsing failed after retries' in err)


def signature(res):
    data = res.get('data') if isinstance(res.get('data'), dict) else {}
    usage = res.get('usage') if isinstance(res.get('usage'), dict) else {}
    raw = data.get('raw_response', None)
    ct = usage.get('completion_tokens', usage.get('output_tokens', usage.get('output', None)))
    err = (str(data.get('error', '')) + ' ' + str(res.get('error', ''))).lower()
    if 'content_filter' in err:
        return 'content_filter'
    if raw == '':
        return 'empty_response'
    if isinstance(ct, int) and ct <= 2:
        return 'zero_output_tokens'
    if 'json parsing failed after retries' in err:
        return 'parse_fail_after_retries'
    return 'other'


def load_context():
    """(gene, pmid) -> {database, organism, product} from any curated CSVs we have (best-effort)."""
    ctx = {}
    def add(g, p, db='', org='', prod=''):
        k = (str(g).strip(), str(p).split('.')[0].strip())
        cur = ctx.setdefault(k, {'database': '', 'organism': '', 'product': ''})
        if db and not cur['database']:
            cur['database'] = db
        if org and not cur['organism']:
            cur['organism'] = org
        if prod and not cur['product']:
            cur['product'] = prod
    prio = 'C:/Users/jtzve/Desktop/PD_PIPELINE/curated_data/batch_needs_description_outstanding.tsv'
    if os.path.exists(prio):
        d = pd.read_csv(prio, sep='\t', dtype=str)
        for _, r in d.iterrows():
            add(r.get('gene_id'), r.get('pmid'), r.get('database', ''), r.get('organism', ''), r.get('existing_product', ''))
    for f in glob.glob('curated_data/all_PDs_with_PMID_2026*preprocessed*.csv'):
        try:
            d = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        if 'Gene ID' not in d.columns or 'pmid_CLEAN' not in d.columns:
            continue
        for _, r in d.iterrows():
            add(r.get('Gene ID'), r.get('pmid_CLEAN'), r.get('Database', ''), r.get('Organism', ''), r.get('Product', ''))
    return ctx


def main():
    os.makedirs(os.path.join(OUTDIR, 'raw'), exist_ok=True)
    ctx = load_context()
    seen = set()   # (pmid, gene, stage) dedupe
    rows = []
    for d in CACHE_DIRS:
        for fp in glob.glob(os.path.join(d, '*.json')):
            pmid = os.path.basename(fp).split('.')[0]
            try:
                data = json.load(open(fp, encoding='utf-8'))
            except Exception:
                continue
            for stage in STAGES:
                for gene, gene_node in (data.get(stage) or {}).items():
                    if not isinstance(gene_node, dict):
                        continue
                    for mk, rec in gene_node.items():
                        if mk == 'gene_ID' or not isinstance(rec, dict):
                            continue
                        if not is_sonnet45(mk) or rec.get('success') is not False:
                            continue
                        if not looks_guardrail(rec):
                            continue
                        key = (pmid, gene, stage)
                        if key in seen:
                            continue
                        seen.add(key)
                        rd = rec.get('data') if isinstance(rec.get('data'), dict) else {}
                        usage = rec.get('usage') if isinstance(rec.get('usage'), dict) else {}
                        raw = rd.get('raw_response', '')
                        c = ctx.get((str(gene).strip(), str(pmid).strip()), {})
                        if raw:
                            with open(os.path.join(OUTDIR, 'raw', f"{pmid}__{gene}__{stage}.txt"), 'w', encoding='utf-8') as fh:
                                fh.write(raw)
                        rows.append({
                            'pmid': pmid, 'gene_id': gene, 'stage': stage, 'model': mk,
                            'signature': signature(rec),
                            'output_tokens': usage.get('completion_tokens', usage.get('output_tokens', '')),
                            'seconds': rec.get('seconds', ''),
                            'database': c.get('database', ''), 'organism': c.get('organism', ''),
                            'product': c.get('product', ''),
                            'source_dir': d,
                            'raw_snippet': (raw or '')[:200].replace('\n', ' '),
                        })

    fields = ['pmid', 'gene_id', 'stage', 'signature', 'model', 'output_tokens', 'seconds',
              'database', 'organism', 'product', 'source_dir', 'raw_snippet']
    out_csv = os.path.join(OUTDIR, 'guardrail_blocked_sonnet45.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    df = pd.DataFrame(rows)
    lines = ['=' * 60, f'GUARDRAIL-BLOCKED SONNET-4.5 CASES: {len(rows)}', '=' * 60]
    if len(df):
        lines.append('\nby stage:\n' + df['stage'].value_counts().to_string())
        lines.append('\nby signature:\n' + df['signature'].value_counts().to_string())
        lines.append('\nunique gene-paper pairs blocked: %d' % df.groupby(['pmid', 'gene_id']).ngroups)
        lines.append('\nby database:\n' + df['database'].replace('', '(unknown)').value_counts().to_string())
        lines.append('\nby source_dir:\n' + df['source_dir'].value_counts().to_string())
    txt = '\n'.join(lines)
    open(os.path.join(OUTDIR, 'summary.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)
    print(f"\nwrote {out_csv}  (+ raw/ snippets, summary.txt)")


if __name__ == '__main__':
    main()
