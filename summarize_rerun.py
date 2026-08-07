"""
summarize_rerun.py  <input.csv> <cache_dir>

Post-run report over the supplement-aware batch. Categorises every input pair by outcome and, in
particular, flags pairs that failed on an LLM GUARDRAIL (empty output / content-filter / refusal
signature: blank raw_response, <=2 completion tokens, $0 cost, or an error naming content_filter /
refusal / "JSON parsing failed after retries"). Writes:
  <cache_dir>/../rerun_summary.txt        human-readable summary
  <cache_dir>/../guardrail_failures.csv   the guardrail-blocked pairs (pmid, gene, db, step)
  <cache_dir>/../outcomes.csv             per-pair outcome table
"""
import os
import sys
import json
import glob
import pandas as pd

STEPS = ['getGeneSummary', 'generatePDs', 'verifyPDs']


def _looks_guardrail(res):
    data = res.get('data') if isinstance(res.get('data'), dict) else {}
    usage = res.get('usage') if isinstance(res.get('usage'), dict) else {}
    raw = data.get('raw_response', None)
    ct = usage.get('completion_tokens', usage.get('output', None))
    err = (str(data.get('error', '')) + ' ' + str(res.get('error', ''))).lower()
    return (raw == '' or (isinstance(ct, int) and ct <= 2)
            or 'content_filter' in err or 'refus' in err
            or 'json parsing failed after retries' in err)


def _step_state(step_node, gene):
    """Return ('success'|'guardrail'|'fail'|None, in_passing_bool) for a gene at one step."""
    g = step_node.get(gene) if isinstance(step_node, dict) else None
    if not isinstance(g, dict):
        return None, False
    in_passing = False
    # success?
    for mk, res in g.items():
        if isinstance(res, dict) and res.get('success') and res.get('data'):
            data = res['data'] if isinstance(res['data'], dict) else {}
            if data.get('only_in_passing') is True:
                in_passing = True
            return 'success', in_passing
    # failure kind
    kind = None
    for mk, res in g.items():
        if isinstance(res, dict) and res.get('success') is False:
            kind = 'guardrail' if _looks_guardrail(res) else 'fail'
            if kind == 'guardrail':
                break
    return kind, in_passing


def classify(paper, gene):
    s, in_passing = _step_state(paper.get('getGeneSummary', {}), gene)
    gp, _ = _step_state(paper.get('generatePDs', {}), gene)
    vp, _ = _step_state(paper.get('verifyPDs', {}), gene)
    if vp == 'success':
        return 'completed', ''
    if gp == 'success':
        return 'pds_no_verify', ''
    if s == 'success':
        if in_passing:
            return 'in_passing', ''
        if gp == 'guardrail':
            return 'guardrail_failed', 'generatePDs'
        if gp == 'fail':
            return 'other_failed', 'generatePDs'
        return 'summary_only', ''
    if s == 'guardrail':
        return 'guardrail_failed', 'getGeneSummary'
    if s == 'fail':
        return 'other_failed', 'getGeneSummary'
    return 'not_processed', ''


def main():
    inp, cache = sys.argv[1], sys.argv[2]
    root = os.path.dirname(cache.rstrip('/\\'))
    want = pd.read_csv(inp, dtype=str)
    db_of = {(r['Gene ID'], str(r['pmid_CLEAN'])): r['Database'] for _, r in want.iterrows()}

    seen = {}
    for f in glob.glob(os.path.join(cache, '*.json')):
        pmid = os.path.splitext(os.path.basename(f))[0]
        try:
            paper = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        genes = set()
        for st in STEPS:
            node = paper.get(st, {})
            if isinstance(node, dict):
                genes.update(k for k in node if isinstance(node.get(k), dict))
        for gene in genes:
            outcome, step = classify(paper, gene)
            seen[(gene, str(pmid))] = (outcome, step)

    rows = []
    for (gene, pmid), db in db_of.items():
        outcome, step = seen.get((gene, pmid), ('not_processed', ''))
        rows.append({'pmid': pmid, 'gene_id': gene, 'database': db,
                     'outcome': outcome, 'guardrail_step': step})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, 'outcomes.csv'), index=False)
    gf = out[out.outcome == 'guardrail_failed']
    gf.to_csv(os.path.join(root, 'guardrail_failures.csv'), index=False)

    L = ['=' * 66, 'SUPPLEMENT-AWARE RE-RUN — OUTCOME SUMMARY', '=' * 66]
    L.append(f"input pairs : {len(out)}")
    L.append("\noutcome counts:")
    L.append(out['outcome'].value_counts().to_string())
    L.append(f"\nGUARDRAIL-BLOCKED pairs: {len(gf)}")
    if len(gf):
        L.append("  by step:")
        L.append(gf['guardrail_step'].value_counts().to_string())
        L.append("  by database:")
        L.append(gf['database'].value_counts().to_string())
    L.append("\noutcomes by database:")
    L.append(pd.crosstab(out['database'], out['outcome']).to_string())
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(root, 'rerun_summary.txt'), 'w', encoding='utf-8').write(txt)
    print(f"\nsaved -> {root}/rerun_summary.txt, guardrail_failures.csv, outcomes.csv")


if __name__ == '__main__':
    main()
