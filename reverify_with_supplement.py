"""
reverify_with_supplement.py  [workers] [--scope all|unlock|unsupported] [--limit N]

Re-runs ONLY the verifyPDs stage for the supplement-affected pairs, with the verifier prompt
enriched with the SAME gene-filtered supplementary evidence the summary stage saw.

WHY: the original pipeline injected supplement text into getGeneSummary only. verifyPDs judged each
PD against the MAIN TEXT alone, so supplement-derived PDs (especially "unlock" genes absent from the
main text) were wrongly marked UNSUPPORTED (~17% of unlock RPDs vs ~2% baseline). This re-runs the
final gate with the missing evidence, reusing the cached getGeneSummary + generatePDs outputs.

SAFE / REVERSIBLE:
- SOURCE cache (out/VPDB_PD_supplementary_rerun/cache) is opened READ-ONLY and never mutated.
- Output goes to a NEW folder (out/VPDB_PD_verify_reverify/cache). Each paper JSON is copied; the
  original verify node is preserved under `verifyPDs_preSUPP`, the enriched result overwrites
  `verifyPDs`. So every file carries both before + after.
- Resumable: a gene already carrying `verifyPDs_preSUPP` in the output is skipped.
- A per-gene before/after CSV is written for analysis.

Scope (default = all): which genes to re-verify among those with non-empty supplement text.
  all         : every gene the supplement contributes evidence to (catches UNSUPPORTED + silent downgrades)
  unlock      : only genes absent from the main text (from the preprocessed flag)
  unsupported : only genes whose ORIGINAL RPD came back UNSUPPORTED/empty
"""
import os
import sys
import csv
import json
import copy
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
import pandas as pd
import STEP_3_batch_processing as S3

SRC = Path('out/VPDB_PD_supplementary_rerun/cache')
DST = Path('out/VPDB_PD_verify_reverify/cache')
PREPROC = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
RESULTS = 'out/VPDB_PD_verify_reverify/reverify_comparison.csv'
_lock = threading.Lock()


def _norm(pmid):
    return str(pmid).split('.')[0].strip()


def load_gene_meta():
    """(pmid, gene_id) -> {'db': host_db, 'unlock': bool} from the preprocessed CSV."""
    meta = {}
    with open(PREPROC, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pmid = _norm(r.get('pmid_CLEAN'))
            gid = r.get('Gene ID')
            if not pmid or not gid:
                continue
            ait = str(r.get('alias_in_text', '')).upper() == 'TRUE'
            try:
                sm = float(r.get('suppl_mentions') or 0)
            except ValueError:
                sm = 0
            meta[(pmid, gid)] = {'db': r.get('Database', ''), 'unlock': (not ait) and sm > 0}
    return meta


def _rpd_status(node_data):
    if not isinstance(node_data, dict):
        return '', ''
    rpd = node_data.get('RPD')
    if isinstance(rpd, dict):
        return str(rpd.get('verification_status', '')).upper(), str(rpd.get('description', ''))
    return '', ''


def reverify_paper(pmid, meta, scope):
    """Return list of comparison rows for one paper; writes the enriched output JSON."""
    src_fp = SRC / f"{pmid}.json"
    dst_fp = DST / f"{pmid}.json"

    # Resume from a partially-written output if present, else start from the source copy.
    if dst_fp.exists():
        paper = json.load(open(dst_fp, encoding='utf-8'))
    else:
        paper = json.load(open(src_fp, encoding='utf-8'))

    gen_node = paper.get('generatePDs', {})
    ver_node = paper.setdefault('verifyPDs', {})
    ver_pre_node = paper.setdefault('verifyPDs_preSUPP', {})
    if not gen_node:
        return []

    # Genes worth re-verifying: those that have generated PDs.
    genes = [g for g in gen_node if g != 'gene_ID']
    if not genes:
        return []

    # Fetch paper text once per paper.
    try:
        paper_text = S3.get_paper_text(pmid)
    except Exception as e:
        return [{'pmid': pmid, 'gene_id': g, 'action': 'skip', 'note': f'paper fetch failed: {e}'}
                for g in genes]

    rows = []
    changed_any = False
    for gene in genes:
        # Already re-verified in a previous run of THIS script?
        if gene in ver_pre_node:
            continue

        km = meta.get((pmid, gene), {})
        host_db = km.get('db', '')
        is_unlock = km.get('unlock', False)

        gmodels = gen_node.get(gene, {})
        # pick the generatePDs model block that succeeded
        gmk = next((k for k in gmodels if k != 'gene_ID' and isinstance(gmodels[k], dict)
                    and gmodels[k].get('success')), None)
        if not gmk:
            continue
        pd_data = gmodels[gmk].get('data')
        if not isinstance(pd_data, dict) or not pd_data.get('PDs'):
            continue

        # Original verify status (for scope filter + before/after).
        vmodels = ver_node.get(gene, {})
        vmk = next((k for k in vmodels if k != 'gene_ID' and isinstance(vmodels[k], dict)), None)
        old_status, old_desc = _rpd_status(vmodels.get(vmk, {}).get('data') if vmk else {})

        if scope == 'unlock' and not is_unlock:
            continue
        if scope == 'unsupported' and old_status not in ('UNSUPPORTED', ''):
            continue

        # Gene-filtered supplement evidence (same call the summary stage used).
        try:
            aliases = S3.get_gene_synonyms(gene, paper_text, host_db) if host_db else None
        except Exception:
            aliases = None
        try:
            suppl = S3.get_supplementary_text(pmid, gene, aliases, host_db,
                                              caps=(S3.SUPPLEMENTARY_CAPS or None))
        except Exception:
            suppl = ""

        if not suppl:
            # Supplement contributed nothing for this gene -> verifier was not actually deprived.
            rows.append({'pmid': pmid, 'gene_id': gene, 'action': 'skip_no_suppl',
                         'is_unlock': is_unlock, 'old_status': old_status})
            continue

        gene_display = f"{gene}, also known as {', '.join(aliases)}" if aliases else gene
        new_result, usage, secs = S3.verifyPDs(
            suggested_pds=pd_data,
            paper_text=paper_text,
            gene_text=gene_display,
            use_caching=True,
            supplementary_text=suppl,
        )
        if not new_result or not isinstance(new_result, dict) or 'error' in new_result:
            rows.append({'pmid': pmid, 'gene_id': gene, 'action': 'reverify_failed',
                         'is_unlock': is_unlock, 'old_status': old_status})
            continue

        # Preserve the original verify node, then overwrite with the enriched one.
        vmodel_key = vmk or S3.PD_VERIFIER_MODEL
        if gene in ver_node:
            ver_pre_node[gene] = copy.deepcopy(ver_node[gene])
        else:
            ver_pre_node[gene] = {'gene_ID': gene}
        ver_node[gene] = {
            'gene_ID': gene,
            vmodel_key: {'model': vmodel_key, 'success': True, 'data': new_result,
                         'usage': usage or {}, 'seconds': secs},
        }
        new_status, new_desc = _rpd_status(new_result)
        changed_any = True
        rows.append({
            'pmid': pmid, 'gene_id': gene, 'action': 'reverified', 'is_unlock': is_unlock,
            'old_status': old_status, 'new_status': new_status,
            'rescued': old_status in ('UNSUPPORTED', '') and new_status == 'SUPPORTED',
            'desc_changed': old_desc.strip() != new_desc.strip(),
            'old_desc': old_desc, 'new_desc': new_desc,
        })

    if changed_any or not dst_fp.exists():
        with _lock:
            with open(dst_fp, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)
    return rows


def main():
    args = sys.argv[1:]
    workers = 8
    scope = 'all'
    limit = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--scope':
            scope = args[i + 1]; i += 2
        elif a == '--limit':
            limit = int(args[i + 1]); i += 2
        elif a.isdigit():
            workers = int(a); i += 1
        else:
            i += 1
    assert scope in ('all', 'unlock', 'unsupported'), f"bad scope {scope}"

    DST.mkdir(parents=True, exist_ok=True)
    S3.FETCH_SUPPLEMENTARY = True
    meta = load_gene_meta()

    pmids = sorted({_norm(fp.stem) for fp in SRC.glob('*.json')})
    if limit:
        pmids = pmids[:limit]
    print(f"{len(pmids)} papers | scope={scope} | {workers} threads", flush=True)

    all_rows, done = [], [0]

    def work(pmid):
        try:
            r = reverify_paper(pmid, meta, scope)
        except Exception as e:
            r = [{'pmid': pmid, 'gene_id': '', 'action': 'paper_error', 'note': str(e)}]
        with _lock:
            done[0] += 1
            all_rows.extend(r)
            resc = sum(1 for x in all_rows if x.get('rescued'))
            rev = sum(1 for x in all_rows if x.get('action') == 'reverified')
            if done[0] % 25 == 0:
                pd.DataFrame(all_rows).to_csv(RESULTS, index=False)
                print(f"  [{done[0]}/{len(pmids)}] papers | reverified={rev} rescued={resc}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p) for p in pmids]
        for f in as_completed(futs):
            f.result()

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS, index=False)
    rev = df[df['action'] == 'reverified'] if len(df) else df
    print("\n" + "=" * 60)
    print(f"RE-VERIFY DONE ({scope}) -> {DST}")
    if len(rev):
        print(f"  re-verified genes : {len(rev)}")
        print(f"  rescued (UNSUPPORTED/empty -> SUPPORTED): {int(rev['rescued'].sum())}")
        print(f"  description changed: {int(rev['desc_changed'].sum())}")
        print(f"  new-status breakdown: {rev['new_status'].value_counts().to_dict()}")
    print(f"  comparison CSV -> {RESULTS}", flush=True)


if __name__ == '__main__':
    main()
