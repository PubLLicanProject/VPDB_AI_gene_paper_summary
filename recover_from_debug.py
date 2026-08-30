"""
recover_from_debug.py  <cache_dir>

Recovers summary-stage parse failures from their debug_failed_parse_*.txt files: the LLM produced a
(near-)valid summary that just didn't parse. For each debug file we extract the original content,
re-parse it (with a brace/bracket truncation-repair fallback), match it back to its gene by comparing
against the gene's stored (truncated) raw_response, and patch the gene's getGeneSummary result to
success=True with the recovered data. The gene can then be re-run cheaply (summary loaded, only PDs +
verify remain). Prints how many recovered; does not touch anything it cannot confidently match+parse.
"""
import os
import re
import sys
import json
import glob

sys.path.insert(0, '.')
from pipeline.utils import extract_json

CACHE = sys.argv[1] if len(sys.argv) > 1 else 'out/VPDB_PD_supplementary_rerun/cache'


def _norm(s):
    return re.sub(r'\s+', '', str(s))


def _try_parse(content):
    p = extract_json(content)
    if isinstance(p, dict):
        return p
    # repair a truncated body: strip fences, trim to a point where braces/brackets balance
    s = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', content.strip(), flags=re.DOTALL)
    for end in range(len(s), max(0, len(s) - 3000), -1):
        frag = s[:end]
        ob = frag.count('{') - frag.count('}')
        oq = frag.count('[') - frag.count(']')
        if ob >= 0 and oq >= 0 and (ob or oq):
            cand = frag.rstrip().rstrip(',') + ']' * oq + '}' * ob
            try:
                d = json.loads(cand)
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
    return None


def main():
    # index failed-summary genes that have content: norm(raw)[:250] -> (file, pmid, gene, model_key)
    index = {}
    for f in glob.glob(os.path.join(CACHE, '*.json')):
        pmid = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        for gene, models in (d.get('getGeneSummary', {}) or {}).items():
            if not isinstance(models, dict):
                continue
            for mk, res in models.items():
                if isinstance(res, dict) and res.get('success') is False:
                    raw = (res.get('data', {}) or {}).get('raw_response', '') if isinstance(res.get('data'), dict) else ''
                    if raw and len(_norm(raw)) > 40:
                        index[_norm(raw)[:250]] = (f, pmid, gene, mk)

    dbg_files = glob.glob(os.path.join(CACHE, 'debug_failed_parse_*.txt'))
    print(f"debug files: {len(dbg_files)} | indexed failed-summary genes with content: {len(index)}")

    recovered, unmatched, unparsed, patched_files = 0, 0, 0, {}
    for dbg in dbg_files:
        txt = open(dbg, encoding='utf-8').read()
        m = re.search(r'Original content:\n(.*?)\n\nSchema:', txt, re.DOTALL)
        content = m.group(1) if m else txt
        key = _norm(content)[:250]
        match = index.get(key)
        if not match:
            for k, v in index.items():          # fuzzy prefix match
                if k[:150] == key[:150]:
                    match = v
                    break
        if not match:
            unmatched += 1
            continue
        parsed = _try_parse(content)
        if not isinstance(parsed, dict) or not parsed.get('GeneSummary'):
            unparsed += 1
            continue
        f, pmid, gene, mk = match
        d = patched_files.get(f) or json.load(open(f, encoding='utf-8'))
        node = d['getGeneSummary'][gene][mk]
        node['success'] = True
        node['data'] = parsed
        node['recovered_from_debug'] = os.path.basename(dbg)
        patched_files[f] = d
        recovered += 1
        print(f"  recovered {pmid} / {gene}  ({len(parsed.get('GeneSummary', []))} bullets)")

    for f, d in patched_files.items():
        json.dump(d, open(f, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"\nRECOVERED {recovered} summaries (patched {len(patched_files)} json files); "
          f"unmatched {unmatched}, unparseable {unparsed}")
    print("Next: re-run these genes to add PDs+verify (summary now loads from cache).")


if __name__ == '__main__':
    main()
