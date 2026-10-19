"""
get_user_comments_by_id.py  INPUT  OUT_DIR  [--sleep SECONDS]

Fetch VEuPathDB user comments by Comment ID and capture everything the API returns.

INPUT: a .tsv/.txt/.csv with a 'Comment ID' column (e.g. a VEuPathDB user-comment list export).
Each comment is fetched individually from the portal endpoint
    https://veupathdb.org/veupathdb/service/user-comments/{id}
which (unlike the per-site endpoint) also returns `aiProvenance` — the original AI-generated
headline/content for comments drafted with the PD tool.

OUT_DIR gets:
  raw/{comment_id}.json        full API response, untouched
  user_comments_full.tsv       one row per comment: the 31 columns used by
                               all_user_comments_with_pmid_fetched.tsv (same names, same order),
                               followed by every other field returned by the API, flattened
                               ('api.<path>'; lists of objects as JSON strings)
  fetch_status.csv             per-ID status (fetched / already_fetched / http_<code> / error)

Requires VEUPATHDB_API_KEY in .env (the service 401s without it since the Aug 2026 release); sent as
`Authorization: Bearer <key>`. Resumable: IDs with an existing raw/{id}.json are not re-fetched.
Successor to get__all_UserComments_for_ORganism_withContent.py (LLM_testbed_JT), which fetched per gene
from each site and kept only a fixed column subset.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

PORTAL_URL = "https://veupathdb.org/veupathdb/service/user-comments/{}"
HTTP_TIMEOUT = 60


def _join(items, key=None, sep=', '):
    vals = [(i.get(key) if key else i) for i in (items or [])]
    return sep.join(str(v) for v in vals if v not in (None, ''))


def legacy_columns(c: dict) -> dict:
    """The 31 columns the API returns, same names and order.

    The distributed all_user_comments_with_pmid_fetched.tsv has 27: the four
    commenter-identity columns below are stripped before release.
    """
    refs = c.get('pubMedRefs') or []
    loc = c.get('location') or {}
    ranges = loc.get('ranges') or []
    target = c.get('target') or {}
    author = c.get('author') or {}
    return {
        'Comment ID': c.get('id'),
        'Gene ID': target.get('id'),
        'Database': (c.get('project') or {}).get('name'),
        'Organism': c.get('organism'),
        'Headline': c.get('headline'),
        'Content': c.get('content'),
        'Comment Date': pd.to_datetime(c['commentDate'], unit='ms') if c.get('commentDate') else None,
        'Categories': _join(c.get('categories')),
        # Commenter identity. Fetched because the API returns it, but STRIPPED
        # from the tables distributed with this repository: no pipeline step
        # reads these four fields, and shipping them would republish the names,
        # organisations and internal user IDs of every VEuPathDB commenter.
        # 'PubMed Authors' below is a different thing: publication authors,
        # ordinary bibliographic metadata, and kept.
        'Author First Name': author.get('firstName'),
        'Author Last Name': author.get('lastName'),
        'Author Organization': author.get('organization'),
        'Author User ID': author.get('userId'),
        'Review Status': c.get('reviewStatus'),
        'Conceptual': c.get('conceptual'),
        'PubMed IDs': _join(refs, 'id'),
        'PubMed Titles': _join(refs, 'title', ' | '),
        'PubMed Journals': _join(refs, 'journal', ' | '),
        'PubMed Authors': _join(refs, 'author', ' | '),
        'Related Stable IDs': _join(c.get('relatedStableIds')),
        'Digital Object IDs': _join(c.get('digitalObjectIds')),
        'Project Name': (c.get('project') or {}).get('name'),
        'Project Version': (c.get('project') or {}).get('version'),
        'External Database': (c.get('externalDatabase') or {}).get('name'),
        'External Database Version': (c.get('externalDatabase') or {}).get('version'),
        'Location Reverse': loc.get('reverse'),
        'Location Coordinate Type': loc.get('coordinateType'),
        'GenBank Accessions': _join(c.get('genBankAccessions')),
        'Target ID': target.get('id'),
        'Target Type': target.get('type'),
        'Location Start': ranges[0].get('start') if ranges else None,
        'Location End': ranges[0].get('end') if ranges else None,
    }


def flatten(obj, prefix='api.') -> dict:
    """Every field: nested objects -> dotted keys; scalar lists -> '; '-joined; object lists -> JSON."""
    out = {}
    for k, v in obj.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        elif isinstance(v, list):
            if any(isinstance(x, (dict, list)) for x in v):
                out[key] = json.dumps(v, ensure_ascii=False)
            else:
                out[key] = '; '.join(str(x) for x in v)
            out[key + '.count'] = len(v)
        else:
            out[key] = v
    return out


def read_ids(path: str) -> list:
    sep = ',' if path.lower().endswith('.csv') else '\t'
    df = pd.read_csv(path, sep=sep, dtype=str)
    if 'Comment ID' not in df.columns:
        sys.exit(f"'Comment ID' column not found in {path} (columns: {list(df.columns)})")
    return list(dict.fromkeys(df['Comment ID'].dropna().str.strip()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('out_dir')
    ap.add_argument('--sleep', type=float, default=0.5)
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv('VEUPATHDB_API_KEY', '').strip()
    if not key:
        sys.exit('VEUPATHDB_API_KEY missing from .env — the user-comments service requires it.')
    headers = {'Authorization': f'Bearer {key}'}

    raw_dir = os.path.join(args.out_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    ids = read_ids(args.input)
    print(f'{len(ids)} comment IDs from {args.input}')

    status = []
    for n, cid in enumerate(ids, 1):
        raw_path = os.path.join(raw_dir, f'{cid}.json')
        if os.path.exists(raw_path):
            status.append({'Comment ID': cid, 'fetch_status': 'already_fetched', 'detail': ''})
            continue
        try:
            r = requests.get(PORTAL_URL.format(cid), headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 401:
                sys.exit('401 Unauthorized — check VEUPATHDB_API_KEY (stopping; nothing else will work).')
            if r.status_code != 200:
                status.append({'Comment ID': cid, 'fetch_status': f'http_{r.status_code}', 'detail': r.text[:200]})
                print(f'[{n}/{len(ids)}] {cid}: HTTP {r.status_code}')
            else:
                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump(r.json(), f, ensure_ascii=False, indent=1)
                status.append({'Comment ID': cid, 'fetch_status': 'fetched', 'detail': ''})
                print(f'[{n}/{len(ids)}] {cid}: ok')
        except Exception as e:
            status.append({'Comment ID': cid, 'fetch_status': 'error', 'detail': str(e)[:200]})
            print(f'[{n}/{len(ids)}] {cid}: error {e}')
        time.sleep(args.sleep)

    rows = []
    for cid in ids:
        raw_path = os.path.join(raw_dir, f'{cid}.json')
        if os.path.exists(raw_path):
            c = json.load(open(raw_path, encoding='utf-8'))
            rows.append({**legacy_columns(c), **flatten(c)})

    out_tsv = os.path.join(args.out_dir, 'user_comments_full.tsv')
    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    st = pd.DataFrame(status)
    st.to_csv(os.path.join(args.out_dir, 'fetch_status.csv'), index=False)
    print(f'\n{len(rows)}/{len(ids)} comments -> {out_tsv}')
    print(st['fetch_status'].value_counts().to_string())


if __name__ == '__main__':
    main()
