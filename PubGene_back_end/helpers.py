# -----------------------------------------------------------------------------
# Shared helper functions for PubMed parsing, VEuPathDB alias lookup,
# prompt utilities, JSON cleanup, and HTML rendering.
# -----------------------------------------------------------------------------

import os
import re
import json
import html
import requests
from typing import List, Dict, Any
from typing import Iterable, Tuple
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import time
# Load environment variables from .env file
# dotenv_path = join(dirname(__file__), 'v2.env')
load_dotenv() #- include API KEys


HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "180"))



# ---------------------------
# Cache helpers
# --------------------------

CACHE_DIR = Path(os.getenv("CACHE_DIR") or Path(__file__).resolve().parent / "cache")
CACHE_VERSION = os.getenv("CACHE_VERSION", "1")

def _ttl(env_name: str) -> int:
    """Return TTL seconds; 0 or missing means 'no expiry'."""
    val = os.getenv(env_name, "0").strip()
    try:
        x = int(val)
        return max(0, x)
    except Exception:
        return 0

def _make_key_hash(key_data: dict) -> str:
    # Ensure stable ordering
    s = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_get(namespace: str, key_data: dict, max_age_s: int = 0) -> dict | None:
    """
    Read JSON payload from cache. Returns None if missing/expired.
    """
    key_data = dict(key_data or {})
    key_data["__v"] = CACHE_VERSION  # bump to invalidate all
    h = _make_key_hash(key_data)
    ns_dir = CACHE_DIR / namespace
    f = ns_dir / f"{h}.json"
    if not f.exists():
        return None
    if max_age_s and (time.time() - f.stat().st_mtime > max_age_s):
        return None
    try:
        with f.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("data") if isinstance(payload, dict) else payload
    except Exception:
        return None

def cache_set(namespace: str, key_data: dict, data: dict) -> None:
    """
    Write JSON payload atomically to cache. Best-effort; errors ignored.
    """
    try:
        key_data = dict(key_data or {})
        key_data["__v"] = CACHE_VERSION
        h = _make_key_hash(key_data)
        ns_dir = CACHE_DIR / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        tmp = ns_dir / f".{h}.tmp.{os.getpid()}"
        final = ns_dir / f"{h}.json"
        blob = {
            "key": key_data,
            "created_at": int(time.time()),
            "data": data,
        }
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False)
        os.replace(str(tmp), str(final))  # atomic on same fs
    except Exception:
        pass



# ---------------------------
# UI new helpers 
# --------------------------
def ui_css() -> str:
    # Hide the native marker and make <summary> look like a blue link.
    return """<style id="ai-css">
    /* Remove summary triangle markers across browsers */
    details > summary { list-style: none; cursor: pointer; }
    details > summary::-webkit-details-marker { display: none; }
    /* Link-like clickable summaries */
    details > summary { color: #0b63ff; text-decoration: underline; }
    details[open] > summary { text-decoration: none; }

    /* Friendly error/info boxes */
    .ai-error { border: 1px solid #ef4444; background:#fff1f2; color:#991b1b; padding:12px; border-radius:8px; }
    .ai-info  { border: 1px solid #2563eb; background:#eff6ff; color:#1e40af; padding:12px; border-radius:8px; }
    
    /* Aliases in paper */
    .aliases-line { margin: 6px 0 10px 0; }

    /* Evidence-based bullets */
    .evidence-summary { margin: 8px 0 0 0; padding-left: 1.25rem; list-style-position: outside; }
    .evidence-summary li { margin-bottom: 8px; }
    .quotes { margin-left: 18px; }
    blockquote { margin: 4px 0; padding-left: 10px; border-left: 3px solid #cbd5e1; }

    /* PD layout polish */
    .pd-item { margin: 10px 0; }
    .pd-title { margin-bottom: 4px; }
    .pd-kv { margin-top: 6px; }
    .pd-label { font-weight: 600; margin-right: 4px; }
    .pd-value code { padding: 1px 4px; border: 1px solid #cbd5e1; border-radius: 4px; background: #f8fafc; }

    /* Indent the Additional Inferences list to align with headings */
    .additional-inferences > ul { padding-left: 1.25rem; margin-left: 0; list-style-position: outside; }
   
    /* PD bullets indent */
    .pd-suggestions ul { padding-left: 1.25rem; margin-left: 0; list-style-position: outside; }
    .pd-suggestions li { margin-bottom: 8px; }

    </style>"""

def render_error_html(message: str, hint: str = "") -> str:
    import html as _html
    msg = _html.escape(message)
    hint_html = f"<br><small>{_html.escape(hint)}</small>" if hint else ""
    return f"<div class='ai-error'><strong>Sorry—something went wrong.</strong><br>{msg}{hint_html}</div>"


# ----------------------------
# PubMed helpers
# ----------------------------
PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
PUBMED_SECTIONS = {"FIG", "TABLE", "RESULTS", "CONCL", "DISCUSSION", "SUPPL"}

def get_pubmed_json(pmid: str) -> Any:
    if pmid.lower().endswith(".pdf"):
        import pymupdf4llm
        file = "/home/tony/uploads/" + pmid
        text = pymupdf4llm.to_markdown(file)
        return {"fulltext": text}
    url = PUBMED_BASE_URL + str(pmid)
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
   
    ct = (r.headers.get("Content-Type") or "").lower()
    if ct.startswith("application/json"):
        return r.json()

    # Heuristics for common PMC responses that come back as text/html or text/plain
    # text = (r.text or "").strip().lower()
    # if "no documents found" in text or "no result" in text:
    #     raise ValueError(f"PMID {pmid} not found in PMC Open Access (BioC). Please ensure you have supplied the correct PMID number.")
    # if "not open" in text or "permission" in text or "restricted" in text:
    #     raise ValueError(f"PMID {pmid} is not open-access in PMC (BioC JSON unavailable).")

    # Default fallback when content isn't JSON
    raise ValueError(f"PMID {pmid} is not available as PMC BioC JSON (full text likely unavailable).")

def parse_pubmed_json(pubmed_json: Any) -> str:

    if "fulltext" in pubmed_json:
        return pubmed_json["fulltext"]
    out_lines = []
    for doc in pubmed_json:
        for document in doc.get("documents", []):
            for passage in document.get("passages", []):
                section = passage.get("infons", {}).get("section_type", "")
                if section and section.upper() in PUBMED_SECTIONS:
                    txt = passage.get("text")
                    if txt:
                        out_lines.append(txt)
    return "\n".join(out_lines)

# ----------------------------
# VEuPathDB alias lookup
# ----------------------------
def get_vpdb_alias(gene_id: str, host_db: str) -> List[str]:
    if not host_db or not gene_id:
        return []

    host_db_lc = host_db.strip().lower()

    host_db_to_path_prefix = {
        "microbiomedb": "mbio",
        "clinepidb": "ce",
        "amoebadb": "amoeba",
        "cryptodb": "cryptodb",
        "fungidb": "fungidb",
        "giardiadb": "giardiadb",
        "hostdb": "hostdb",
        "microsporidiadb": "micro",
        "piroplasmadb": "piro",
        "plasmodb": "plasmo",
        "toxodb": "toxo",
        "trichdb": "trichdb",
        "tritrypdb": "tritrypdb",
        "vectorbase": "vectorbase",
        "orthomcl": "orthomcl",
    }
    host_db_to_project = {
        "microbiomedb": "MicrobiomeDB",
        "clinepidb": "ClinEpiDB",
        "amoebadb": "AmoebaDB",
        "cryptodb": "CryptoDB",
        "fungidb": "FungiDB",
        "giardiadb": "GiardiaDB",
        "hostdb": "HostDB",
        "microsporidiadb": "MicrosporidiaDB",
        "piroplasmadb": "PiroplasmaDB",
        "plasmodb": "PlasmoDB",
        "toxodb": "ToxoDB",
        "trichdb": "TrichDB",
        "tritrypdb": "TriTrypDB",
        "vectorbase": "VectorBase",
        "orthomcl": "OrthoMCL",
    }

    path_prefix = host_db_to_path_prefix.get(host_db_lc)
    project = host_db_to_project.get(host_db_lc)
    if not path_prefix or not project:
        return []

    url = f"https://{host_db_lc}.org/{path_prefix}/service/record-types/gene/records"
    headers = {"content-type": "application/json"}
    data = {
        "attributes": [],
        "primaryKey": [
            {"name": "source_id", "value": gene_id},
            {"name": "project_id", "value": project},
        ],
        "tables": ["AllProducts", "Alias"],
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return []
        payload = resp.json() or {}
    except Exception:
        return []

    alias_set = set()
    for row in (payload.get("tables", {}).get("Alias", []) or []):
        val = row.get("alias")
        if val and val != gene_id:
            alias_set.add(val)
    return sorted(alias_set)

# ----------------------------
# Alias helpers for prompts
# ----------------------------
def _count_substrings(paper: str, alias: str) -> int:
    """
    Count tokenized matches of an alias in paper text, conservatively.
    - For plain <letters><digits> tokens (e.g. 'Nd6', 'DHHC16'), allow an optional hyphen
      between the letter and digit parts (matches 'Nd6' or 'Nd-6').
    - For aliases containing underscores/hyphens/spaces (e.g. 'PF3D7_1133400'),
      treat any run of separators in the alias as '[-_\\s]+' in text so
      'PF3D7_1133400', 'PF3D7-1133400', or 'PF3D7 1133400' all match.
    - Require non-alphanumeric boundaries on both sides to avoid partial hits.
    """
    if not paper or not alias:
        return 0

    # Case 1: letters+digits only → allow optional hyphen before digits
    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", alias)
    if m:
        letters, digits = m.groups()
        inner = f"{re.escape(letters)}-?{re.escape(digits)}"
    else:
        # Case 2: general alias with possible separators → collapse each run of separators
        # into a flexible '[-_\s]+' matcher; escape all other characters.
        parts = []
        i = 0
        while i < len(alias):
            ch = alias[i]
            if ch in "_- ":
                # consume a whole run of separators
                while i < len(alias) and alias[i] in "_- ":
                    i += 1
                parts.append(r"[-_\s]+")
                continue
            parts.append(re.escape(ch))
            i += 1
        inner = "".join(parts)

    pattern = rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])"
    return len(re.findall(pattern, paper, flags=re.IGNORECASE))

def get_gene_synonyms_in_paper(gene_id: str, paper: str, host_db: str) -> List[str]:
    aliases = get_vpdb_alias(gene_id, host_db)
    alias_counts = {a: _count_substrings(paper, a) for a in aliases}
    alias_counts = {a: c for a, c in alias_counts.items() if c > 0}
    top = sorted(alias_counts, key=alias_counts.get, reverse=True)
    return top[:3]

def aliases_mentioned_in_paper(gene_id: str, host_db: str, paper_text: str) -> List[str]:
    """
    Returns names actually mentioned in text:
      - gene_id first (only if mentioned, underscore↔hyphen tolerant),
      - then aliases that occur in the text (by frequency desc, then A→Z),
      - de-duplicated case-insensitively, order-preserving.
    """
    if not paper_text or not gene_id:
        return []

    names: List[str] = []

    # gene_id hits (try hyphen variant too)
    hits = _count_substrings(paper_text, gene_id)
    if hits == 0 and "_" in gene_id:
        hits = _count_substrings(paper_text, gene_id.replace("_", "-"))
    if hits > 0:
        names.append(gene_id)

    # aliases actually mentioned
    aliases = get_gene_synonyms_in_paper(gene_id, paper_text, host_db) or []
    # get_gene_synonyms_in_paper already returns aliases seen in text, sorted by count desc
    # De-dup vs gene_id (case-insensitive) and among aliases
    seen = {n.lower() for n in names}
    for a in aliases:
        if a.lower() not in seen:
            seen.add(a.lower())
            names.append(a)

    return names
# def gene_for_prompt(gene_id: str, synonyms: List[str]) -> str:
#     if not synonyms:
#         return gene_id
#     return f"{gene_id} (also known as " + " or ".join(synonyms) + ")"
def gene_for_prompt(gene_id: str, synonyms: List[str]) -> str:
    """
    Build a prompt name from names actually mentioned in text.
      - If only one name -> return it.
      - If gene_id is among names -> show gene_id first, then the others.
      - Else -> use first mention, then list the rest.
    """
    # de-duplicate case-insensitively, preserve order
    seen = set()
    names = []
    for s in synonyms:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            names.append(s)

    if len(names) == 1:
        return names[0]

    gid_lower = gene_id.lower()
    if any(n.lower() == gid_lower for n in names):
        other_synonyms = [n for n in names if n.lower() != gid_lower]
        return f"{gene_id} (also known as " + " or ".join(other_synonyms) + ")"

    return f"{names[0]} (also known as " + " or ".join(names[1:]) + ")"
# ----------------------------
# Utility
# ----------------------------
def strip_fenced_json(text: str) -> str:
    txt = text.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL)
    return txt

# ----------------------------
# HTML renderers for Extended Summary
# ----------------------------
def render_evidence_based_section(gene_summary: List[Dict[str, Any]]) -> str:
    items = []
    for row in gene_summary:
        bp = html.escape(row.get("bullet_point", ""))
        loc = html.escape(row.get("evidence_location", ""))
        quotes = [html.escape(q) for q in row.get("supporting_quotes", [])]
        quotes_html = "".join([f"<li><blockquote>{q}</blockquote></li>" for q in quotes])
        item_html = (
            f"<li>{bp}"
            f"<details><summary>Show supporting quotes &amp; location</summary>"
            f"<div class='evidence-block'><em>Evidence location:</em> {loc}"
            f"<ul class='quotes'>{quotes_html}</ul></div>"
            f"</details></li>"
        )
        items.append(item_html)
    return "<ul class='evidence-summary'>" + "".join(items) + "</ul>"

def render_additional_inferences(inferences: List[str]) -> str:
    bullets = "".join([f"<li>{html.escape(x)}</li>" for x in inferences])
    return (
        "<details class='additional-inferences'>"
        "<summary>Show additional inferences</summary>"
        f"<ul>{bullets}</ul>"
        "</details>"
    )

def build_extended_summary_html(parsed: Dict[str, Any]) -> str:
    evidence_html = render_evidence_based_section(parsed.get("GeneSummary", []) or [])
    addl_html = render_additional_inferences(parsed.get("AdditionalInferences", []) or [])
    return (
        "<section class='extended-summary'>"
        "<h4>Evidence-based findings</h4>"
        f"{evidence_html}"
	"<h4>Additional inferences</h4>"
        f"{addl_html}"
        "</section>"
    )


### hetting pds from vpdb directly
def _veupath_paths(host_db_lc: str) -> Tuple[str, str]:
    """
    Returns (domain_prefix, path_prefix) for a VEuPathDB 'host_db'.
    """
    # Keep in sync with your other maps
    host_db_to_path_prefix = {
        "microbiomedb": "mbio",
        "clinepidb": "ce",
        "amoebadb": "amoeba",
        "cryptodb": "cryptodb",
        "fungidb": "fungidb",
        "giardiadb": "giardiadb",
        "hostdb": "hostdb",
        "microsporidiadb": "micro",
        "piroplasmadb": "piro",
        "plasmodb": "plasmo",
        "toxodb": "toxo",
        "trichdb": "trichdb",
        "tritrypdb": "tritrypdb",
        "vectorbase": "vectorbase",
        "orthomcl": "orthomcl",
    }
    return host_db_lc, host_db_to_path_prefix.get(host_db_lc, host_db_lc)

def fetch_curated_pd_map(host_db: str, gene_ids: Iterable[str]) -> Dict[str, List[str]]:
    """
    Fetch curated Product Description(s) from VEuPathDB for the given gene ids.
    Returns: { gene_id : [product, ...] }  (empty list if none)
    Uses the 2-step dataset-id + report flow in a single requests.Session().
    """
    if not host_db:
        return {}
    ids = [g for g in (gene_ids or []) if g]
    if not ids:
        return {}

    host_db_lc = host_db.strip().lower()
    domain, path_prefix = _veupath_paths(host_db_lc)

    # Optional cache
    if os.getenv("FORCE_REFRESH") not in ("1", "true", "TRUE"):
        key = {"host_db": host_db_lc, "ids": sorted(ids)}
        cached = cache_get("veupath_pd", key, max_age_s=_ttl("PD_FETCH_CACHE_MAX_AGE"))
        if cached is not None:
            return cached

    s = requests.Session()
    base = f"https://{domain}.org/{path_prefix}"

    # Step 1: create a dataset from the id list
    try:
        ds_payload = {"sourceType": "idList", "sourceContent": {"ids": ids}}
        r1 = s.post(f"{base}/service/users/current/datasets", json=ds_payload, timeout=HTTP_TIMEOUT)
        r1.raise_for_status()
        ds_id = (r1.json() or {}).get("id")
        if not ds_id:
            raise ValueError("VEuPathDB: dataset id not returned")
    except Exception as e:
        # No cache write on failure
        return {g: [f"{e}"] for g in ids}

    # Step 2: request a JSON report over that dataset
    # We ask for the primary_key + AllProducts (which holds 'product')
    report_payload = {
        "searchConfig": {"parameters": {"ds_gene_ids": str(ds_id)}},
        "reportConfig": {"attributes": ["primary_key"], "tables": ["AllProducts"]},
    }

    out: Dict[str, List[str]] = {g: [] for g in ids}
    try:
        r2 = s.post(
            f"{base}/service/record-types/transcript/searches/GeneByLocusTag/reports/standard",
            json=report_payload,
            timeout=HTTP_TIMEOUT,
        )
        r2.raise_for_status()
        data = r2.json() or {}
        for rec in data.get("records", []):
            attrs = rec.get("attributes", {})
            key = attrs.get("primary_key") or attrs.get("source_id")
            if not key:
                continue
            products = []
            # Standard structure: tables -> "AllProducts" -> list of rows, each with 'product'
            t = rec.get("tables", {}).get("AllProducts") or []
            for row in t:
                p = row.get("product")
                if p:
                    products.append(str(p))
            # de-dup preserve order
            seen = set()
            uniq = []
            for p in products:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            out.setdefault(key, [])
            out[key].extend(uniq)
    except Exception:
        # Keep 'out' with empty lists on failure
        pass

    # Cache on success path (best-effort)
    try:
        cache_set("veupath_pd", {"host_db": host_db_lc, "ids": sorted(ids)}, out)
    except Exception:
        pass
    return out

def fetch_curated_pd_single(host_db: str, gene_id: str) -> List[str]:
    """
    Convenience wrapper: single gene → list of curated PD strings (possibly empty).
    """
    m = fetch_curated_pd_map(host_db, [gene_id])
    return m.get(gene_id, [])

print(fetch_curated_pd_single("ToxoDB","TGME49_248640"))