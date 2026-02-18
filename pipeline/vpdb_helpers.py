"""
Helper functions to fetch alias and check if in paper

"""

import requests
import re

from typing import List, Dict, Tuple, Optional, Any





HTTP_TIMEOUT = 180

def get_vpdb_alias(gene_id: str, host_db: str) -> List[str]:
    """Fetch aliases from VEuPathDB."""
    if not host_db or not gene_id:
        return []

    host_db_lc = host_db.strip().lower()

    db_mappings = {
        "microbiomedb": ("mbio", "MicrobiomeDB"),
        "clinepidb": ("ce", "ClinEpiDB"),
        "amoebadb": ("amoeba", "AmoebaDB"),
        "cryptodb": ("cryptodb", "CryptoDB"),
        "fungidb": ("fungidb", "FungiDB"),
        "giardiadb": ("giardiadb", "GiardiaDB"),
        "hostdb": ("hostdb", "HostDB"),
        "microsporidiadb": ("micro", "MicrosporidiaDB"),
        "piroplasmadb": ("piro", "PiroplasmaDB"),
        "plasmodb": ("plasmo", "PlasmoDB"),
        "toxodb": ("toxo", "ToxoDB"),
        "trichdb": ("trichdb", "TrichDB"),
        "tritrypdb": ("tritrypdb", "TriTrypDB"),
        "vectorbase": ("vectorbase", "VectorBase"),
        "orthomcl": ("orthomcl", "OrthoMCL"),
    }

    if host_db_lc not in db_mappings:
        return []

    path_prefix, project = db_mappings[host_db_lc]
    url = f"https://{host_db_lc}.org/{path_prefix}/service/record-types/gene/records"

    data = {
        "attributes": [],
        "primaryKey": [
            {"name": "source_id", "value": gene_id},
            {"name": "project_id", "value": project},
        ],
        "tables": ["AllProducts", "Alias"],
    }

    try:
        resp = requests.post(url, headers={"content-type": "application/json"},
                             json=data, timeout=HTTP_TIMEOUT)
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


def _count_substrings(paper: str, alias: str) -> int:
    """Count occurrences of alias in paper with flexible matching."""
    if not paper or not alias:
        return 0

    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", alias)
    if m:
        letters, digits = m.groups()
        inner = f"{re.escape(letters)}-?{re.escape(digits)}"
    else:
        parts = []
        i = 0
        while i < len(alias):
            if alias[i] in "_- ":
                while i < len(alias) and alias[i] in "_- ":
                    i += 1
                parts.append(r"[-_\s]+")
                continue
            parts.append(re.escape(alias[i]))
            i += 1
        inner = "".join(parts)

    pattern = rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])"
    return len(re.findall(pattern, paper, flags=re.IGNORECASE))


def get_gene_synonyms(gene_id: str, paper: str, host_db: str) -> List[str]:
    """Get top 3 most common aliases in paper."""
    aliases = get_vpdb_alias(gene_id, host_db)
    alias_count = {a: _count_substrings(paper, a) for a in aliases if a != gene_id}
    sorted_aliases = sorted(alias_count, key=alias_count.get, reverse=True)
    return sorted_aliases[:3]


def check_gene_in_text(gene_id: str, paper: str, host_db: str) -> Tuple[bool, int]:
    """
    Check if gene ID or any of its aliases appear in paper text.

    Args:
        gene_id: Gene identifier
        paper: Paper text to search in
        host_db: Database to fetch aliases from

    Returns:
        Tuple of (found: bool, total_mentions: int)
    """
    if not paper:
        return False, 0

    # Check gene ID itself
    gene_count = _count_substrings(paper, gene_id)

    # Check aliases
    aliases = get_vpdb_alias(gene_id, host_db)
    alias_counts = {a: _count_substrings(paper, a) for a in aliases}

    total_mentions = gene_count + sum(alias_counts.values())
    found = total_mentions > 0

    return found, total_mentions
