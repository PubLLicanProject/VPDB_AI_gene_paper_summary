#!/usr/bin/env python
"""
Extract data from pipeline variation comparison results
Handles variation-specific keys like generatePDs_Option_1_baseline
AND PD_without_summary (direct generation without summary step)
"""
import os, json, csv, pathlib, re

# CONFIG
STATUS_DIR = "./out/with_vs_withou_summary_verification_species_balanced"  # Your output directory
OUT_DIR = "./out/with_vs_withou_summary_verification_species_balanced/extracted"
GENE_CSV = pathlib.Path(OUT_DIR, "GeneSummary_rows_all_variations.csv")
PD_CSV = pathlib.Path(OUT_DIR, "PD_rows_all_variations.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Define variation patterns to extract
VARIATION_PATTERNS = [
    "Option_1_baseline",
    "Option_2_verify_summary",
    "Option_3_verify_pds",
    "Option_4_select_pd",
    "Option_2+3_combined",
    "Option_2+4_combined"
]


# UTILITY FUNCTIONS
def safe_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, dict) and "items" in val:
        return val["items"]
    return []


def blank(x):
    return "" if x is None else str(x)


def has_error(data):
    """Check if data contains an error response"""
    if not isinstance(data, dict):
        return False
    return "error" in data or "message" in data or "text" in data


def extract_variation_from_key(key):
    """Extract variation name from keys like 'generatePDs_Option_1_baseline'"""
    for pattern in VARIATION_PATTERNS:
        if pattern in key:
            return pattern
    return "baseline"  # Default for original keys without variation suffix


def _pd_row_core(pmid, gene_id, model, variation, pd_type, pd_dict, schema_followed=True):
    """Core function to create a PD row with variation tracking"""
    if pd_dict is None:
        pd_dict = {}
    elif isinstance(pd_dict, (list, tuple)):
        pd_dict = pd_dict[0] if pd_dict else {}
    elif not isinstance(pd_dict, dict):
        pd_dict = {"description": str(pd_dict)}

    return {
        "pmid": pmid,
        "gene_ID": gene_id,
        "model": model,
        "variation": variation,
        "PD_type": pd_type,
        "description": blank(pd_dict.get("description")),
        "verification_status": blank(pd_dict.get("verification_status")),
        "evidence_code": blank(pd_dict.get("evidence_code")),
        "code_reason": blank(pd_dict.get("code_reason")),
        "audit_status": blank(pd_dict.get("audit_status")),
        "audit_reason": blank(pd_dict.get("audit_reason")),
        "schema_followed": str(schema_followed),
    }


def pd_rows_from_verify(pmid, gene_id, model, variation, vpd_data):
    """Extract PD rows from verifyPDs data with variation tracking"""
    if not isinstance(vpd_data, dict):
        return
    if has_error(vpd_data):
        return

    # Handle RPD
    if "RPD" in vpd_data:
        rpd_schema_ok = isinstance(vpd_data["RPD"], dict)
        yield _pd_row_core(pmid, gene_id, model, variation, "RPD", vpd_data["RPD"], rpd_schema_ok)

    # Handle APDs
    apds = vpd_data.get("APDs", [])

    if apds:
        if isinstance(apds, list):
            for apd in apds:
                apd_schema_ok = isinstance(apd, dict)
                yield _pd_row_core(pmid, gene_id, model, variation, "APD", apd, apd_schema_ok)
        elif isinstance(apds, dict):
            apds_schema_ok = False
            if "description" in apds or "verification_status" in apds:
                yield _pd_row_core(pmid, gene_id, model, variation, "APD", apds, apds_schema_ok)
            else:
                for key, apd_value in apds.items():
                    if isinstance(apd_value, dict):
                        yield _pd_row_core(pmid, gene_id, model, variation, "APD", apd_value, apds_schema_ok)
        else:
            yield _pd_row_core(pmid, gene_id, model, variation, "APD", {"description": str(apds)}, False)


def gene_row_from_summary(pmid, gene_id, model, variation, bullet, schema_followed=True):
    """Create a gene summary row with variation tracking"""
    return {
        "pmid": pmid,
        "gene_ID": gene_id,
        "model": model,
        "variation": variation,
        "bullet_point": blank(bullet.get("bullet_point")),
        "evidence_location": blank(bullet.get("evidence_location")),
        "supporting_quotes": ";;;;;".join(bullet.get("supporting_quotes", [])),
        "verification_status": blank(bullet.get("verification_status", "")),
        "reason": blank(bullet.get("reason", "")),
        "schema_followed": str(schema_followed),
    }


# MAIN EXTRACTION
gene_rows, pd_rows = [], []

for fname in sorted(os.listdir(STATUS_DIR)):
    if not fname.endswith(".json"):
        continue
    pmid = fname.split(".")[0]

    try:
        data = json.load(open(os.path.join(STATUS_DIR, fname), encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"⚠️  Skipped malformed JSON → {fname}")
        continue

    # 0. getGeneSummary → bullet-point rows (shared across all variations)
    for gene, models in data.get("getGeneSummary", {}).items():
        gene_id = models.get("gene_ID", gene)
        for mdl_name, mdl_block in models.items():
            if mdl_name == "gene_ID":
                continue
            if not mdl_block.get("success"):
                continue

            d = mdl_block.get("data", {})
            if has_error(d):
                continue

            schema_ok = True
            if isinstance(d, dict) and "GeneSummary" in d:
                summary_data = d["GeneSummary"]
                if isinstance(summary_data, list):
                    for bullet in summary_data:
                        bullet_schema_ok = isinstance(bullet, dict)
                        gene_rows.append(
                            gene_row_from_summary(pmid, gene_id, mdl_name, "base_summary", bullet, bullet_schema_ok))
                else:
                    schema_ok = False
                    if isinstance(summary_data, dict):
                        gene_rows.append(
                            gene_row_from_summary(pmid, gene_id, mdl_name, "base_summary", summary_data, schema_ok))

    # 1. verifyGeneSummary (shared across variations that use it)
    for gene, models in data.get("verifyGeneSummary", {}).items():
        gene_id = models.get("gene_ID", gene)
        for mdl_name, mdl_block in models.items():
            if mdl_name == "gene_ID":
                continue
            if not mdl_block.get("success"):
                continue

            d = mdl_block.get("data", {})
            if has_error(d):
                continue

            v_summary = d.get("VerifiedSummary") or d if isinstance(d, dict) else d
            schema_ok = isinstance(v_summary, list)

            if schema_ok:
                for bullet in v_summary:
                    bullet_schema_ok = isinstance(bullet, dict)
                    gene_rows.append(
                        gene_row_from_summary(pmid, gene_id, mdl_name, "verified_summary", bullet, bullet_schema_ok))
            elif isinstance(v_summary, dict):
                gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, "verified_summary", v_summary, False))

    # 2. PD_without_summary → RPD / APD rows (direct generation without summary)
    for gene, models in data.get("PD_without_summary", {}).items():
        gene_id = models.get("gene_ID", gene)
        for mdl_name, mdl_block in models.items():
            if mdl_name == "gene_ID":
                continue
            if not mdl_block.get("success"):
                continue

            d = mdl_block.get("data", {})
            if has_error(d):
                continue

            # PD_without_summary uses same format as verifyPDs (RPD + APDs)
            for row in pd_rows_from_verify(pmid, gene_id, mdl_name, "no_summary_step", d):
                pd_rows.append(row)

    # 3. generatePDs_* → raw_PD rows (variation-specific)
    for key in data.keys():
        if key.startswith("generatePDs"):
            variation = extract_variation_from_key(key)

            for gene, models in data.get(key, {}).items():
                gene_id = models.get("gene_ID", gene)
                for mdl_name, mdl_block in models.items():
                    if mdl_name == "gene_ID":
                        continue
                    if not mdl_block.get("success"):
                        continue

                    d = mdl_block.get("data", {})
                    if has_error(d):
                        continue

                    pds_data = d.get("PDs", [])
                    schema_ok = isinstance(pds_data, list)

                    if schema_ok:
                        for raw_pd in pds_data:
                            pd_schema_ok = isinstance(raw_pd, dict)
                            pd_rows.append(
                                _pd_row_core(pmid, gene_id, mdl_name, variation, "raw_PD", raw_pd, pd_schema_ok)
                            )
                    elif isinstance(pds_data, dict):
                        pd_rows.append(
                            _pd_row_core(pmid, gene_id, mdl_name, variation, "raw_PD", pds_data, False)
                        )

    # 4. verifyPDs_* → RPD / APD rows (variation-specific)
    for key in data.keys():
        if key.startswith("verifyPDs"):
            variation = extract_variation_from_key(key)

            for gene, models in data.get(key, {}).items():
                gene_id = models.get("gene_ID", gene)
                for mdl_name, mdl_block in models.items():
                    if mdl_name == "gene_ID":
                        continue
                    if not mdl_block.get("success"):
                        continue

                    d = mdl_block.get("data", {})
                    if has_error(d):
                        continue

                    for row in pd_rows_from_verify(pmid, gene_id, mdl_name, variation, d):
                        pd_rows.append(row)

    # 5. selectPD_* → selected_PD rows (variation-specific)
    for key in data.keys():
        if key.startswith("selectPD"):
            variation = extract_variation_from_key(key)

            for gene, models in data.get(key, {}).items():
                gene_id = models.get("gene_ID", gene)
                for mdl_name, mdl_block in models.items():
                    if mdl_name == "gene_ID":
                        continue
                    if not mdl_block.get("success"):
                        continue

                    d = mdl_block.get("data", {})
                    if has_error(d):
                        continue

                    sel_pd = d.get("recommended_product") or d.get("selected_PD") or d.get("selectedPD")
                    if sel_pd:
                        schema_ok = isinstance(sel_pd, dict)
                        pd_rows.append(
                            _pd_row_core(pmid, gene_id, mdl_name, variation, "selected_PD", sel_pd, schema_ok)
                        )


# WRITE CSVs
def write_csv(path, rows, fieldnames):
    if not rows:
        print(f"⚠️  No rows to write for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    GENE_CSV,
    gene_rows,
    ["pmid", "gene_ID", "model", "variation", "bullet_point", "evidence_location",
     "supporting_quotes", "verification_status", "reason", "schema_followed"],
)

write_csv(
    PD_CSV,
    pd_rows,
    ["pmid", "gene_ID", "model", "variation", "PD_type", "description",
     "verification_status", "evidence_code", "code_reason",
     "audit_status", "audit_reason", "schema_followed"],
)

# Summary statistics
schema_issues_gene = sum(1 for row in gene_rows if row.get("schema_followed") == "False")
schema_issues_pd = sum(1 for row in pd_rows if row.get("schema_followed") == "False")

print(f"✅  Gene-summary rows written → {GENE_CSV}  ({len(gene_rows)} rows)")
print(f"    Schema issues detected: {schema_issues_gene} rows")
print(f"✅  Product-description rows written → {PD_CSV}  ({len(pd_rows)} rows)")
print(f"    Schema issues detected: {schema_issues_pd} rows")

# Variation-specific statistics
print("\n📊 Breakdown by variation:")
from collections import Counter

gene_variations = Counter(row["variation"] for row in gene_rows)
pd_variations = Counter(row["variation"] for row in pd_rows)

print("\nGene summary rows by variation:")
for var, count in sorted(gene_variations.items()):
    print(f"  {var}: {count}")

print("\nPD rows by variation:")
for var, count in sorted(pd_variations.items()):
    print(f"  {var}: {count}")

# Optional: Print models with schema issues
if schema_issues_gene > 0 or schema_issues_pd > 0:
    gene_issue_models = set(row["model"] for row in gene_rows if row.get("schema_followed") == "False")
    pd_issue_models = set(row["model"] for row in pd_rows if row.get("schema_followed") == "False")
    all_issue_models = gene_issue_models.union(pd_issue_models)
    if all_issue_models:
        print(f"\n⚠️  Models with schema compliance issues: {', '.join(sorted(all_issue_models))}")