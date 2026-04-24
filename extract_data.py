"""
Extract data from multi-model comparison results
Handles OpenRouter model naming and error responses
"""
import os, json, csv, pathlib

# CONFIG
STATUS_DIR = "./out/summaries"
OUT_DIR = "./out/summaries/extracted"
GENE_CSV = pathlib.Path(OUT_DIR, "Summary_Quotes_by_model_full.csv")
PD_CSV = pathlib.Path(OUT_DIR, "PD_rows_all_models_full.csv")

os.makedirs(OUT_DIR, exist_ok=True)


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


def _pd_row_core(pmid, gene_id, model, pd_type, pd_dict, schema_followed=True):
    """Core function to create a PD row with schema_followed tracking"""
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
        "PD_type": pd_type,
        "description": blank(pd_dict.get("description")),
        "verification_status": blank(pd_dict.get("verification_status")),
        "evidence_code": blank(pd_dict.get("evidence_code")),
        "code_reason": blank(pd_dict.get("code_reason")),
        "audit_status": blank(pd_dict.get("audit_status")),
        "audit_reason": blank(pd_dict.get("audit_reason")),
        "schema_followed": str(schema_followed),
    }


def pd_rows_from_verify(pmid, gene_id, model, vpd_data):
    """Extract PD rows from verifyPDs data with improved APDs handling"""
    if not isinstance(vpd_data, dict):
        return
    if has_error(vpd_data):
        return

    # Track if schema was followed correctly
    schema_issues = []

    # Handle RPD
    if "RPD" in vpd_data:
        rpd_schema_ok = isinstance(vpd_data["RPD"], dict)
        yield _pd_row_core(pmid, gene_id, model, "RPD", vpd_data["RPD"], rpd_schema_ok)

    # Handle APDs with improved logic
    apds = vpd_data.get("APDs", [])
    apds_schema_ok = True

    # Check if APDs is in the expected format (should be a list)
    if apds:  # Only process if APDs exists
        if isinstance(apds, list):
            # Correct format - APDs is a list
            for apd in apds:
                apd_schema_ok = isinstance(apd, dict)
                yield _pd_row_core(pmid, gene_id, model, "APD", apd, apd_schema_ok)
        elif isinstance(apds, dict):
            # Wrong format - APDs is a single dict instead of a list
            # This is the issue with qwen/qwen3-max in your example
            apds_schema_ok = False

            # Check if it looks like a single APD entry
            if "description" in apds or "verification_status" in apds:
                # It's a single APD dict that should have been in a list
                yield _pd_row_core(pmid, gene_id, model, "APD", apds, apds_schema_ok)
            else:
                # It might be a dict containing multiple APDs with keys
                for key, apd_value in apds.items():
                    if isinstance(apd_value, dict):
                        yield _pd_row_core(pmid, gene_id, model, "APD", apd_value, apds_schema_ok)
        else:
            # Unexpected format
            apds_schema_ok = False
            # Try to convert to string representation
            yield _pd_row_core(pmid, gene_id, model, "APD", {"description": str(apds)}, apds_schema_ok)


def gene_row_from_summary(pmid, gene_id, model, bullet, schema_followed=True):
    """Create a gene summary row with schema tracking"""
    return {
        "pmid": pmid,
        "gene_ID": gene_id,
        "model": model,
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
        print("⚠️  Skipped malformed JSON →", fname)
        continue

    # 0. getGeneSummary → bullet-point rows
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
                # Check if GeneSummary is a list as expected
                if isinstance(summary_data, list):
                    for bullet in summary_data:
                        bullet_schema_ok = isinstance(bullet, dict)
                        gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, bullet, bullet_schema_ok))
                else:
                    # Schema not followed - GeneSummary should be a list
                    schema_ok = False
                    if isinstance(summary_data, dict):
                        gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, summary_data, schema_ok))

    # 1. verifyGeneSummary
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
                    gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, bullet, bullet_schema_ok))
            elif isinstance(v_summary, dict):
                gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, v_summary, False))

    # 2. verifyPDs → RPD / APD rows
    for gene, models in data.get("verifyPDs", {}).items():
        gene_id = models.get("gene_ID", gene)
        for mdl_name, mdl_block in models.items():
            if mdl_name == "gene_ID":
                continue
            if not mdl_block.get("success"):
                continue

            d = mdl_block.get("data", {})
            if has_error(d):
                continue

            for row in pd_rows_from_verify(pmid, gene_id, mdl_name, d):
                pd_rows.append(row)

    # 3. selectPD → selected_PD rows
    for sel_key in ("selectPD", "selectPDs"):
        for gene, models in data.get(sel_key, {}).items():
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
                        _pd_row_core(pmid, gene_id, mdl_name, "selected_PD", sel_pd, schema_ok)
                    )

    # 4. generatePDs → raw_PD rows
    for gene, models in data.get("generatePDs", {}).items():
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
                        _pd_row_core(pmid, gene_id, mdl_name, "raw_PD", raw_pd, pd_schema_ok)
                    )
            elif isinstance(pds_data, dict):
                # Single PD that should have been in a list
                pd_rows.append(
                    _pd_row_core(pmid, gene_id, mdl_name, "raw_PD", pds_data, False)
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
    ["pmid", "gene_ID", "model", "bullet_point", "evidence_location",
     "supporting_quotes", "verification_status", "reason", "schema_followed"],
)

write_csv(
    PD_CSV,
    pd_rows,
    ["pmid", "gene_ID", "model", "PD_type", "description",
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

# Optional: Print models with schema issues
if schema_issues_gene > 0 or schema_issues_pd > 0:
    gene_issue_models = set(row["model"] for row in gene_rows if row.get("schema_followed") == "False")
    pd_issue_models = set(row["model"] for row in pd_rows if row.get("schema_followed") == "False")
    all_issue_models = gene_issue_models.union(pd_issue_models)
    if all_issue_models:
        print(f"\n⚠️  Models with schema compliance issues: {', '.join(sorted(all_issue_models))}")