"""
extract_compare.py

Reuses the EXACT extraction logic from extract_data.py (row-building copied verbatim) but as a callable
extract_rows(status_dir), to build a with-vs-without-supplementary comparison:

  1) _noSUPP  = existing full CSVs  +  extraction of out/VPDB_PD_set_expansion/cache (no supplement)
  2) _withSUPP = extraction of out/VPDB_PD_supplementary_rerun/cache (with supplement)
  3) For (pmid, gene_ID) present in both, flag RPDs whose description changed.

Outputs into out/wiht_vs_without_supplementary_comapriso/.
"""
import os
import csv
import json

# ---- verbatim helpers from extract_data.py ------------------------------------------------
def blank(x):
    return "" if x is None else str(x)


def has_error(data):
    if not isinstance(data, dict):
        return False
    return "error" in data or "message" in data or "text" in data


def _pd_row_core(pmid, gene_id, model, pd_type, pd_dict, schema_followed=True):
    if pd_dict is None:
        pd_dict = {}
    elif isinstance(pd_dict, (list, tuple)):
        pd_dict = pd_dict[0] if pd_dict else {}
    elif not isinstance(pd_dict, dict):
        pd_dict = {"description": str(pd_dict)}
    return {
        "pmid": pmid, "gene_ID": gene_id, "model": model, "PD_type": pd_type,
        "description": blank(pd_dict.get("description")),
        "verification_status": blank(pd_dict.get("verification_status")),
        "evidence_code": blank(pd_dict.get("evidence_code")),
        "code_reason": blank(pd_dict.get("code_reason")),
        "audit_status": blank(pd_dict.get("audit_status")),
        "audit_reason": blank(pd_dict.get("audit_reason")),
        "schema_followed": str(schema_followed),
    }


def pd_rows_from_verify(pmid, gene_id, model, vpd_data):
    if not isinstance(vpd_data, dict) or has_error(vpd_data):
        return
    if "RPD" in vpd_data:
        yield _pd_row_core(pmid, gene_id, model, "RPD", vpd_data["RPD"], isinstance(vpd_data["RPD"], dict))
    apds = vpd_data.get("APDs", [])
    if apds:
        if isinstance(apds, list):
            for apd in apds:
                yield _pd_row_core(pmid, gene_id, model, "APD", apd, isinstance(apd, dict))
        elif isinstance(apds, dict):
            if "description" in apds or "verification_status" in apds:
                yield _pd_row_core(pmid, gene_id, model, "APD", apds, False)
            else:
                for _, apd_value in apds.items():
                    if isinstance(apd_value, dict):
                        yield _pd_row_core(pmid, gene_id, model, "APD", apd_value, False)
        else:
            yield _pd_row_core(pmid, gene_id, model, "APD", {"description": str(apds)}, False)


def gene_row_from_summary(pmid, gene_id, model, bullet, schema_followed=True):
    return {
        "pmid": pmid, "gene_ID": gene_id, "model": model,
        "bullet_point": blank(bullet.get("bullet_point")),
        "evidence_location": blank(bullet.get("evidence_location")),
        "supporting_quotes": ";;;;;".join(bullet.get("supporting_quotes", [])),
        "verification_status": blank(bullet.get("verification_status", "")),
        "reason": blank(bullet.get("reason", "")),
        "schema_followed": str(schema_followed),
    }


def extract_rows(status_dir):
    """Identical logic to extract_data.py's main loop, parameterised on the cache dir."""
    gene_rows, pd_rows = [], []
    for fname in sorted(os.listdir(status_dir)):
        if not fname.endswith(".json"):
            continue
        pmid = fname.split(".")[0]
        try:
            data = json.load(open(os.path.join(status_dir, fname), encoding="utf-8"))
        except json.JSONDecodeError:
            print("skip malformed:", fname)
            continue

        for gene, models in data.get("getGeneSummary", {}).items():
            gene_id = models.get("gene_ID", gene)
            for mdl_name, mdl_block in models.items():
                if mdl_name == "gene_ID" or not mdl_block.get("success"):
                    continue
                d = mdl_block.get("data", {})
                if has_error(d):
                    continue
                if isinstance(d, dict) and "GeneSummary" in d:
                    sm = d["GeneSummary"]
                    if isinstance(sm, list):
                        for bullet in sm:
                            gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, bullet, isinstance(bullet, dict)))
                    elif isinstance(sm, dict):
                        gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, sm, False))

        for gene, models in data.get("verifyGeneSummary", {}).items():
            gene_id = models.get("gene_ID", gene)
            for mdl_name, mdl_block in models.items():
                if mdl_name == "gene_ID" or not mdl_block.get("success"):
                    continue
                d = mdl_block.get("data", {})
                if has_error(d):
                    continue
                v = d.get("VerifiedSummary") or d if isinstance(d, dict) else d
                if isinstance(v, list):
                    for bullet in v:
                        gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, bullet, isinstance(bullet, dict)))
                elif isinstance(v, dict):
                    gene_rows.append(gene_row_from_summary(pmid, gene_id, mdl_name, v, False))

        for gene, models in data.get("verifyPDs", {}).items():
            gene_id = models.get("gene_ID", gene)
            for mdl_name, mdl_block in models.items():
                if mdl_name == "gene_ID" or not mdl_block.get("success"):
                    continue
                d = mdl_block.get("data", {})
                if has_error(d):
                    continue
                for row in pd_rows_from_verify(pmid, gene_id, mdl_name, d):
                    pd_rows.append(row)

        for sel_key in ("selectPD", "selectPDs"):
            for gene, models in data.get(sel_key, {}).items():
                gene_id = models.get("gene_ID", gene)
                for mdl_name, mdl_block in models.items():
                    if mdl_name == "gene_ID" or not mdl_block.get("success"):
                        continue
                    d = mdl_block.get("data", {})
                    if has_error(d):
                        continue
                    sel_pd = d.get("recommended_product") or d.get("selected_PD") or d.get("selectedPD")
                    if sel_pd:
                        pd_rows.append(_pd_row_core(pmid, gene_id, mdl_name, "selected_PD", sel_pd, isinstance(sel_pd, dict)))

        for gene, models in data.get("generatePDs", {}).items():
            gene_id = models.get("gene_ID", gene)
            for mdl_name, mdl_block in models.items():
                if mdl_name == "gene_ID" or not mdl_block.get("success"):
                    continue
                d = mdl_block.get("data", {})
                if has_error(d):
                    continue
                pds = d.get("PDs", [])
                if isinstance(pds, list):
                    for raw_pd in pds:
                        pd_rows.append(_pd_row_core(pmid, gene_id, mdl_name, "raw_PD", raw_pd, isinstance(raw_pd, dict)))
                elif isinstance(pds, dict):
                    pd_rows.append(_pd_row_core(pmid, gene_id, mdl_name, "raw_PD", pds, False))
    return gene_rows, pd_rows


GENE_FIELDS = ["pmid", "gene_ID", "model", "bullet_point", "evidence_location",
               "supporting_quotes", "verification_status", "reason", "schema_followed"]
PD_FIELDS = ["pmid", "gene_ID", "model", "PD_type", "description", "verification_status",
             "evidence_code", "code_reason", "audit_status", "audit_reason", "schema_followed"]
OUTDIR = "out/wiht_vs_without_supplementary_comapriso"


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) _noSUPP = existing full + set_expansion (no supplement)
    base_gene = read_csv("out/summaries/extracted/Summary_Quotes_by_model_full.csv")
    base_pd = read_csv("out/summaries/extracted/PD_rows_all_models_full.csv")
    se_gene, se_pd = extract_rows("out/VPDB_PD_set_expansion/cache")
    nosupp_gene, nosupp_pd = base_gene + se_gene, base_pd + se_pd
    write_csv(f"{OUTDIR}/Summary_Quotes_by_model_full_noSUPP.csv", nosupp_gene, GENE_FIELDS)
    write_csv(f"{OUTDIR}/PD_rows_all_models_full_noSUPP.csv", nosupp_pd, PD_FIELDS)
    print(f"_noSUPP: summary {len(nosupp_gene)} (base {len(base_gene)} + set_exp {len(se_gene)}), "
          f"PD {len(nosupp_pd)} (base {len(base_pd)} + set_exp {len(se_pd)})")

    # 2) _withSUPP = supplementary rerun (with supplement)
    ws_gene, ws_pd = extract_rows("out/VPDB_PD_supplementary_rerun/cache")
    write_csv(f"{OUTDIR}/Summary_Quotes_by_model_full_withSUPP.csv", ws_gene, GENE_FIELDS)
    write_csv(f"{OUTDIR}/PD_rows_all_models_full_withSUPP.csv", ws_pd, PD_FIELDS)
    print(f"_withSUPP: summary {len(ws_gene)}, PD {len(ws_pd)}")

    # 3) RPD change for (pmid, gene_ID) present in both
    def rpd_map(rows):
        m = {}
        for r in rows:
            if r.get("PD_type") == "RPD":
                m[(r["pmid"], r["gene_ID"])] = r["description"]  # one RPD per gene
        return m
    no_rpd, ws_rpd = rpd_map(nosupp_pd), rpd_map(ws_pd)
    shared = sorted(set(no_rpd) & set(ws_rpd))
    flagged = [{"pmid": p, "gene_ID": g,
                "RPD_noSUPP": no_rpd[(p, g)], "RPD_withSUPP": ws_rpd[(p, g)],
                "changed": no_rpd[(p, g)].strip() != ws_rpd[(p, g)].strip()}
               for (p, g) in shared]
    write_csv(f"{OUTDIR}/RPD_change_shared.csv", flagged,
              ["pmid", "gene_ID", "changed", "RPD_noSUPP", "RPD_withSUPP"])
    changed = [f for f in flagged if f["changed"]]
    print(f"\nRPD comparison: {len(shared)} genes with an RPD in BOTH; "
          f"CHANGED = {len(changed)} ({100*len(changed)/max(len(shared),1):.0f}%)")
    print("flagged file -> RPD_change_shared.csv  (column 'changed')")
    for f in changed[:5]:
        print(f"  {f['pmid']}/{f['gene_ID']}:")
        print(f"     no-supp : {f['RPD_noSUPP'][:90]}")
        print(f"     w/ supp : {f['RPD_withSUPP'][:90]}")


if __name__ == "__main__":
    main()
