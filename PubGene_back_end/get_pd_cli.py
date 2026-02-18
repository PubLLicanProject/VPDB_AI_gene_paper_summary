#!/usr/bin/env python3
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv() 

import os, sys, json, argparse

# Import summary to get bullets+paper; and PDs to run gen→verify
from summary_to_html import run_summary
from pds_to_html import generate_pds_html  # expects paper_text
from helpers import ui_css, render_error_html
# test changes on serv
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--gene_id", required=True)
    ap.add_argument("--host_db", required=True)
    ap.add_argument("--n_pds", type=int, default=int(os.getenv("MAX_PDS", "3")))
    args = ap.parse_args()

    try:
        s = run_summary(args.pmid, args.gene_id, args.host_db)
        bullets = s.get("evidence_bullets", [])
        paper_text = s.get("paper_text", "")
        reason = (s.get("raw_json") or {}).get("reason")

        if reason == "no_aliases_and_no_text_hit":
            html_block = ui_css() + "<div class='ai-info'>Skipped: gene ID not found in the selected database and not detected in the paper text. Please verify Gene ID and database.</div>"

        elif reason == "aliases_exist_but_no_text_hit":
            html_block = ui_css() + "<div class='ai-info'>Skipped: gene exists in the selected database, but is not mentioned in this paper (no ID or alias hits).</div>"


        elif not paper_text.strip():
            html_block = ui_css() + "<div class='ai-info'>Verification skipped: paper text unavailable.</div>"

        else:
            p = generate_pds_html(args.gene_id, args.pmid, bullets, paper_text, n_pds=args.n_pds, host_db=args.host_db, )
            html_block = ui_css() + (p.get("product_descriptions_html", "")
                                     or "<div class='ai-info'>No verified product descriptions.</div>")

        out = {"summary": html_block, "prompt": "ok"}
    except Exception as e:
        out = {"summary": ui_css() + render_error_html("Unexpected error in PD generator.", str(e)), "prompt": "error"}

    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_dict = {"summary": str(e)}

        print(json.dumps(error_dict))

