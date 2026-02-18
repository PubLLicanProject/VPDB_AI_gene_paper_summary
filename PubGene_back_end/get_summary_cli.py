#!/usr/bin/env python3
from dotenv import load_dotenv
# Load environment variables from .env file

load_dotenv() 

import os, sys, json, argparse, html

# Import your pipeline pieces
from summary_to_html import run_summary
# Add new error message helpers
from helpers import ui_css, render_error_html


def make_collapsed_summary_html(short_html: str, extended_html: str, aliases_line_html: str = "") -> str:
    """
    Show only the short summary by default. Extended summary is hidden
    behind a <details> block; inside it we keep the structure produced
    by summary_to_html (Evidence-based Summary + Additional Inferences).
    """
    return f"""
<div class="ai-summary">
  <div class="short-summary">
    {short_html}
  </div>
{aliases_line_html}
  <details class="extended-wrapper">
    <summary>Show extended summary</summary>
    {extended_html}
  </details>
</div>
""".strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--gene_id", required=True)
    ap.add_argument("--host_db", required=True)
    args = ap.parse_args()
# add try except for error handling
    try:
        s = run_summary(args.pmid, args.gene_id, args.host_db)
        short_html = s.get("short_summary_html", "")
        extended_html = s.get("extended_summary_html", "")
        names = s.get("aliases_mentioned") or []
        aliases_line_html = ""
        if names:
            alias_str = ", ".join(html.escape(x) for x in names)
            aliases_line_html = (
                "<div class='aliases-line'>"
                "<strong>Aliases mentioned in paper</strong>: "
                f"<em>{alias_str}</em>"
                "</div>"
            )
        combined_html = ui_css() + make_collapsed_summary_html(short_html, extended_html, aliases_line_html)
        out = {"summary": combined_html, "prompt": "ok"}
    except Exception as e:
        out = {"summary": ui_css() + render_error_html("Unexpected error in summariser.", str(e)), "prompt": "error"}

    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()

