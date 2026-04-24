from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


STATUS_DIR = Path("./out/user_comment_evaluation")
STAGE_KEY = "evaluateUserCommentAlignment"
MODEL_KEY = "anthropic/claude-sonnet-4-5"

REPORT_TSV = Path("./out/user_comment_evaluation/legacy_invalid_comment_evals.tsv")
DRY_RUN = False


def _get_comment_raw_response(comment_payload: Dict[str, Any]) -> str:
    if not isinstance(comment_payload, dict):
        return ""

    if isinstance(comment_payload.get("raw_response"), str):
        return comment_payload["raw_response"]

    evaluation = comment_payload.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("raw_response"), str):
        return evaluation["raw_response"]

    return ""


def _get_comment_parsed(comment_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(comment_payload, dict):
        return None

    parsed = comment_payload.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    evaluation = comment_payload.get("evaluation")
    if isinstance(evaluation, dict):
        inner = evaluation.get("parsed")
        if isinstance(inner, dict):
            return inner
        if "assessment_possible" in evaluation:
            return evaluation

    return None


def _is_known_fake_no_data(parsed: Dict[str, Any] | None) -> bool:
    if not isinstance(parsed, dict):
        return False

    return (
        parsed.get("assessment_possible") is False
        and parsed.get("assessment_limitations") == "No data provided to assess"
        and parsed.get("comment_information_density_score") == 0
        and parsed.get("short_vs_headline_score") == 0
        and parsed.get("short_vs_content_score") == 0
        and parsed.get("extended_vs_content_score") == 0
        and parsed.get("pipeline_usefulness_for_submission_score") == 0
        and parsed.get("user_comment_usefulness_for_submission_score") == 0
    )


def _is_legacy_invalid_comment_eval(comment_payload: Dict[str, Any]) -> Tuple[bool, str]:
    raw_response = (_get_comment_raw_response(comment_payload) or "").strip()
    parsed = _get_comment_parsed(comment_payload)

    if raw_response in {"", "{", "["}:
        return True, f"truncated_raw_response:{repr(raw_response)}"

    if _is_known_fake_no_data(parsed):
        return True, "known_fake_no_data_formatter_payload"

    return False, ""


def _recalculate_model_block_summary(model_block: Dict[str, Any]) -> None:
    data = model_block.get("data", {})
    comment_results = data.get("comment_results", {})

    success_count = 0
    failed_count = 0
    total_seconds = 0.0

    for item in comment_results.values():
        if not isinstance(item, dict):
            continue
        if item.get("success") is True:
            success_count += 1
        else:
            failed_count += 1
        total_seconds += float(item.get("seconds", 0.0) or 0.0)

    model_block["usage"] = {
        "comment_count": len(comment_results),
        "successful_comment_count": success_count,
        "failed_comment_count": failed_count,
    }
    model_block["seconds"] = round(total_seconds, 3)

    # Keep model-level success meaning "this model has records saved"
    model_block["success"] = True


def scan_and_patch_legacy_invalid_comment_evals(
    status_dir: Path = STATUS_DIR,
    report_tsv: Path = REPORT_TSV,
    dry_run: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for json_path in sorted(status_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            rows.append({
                "pmid": json_path.stem,
                "gene_id": None,
                "model": None,
                "comment_id": None,
                "action": "read_error",
                "reason": str(e),
                "file": str(json_path),
            })
            continue

        stage_block = payload.get(STAGE_KEY, {})
        if not isinstance(stage_block, dict):
            continue

        file_changed = False

        for gene_id, gene_block in stage_block.items():
            if not isinstance(gene_block, dict):
                continue

            for model_key, model_block in gene_block.items():
                if model_key == "gene_ID" or not isinstance(model_block, dict):
                    continue

                comment_results = ((model_block.get("data") or {}).get("comment_results") or {})
                if not isinstance(comment_results, dict):
                    continue

                for comment_id, comment_payload in comment_results.items():
                    if not isinstance(comment_payload, dict):
                        continue

                    is_bad, reason = _is_legacy_invalid_comment_eval(comment_payload)
                    if not is_bad:
                        continue

                    rows.append({
                        "pmid": json_path.stem,
                        "gene_id": gene_id,
                        "model": model_key,
                        "comment_id": comment_id,
                        "action": "would_patch" if dry_run else "patched",
                        "reason": reason,
                        "file": str(json_path),
                    })

                    if dry_run:
                        continue

                    # Preserve old payload for audit
                    if "legacy_invalid_saved_payload" not in comment_payload:
                        comment_payload["legacy_invalid_saved_payload"] = comment_payload.get("evaluation")

                    comment_payload["success"] = False
                    comment_payload["error"] = f"Legacy invalid STEP 4 evaluation ({reason})"

                    # If raw_response was nested under evaluation, keep the structure but mark nested success false too
                    if isinstance(comment_payload.get("evaluation"), dict):
                        comment_payload["evaluation"]["success"] = False
                        if not comment_payload["evaluation"].get("error"):
                            comment_payload["evaluation"]["error"] = f"Legacy invalid STEP 4 evaluation ({reason})"

                    file_changed = True

                if file_changed:
                    _recalculate_model_block_summary(model_block)

        if file_changed and not dry_run:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

    df = pd.DataFrame(rows)
    if not df.empty:
        report_tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_tsv, sep="\t", index=False)

    return df


if __name__ == "__main__":
    df = scan_and_patch_legacy_invalid_comment_evals(
        status_dir=STATUS_DIR,
        report_tsv=REPORT_TSV,
        dry_run=DRY_RUN,
    )
    print(df.head())
    print(f"Rows flagged: {len(df)}")