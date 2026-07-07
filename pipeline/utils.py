from __future__ import annotations
import json
import re


from typing import Any, Dict, Optional, Mapping, Union

def collect_bullets(summary_json: dict) -> list:
    """
    Extract bullet points from summary JSON.

    Args:
        summary_json: Summary output from getGeneSummary

    Returns:
        List of bullet point strings
    """
    if not summary_json or not isinstance(summary_json, dict):
        return []

    # Extract from GeneSummary key
    if "GeneSummary" in summary_json:
        bullets = []
        for item in summary_json["GeneSummary"]:
            if isinstance(item, dict) and "bullet_point" in item:
                bullets.append(item["bullet_point"])
        print(f"BULLETS: {bullets}")
        return bullets

    return []


def extract_json(text: str) -> Union[Dict[str, Any], list, str]:
    """
    Parse JSON from LLM output robustly:
    1) Prefer fenced ```json ... ``` blocks anywhere in the text.
    2) Else extract the first {...} or [...] region and attempt json.loads.
    Returns dict/list on success, else raw text on failure.
    """
    if not isinstance(text, str):
        return text

    s = text.strip()

    # 1) Extract fenced JSON block anywhere (not just at start/end)
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse fenced JSON. ({e})")
            return text

    # 2) Extract first object/array region
    obj_start = s.find("{")
    obj_end = s.rfind("}")
    arr_start = s.find("[")
    arr_end = s.rfind("]")

    candidates = []
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(s[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(s[arr_start : arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3) Try direct parse as last attempt
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"⚠️ Could not parse JSON – saving raw text. ({e})")
        return text


def gene_to_prompt(gene, genes):
    """
    Converts a gene ID and a list of synonyms into a readable string for prompts.

    Args:
        gene (str): The gene ID.
        genes (list of str): A list of synonyms for the gene.

    Returns:
        str: A formatted string that includes the gene ID and its synonyms.
    """

    if len(genes) ==0:
        print(f"GENE and aliases: {gene}")
        return gene
    else:
        aliases_for_prompt = gene + " (also known as " + " or ".join(genes) +")"
        print(f"GENE and aliases: {aliases_for_prompt}")
        return aliases_for_prompt


def check_if_ok(status_data):
    ok = (
            isinstance(status_data, dict)  # right type
            and status_data.get("success") is True  # flagged OK
            and isinstance(status_data.get("data"), dict)  # data exists
            and "error" not in status_data["data"]  # no nested error
    )
    if ok:
        return True
    return False


def check_if_in_passing(summary_data):
    """
        Check if gene is only mentioned in passing based on summary data.
        # Older versions do not have that flag so we just process them
    """
    if not isinstance(summary_data, Mapping):
        return False
    return summary_data.get("only_in_passing") is True


def _unverified_to_select_candidates(suggested: dict) -> dict:
    """
    Convert generatePDs output into {"PDs": [{"description","evidence_code","code_reason"}, ...]}
    so it can be fed into selectPD(). Returns an empty "PDs" list if nothing usable.

    Args:
        suggested: Output from generatePDs stage

    Returns:
        Dict with "PDs" key containing list of PD candidates

    Example input:
        {
          "gene_id": "PF3D7_1355700",
          "short_summary": "...",
          "PDs": [
            {
              "description": "NIF-like phosphatase NIF3, putative",
              "evidence_code": "ISA",
              "code_reason": "Classification based on..."
            }
          ]
        }
    """
    try:
        if not isinstance(suggested, dict):
            return {"PDs": []}

        # Extract PDs list directly
        pds = suggested.get("PDs", [])

        if not isinstance(pds, list):
            return {"PDs": []}

        def _extract_triplet(d):
            if not isinstance(d, dict):
                return None
            desc = d.get("description") or d.get("Description")
            ev = d.get("evidence_code") or d.get("evidenceCode") or d.get("evidence")
            reason = d.get("code_reason") or d.get("reason") or d.get("evidence_reason")
            if not isinstance(desc, str) or not desc.strip():
                return None
            return {"description": desc, "evidence_code": (ev or ""), "code_reason": (reason or "")}

        candidates = []
        for item in pds:
            t = _extract_triplet(item)
            if t:
                candidates.append(t)

        # De-duplicate by description
        seen = set()
        unique = []
        for c in candidates:
            key = c["description"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return {"PDs": unique}
    except Exception:
        return {"PDs": []}
def _verified_to_select_candidates(verified: dict) -> dict:
    """
    Convert verifyPDs output into {"PDs": [{"description","evidence_code","code_reason"}, ...]}
    so it can be fed into selectPD(). Returns an empty "PDs" list if nothing usable.
    """
    try:
        if not isinstance(verified, dict):
            return {"PDs": []}

        def _extract_triplet(d):
            if not isinstance(d, dict):
                return None
            desc = d.get("description") or d.get("Description")
            ev = d.get("evidence_code") or d.get("evidenceCode") or d.get("evidence")
            reason = d.get("code_reason") or d.get("reason") or d.get("evidence_reason")
            if not isinstance(desc, str) or not desc.strip():
                return None
            return {"description": desc, "evidence_code": (ev or ""), "code_reason": (reason or "")}

        candidates = []

        # Include RPD if present
        rpd = verified.get("RPD")
        rpd_triplet = _extract_triplet(rpd)
        if rpd_triplet:
            candidates.append(rpd_triplet)

        # Include APDs (could be list or single)
        apds = verified.get("APDs")
        if isinstance(apds, list):
            for item in apds:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)
        else:
            t = _extract_triplet(apds)
            if t:
                candidates.append(t)

        # Fallback: sometimes verify step may echo a "PDs" list
        if not candidates and isinstance(verified.get("PDs"), list):
            for item in verified["PDs"]:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)

        # De-duplicate by description
        seen = set()
        unique = []
        for c in candidates:
            key = c["description"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return {"PDs": unique}
    except Exception:
        return {"PDs": []}

# helpers to  track token usage for costing
def _pydantic_extra(obj: Any) -> Dict[str, Any]:
    """
    Pydantic v2 stores unknown fields on either `.model_extra` or `.__pydantic_extra__`.
    OpenRouter adds fields like `cost` that may land there depending on SDK versions.
    """
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra
    extra = getattr(obj, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        return extra
    return {}


def _safe_dump(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    md = getattr(obj, "model_dump", None)
    if callable(md):
        try:
            return md()
        except Exception:
            return {}
    d = getattr(obj, "dict", None)
    if callable(d):
        try:
            return d()
        except Exception:
            return {}
    return {}


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default
def _extract_usage(provider, response):
    """
    Return {"input": int, "output": int, "total": int} from SDK response objects.
    Supports Anthropic, OpenAI Chat Completions, and OpenAI Responses API.
    """
    try:
        if provider == "openrouter":
            # 1) Try response.usage first (OpenAI SDK object)
            usage_obj = getattr(response, "usage", None)
            usage = _safe_dump(usage_obj)

            # 2) Merge in any "extra" fields on usage (e.g., cost)
            usage.update(_pydantic_extra(usage_obj))

            # 3) If usage still empty, fall back to dumping whole response and reading "usage"
            if not usage:
                resp_dump = _safe_dump(response)
                usage = resp_dump.get("usage") or {}

            # 4) Normalize nested details (may be objects)
            ptd = usage.get("prompt_tokens_details") or {}
            ctd = usage.get("completion_tokens_details") or {}
            if not isinstance(ptd, dict):
                ptd = _safe_dump(ptd) | _pydantic_extra(ptd)
            if not isinstance(ctd, dict):
                ctd = _safe_dump(ctd) | _pydantic_extra(ctd)

            prompt_tokens = _as_int(usage.get("prompt_tokens", 0))
            completion_tokens = _as_int(usage.get("completion_tokens", 0))
            total_tokens = _as_int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
            )

            # Extract caching/cost fields per OpenRouter docs
            cached_tokens = _as_int(ptd.get("cached_tokens", 0))
            cache_write_tokens = _as_int(ptd.get("cache_write_tokens", 0))
            audio_tokens = _as_int(ptd.get("audio_tokens", 0))
            reasoning_tokens = _as_int(ctd.get("reasoning_tokens", 0))

            cost = usage.get("cost", None)
            cost_details = usage.get("cost_details", None)

            # OpenRouter sometimes includes `cache_discount` at the response root; preserve if present.
            resp_extra = _pydantic_extra(response)
            cache_discount = resp_extra.get("cache_discount")

            # Helpful for later audit via /generation endpoint
            generation_id = getattr(response, "id", None) or resp_extra.get("id")

            out = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "audio_tokens": audio_tokens,
                },
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                "cost": cost,
                "cost_details": cost_details,
            }
            if cache_discount is not None:
                out["cache_discount"] = cache_discount
            if generation_id:
                out["generation_id"] = generation_id

            # Back-compat aliases (so older code expecting input/output/total still works)
            out["input"] = prompt_tokens
            out["output"] = completion_tokens
            out["total"] = total_tokens
            return out

        if provider == "anthropic":
            u = response.usage
            return {"input": u.input_tokens,
                    "output": u.output_tokens,
                    "total": u.input_tokens + u.output_tokens}
        elif provider in ("openai", "openai-chat"):
            u = response.usage
            return {"input": u.prompt_tokens,
                    "output": u.completion_tokens,
                    "total": u.total_tokens}
        elif provider in ("openai-responses",):
            u = response.usage
            # Responses API typically exposes input_tokens/output_tokens[/total_tokens]
            input_tokens = getattr(u, "input_tokens", 0)
            output_tokens = getattr(u, "output_tokens", 0)
            total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens)
            return {"input": input_tokens, "output": output_tokens, "total": total_tokens}
    except AttributeError:
        pass
    return {"input": 0, "output": 0, "total": 0}


def _to_responses_input(messages):
    """
    Convert [{"role":..,"content": <str|blocks>}, ...] into Responses API format:
    each content block must use type 'input_text' (or 'input_image').
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        blocks = []
        if isinstance(content, str):
            blocks = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            for b in content:
                btype = b.get("type")
                if btype == "text":
                    blocks.append({"type": "input_text", "text": b.get("text", "")})
                elif btype in ("image_url", "input_image"):
                    # minimal mapping from Chat image part to Responses
                    url = b.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if isinstance(url, str) and url:
                        blocks.append({"type": "input_image", "image_url": url})
                # ignore unknown block types
        else:
            blocks = [{"type": "input_text", "text": str(content)}]
        out.append({"role": role, "content": blocks})
    return out

def _responses_supports_temperature(model: str) -> bool:
    """
    Return False for newer deterministic Responses models that reject `temperature`.
    Extend the prefixes as needed if you see similar errors with other models.
    """
    no_temp_prefixes = ("gpt-5",)  # add more if needed
    return not any(model.startswith(p) for p in no_temp_prefixes)

def _verified_to_select_candidates(verified: dict) -> dict:
    """
    Convert verifyPDs output into {"PDs": [{"description","evidence_code","code_reason"}, ...]}
    so it can be fed into selectPD(). Returns an empty "PDs" list if nothing usable.
    """
    try:
        if not isinstance(verified, dict):
            return {"PDs": []}

        def _extract_triplet(d):
            if not isinstance(d, dict):
                return None
            desc = d.get("description") or d.get("Description")
            ev = d.get("evidence_code") or d.get("evidenceCode") or d.get("evidence")
            reason = d.get("code_reason") or d.get("reason") or d.get("evidence_reason")
            if not isinstance(desc, str) or not desc.strip():
                return None
            return {"description": desc, "evidence_code": (ev or ""), "code_reason": (reason or "")}

        candidates = []

        # Include RPD if present
        rpd = verified.get("RPD")
        rpd_triplet = _extract_triplet(rpd)
        if rpd_triplet:
            candidates.append(rpd_triplet)

        # Include APDs (could be list or single)
        apds = verified.get("APDs")
        if isinstance(apds, list):
            for item in apds:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)
        else:
            t = _extract_triplet(apds)
            if t:
                candidates.append(t)

        # Fallback: sometimes verify step may echo a "PDs" list
        if not candidates and isinstance(verified.get("PDs"), list):
            for item in verified["PDs"]:
                t = _extract_triplet(item)
                if t:
                    candidates.append(t)

        # De-duplicate by description
        seen = set()
        unique = []
        for c in candidates:
            key = c["description"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return {"PDs": unique}
    except Exception:
        return {"PDs": []}

# TODO: add check for mentioned in passing