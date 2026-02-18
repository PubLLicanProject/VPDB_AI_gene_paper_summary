import json
import re

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


def extract_json(text: str):
    """
    Strip ```json fences if present and json-load the remainder.
    Returns a dict/list on success, or the raw text on failure.
    Use on LLM outputs to get string as json
    """
    text = text.strip()
    # remove ```json … ``` or plain ``` … ``` fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
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
    """
    return summary_data["only_in_passing"] is True


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

# helper to  track token usage for costing
def _extract_usage(provider, response):
    """
    Return {"input": int, "output": int, "total": int} from SDK response objects.
    Supports Anthropic, OpenAI Chat Completions, and OpenAI Responses API.
    """
    try:
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
    no_temp_prefixes = ("gpt-5")  # add more if needed
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