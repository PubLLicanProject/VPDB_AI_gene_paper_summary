# pds_to_html.py — Generate PDs then verify them (Anthropic-only)
# -----------------------------------------------------------------------------
# Stage 1: generate PDs from evidence bullets
# Stage 2: verify the PDs against the paper text (verifyPDs) and keep only verified
#
# Inputs:
#   - gene_id: str
#   - evidence_bullets: List[str]
#   - paper_text: str  (required for verification)
#   - n_pds: int       (default: env MAX_PDS or 3)
#
# Output dict:
#   {
#     "product_descriptions_html": "<div>...</div>",   # ONLY verified PDs
#     "generated_json": {...},   # raw generated PD set (pre-verification)
#     "verified_json": {...}     # verifyPDs JSON
#   }
#
# Env:
#   - ANTHROPIC_API_KEY (required)
#   - PD_MODEL          (default "claude-sonnet-4-20250514")
#   - MAX_PDS           (default 3)
# -----------------------------------------------------------------------------

from dotenv import load_dotenv
# Load environment variables from .env file
# dotenv_path = join(dirname(__file__), 'v2.env')
load_dotenv() 

import os
import json
import html
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
from helpers import ui_css
from helpers import strip_fenced_json, cache_get, cache_set, _ttl, CACHE_VERSION
from helpers import get_gene_synonyms_in_paper, _count_substrings, gene_for_prompt
import hashlib, json


# ---- config -----------------------------------------------------------------
PD_MODEL = os.getenv("PD_MODEL", "claude-sonnet-4-20250514")
MAX_PDS = int(os.getenv("MAX_PDS", "3"))

EVIDENCE_CODE_DEFS = {
    "EXP": "Inferred from Experiment",
    "IDA": "Inferred from Direct Assay",
    "IPI": "Inferred from Physical Interaction",
    "IMP": "Inferred from Mutant Phenotype",
    "IGI": "Inferred from Genetic Interaction",
    "IEP": "Inferred from Expression Pattern",
    "HTP": "Inferred from High Throughput Experiment",
    "HDA": "Inferred from High Throughput Direct Assay",
    "HMP": "Inferred from High Throughput Mutant Phenotype",
    "HGI": "Inferred from High Throughput Genetic Interaction",
    "HEP": "Inferred from High Throughput Expression Pattern",
    "ISS": "Inferred from Sequence or Structural Similarity",
    "ISO": "Inferred from Sequence Orthology",
    "ISA": "Inferred from Sequence Alignment",
    "ISM": "Inferred from Sequence Model",
    "IGC": "Inferred from Genomic Context",
    "RCA": "Reviewed Computational Analysis",
    "TAS": "Traceable Author Statement",
    "NAS": "Non-traceable Author Statement",
    "IC":  "Inferred by Curator",
    "IEA": "Inferred from Electronic Annotation",
}
# ---- small utils -------------------------------------------------------------
def _strip_fenced_json(text: str) -> str:
    import re
    txt = text.strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL)

def _anthropic_call(system_prompt: str, user_messages: List[str], model: str) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Support 1 or 2 user messages (first treated as "context")
    if len(user_messages) > 1:
        content = [
            {"type": "text", "text": user_messages[0], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_messages[1]},
        ]
    else:
        content = [{"type": "text", "text": user_messages[0]}]
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=6000,
        temperature=0,
    )
    return resp.content[0].text if resp.content else ""

# =============================================================================
#                               STAGE 1: GENERATE
# =============================================================================
GENERATE_PDS_SYSTEM = (
    "ROLE: You are an expert gene curation assistant specializing in synthesizing "
    "scientific literature into standardized gene and protein functional annotations.\n\n"
    "BACKGROUND: You will analyze a comprehensive gene summary derived from a PubMed publication. "
    "This summary contains experimental evidence and conclusions for a single gene, "
    "prepared by another expert assistant.\n\n"
    "OBJECTIVES:\n"
    "1. Generate a concise one-sentence summary capturing key novel findings.\n"
    "2. Generate up to [N_PDs] product descriptions (PDs) for the curated gene.\n\n"
    "ANALYSIS WORKFLOW:\n"
    "STEP 1: Information Extraction\n"
    "- Compile all gene names, synonyms, locus/ORF identifiers\n"
    "- Catalog all functional evidence presented\n"
    "- Identify experimental vs. computational evidence\n"
    "STEP 2: Description Brainstorming"
    "- Refer to the TEMPLATES, and GUIDELINES to generate one appropriate PD at a time,"
    f" up to a total of {MAX_PDS}."
    "- Prioritise PDs founded on experimental evidence over computational predictions.\n"
    "- Only create additional PDs when multiple valid alternatives exist, ensuring PDs "
    "are distinct and scientifically meaningful\n"
    "- Initially assign a \"putative\" qualifier to all PDs, then review if it needs to be removed as"
    "described in the RULES.\n"
    "Step 3: Evidence code assignment \n"
    "Assign appropriate GO evidence codes with justification for each PD - refer to the EVIDENCE CODES "
    "section for more details. Explain briefly why the code was selected as evidence_reason.\n\n"
    "<TEMPLATES>"
    "- Experimentally characterized gene function → \"<Function name>\", e.g. sulfotransferase, phosphatase, etc.\n "
    "- Function known in ortholog/paralog but not tested specifically in gene → \"<Function name>, putative\"; "
    "- Domain/motif identified → \"<Domain> family protein, putative\" OR \"<Domain> containing protein\"\n"
    "Caveat: if a certain domain or other relevant biology/biochemistry is conserved and you "
    "are aware that such a feature is always linked to certain function, indicate this in "
    "the description (e.g. if you know that a DHHC domain confers palmitoyltransferase activity, "
    "assigning \"palmitoyltransferase DHHC9\" would be preferred over \"DHHC-containing protein DHHC9\"; "
    "If it will not result in breaking the max number of PDs, assigning both would be recommended).\n"
    "- Gene is conserved, but has unknown function → \"hypothetical protein, conserved\"\n"
    "- No conservation → \"hypothetical protein\"\n"
    "- No conservation and Dubious small ORF → \"hypothetical protein, unlikely\"\n"
    "- Insufficient information about the gene in the paper → \"no relevant gene info in this paper\"\n"
    "</TEMPLATES>\n\n"
    "<RULES>\n"
    "Rule 1: PDs should ideally be a succinct phrase describing the gene function/role.\n"
    "Rule 2: Never use gene symbols alone as product descriptions.\n"
    "✓ Correct: \"Fructose 1,6-bisphosphatase FBP1\""
    "✗ Incorrect: \"FBP1\""
    "Rule 3: Be specific, but concise. E.g., if supported by evidence, \"Fructose 1,6-bisphosphatase FBP1\" "
    "is preferred over a generic \"Phosphatase FBP1\" \n"
    "Rule 4: Remove \"putative\" qualifiers when experimental evidence or confirms function; "
    "Retain \"putative\" for predictions or indirect evidence, except when your know\n"
    "Rule 5: All descriptions require evidence_code and code_reason\n"
    "Rule 6: When multiple PDs could be relevant, prioritise the author-provided PD first and quote "
    "it verbatim. Next, prioritise PDs pertaining to gene function, followed by domain info, and "
    "finally any other PDs within the maximum limit of PDs generated that are informative and relevant.\n"
    "Rule 7: Avoid species information in the PDs. For example, \"Plasmodium falciparum Fructose 1,6-bisphosphatase PfFBP1\""
    "should be corrected to \"Fructose 1,6-bisphosphatase FBP1\". Note that the Pf at the start of PfFBP1 "
    "in this case is what the author of the summary used to denote the species. Care should be taken - DO NOT"
    "strip actual parts of the gene symbol that are NOT species-related."
    "</RULES>\n\n"
    "<EVIDENCE CODES>\n"
    "Use Gene Ontology evidence codes exactly as defined; They are supplied below as Code :: Definition :: Summary:\n"
    "EXPERIMENTAL CODES:\n"
    "EXP :: Inferred from Experiment :: Parent of IDA, IPI, IMP, IGI & IEP; use a child term if possible.\n"
    "IDA :: Inferred from Direct Assay :: Enzyme assays; in vitro reconstitution; immunofluorescence; binding assays.\n"
    "IPI :: Inferred from Physical Interaction :: Two hybrid, co purification, co immunoprecipitation, ion/protein binding experiments.\n"
    "IMP :: Inferred from Mutant Phenotype :: Gene knockout/over expression; antisense; inhibitors.\n"
    "IGI :: Inferred from Genetic Interaction :: Suppressors, synthetic lethals, complementation, rescue.\n"
    "IEP :: Inferred from Expression Pattern :: Northern/Western, etc.\n"
    "HTP :: Inferred from High Throughput Experiment :: Parent of HDA, HMP, HGI & HEP.\n"
    "HDA :: Inferred from High Throughput Direct Assay :: Proteomics, fluorescence microscopy, etc.\n"
    "HMP :: Inferred from High Throughput Mutant Phenotype :: Large scale perturbation screens.\n"
    "HGI :: Inferred from High Throughput Genetic Interaction :: Large scale genetic screens.\n"
    "HEP :: Inferred from High Throughput Expression Pattern :: Microarray, RNA seq, proteomics.\n"
    "COMPUTATIONAL CODES:\n"
    "ISS :: Inferred from Sequence or Structural Similarity :: Parent of ISO, ISA & ISM.\n"
    "ISO :: Inferred from Sequence Orthology :: Annotation transfer via orthology.\n"
    "ISA :: Inferred from Sequence Alignment :: Based on pairwise or multiple alignment.\n"
    "ISM :: Inferred from Sequence Model :: HMMs, tRNAscan SE, Rfam, etc.\n"
    "IGC :: Inferred from Genomic Context :: Operons, synteny, pathway context.\n"
    "RCA :: Reviewed Computational Analysis :: Predictions from integrated large scale data.\n"
    "CURATOR JUDGEMENT CODES:\n"
    "TAS :: Traceable Author Statement :: Statement in review attributed to a cited primary source.\n"
    "NAS :: Non traceable Author Statement :: Database record or publication statement without traceable source.\n"
    "IC  :: Inferred by Curator :: Logical inference from other GO annotations.\n"
    "IEA :: Inferred from Electronic Annotation :: Unreviewed computational transfer (e.g. InterPro2GO).\n"
    "</EVIDENCE CODES>\n\n"
    "OUTPUT FORMAT:\n"
    "Double-check you have followed the templates and have fulfilled all rules. Return JSON using "
    "the following schema:\n"
    "[JSON_SCHEMA]\n\n"
    "QUALITY CHECKLIST:\n"
    "- Are PDs genuinely distinct alternatives?\n"
    "- Are \"putative\" qualifiers used appropriately?\n"
    "- Do evidence codes match the experimental approaches described?\n"
    "- Is species information correctly stripped away? If in doubt, it is best to leave it in.\n"
    "OUTPUT (JSON only):\n"
    "{\n"
    "  \"gene_id\": \"...\",\n"
    "  \"PDs\": [\n"
    "    {\"description\": \"...\", \"evidence_code\": \"EXP|IDA|IPI|IMP|IGI|IEP|HTP|HDA|HMP|HGI|HEP|ISS|ISO|ISA|ISM|IGC|RCA|TAS|NAS|IC|IEA\", \"code_reason\": \"...\"}\n"
    "  ]\n"
    "}\n"
)

def _build_generate_prompt(gene_text: str, bullets: List[str], n_pds: int) -> Tuple[str, List[str]]:
    sys = GENERATE_PDS_SYSTEM.replace("{MAX_PDS}", str(n_pds))
    bullets_text = "\n".join(f"- {b}" for b in bullets)
    user = (
        f"Gene: {gene_text}\n"
        "Here are the direct-evidence bullets you must base PDs on:\n"
        f"{bullets_text}\n\n"
        f"Generate up to {n_pds} PDs and return JSON only."
    )
    return sys, [user]

def _generate_pds_json(gene_text: str, bullets: List[str], n_pds: int) -> Dict[str, Any]:
    sys, user_msgs = _build_generate_prompt(gene_text, bullets, n_pds)
    raw = _anthropic_call(sys, user_msgs, model=PD_MODEL)
    txt = _strip_fenced_json(raw)
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            raise ValueError("PD generation output was not a JSON object")
        return parsed
    except Exception as e:
        return {"error": f"PD generation failed: {e}", "_raw": raw}

# =============================================================================
#                               STAGE 2: VERIFY
# =============================================================================
# The following prompt + schema mirror your original `verifyPDs` stage.
# (Adapted from pipeline_runner_single.py → "verifyPDs")  ← see note in chat.
VERIFY_PDS_SYSTEM = (
    "ROLE: You audit gene curation records and specialise in scrutinising product descriptions\n"
    "BACKGROUND: You will be provided a scientific paper and a set of gene product descriptions that "
    "have been generated by an apprentice curator for a gene of interest.\n"
    "GOALS:\n"
    "- Assess if the paper text presents sufficient evidence for the supplied product descriptions (PDs)"
    "to be assigned to the specified gene.\n"
    "- Determine if the PDs need to be edited as per the SENIOR AUDITOR EXAMPLES supplied.\n"
    "- Select one recommended PD as most suitable based on the HIERARCHY supplied.\n\n"
    "WORKFLOW:\n"
    "1) For each PD supplied, check if it is supported by the paper text.\n"
    "If yes, do the following:\n"
    "- assign it a verification_status SUPPORTED\n"
    "- add evicence_location that lists all sections of the paper that support the assignment"
    "(e.g. Fig 1, Results paragraph 3, etc.)\n"
    "- select the most suitable evidence_code from the EVIDENCE CODES below.\n"
    "- provide a reason that briefly explain why this code was selected.\n\n"
    "If not supported, assign verification_status as UNSUPPORTED and provide no further information for it.\n"
    "2) Consult the SENIOR AUDITOR EXAMPLES and check whether each SUPPORTED PD would pass an audit:\n"
    "- if yes, assign audit_check as PASS.\n"
    "- if not, modify the PD so it would pass and assign audit_check as EDITED\n"
    "- provide a short edit_reason (<20 words)\n"
    "3) Select a recommended PD from your final edited list, based on the HIERARCHY provided.\n\n"
    "<EVIDENCE CODES>\n"
    "EXP :: Inferred from Experiment :: Parent of IDA, IPI, IMP, IGI & IEP; use a child term if possible.\n"
    "IDA :: Inferred from Direct Assay :: Enzyme assays; in vitro reconstitution; immunofluorescence; binding assays.\n"
    "IPI :: Inferred from Physical Interaction :: Two hybrid, co purification, co immunoprecipitation, ion/protein binding experiments.\n"
    "IMP :: Inferred from Mutant Phenotype :: Gene knockout/over expression; antisense; inhibitors.\n"
    "IGI :: Inferred from Genetic Interaction :: Suppressors, synthetic lethals, complementation, rescue.\n"
    "IEP :: Inferred from Expression Pattern :: Northern/Western, etc.\n"
    "HTP :: Inferred from High Throughput Experiment :: Parent of HDA, HMP, HGI & HEP.\n"
    "HDA :: Inferred from High Throughput Direct Assay :: Proteomics, fluorescence microscopy, etc.\n"
    "HMP :: Inferred from High Throughput Mutant Phenotype :: Large scale perturbation screens.\n"
    "HGI :: Inferred from High Throughput Genetic Interaction :: Large scale genetic screens.\n"
    "HEP :: Inferred from High Throughput Expression Pattern :: Microarray, RNA seq, proteomics.\n"
    "ISS :: Inferred from Sequence or Structural Similarity :: Parent of ISO, ISA & ISM.\n"
    "ISO :: Inferred from Sequence Orthology :: Annotation transfer via orthology.\n"
    "ISA :: Inferred from Sequence Alignment :: Based on pairwise or multiple alignment.\n"
    "ISM :: Inferred from Sequence Model :: HMMs, tRNAscan SE, Rfam, etc.\n"
    "IGC :: Inferred from Genomic Context :: Operons, synteny, pathway context.\n"
    "RCA :: Reviewed Computational Analysis :: Predictions from integrated large scale data.\n"
    "TAS :: Traceable Author Statement :: Statement in review attributed to a cited primary source.\n"
    "NAS :: Non traceable Author Statement :: Database record or publication statement without traceable source.\n"
    "IC  :: Inferred by Curator :: Logical inference from other GO annotations.\n"
    "IEA :: Inferred from Electronic Annotation :: Unreviewed computational transfer (e.g. InterPro2GO).\n"
    "</EVIDENCE CODES>\n\n"
    "<SENIOR AUDITOR EXAMPLES>\n\n"
    "EXAMPLE 1:\n"
    "Description - Rhoptry secretion complex protein TgNd6\n"
    "Corrected -  Rhoptry secretion complex protein Nd6\n"
    "Reason - species specific information should be avoided\n"
    "Explanation for future auditors – in this case Tg in TgNd6 stands for trypanosoma gondii, "
    "we don’t want to include this but we do keep the symbol.\n\n"
    "EXAMPLE 2\n"
    "Description - Palmitoyl acyl transferase PfDHHC16, putative\n"
    "Corrected -  palmitoyltransferase DHHC16\n"
    "Reason - species specific information should be avoided\n"
    "Explanation for future auditors – in this case Pf in PfDHHC16 stands for plasmodium falciparum, "
    "we don’t want to include this. We do keep the gene symbol as part of the description since this "
    "is common practice in Toxoplasma curation. This is not a putative assignment, since based on "
    "our knowledge of DHHC domains, we can infer the palmitoyltransferase activity without the need "
    "for conducting an experimental assay, although having said assay would be even stronger evidence.\n\n"
    "EXAMPLE 3\n"
    "Description - Iron-sulfur cluster assembly scaffold protein, putative\n"
    "Corrected - iron-sulfur cluster assembly protein ISU1\n"
    "Reason – conciseness and disambiguation are preferred\n"
    "Explanation for future auditors – we want the description to be succinct (no need to specify scaffold) and to include "
    "the symbol if possible for disambiguation (add ISU1, it could have been ISU2, ISU3, etc…). "
    "In the supplied summary that is not shown here there was experimental evidence for the "
    "involvement of this protein in the iron-sulfur cluster assembly, so this is no longer a putative "
    "description.\n\n"
    "EXAMPLE 4\n"
    "Description - transcription factor Ace2\n"
    "Corrected - Conidiophore development regulator Ace2\n"
    "Reason – gene function should be specific\n"
    "Explanation for future auditors – while the paper provided evidence that Ace2 is a transcription"
    "factor, it specifically mentioned functions it regulated. The primary focus was Ace2's role in conidiation,"
    "so this was the best possible PD assignment, but other specific function PDs would also be acceptable.\n\n"
    "EXAMPLE 5\n"
    "Description - C6 zinc finger domain transcription factor, putative\n"
    "Corrected - Sclerotia-like\n"
    "Reason – named by authors based on homology to gene in another species\n"
    "Explanation for future auditors – The original descripiton is excellent, but the number one "
    "priority in the hierarchy is to accept author-suggested PDs IF they follow good naming conventions. "
    "The author-suggested description is taken forward to ensure concistency accross species namings.\n\n"
    "EXAMPLE 6\n"
    "Description -GAPM family protein GAPM1B\n"
    "Corrected - glideosome-associated protein with multiple-membrane spans GAPM1B\n"
    "Reason – without specificity the original description is pretty much useles\n"
    "Explanation for future auditors – as GAPM is part of the symbol, specifying this is GAPM family"
    "protein brings nothing new to the PD. Instead, focusing on the novel info from the paper that"
    "it has multiple membrane spans is more informative.\n"
    "EXAMPLE 7\n"
    "Description -serine/threonine protein phosphatase PPM2A, putative\n"
    "Corrected - PP2C family protein phosphatase PPM2A\n"
    "Reason – known function should be preferred over putative\n"
    "Explanation for future auditors – here putative was assigned because it was not certain if the "
    "gene is a serine/threonine phosphatase. It is better to present the family and known but slightly"
    "less specific \"protein phosphatse\" function than assign putative PDs where possible. A putative"
    "PD should only be selected if no novel non-putative information can be extracted.\n\n"
    "</SENIOR AUDITOR EXAMPLES>\n\n"
    "<HIERARCHY>\n"
    "1. author-suggested PD\n"
    "2. describes non-putative function and is specific\n"
    "3. describes non-putative structure/domain and is specific\n"
    "4. describes non-putative function\n"
    "5. describes non-putative structure/domain\n"
    "6. describes putative function and is specific\n"
    "7. describes putative structure/domain and is specific\n"
    "8. describes putative function\n"
    "9. describes putative structure/domain\n"
    "10. conserved, but with unknown function/structure\n"
    "11. anything else that is less insightful\n"
    "</HIERARCHY>\n"
    "OUTPUT FORMAT:\n"
    "Respond in JSON using the folloowing schema:\n"
    "{\n"
    "  \"gene_id\":\"...\",\n"
    "  \"RPD\": {\"verification_status\":\"SUPPORTED\",\"description\":\"...\",\"evidence_code\":\"...\",\"code_reason\":\"...\",\"audit_status\":\"PASS|EDITED\",\"audit_reason\":\"...\"},\n"
    "  \"APDs\": [\n"
    "    {\"verification_status\":\"SUPPORTED|UNSUPPORTED\",\"description\":\"...\",\"evidence_code\":\"...\",\"code_reason\":\"...\",\"audit_status\":\"PASS|EDITED\",\"audit_reason\":\"...\"}\n"
    "  ]\n"
    "}\n"
)

def _build_verify_prompt(gene_text: str, paper_text: str, suggested: Dict[str, Any]) -> Tuple[str, List[str]]:
    # Flatten PDs into a neat numbered block with evidence codes, as your pipeline does.
    lines = []
    for i, pd in enumerate((suggested.get("PDs") or []), start=1):
        desc = pd.get("description", "")
        ev = pd.get("evidence_code", "")
        lines.append(f"{i}. {desc}  [evidence_code: {ev}]")
    pds_block = "\n".join(lines)

    system = VERIFY_PDS_SYSTEM
    user_messages = [
        "Do not respond to this message. Here is the paper text:\n" + paper_text,
        (
            f"The gene of interest is {gene_text}. Here are the suggested PDs:\n"
            "<suggested_descriptions>\n"
            f"{pds_block}\n"
            "</suggested_descriptions>\n"
            "Follow your workflow and return parsable JSON in the OUTPUT format provided in the system prompt."
        ),
    ]
    return system, user_messages

def _verify_pds_json(gene_text: str, paper_text: str, suggested: Dict[str, Any]) -> Dict[str, Any]:
    sys, user_msgs = _build_verify_prompt(gene_text, paper_text, suggested)
    raw = _anthropic_call(sys, user_msgs, model=PD_MODEL)
    txt = _strip_fenced_json(raw)
    try:
        verified = json.loads(txt)
        if not isinstance(verified, dict):
            raise ValueError("verifyPDs output was not a JSON object")
        return verified
    except Exception as e:
        return {"error": f"PD verification failed: {e}", "_raw": raw}

# ---- convert verify output → list of PD triplets -----------------------------
def _verified_to_candidates(verified: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert verifyPDs output into a unique list[ {description,evidence_code,code_reason} ]
    Only include items with verification_status == SUPPORTED (for APDs);
    RPD is included by definition and assumed SUPPORTED per schema.
    Mirrors the helper in your pipeline.  (Adapted)
    """
    def _triplet(d: Dict[str, Any]) -> Dict[str, str] | None:
        if not isinstance(d, dict):
            return None
        desc = (d.get("description") or "").strip()
        if not desc:
            return None
        return {
            "description": desc,
            "evidence_code": (d.get("evidence_code") or "").strip(),
            "code_reason": (d.get("code_reason") or "").strip(),
        }

    cands: List[Dict[str, str]] = []

    # RPD
    rpd = verified.get("RPD")
    t = _triplet(rpd)
    if t:
        cands.append(t)

    # APDs can be list or single dict
    apds = verified.get("APDs")
    if isinstance(apds, list):
        for item in apds:
            if isinstance(item, dict) and item.get("verification_status") == "SUPPORTED":
                t = _triplet(item)
                if t:
                    cands.append(t)
    elif isinstance(apds, dict):
        if apds.get("verification_status") == "SUPPORTED":
            t = _triplet(apds)
            if t:
                cands.append(t)

    # De-duplicate by description
    seen = set()
    unique = []
    for c in cands:
        key = c["description"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# ---- HTML renderer -----------------------------------------------------------
def _pds_html(pd_list: List[Dict[str, str]]) -> str:
    if not pd_list:
        return "<div class='pd-suggestions'><p>No verified product descriptions.</p></div>"
    items = []
    for pd in pd_list[:MAX_PDS]:
        desc = html.escape(pd.get("description", ""))
        ev = html.escape(pd.get("evidence_code", ""))
        defn = EVIDENCE_CODE_DEFS.get(ev.upper(), "")
        defn_html = f" ({html.escape(defn).lower()})" if defn else ""
        reason = html.escape(pd.get("code_reason", ""))

        items.append(
            "<li class='pd-item'>"
            f"  <div class='pd-title'><strong>{desc}</strong></div>"
            "  <details class='pd-why'>"
            "    <summary>Why was this Product Description assigned?</summary>"
            "    <div class='pd-kv'><span class='pd-label'>Evidence code:</span> "
            f"      <span class='pd-value'><code>{ev}</code>{defn_html}</span></div>"
            "    <div class='pd-kv'><span class='pd-label'>Reason for code:</span> "
            f"      <span class='pd-value'>{reason}</span></div>"
            "  </details>"
            "</li>"
        )
    return "<div class='pd-suggestions'><ul>" + "".join(items) + "</ul></div>"

# ---- public entrypoint -------------------------------------------------------
def generate_pds_html(
    gene_id: str,
    pmid: str,
    evidence_bullets: List[str],
    paper_text: str,
    n_pds: int = MAX_PDS,
    host_db: str | None = None,
) -> Dict[str, Any]:
    """
    Two-stage PD flow:
      1) Generate PDs from evidence bullets
      2) Verify PDs against the paper text (only verified appear in HTML)
    """
    # check for cache and if we want to update it or not
    if os.getenv("FORCE_REFRESH") not in ("1", "true", "TRUE"):
        e_blob = json.dumps(evidence_bullets, sort_keys=True, ensure_ascii=False) + "\n" + (paper_text or "")
        e_hash = hashlib.sha256(e_blob.encode("utf-8")).hexdigest()
        key = {
            "pmid": pmid,
            "gene_id": gene_id,
            "n_pds": n_pds,
            "model": PD_MODEL,
            "evidence_hash": e_hash,
        }
        cached = cache_get("pds", key, max_age_s=_ttl("PDS_CACHE_MAX_AGE"))
        if cached:
            return cached

    # Build gene_text from names actually mentioned in the paper
    mentions = []
    hits = _count_substrings(paper_text, gene_id)
    if hits == 0 and "_" in gene_id:
        hits = _count_substrings(paper_text, gene_id.replace("_", "-"))
    if hits > 0:
        mentions.append(gene_id)
    if host_db:
        aliases_in_text = get_gene_synonyms_in_paper(gene_id, paper_text, host_db)
        for a in aliases_in_text:
            if a.lower() not in (n.lower() for n in mentions):
                mentions.append(a)

    gene_text = gene_for_prompt(gene_id, mentions) if mentions else gene_id  # mentions should exist by earlier gating
    # Stage 1
    gen_json = _generate_pds_json(gene_text, evidence_bullets, n_pds)

    # If generation failed, stop early
    if not isinstance(gen_json, dict) or "PDs" not in gen_json:
        return {
            "product_descriptions_html": ui_css() +  "<div class='ai-error'>Could not generate product descriptions.</div>",
            "generated_json": gen_json,
            "verified_json": {},
        }

    # Stage 2 (requires paper_text)
    if not paper_text or not paper_text.strip():
        out =  {
            "product_descriptions_html": ui_css() +  "<div class='ai-info'>Verification skipped (no paper text supplied).</div>",
            "generated_json": gen_json,
            "verified_json": {},
        }
        return out

    verified = _verify_pds_json(gene_text, paper_text, gen_json)
    # Pull verified candidates
    verified_list = _verified_to_candidates(verified)
    html_block = _pds_html(verified_list)

    out =  {
        "product_descriptions_html": html_block,
        "generated_json": gen_json,
        "verified_json": verified,
    }

    # save to cache
    try:
        e_blob = json.dumps(evidence_bullets, sort_keys=True, ensure_ascii=False) + "\n" + (paper_text or "")
        e_hash = hashlib.sha256(e_blob.encode("utf-8")).hexdigest()
        key = {
            "pmid": pmid,
            "gene_id": gene_id,
            "n_pds": n_pds,
            "model": PD_MODEL,
            "evidence_hash": e_hash,
        }
        cache_set("pds", key, out)
    finally:
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--gene_id", required=True)
    ap.add_argument("--bullets_json", required=True, help="JSON array of bullet strings")
    ap.add_argument("--paper_file", help="Path to a file containing paper text (required for verification)")
    ap.add_argument("--n_pds", type=int, default=MAX_PDS)
    args = ap.parse_args()

    bullets = json.loads(args.bullets_json)
    paper_text = ""
    if args.paper_file:
        with open(args.paper_file, "r", encoding="utf-8") as f:
            paper_text = f.read()

    out = generate_pds_html(args.gene_id, bullets, paper_text, n_pds=args.n_pds)
    print(json.dumps(out, indent=2))
