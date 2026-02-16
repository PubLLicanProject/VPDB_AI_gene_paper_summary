"""
Gene Summary -> brainstorm PD -> select PD Pipeline
Used to process all user comments witha tagged paper on VPDB
Processes gene-paper pairs from CSV with batch processing and intelligent caching.
NB: uses claude-sonnet-4-20250514 as our best performing model
"""

import os
import time
import json
import requests
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

#######################################################################################################################
#                                           CONFIGURATION                                                             #
#######################################################################################################################

# API Setup
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Models
SUMMARY_MODEL = "claude-sonnet-4-20250514"
PD_GENERATOR_MODEL = "claude-sonnet-4-20250514"
PD_VERIFIER_MODEL = "claude-sonnet-4-20250514"
FORMATTER_MODEL = "claude-sonnet-4-20250514"


# Processing Parameters
MAX_TOKENS = 20000
MODEL_TEMP = 0
N_QUOTES = 2
MAX_RETRY = 3
HTTP_TIMEOUT = 30
N_PDs = 3  # Maximum number of PDs to generate

# Output
OUT_DIR = Path("./out/VPDB_UserComments_batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing Strategy
MIN_GENES_FOR_CACHING = 3  # Use caching if paper has 3+ genes
USE_BATCH_FOR_LOW_DENSITY = True  # Use batch API for papers with <3 genes (falls back to standard if unavailable)
OVERWRITE_EXISTING = False  # Set to True to reprocess already-completed pairs

# CSV Column Mapping (adjust these if your CSV has different column names)
CSV_COLUMNS = {
    'gene_id': 'Gene ID',        # Column containing gene IDs (e.g., "PF3D7_1234")
    'pmid': 'PMID_Final',        # Column containing PubMed IDs (e.g., "12345678")
    'database': 'Database',       # Column containing database names (e.g., "plasmodb", "toxodb")
    # Optional columns for filtering (will be used if present):
    'paper_available': 'paper_available',  # Boolean: whether paper is available in PMC
    'alias_in_text': 'alias_in_text'       # Boolean: whether gene alias found in paper
}

#######################################################################################################################
#                                    PAPER FETCHING (UPDATED FUNCTIONS)                                               #
#######################################################################################################################

PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
PUBMED_SECTIONS = ['RESULTS', 'FIG', 'DISCUSSION', 'DISCUSS', 'CONCLUSION', 'FIGURE', 'CONCL', 'TABLE', 'SUPPL']


def parse_pubmed_json(pubmed_json):
    """Parse PubMed JSON to extract relevant sections."""
    document_text = ""
    for doc in pubmed_json:
        for document in doc.get("documents", []):
            for passage in document.get("passages", []):
                section_type = passage.get("infons", {}).get("section_type", "")
                if section_type.upper() in {s.upper() for s in PUBMED_SECTIONS}:
                    if "text" in passage:
                        document_text += passage["text"] + "\n"
    return document_text


def get_pubmed_json(pubmed_id):
    """Fetch PubMed JSON with improved error handling."""
    url = PUBMED_BASE_URL + str(pubmed_id) + "/unicode"
    response = requests.get(url, timeout=HTTP_TIMEOUT)

    if response.status_code == 200:
        if not response.headers.get('Content-Type', '').startswith('application/json'):
            raise ValueError(f"Paper {pubmed_id} not available in PMC Open Access subset")
        return response.json()

    raise ValueError(f"Paper fetch failed for {pubmed_id}, status: {response.status_code}")


def get_paper_text(pubmed_id: str) -> str:
    """Fetch and parse paper text."""
    pubmed_json = get_pubmed_json(pubmed_id)
    return parse_pubmed_json(pubmed_json)


def check_paper_available(pubmed_id: str) -> bool:
    """
    Check if paper is available via PMC API.

    Args:
        pubmed_id: PubMed ID

    Returns:
        True if paper can be fetched, False otherwise
    """
    try:
        get_paper_text(pubmed_id)
        return True
    except Exception:
        return False


#######################################################################################################################
#                                    GENE ALIAS RETRIEVAL (UPDATED FUNCTIONS)                                         #
#######################################################################################################################

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


#######################################################################################################################
#                                    HELPER FUNCTIONS FOR PD PIPELINE                                                 #
#######################################################################################################################

def get_prompt_and_replace(stage_key: str, replacements: dict, prompt_type: str) -> str:
    """
    Get prompt from global_prompts_and_schema and replace placeholders.

    Args:
        stage_key: Key in global_prompts_and_schema (e.g., "generatePDs")
        replacements: Dict of placeholder -> value mappings
        prompt_type: "SystemPrompt" or "UserPrompts"

    Returns:
        Processed prompt string with replacements applied
    """
    prompt = global_prompts_and_schema[stage_key][prompt_type]

    # Handle UserPrompts as list (join them)
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)

    # Apply replacements
    for key, value in replacements.items():
        # Handle both dict and non-dict values
        if isinstance(value, dict):
            value = json.dumps(value, indent=2)
        else:
            value = str(value)
        prompt = prompt.replace(f"[{key}]", value)

    return prompt


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
        return bullets

    return []


#######################################################################################################################
#                                         PROMPTS AND SCHEMA                                                          #
#######################################################################################################################
# TODO: modify prompts to ensure species information is taken into account, author-assigned PD are high prio, and evidence codes are carefully selected
global_prompts_and_schema = {
#### STAGE 1: generate a gene-specific summary
    "getGeneSummary": {
        "SystemPrompt":  "ROLE: You are a scientist conducting a systematic literature review focused on ONE SPECIFIC gene of interest.\n"
        "GOAL: Analyze the provided research paper and extract ALL information relevant to the specified gene. The gene"
        "or one of its aliases is confirmed to be present in the text.\n"
        "INSTRUCTIONS: Follow this step-by-step workflow:\n"
        "1. EXPERIMENTAL ANALYSIS\n"
        "   - Identify and describe all experiments involving the specified gene\n"
        "   - Report experimental methods, results, and conclusions related to gene activity or expression\n"
        "   - Note any functional studies, assays, or measurements"
        "   - Be especially careful to report if conclusions about the gene were drawn based on an ortholog from a different species\n\n"
        "2. CONTEXTUAL RELEVANCE\n"
        "   - Extract any additional information relevant to understanding this gene's role\n"
        "   - Include regulatory mechanisms, interactions, pathways, or clinical significance\n"
        "   - Consider implications for gene function or therapeutic potential\n\n"
        "3. KEY FINDINGS SYNTHESIS\n"
        "   - Summarize discoveries as numbered bullet points\n"
        "   - Include one bullet point that lists all names given to the gene by the author.\n"
        "   - Prioritize findings by significance and directness of evidence\n\n"
        "4. EVIDENCE CLASSIFICATION\n"
        "   For each bullet point, determine:\n"
        "   - Is this based on DIRECT EVIDENCE from the text?\n"
        "   - Is this an INFERENCE you made from the data?\n"
        "   For direct evidence:\n"
        "   - Specify source location (e.g. Figure 1, Table 4, Results section - paragraph 4, Supplementary material, or Discussion)\n"
        "   - Extract up to [N_QUOTES] verbatim quotes supporting each point\n\n"
        "5. STRUCTURED OUTPUT\n"
        "   Organize findings into:\n"
        "   - <Gene Summary>: Direct evidence-based bullet points with citations and quotes\n"
        "   - <Additional Inferences>: Your analytical interpretations of the data\n\n"
        "   - <Short Summary>: A concise one-sentence summary capturing and synthesising key novel findings from the "
        "Gene Summary bullet points \n\n"
        "6. MINIMAL MENTION HANDLING\n"
        "   If the gene appears only peripherally without substantial findings, provide:\n"
        "   - Single bullet point: \"Mentioned in passing\"\n"
        "   - Include context and direct quotes showing where/how it's referenced\n"
        "   - Flag in the JSON schema only_in_passing: TRUE\n"
        "7. FORMAT COMPLIANCE"
        "Structure your response according to the provided JSON schema:\n"
        "[JSON_SCHEMA]\n"
        "Remember: Focus exclusively on the specified gene. Be precise about distinguishing direct evidence from analytical inferences.",

        "UserPrompts": [
            "Do not respond to this message. Here is the paper text: \n [PAPER_TEXT]",
            "Please generate a summary for [GENE] based on the supplied paper text."
        ],

        "ValidationSchema": {
                    "type": "object",
                    "required": [
                        "Aliases_in_paper",
                        "GeneSummary",
                        "AdditionalInferences",
                        "ShortSummary"
                    ],
                    "properties": {
                        "Aliases_in_paper": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of known aliases or synonyms associated with the gene found in the paper."
                        },
                        "GeneSummary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                            "bullet_point",
                            "evidence_location",
                            "supporting_quotes",
                            "only_in_passing"
                            ],
                            "properties": {
                                "bullet_point": {
                                    "type": "string",
                                    "description": "A concise bullet-point summary describing the direct evidence about the gene."
                                },
                                "evidence_location": {
                                    "type": "string",
                                    "description": "Where in the paper the evidence can be found (e.g., Figure 1, Table 2, Results paragraph 3, Supplementary, or Discussion paragraph 5)."
                                },
                                "supporting_quotes": {
                                    "type": "array",
                                    "items": {
                                    "type": "string"
                                    },
                                    "maxItems": 2,
                                    "description": "Up to 2 direct quotes from the paper that support the summary bullet point."
                                },
                                    # NEW: add specific flag to schema
                                "only_in_passing":{
                                    "type": "boolean",
                                    "description": "Flags if gene is only mentioned in passing without substantial findings - TRUE if so, FALSE if not."
                                },
                            }
                        },
                        "description": "Bullet points from the <Gene Summary> section, each with its own paired evidence location and supporting quotes."
                        },
                        "AdditionalInferences": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Bullet points from the <Additional Inferences> section, capturing inferred conclusions not directly evidenced."
                        },
                        "ShortSummary":{
                            "type": "string",
                            "description": "The one sentence <Short Summary> that captures key findings."
                        }
                    },
                    "additionalProperties": False
                    },
    },
#### STAGE 2: Brainstorm a set of product descriptions
    "generatePDs": {
        "SystemPrompt": "ROLE: You are an expert gene curation assistant specializing in synthesizing "
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
                        "- Identify experimental vs. computational evidence; be very careful not to mix-up orthology-based evidence as direct experimental evidence - pay attention to the species.\n"
                        "STEP 2: Description Brainstorming"
                        "- Refer to the TEMPLATES, and GUIDELINES to generate one appropriate PD at a time,"
                        " up to a total of [N_PDs]."
                        "- If the authors provide a product description for a gene, prioritise their naming convention above all else.\n"
                        "- Next, prioritise PDs founded on experimental evidence over computational predictions, but DO NOT confuse orthology-based assignments as experimental evidence that comes from another species\n"
                        "- Only create additional PDs when multiple valid alternatives exist, ensuring PDs "
                        "are distinct and scientifically meaningful\n"
                        "- Initially assign a \"putative\" qualifier to all PDs, then review if it needs to be removed as"
                        "described in the RULES.\n"
                        "Step 3: Evidence code assignment \n"
                        "Assign appropriate GO evidence codes with justification for each PD - refer to the EVIDENCE CODES "
                        "section for more details. Explain briefly why the code was selected as evidence_reason. Be especially careful with assigning "
                        "experimental codes - sometimes the experiments are done in another species, in which case \n\n"
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
                        "strip actual parts of the gene symbol that are NOT species-related.\n"
                        "Rule 8: Pay extra attention to the species in which experimetns were performed. If experimental evidence exists in an ortologue, "
                        "but not for the species this gene is associated with, make sure the description is labelled as putative and the correct evidence code is supplied."
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
                        "- Is species information correctly stripped away? If in doubt, it is best to leave it in.\n",

        "UserPrompts":  [
            "Please generate a one sentence summary and product descriptions based on the following summary for [GENE]:\n\n"
            "[SUMMARY]"
        ],
        "ValidationSchema": {
                  "type": "object",
                  "required": ["gene_id", "short_summary", "PDs"],
                  "properties": {
                    "gene_id": {
                      "type": "string",
                      "description": "Locus tag or accession for the gene."
                    },
                    "short_summary": {
                      "type": "string",
                      "description": "One sentence synthesised summary of the gene's MF, BP, and CC."
                    },
                    "PDs": {
                      "type": "array",
                      "description": f"Up to {N_PDs} product descriptions",
                      "items": {
                        "type": "object",
                        "required": ["description", "evidence_code", "code_reason"],
                        "properties": {
                          "description": { "type": "string" },
                          "evidence_code": {
                            "type": "string",
                            "pattern": "^(EXP|IDA|IPI|IMP|IGI|IEP|HTP|HDA|HMP|HGI|HEP|ISS|ISO|ISA|ISM|IGC|RCA|TAS|NAS|IC|IEA)(; ?(EXP|IDA|IPI|IMP|IGI|IEP|HTP|HDA|HMP|HGI|HEP|ISS|ISO|ISA|ISM|IGC|RCA|TAS|NAS|IC|IEA))*$",
                            "description": "GO evidence code supporting the PD."
                          },
                          "code_reason": { "type": "string" }
                        },
                        "additionalProperties": False
                      }
                    }
                  },
                  "additionalProperties": False

        }
    },
#### STAGE 3: Final check - are the brainstormed  PDs supported by the original text which is recommended?
    "verifyPDs": {
        "SystemPrompt": "ROLE: You audit gene curation records and specialise in scrutinising product descriptions\n"
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
                        "- Pay extra attention to the species in which experimetns were performed. If experimental evidence exists in an ortologue, "
                        "but not for the species this gene is associated with, make sure the description is labelled as putative and the correct evidence code is supplied.\n"
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
                        "EXAMPLE 8\n"
                        "Description - apicoplast protein import factor PPP1\n"
                        "Corrected - peripheral plastid protein PPP1\n"
                        "Reason – As per the hierarchy, author-suggested PD is to be prioritiesed.\n"
                        "The authors used this product description when introducing the gene abbreviaiton PPP1 as per a sentence in the paper text. Thus, this would be the recommended product. The alternative description 'apicoplast protein import factor' is also correct, but would be acceptable as an alternative description–\n"
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
                        "[JSON_SCHEMA]",
        "UserPrompts": [
            "Do not respond to this message. Here is the paper text: \n [PAPER_TEXT]",
            "The gene of interest is [GENE]. Here are the suggested PDs:\n"
            "<suggested_descriptions>\n"
            "[PDs]\n"
            "</suggested_descriptions>\n"
            "Follow your workflow and return parsable JSON as per the OUTPUT FORMAT schema supplied. "
        ],
        "ValidationSchema": {
                  "type": "object",
                  "required": ["gene_id", "RPD", "APDs"],
                  "properties": {
                    "gene_id": {
                      "type": "string",
                      "description": "Locus tag or accession for the gene."
                    },
                    "RPD": {
                      "type": "object",
                      "description": "recommended PD",
                      "required": ["verification_status", "description", "evidence_code", "code_reason", "audit_status", "audit_reason"],
                      "properties": {
                        "verification_status": {
                            "type": "string",
                            "description": "SUPPORTED if evidence in text; do not assign UNSUPPORTED as RPD."
                        },

                        "description": {
                            "type": "string",
                            "description": "Product description that was audited (including edits)"
                        },

                        "evidence_code": {
                            "type": "string",
                            "description": "selected evidence code"
                        },

                        "code_reason": {
                            "type": "string",
                            "description": "Reason why you selected this evidence code."
                        },

                        "audit_status": {
                            "type": "string",
                            "description": "PASS or EDITED"
                        },

                        "audit_reason": {
                            "type": "string",
                            "description": "Reason why PD was edited if audit status is EDITED."
                        }

                      },
                      "additionalProperties": False
                    }
                  },
            "APDs": {
                "type": "object",
                "description": "alternative PDs to the RPD",
                "required": ["verification_status", "description", "evidence_code", "code_reason", "audit_status",
                             "audit_reason"],
                "properties": {
                    "verification_status": {
                        "type": "string",
                        "description": "SUPPORTED if evidence in text or UNSUPPORTED if not"
                    },

                    "description": {
                        "type": "string",
                        "description": "Product description that was audited (including edits)"
                    },

                    "evidence_code": {
                        "type": "string",
                        "description": "selected evidence code"
                    },

                    "code_reason": {
                        "type": "string",
                        "description": "Reason why you selected this evidence code."
                    },

                    "audit_status": {
                        "type": "string",
                        "description": "PASS or EDITED"
                    },

                    "audit_reason": {
                        "type": "string",
                        "description": "Reason why PD was edited if audit status is EDITED."
                    }

                },
                "additionalProperties": False
            }
        }

    },
}

# Extract commonly used variables from global_prompts_and_schema
SUMMARY_SYSTEM_PROMPT = global_prompts_and_schema["getGeneSummary"]["SystemPrompt"]
VALIDATION_SCHEMA = global_prompts_and_schema["getGeneSummary"]["ValidationSchema"]

#######################################################################################################################
#                                         UTILITY FUNCTIONS                                                           #
#######################################################################################################################

def is_gene_already_processed(pubmed_id: str, gene_id: str, check_all_steps: bool = True) -> bool:
    """
    Check if gene-paper pair has been fully processed.

    Args:
        pubmed_id: Paper ID
        gene_id: Gene ID
        check_all_steps: If True, checks summary + PDs + verify all complete
                        If False, only checks summary (backward compatible)

    Returns:
        True if processing is complete
    """
    filename = OUT_DIR / f"{pubmed_id}.json"

    if not filename.exists():
        return False

    try:
        with open(filename, 'r') as f:
            paper_data = json.load(f)

        if not check_all_steps:
            # Old behavior - just check summary
            if gene_id in paper_data.get("getGeneSummary", {}):
                gene_data = paper_data["getGeneSummary"][gene_id]
                if SUMMARY_MODEL in gene_data:
                    model_result = gene_data[SUMMARY_MODEL]
                    # Consider it processed if success=True and data is not None
                    if model_result.get("success") and model_result.get("data") is not None:
                        return True
            return False

        # New behavior - check all three steps
        required_steps = [
            ("getGeneSummary", SUMMARY_MODEL),
            ("generatePDs", PD_GENERATOR_MODEL),
            ("verifyPDs", PD_VERIFIER_MODEL)
        ]

        for step_key, model in required_steps:
            step_data = paper_data.get(step_key, {}).get(gene_id, {})
            model_data = step_data.get(model, {})

            if not (model_data.get("success") and model_data.get("data")):
                return False  # This step not complete

        return True  # All steps complete!

    except Exception:
        return False


def save_result(pubmed_id: str, gene_id: str, data: dict, success: bool,
                step_key: str = "getGeneSummary", usage: dict = None,
                seconds: float = None, model: str = None):
    """
    Save result to JSON file (one file per paper with nested genes and pipeline steps).

    Args:
        pubmed_id: Paper ID
        gene_id: Gene ID
        data: Result data (or error dict)
        success: Whether step succeeded
        step_key: Pipeline step name ("getGeneSummary", "generatePDs", "verifyPDs")
        usage: Token usage dict
        seconds: Elapsed time
        model: Model name (defaults to appropriate model for step_key)
    """
    filename = OUT_DIR / f"{pubmed_id}.json"

    # Load existing file if it exists
    if filename.exists():
        with open(filename, 'r') as f:
            paper_data = json.load(f)
    else:
        paper_data = {}

    # Initialize step if needed
    if step_key not in paper_data:
        paper_data[step_key] = {}

    # Add or update this gene's data
    if gene_id not in paper_data[step_key]:
        paper_data[step_key][gene_id] = {"gene_ID": gene_id}

    # Determine model name if not provided
    if model is None:
        if step_key == "getGeneSummary":
            model = SUMMARY_MODEL
        elif step_key == "generatePDs":
            model = PD_GENERATOR_MODEL
        elif step_key == "verifyPDs":
            model = PD_VERIFIER_MODEL
        else:
            model = SUMMARY_MODEL  # fallback

    # Store model results
    paper_data[step_key][gene_id][model] = {
        "model": model,
        "success": success,
        "data": data
    }

    if usage:
        paper_data[step_key][gene_id][model]["usage"] = usage
    if seconds:
        paper_data[step_key][gene_id][model]["seconds"] = seconds

    # Save back to file
    with open(filename, 'w') as f:
        json.dump(paper_data, f, indent=2)


def format_with_retry(content: str, schema: dict, max_attempts: int = MAX_RETRY) -> Optional[dict]:
    """Parse JSON with retry using formatter model."""
    # Try direct parsing first, after stripping markdown
    try:
        # Strip markdown code fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
            cleaned = re.sub(r'\n```\s*$', '', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed: {e}")
        print(f"  First 200 chars: {content[:200]}")

    # Try with formatter model
    for attempt in range(max_attempts):
        try:
            print(f"  Retry {attempt + 1}/{max_attempts} with formatter model...")
            response = anthropic_client.messages.create(
                model=FORMATTER_MODEL,
                max_tokens=4000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"Convert this to valid JSON matching the schema. Output ONLY valid JSON, no markdown:\n\n{content}\n\nSchema:\n{json.dumps(schema)}"
                }]
            )
            result_text = response.content[0].text.strip()
            # Strip markdown again
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\s*\n', '', result_text)
                result_text = re.sub(r'\n```\s*$', '', result_text)
            return json.loads(result_text)
        except Exception as e:
            print(f"  Retry {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                print(f"  ERROR: All retries failed. Saving raw response for debugging.")
                # Save raw response for debugging
                debug_file = OUT_DIR / "debug_failed_parse.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Original content:\n{content}\n\n")
                    f.write(f"Schema:\n{json.dumps(schema, indent=2)}\n")
                print(f"  Raw response saved to: {debug_file}")
                return None
            time.sleep(1)
    return None


#######################################################################################################################
#                                    PD GENERATION FUNCTIONS                                                          #
#######################################################################################################################

def generatePDs(summary_json: Dict[Any, Any], gene_text: str, n_pds: int = N_PDs) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Generate product descriptions from gene summary.

    Args:
        summary_json: Output from getGeneSummary
        gene_text: Gene ID with aliases (e.g., "PF3D7_1234 (also known as ABC, DEF)")
        n_pds: Maximum number of PDs to generate

    Returns:
        Tuple of (parsed_result, usage_dict, elapsed_seconds)
        Returns (None, {}, None) on failure
    """
    # Validate input
    if not isinstance(summary_json, dict):
        print("  ERROR: generatePDs requires dict input from summary")
        return None, {}, None

    # Get schema
    schema = global_prompts_and_schema["generatePDs"]["ValidationSchema"]

    # Extract bullet points
    bullets = collect_bullets(summary_json)
    if not bullets:
        print("  WARNING: No bullet points found in summary")
        return {"error": "No bullets in summary"}, {}, None

    summary_text = "\n".join(bullets)

    # Build prompts
    replacements = {
        "N_PDs": n_pds,
        "GENE": gene_text,
        "JSON_SCHEMA": schema,
        "SUMMARY": summary_text
    }

    system_prompt = get_prompt_and_replace("generatePDs", replacements, "SystemPrompt")
    user_prompt = get_prompt_and_replace("generatePDs", replacements, "UserPrompts")

    # Call API and track timing
    start = time.time()

    try:
        response = anthropic_client.messages.create(
            model=PD_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=MODEL_TEMP,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        elapsed = time.time() - start

        # Track usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Parse with retry
        result = format_with_retry(response.content[0].text, schema)

        if not result:
            print("  ERROR: generatePDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: generatePDs API call failed: {e}")
        return None, {}, elapsed


def verifyPDs(suggested_pds: Dict[Any, Any], paper_text: str, gene_text: str,
              use_caching: bool = True) -> Tuple[Optional[dict], dict, Optional[float]]:
    """
    Verify and select product descriptions against paper evidence.

    IMPORTANT: This function can reuse cached paper text from summary generation!
    When use_caching=True, the paper text will be cached for efficient reuse.

    Args:
        suggested_pds: Output from generatePDs
        paper_text: Full paper text (will be cached if use_caching=True)
        gene_text: Gene ID with aliases
        use_caching: Whether to cache paper text (should be True in batch mode)

    Returns:
        Tuple of (parsed_result, usage_dict, elapsed_seconds)
        Returns (None, {}, None) on failure
    """
    # Validate input
    if not isinstance(suggested_pds, dict):
        print("  ERROR: verifyPDs requires dict input from generatePDs")
        return None, {}, None

    # Get schema
    schema = global_prompts_and_schema["verifyPDs"]["ValidationSchema"]

    # Extract PDs into formatted list
    pds = suggested_pds.get("PDs", [])
    if not pds:
        print("  WARNING: No PDs found to verify")
        return {"error": "No PDs to verify"}, {}, None

    pd_lines = [
        f"{i + 1}. {pd['description']}  "
        f"[evidence_code: {pd.get('evidence_code', 'N/A')}]"
        for i, pd in enumerate(pds)
    ]
    pd_text = "\n".join(pd_lines)

    # Build prompts
    replacements = {
        "GENE": gene_text,
        "JSON_SCHEMA": schema,
        "PAPER_TEXT": paper_text,
        "PDs": pd_text
    }

    system_prompt = get_prompt_and_replace("verifyPDs", replacements, "SystemPrompt")
    user_prompt = get_prompt_and_replace("verifyPDs", replacements, "UserPrompts")

    # Call API with optional caching
    start = time.time()

    try:
        if use_caching:
            # Use caching structure - paper text will be cached
            # This reuses the cache from summary generation if within cache TTL!
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do not respond. Here is the paper text:"},
                        {"type": "text", "text": paper_text, "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "assistant", "content": "I have received the paper text."},
                {"role": "user", "content": user_prompt}
            ]

            response = anthropic_client.messages.create(
                model=PD_VERIFIER_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages
            )

            # Track cache usage
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
            }
        else:
            # No caching - simple call
            response = anthropic_client.messages.create(
                model=PD_VERIFIER_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        elapsed = time.time() - start

        # Parse with retry
        result = format_with_retry(response.content[0].text, schema)

        if not result:
            print("  ERROR: verifyPDs parsing failed after retries")
            return None, usage, elapsed

        return result, usage, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: verifyPDs API call failed: {e}")
        return None, {}, elapsed


#######################################################################################################################
#                                    PROMPT CACHING IMPLEMENTATION                                                    #
#######################################################################################################################

def process_paper_with_caching(pubmed_id: str, gene_list: List[Tuple[str, str]],
                               save: bool = True) -> List[dict]:
    """Process multiple genes from same paper with prompt caching."""
    print(f"\n{'=' * 80}")
    print(f"Paper {pubmed_id}: {len(gene_list)} genes (USING CACHING)")
    print(f"{'=' * 80}")

    # Try to fetch paper - check if available
    paper_available = False
    paper_text = None
    try:
        paper_text = get_paper_text(pubmed_id)
        paper_available = True
    except Exception as e:
        print(f"  ✗ Paper not available in PMC: {e}")
        # Return all genes as failed with paper_available=False
        results = []
        for gene_id, host_db in gene_list:
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": "Paper not available in PMC",
                "paper_available": False,
                "alias_in_text": False,
                "mentions": 0
            })
        return results

    system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
    system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

    results = []

    for i, (gene_id, host_db) in enumerate(gene_list, 1):
        print(f"  [{i}/{len(gene_list)}] {gene_id}...", end=" ")

        # Check if gene/aliases are in paper text
        alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, host_db)

        if not alias_in_text:
            print(f"✗ Gene/aliases not found in text (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": "Gene/aliases not found in paper text",
                "paper_available": True,
                "alias_in_text": False,
                "mentions": 0
            })
            continue

        print(f"({mentions} mentions) ", end="")

        # Check if already processed
        if not OVERWRITE_EXISTING and is_gene_already_processed(pubmed_id, gene_id):
            print(f"✓ Already processed (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "skipped": True,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })
            continue

        start = time.time()

        try:
            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            # Messages with caching
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do not respond. Here is the paper text:"},
                        {"type": "text", "text": paper_text, "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "assistant", "content": "I have received the paper text."},
                {"role": "user", "content": f"Generate summary for {gene_display}."}
            ]

            response = anthropic_client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages
            )

            # Track usage first (needed even if parsing fails)
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
            }

            result = format_with_retry(response.content[0].text, VALIDATION_SCHEMA)

            elapsed = time.time() - start

            if not result:
                # Save raw response for debugging
                print(f"  ✗ Failed to parse response - saving raw output for debugging")
                debug_data = {
                    "error": "JSON parsing failed after retries",
                    "raw_response": response.content[0].text[:1000],  # First 1000 chars
                    "model": SUMMARY_MODEL
                }
                if save:
                    save_result(pubmed_id, gene_id, debug_data, False, "getGeneSummary", usage, elapsed)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "JSON parsing failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue

            if save:
                save_result(pubmed_id, gene_id, result, True, "getGeneSummary", usage, elapsed)

            print(f"✓ Summary ({elapsed:.1f}s)", end=" | ")

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Generate PDs
            # ═══════════════════════════════════════════════════════════════
            pd_result, pd_usage, pd_time = generatePDs(
                summary_json=result,
                gene_text=gene_display,
                n_pds=N_PDs
            )

            if not pd_result:
                print(f"✗ PD gen failed")
                if save:
                    save_result(pubmed_id, gene_id,
                               {"error": "PD generation failed"},
                               False, "generatePDs", pd_usage, pd_time)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "PD generation failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue  # Skip verify if PD gen failed

            # Save PDs
            if save:
                save_result(pubmed_id, gene_id, pd_result, True,
                           "generatePDs", pd_usage, pd_time)

            print(f"✓ PDs ({pd_time:.1f}s)", end=" | ")

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Verify PDs
            # ═══════════════════════════════════════════════════════════════
            verify_result, verify_usage, verify_time = verifyPDs(
                suggested_pds=pd_result,
                paper_text=paper_text,  # Reuses cached paper!
                gene_text=gene_display,
                use_caching=True  # CRITICAL for cache reuse!
            )

            if not verify_result:
                print(f"✗ Verify failed")
                if save:
                    save_result(pubmed_id, gene_id,
                               {"error": "PD verification failed"},
                               False, "verifyPDs", verify_usage, verify_time)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "PD verification failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
                continue

            # Save verified PDs
            if save:
                save_result(pubmed_id, gene_id, verify_result, True,
                           "verifyPDs", verify_usage, verify_time)

            total_time = elapsed + pd_time + verify_time
            cache_info = f"cache={verify_usage['cache_read_input_tokens']:,}" if i > 1 else f"created={verify_usage.get('cache_creation_input_tokens', 0):,}"
            print(f"✓ Verified ({verify_time:.1f}s) | Total: {total_time:.1f}s ({cache_info})")

            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "data": result,
                "usage": usage,
                "summary_time": elapsed,
                "pd_time": pd_time,
                "verify_time": verify_time,
                "total_time": total_time,
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": str(e),
                "paper_available": True,
                "alias_in_text": True,
                "mentions": mentions
            })

    return results


#######################################################################################################################
#                                      BATCH PROCESSING                                                               #
#######################################################################################################################

def create_batch_requests(pairs: List[Tuple[str, str, str]]) -> List[dict]:
    """Create batch requests for low-density papers."""
    batch_requests = []

    for pubmed_id, gene_id, host_db in pairs:
        try:
            paper_text = get_paper_text(pubmed_id)
            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
            system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

            batch_requests.append({
                "custom_id": f"{pubmed_id}_{gene_id}",
                "params": {
                    "model": SUMMARY_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "temperature": MODEL_TEMP,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": f"Do not respond. Here is the paper text:\n\n{paper_text}"},
                        {"role": "assistant", "content": "I have received the paper text."},
                        {"role": "user", "content": f"Generate summary for {gene_display}."}
                    ]
                }
            })
        except Exception as e:
            print(f"  ✗ Failed to prepare {pubmed_id}_{gene_id}: {e}")

    return batch_requests


def process_batch_fallback(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
    """Fallback to standard processing when batch API is unavailable."""
    print(f"Processing {len(pairs)} pairs with standard API (sequential)...")

    results = []
    for i, (pubmed_id, gene_id, host_db) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {pubmed_id}_{gene_id}...", end=" ")

        # Check if already processed
        if not OVERWRITE_EXISTING and is_gene_already_processed(pubmed_id, gene_id):
            print("✓ Already processed (skipped)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": True,
                "skipped": True
            })
            continue

        start = time.time()

        # Initialize diagnostic flags
        paper_available = False
        alias_in_text = False
        mentions = 0

        try:
            # Fetch paper - check if available
            try:
                paper_text = get_paper_text(pubmed_id)
                paper_available = True
            except Exception as e:
                print(f"✗ Paper not available ({time.time() - start:.1f}s)")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": f"Paper not available: {str(e)}",
                    "paper_available": False,
                    "alias_in_text": False,
                    "mentions": 0
                })
                continue

            # Check if gene/aliases are in text
            alias_in_text, mentions = check_gene_in_text(gene_id, paper_text, host_db)

            if not alias_in_text:
                print(f"✗ Gene not in text ({time.time() - start:.1f}s)")
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "Gene/aliases not found in paper text",
                    "paper_available": True,
                    "alias_in_text": False,
                    "mentions": 0
                })
                continue

            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

            # Prepare prompts
            system_prompt = SUMMARY_SYSTEM_PROMPT.replace("[JSON_SCHEMA]", json.dumps(VALIDATION_SCHEMA))
            system_prompt = system_prompt.replace("[N_QUOTES]", str(N_QUOTES))

            # Call API
            response = anthropic_client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=MODEL_TEMP,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Do not respond. Here is the paper text:\n\n{paper_text}"},
                    {"role": "assistant", "content": "I have received the paper text."},
                    {"role": "user", "content": f"Generate summary for {gene_display}."}
                ]
            )

            # Track usage
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            # Parse result
            result = format_with_retry(response.content[0].text, VALIDATION_SCHEMA)
            elapsed = time.time() - start

            if not result:
                print(f"✗ Parse failed ({elapsed:.1f}s)")
                error_data = {"error": "JSON parsing failed", "raw_response": response.content[0].text[:1000]}
                if save:
                    save_result(pubmed_id, gene_id, error_data, False, "getGeneSummary", usage, elapsed)
                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": False,
                    "error": "JSON parsing failed",
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })
            else:
                # Summary success
                if save:
                    save_result(pubmed_id, gene_id, result, True, "getGeneSummary", usage, elapsed)

                print(f"✓ Summary ({elapsed:.1f}s)", end=" | ")

                # Generate PDs
                try:
                    pd_result, pd_usage, pd_time = generatePDs(
                        summary_json=result,
                        gene_text=gene_display,
                        n_pds=N_PDs
                    )

                    if pd_result and save:
                        save_result(pubmed_id, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)
                        print(f"✓ PDs ({pd_time:.1f}s)", end=" | ")

                        # Verify PDs
                        verify_result, verify_usage, verify_time = verifyPDs(
                            suggested_pds=pd_result,
                            paper_text=paper_text,
                            gene_text=gene_display,
                            use_caching=False
                        )

                        if verify_result and save:
                            save_result(pubmed_id, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)
                            total_time = elapsed + pd_time + verify_time
                            print(f"✓ Verified ({verify_time:.1f}s) | Total: {total_time:.1f}s")
                        else:
                            print(f"✗ Verify failed")
                    else:
                        print(f"✗ PD gen failed")
                except Exception as e:
                    print(f"✗ PD error: {e}")

                results.append({
                    "pubmed_id": pubmed_id,
                    "gene_id": gene_id,
                    "success": True,
                    "data": result,
                    "paper_available": True,
                    "alias_in_text": True,
                    "mentions": mentions
                })

        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ Error: {e} ({elapsed:.1f}s)")
            results.append({
                "pubmed_id": pubmed_id,
                "gene_id": gene_id,
                "success": False,
                "error": str(e),
                "paper_available": paper_available,
                "alias_in_text": alias_in_text,
                "mentions": mentions
            })

    return results


def process_batch(pairs: List[Tuple[str, str, str]], save: bool = True) -> List[dict]:
    """Process pairs using batch API (with fallback to standard if unavailable)."""
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING: {len(pairs)} pairs")
    print(f"{'=' * 80}")

    # Check if batch API is available
    if not hasattr(anthropic_client.messages, 'batches'):
        print("Warning: Batch API not available in your Anthropic SDK version.")
        print("Falling back to standard processing...")
        print("To enable batch API, upgrade: pip install --upgrade anthropic")
        print()
        return process_batch_fallback(pairs, save)

    batch_requests = create_batch_requests(pairs)
    if not batch_requests:
        return []

    try:
        batch = anthropic_client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        print(f"Batch submitted: {batch_id}")
    except Exception as e:
        print(f"Batch submission failed: {e}")
        print("Falling back to standard processing...")
        return process_batch_fallback(pairs, save)

    # Wait for completion
    while True:
        batch = anthropic_client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored
        print(f"  Status: {batch.processing_status} | Succeeded: {counts.succeeded}/{total}", end="\r")

        if batch.processing_status in ["ended", "canceled", "expired"]:
            break
        time.sleep(60)

    print(f"\n✓ Batch completed")

    # Retrieve results
    results = []
    for result in anthropic_client.messages.batches.results(batch_id):
        pmid, gene_id = result.custom_id.split("_", 1)

        if result.result.type == "succeeded":
            try:
                content = result.result.message.content[0].text
                parsed = format_with_retry(content, VALIDATION_SCHEMA)

                if not parsed:
                    # Parsing failed
                    error_data = {
                        "error": "JSON parsing failed",
                        "raw_response": content[:1000]
                    }
                    if save:
                        save_result(pmid, gene_id, error_data, False, "getGeneSummary")
                    results.append({
                        "pubmed_id": pmid,
                        "gene_id": gene_id,
                        "success": False,
                        "error": "JSON parsing failed",
                        "paper_available": True,  # Paper was available (batch succeeded)
                        "alias_in_text": True,    # Assumed true if batch API succeeded
                        "mentions": 0             # Unknown for batch
                    })
                else:
                    # Summary success - now process PDs
                    if save:
                        save_result(pmid, gene_id, parsed, True, "getGeneSummary")

                    # For PD steps, we need paper text and gene display
                    # Extract host_db from pairs
                    host_db = next((hdb for p, g, hdb in pairs if p == pmid and g == gene_id), None)

                    paper_available = True
                    alias_in_text = True  # Assumed if summary was generated

                    if host_db:
                        try:
                            # Fetch paper and prepare gene display
                            paper_text = get_paper_text(pmid)

                            # Check gene in text
                            alias_in_text, mention_count = check_gene_in_text(gene_id, paper_text, host_db)

                            aliases = get_gene_synonyms(gene_id, paper_text, host_db)
                            gene_display = f"{gene_id}, also known as {', '.join(aliases)}" if aliases else gene_id

                            # Generate PDs
                            pd_result, pd_usage, pd_time = generatePDs(
                                summary_json=parsed,
                                gene_text=gene_display,
                                n_pds=N_PDs
                            )

                            if pd_result and save:
                                save_result(pmid, gene_id, pd_result, True, "generatePDs", pd_usage, pd_time)

                            # Verify PDs (use caching even though batch papers are low-density)
                            if pd_result:
                                verify_result, verify_usage, verify_time = verifyPDs(
                                    suggested_pds=pd_result,
                                    paper_text=paper_text,
                                    gene_text=gene_display,
                                    use_caching=False  # Don't use caching for batch (single genes)
                                )

                                if verify_result and save:
                                    save_result(pmid, gene_id, verify_result, True, "verifyPDs", verify_usage, verify_time)

                        except Exception as e:
                            print(f"  Warning: PD processing failed for {gene_id}: {e}")

                    results.append({
                        "pubmed_id": pmid,
                        "gene_id": gene_id,
                        "success": True,
                        "data": parsed,
                        "paper_available": paper_available,
                        "alias_in_text": alias_in_text,
                        "mentions": mention_count if alias_in_text else 0
                    })
            except Exception as e:
                results.append({
                    "pubmed_id": pmid,
                    "gene_id": gene_id,
                    "success": False,
                    "error": str(e),
                    "paper_available": True,  # Batch succeeded so paper was available
                    "alias_in_text": None,    # Unknown if exception occurred
                    "mentions": 0
                })
        else:
            results.append({
                "pubmed_id": pmid,
                "gene_id": gene_id,
                "success": False,
                "error": result.result.error.message,
                "paper_available": None,  # Unknown if batch failed
                "alias_in_text": None,
                "mentions": 0
            })

    return results


#######################################################################################################################
#                                      MAIN PROCESSING                                                                #
#######################################################################################################################

def process_from_csv(csv_path: str, save: bool = True) -> pd.DataFrame:
    """
    Process gene-paper pairs from CSV file.

    CSV must have columns: Gene ID, PMID_clean, Database, paper_available, alias_in_text
    Only processes rows where paper_available==TRUE and alias_in_text==TRUE

    Saves results as {pmid}.json with structure:
    {
      "getGeneSummary": {
        "gene_id": {
          "gene_ID": "gene_id",
          "model_name": {
            "model": "model_name",
            "success": true/false,
            "data": {...},
            "usage": {...},
            "seconds": float
          }
        }
      }
    }
    """
    print(f"\n{'=' * 80}")
    print(f"LOADING DATA FROM: {csv_path}")
    print(f"{'=' * 80}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    # Show available columns for debugging
    print(f"CSV columns: {', '.join(df.columns.tolist())}")

    # Check for required columns with flexible naming
    gene_col = CSV_COLUMNS['gene_id']
    pmid_col = CSV_COLUMNS['pmid']
    db_col = CSV_COLUMNS['database']

    required_columns = [gene_col, pmid_col, db_col]
    missing_required = [col for col in required_columns if col not in df.columns]

    if missing_required:
        print(f"\n❌ ERROR: Missing required columns: {missing_required}")
        print(f"\nExpected column names (adjust in CSV_COLUMNS at top of script):")
        print(f"  - Gene ID column: '{gene_col}'")
        print(f"  - PMID column: '{pmid_col}'")
        print(f"  - Database column: '{db_col}'")
        print(f"\nYour CSV has: {', '.join(df.columns.tolist())}")
        print(f"\nEither:")
        print(f"  1. Rename your CSV columns to match expected names, OR")
        print(f"  2. Update CSV_COLUMNS dictionary at line ~50 in the script")
        return pd.DataFrame()

    # Optional filter columns (if they exist, use them; if not, skip filtering)
    filter_applied = False
    paper_avail_col = CSV_COLUMNS.get('paper_available')
    alias_col = CSV_COLUMNS.get('alias_in_text')

    if paper_avail_col in df.columns and alias_col in df.columns:
        valid_df = df[(df[paper_avail_col] == True) & (df[alias_col] == True)].copy()
        print(f"Valid pairs ({paper_avail_col}=TRUE, {alias_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    elif paper_avail_col in df.columns:
        valid_df = df[df[paper_avail_col] == True].copy()
        print(f"Valid pairs ({paper_avail_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    elif alias_col in df.columns:
        valid_df = df[df[alias_col] == True].copy()
        print(f"Valid pairs ({alias_col}=TRUE): {len(valid_df)}")
        filter_applied = True
    else:
        valid_df = df.copy()
        print(f"⚠ No filtering applied (optional filter columns not found)")
        print(f"Processing all {len(valid_df)} rows")

    if len(valid_df) == 0:
        print("No valid pairs to process after filtering!")
        return pd.DataFrame()

    # Group by paper
    grouped = defaultdict(list)
    for _, row in valid_df.iterrows():
        pmid = str(row[pmid_col])
        gene_id = str(row[gene_col])
        database = str(row[db_col]).lower()
        grouped[pmid].append((gene_id, database))

    # Deduplicate genes within each paper
    total_pairs = sum(len(genes) for genes in grouped.values())
    for pmid in grouped:
        grouped[pmid] = list(set(grouped[pmid]))  # Remove duplicates
    unique_pairs = sum(len(genes) for genes in grouped.values())

    print(f"Unique papers: {len(grouped)}")
    if total_pairs > unique_pairs:
        print(f"  Removed {total_pairs - unique_pairs} duplicate gene entries")

    # Analyze distribution
    genes_per_paper = [len(genes) for genes in grouped.values()]
    avg_genes = sum(genes_per_paper) / len(genes_per_paper)
    high_density = sum(1 for x in genes_per_paper if x >= MIN_GENES_FOR_CACHING)
    low_density = len(genes_per_paper) - high_density

    print(f"\nDistribution:")
    print(f"  Average genes per paper: {avg_genes:.2f}")
    print(f"  Papers with {MIN_GENES_FOR_CACHING}+ genes (caching): {high_density}")
    print(f"  Papers with <{MIN_GENES_FOR_CACHING} genes (batch): {low_density}")

    # Process
    all_results = []

    # High-density papers with caching
    high_density_papers = {pmid: genes for pmid, genes in grouped.items()
                           if len(genes) >= MIN_GENES_FOR_CACHING}

    if high_density_papers:
        print(f"\n{'=' * 80}")
        print(f"PHASE 1: CACHING ({len(high_density_papers)} papers)")
        print(f"{'=' * 80}")
        for pmid, genes in high_density_papers.items():
            results = process_paper_with_caching(pmid, genes, save=save)
            all_results.extend(results)

            # Print summary for this paper
            processed = sum(1 for r in results if not r.get("skipped", False))
            skipped = sum(1 for r in results if r.get("skipped", False))
            if skipped > 0:
                print(f"  Paper {pmid}: {processed} processed, {skipped} skipped (already done)")

    # Low-density papers with batch
    if USE_BATCH_FOR_LOW_DENSITY:
        low_density_pairs = [(pmid, g[0], g[1]) for pmid, genes in grouped.items()
                             if len(genes) < MIN_GENES_FOR_CACHING for g in genes]

        # Filter out already-processed pairs if not overwriting
        if not OVERWRITE_EXISTING:
            original_count = len(low_density_pairs)
            low_density_pairs = [(pmid, gene_id, host_db) for pmid, gene_id, host_db in low_density_pairs
                                 if not is_gene_already_processed(pmid, gene_id)]
            skipped_count = original_count - len(low_density_pairs)
            if skipped_count > 0:
                print(f"  Skipping {skipped_count} already-processed pairs")

        # Deduplicate pairs (in case same pair appears multiple times in CSV)
        original_count = len(low_density_pairs)
        low_density_pairs = list(set(low_density_pairs))  # Remove duplicates
        duplicate_count = original_count - len(low_density_pairs)
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate pairs from batch")

        if low_density_pairs:
            print(f"\n{'=' * 80}")
            print(f"PHASE 2: BATCH PROCESSING")
            print(f"{'=' * 80}")
            results = process_batch(low_density_pairs, save=save)
            all_results.extend(results)

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'pmid': r['pubmed_id'],
            'gene_id': r['gene_id'],
            'success': r['success'],
            'paper_available': r.get('paper_available', None),
            'alias_in_text': r.get('alias_in_text', None),
            'mentions': r.get('mentions', 0),
            'error': r.get('error', '')
        }
        for r in all_results
    ])

    # Generate detailed step completion log
    step_log_data = []
    for _, row in results_df.iterrows():
        pmid = row['pmid']
        gene_id = row['gene_id']

        # Initialize status for each step
        step_status = {
            'pmid': pmid,
            'gene_id': gene_id,
            'paper_available': row.get('paper_available'),
            'alias_in_text': row.get('alias_in_text'),
            'Summary': False,
            'generatePDs': False,
            'verifyPDs': False
        }

        # Check completion status from saved files
        filepath = OUT_DIR / f"{pmid}.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Check each step
                for step_key, model in [
                    ("getGeneSummary", SUMMARY_MODEL),
                    ("generatePDs", PD_GENERATOR_MODEL),
                    ("verifyPDs", PD_VERIFIER_MODEL)
                ]:
                    step_data = data.get(step_key, {}).get(gene_id, {})
                    model_data = step_data.get(model, {})

                    # Map step_key to column name
                    if step_key == "getGeneSummary":
                        col_name = "Summary"
                    else:
                        col_name = step_key

                    if model_data.get("success") and model_data.get("data"):
                        step_status[col_name] = True
            except:
                pass

        step_log_data.append(step_status)

    step_log_df = pd.DataFrame(step_log_data)

    print(f"\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total gene-paper pairs: {len(results_df)}")

    if not step_log_df.empty:
        # Diagnostic statistics
        paper_avail_count = step_log_df['paper_available'].sum() if 'paper_available' in step_log_df.columns else 0
        alias_count = step_log_df['alias_in_text'].sum() if 'alias_in_text' in step_log_df.columns else 0

        print(f"\nDiagnostic checks:")
        print(f"  ✓ Paper available:    {paper_avail_count}/{len(step_log_df)}")
        print(f"  ✓ Gene in text:       {alias_count}/{len(step_log_df)}")

        print(f"\nPipeline completion by step:")
        print(f"  ✓ Summary generated:  {step_log_df['Summary'].sum()}/{len(step_log_df)}")
        print(f"  ✓ PDs generated:      {step_log_df['generatePDs'].sum()}/{len(step_log_df)}")
        print(f"  ✓ PDs verified:       {step_log_df['verifyPDs'].sum()}/{len(step_log_df)}")
        print(f"  ✗ Failed:             {(~results_df['success']).sum()}/{len(step_log_df)}")

        # Calculate average times if available
        successful = [r for r in all_results if r.get('total_time')]
        if successful:
            avg_time = sum(r['total_time'] for r in successful) / len(successful)
            print(f"\nAverage time per gene: {avg_time:.1f}s")

    # Save step completion log
    if not step_log_df.empty:
        log_path = csv_path.replace('.csv', '_step_log.csv')
        step_log_df.to_csv(log_path, index=False)
        print(f"\nStep completion log saved to: {log_path}")

    return results_df


#######################################################################################################################
#                                              MAIN                                                                   #
#######################################################################################################################

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <csv_file_path>")
        print("\nExample: python pipeline.py gene_paper_pairs.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Process
    results_df = process_from_csv(csv_path, save=True)

    # Save results summary
    if not results_df.empty:
        output_path = csv_path.replace('.csv', '_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")