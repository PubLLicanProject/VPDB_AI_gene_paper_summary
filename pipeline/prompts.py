import os
import time
import json
from pathlib import Path
from openai import OpenAI  # if processing using GPT
from anthropic import Anthropic  # if processing suing Claude
from dotenv import load_dotenv
from pipeline.utils import *
from pipeline.utils import (_verified_to_select_candidates,
    _extract_usage,
    _to_responses_input,
    _responses_supports_temperature)
# Load environment variables from .env file
load_dotenv() #- include API KEys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
#######################################################################################################################
#                                         CONSTANTS TO CONFIGURE                                                          #
#######################################################################################################################
OUT_DIR = Path("./out/cache") # where .json status files should be stored
EXAMPLE_PDS_PATH = './curated_data/example_PDs_2025_07_10.txt'

HTTP_TIMEOUT=180

# general pipeline and model config
N_PDs = 3 # PDs to brainstorm
N_QUOTES = 2 # quotes per bullet
model_temp = 0 # NB: can only be set if thinking is not enabled. we use 0 to maximise consistency
# initiate schema as empty string, each workflow stage has different schema
JSON_SCHEMA = "" # schema as string to append to the system prompt; stored in the prompts and schema dict and will be updated for each workflow step as needed.

# model choice for each pipeline step
max_tokens = 20000  # default in Claude tester online, NB: needs lowering for openai models
summary_llm = ["anthropic", "claude-sonnet-4-20250514"]
summary_QC_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_generator_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_picker_llm = ["anthropic", "claude-sonnet-4-20250514"]
PD_QC_llm = ["anthropic", "claude-sonnet-4-20250514"]
# additional model to force correct JSON schema if original LLM fails; attempt this up to max_retry times
formatter_llm = ["anthropic", "claude-sonnet-4-20250514"]
max_retry = 3


# some additional settings for openai reasoning models
REASONING_EFFORT = "medium"   # options: "minimal", "low", "medium", "high"
VERBOSITY = "low"              # options: "low", "medium", "high"
RESPONSES_MAX_OUTPUT_TOKENS = 6000  # hard cap; adjusted to keep outputs shorter

# # list of 200 example product descriptions picked at random; all 10 < characters < 50
# with open(EXAMPLE_PDS_PATH, 'r', encoding='utf-8') as file:
#     PD_EXAMPLES = file.read()
#######################################################################################################################
#                                         PROMPTS AND SCHEMA                                                          #
#######################################################################################################################
# NEW: modified prompts to encourage species information being taken into account, author-assigned PD are high prio,
# and evidence codes are carefully selected

# NEW 2: added explicit boolean flag in summary schema for pre-filtering - only genes not mentioned in passing will get PD generated.

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
                        "ShortSummary",
                        "only_in_passing"
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
                        },
                        # NEW: add specific flag to schema
                        "only_in_passing": {
                            "type": "boolean",
                            "description": "Flags if gene is only mentioned in passing without substantial findings - TRUE if so, FALSE if not."
                        },
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
#### STAGE 3A: Final check - are the brainstormed  PDs supported by the original text; which is recommended?
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
#### STAGE 3B: Selection base don examples - alternative version
    "selectPD": {
        "SystemPrompt": "ROLE: You are an expert auditor that synchronises gene product descriptions with the already "
                        "existing database.\n"
                        "OBJECTIVE: Select the most appropriate gene product description from a set of suggestions "
                        "based on examples of previously assigned descriptions.\n"
                        "Your task is to select the description that best fits the gene, based on the style, tone, and "
                        "content of the example descriptions provided. Consider the following criteria:\n"
                        "- Similarity in length to the examples\n"
                        "- Consistency in tone and style with the examples\n"
                        "- Appropriate level of detail compared to the examples\n"
                        "- Relevance of information included\n"
                        "WORKFLOW\n"
                        "1. Review these examples of previously assigned product descriptions:\n"
                        "<examples>\n"
                        "[EXAMPLES]\n"
                        "</examples>\n"
                        "2. Consider the summary supplied by the user of what we know about the gene.\n"
                        "3. Then go through the list of suggested descriptions supplied by the user and select the most suitable:\n"
                        "3.1. Analyze each suggested description, comparing it to the examples. Consider how well it matches "
                        "the criteria mentioned above.\n"
                        "3.2.Choose the description that you believe best fits the summary "
                        "and aligns with the style of the example descriptions.\n"
                        "3.3. If it can be improved to fit these criteria even better, improve it.\n"
                        "Before providing your final answer, briefly explain your reasoning for choosing this description (<25 words). "
                        "Consider how it compares to the other options and why it's the best fit based on the examples and criteria."
                        "Present your final answer in the following format:\n"
                        "[JSON_SCHEMA]\n"
                        "Respond with nothing but the final JSON.",
        "UserPrompts": [
            "Please read the following gene summary:\n"
            "<summary>\n"
            "[SUMMARY]\n"
            "</summary>\n"
            "Here is a list of potential gene product descriptions:\n"
            "<suggested_descriptions>\n"
            "[PDs]\n"
            "</suggested_descriptions>\n"
            "Please select the most appropriate and respond in JSON."
        ],
        "ValidationSchema": {
                  "type": "object",
                  "required": ["gene_id","recommended_product"],
                  "properties": {
                    "gene_id": {
                      "type": "string",
                      "description": "Locus tag or accession for the gene."
                    },
                    "recommended_product": {
                      "type": "object",
                      "required": ["description", "reasoning"],
                      "properties": {
                        "description": {
                          "type": "string",
                          "description": "Primary (recommended) product description."
                        },

                        "reasoning": {
                          "type": "string",
                          "description": "Reason why you selected this description over the rest."
                        }
                      },
                      "additionalProperties": False
                    }
                  }
        }
    },
#### STAGE 0: PD direct from paper without summary; for testing only
    "PD_without_summary": {
        "SystemPrompt": "ROLE: You are a scientist conducting a systematic literature review focused on ONE SPECIFIC gene of interest.\n"
        "GOAL: Analyze the provided research paper and extract ALL information relevant to the specified gene. Based on that information,"
        "generate a product description (PD) for that gene. The gene or one of its aliases is confirmed to be present in the text.\n"
        "INSTRUCTIONS: Follow this step-by-step workflow:\n"
        "<WORKFLOW>\n"
        "<STAGE_I>\n"
        "STAGE I: Extraction of relevant information\n"
        "1. EXPERIMENTAL ANALYSIS\n"
        "   - Identify and describe all experiments involving the specified gene\n"
        "   - Report experimental methods, results, and conclusions related to gene activity or expression\n"
        "   - Note any functional studies, assays, or measurements\n\n"
        "2. CONTEXTUAL RELEVANCE\n"
        "   - Extract any additional information relevant to understanding this gene's role\n"
        "   - Include regulatory mechanisms, interactions, pathways, or clinical significance\n"
        "   - Consider implications for gene function or therapeutic potential\n\n"
        "   If the gene appears only peripherally without substantial findings, label it as \"mentioned in passing\"\n"
        "</STAGE_I>\n"
        "<STAGE_II>\n"
        "STAGE II: Product Description Brainstorming"
        "- Refer to the TEMPLATES, and GUIDELINES to generate one appropriate PD at a time,"
        " up to a total of [N_PDs].\n"
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
        "</STAGE_II>\n"
        "<STAGE_III>\n"
        "STAGE_III: Audit and selection of most appropriate product description.\n"
        "1) For each PD you brainstorm, check if it is supported by the paper text.\n"
        "If yes, do the following:\n"
        "- assign it a verification_status SUPPORTED\n"
        "- add evicence_location that lists all sections of the paper that support the assignment"
        "(e.g. Fig 1, Results paragraph 3, etc.)\n"
        "- double-check you have selected the most suitable evidence_code from the EVIDENCE CODES above.\n"
        "- provide a reason that briefly explain why this code was selected.\n\n"
        "If not supported, assign verification_status as UNSUPPORTED and provide no further information for it.\n"
        "2) Consult the SENIOR AUDITOR EXAMPLES and check whether each SUPPORTED PD would pass an audit:\n"
        "- if yes, assign audit_check as PASS.\n"
        "- if not, modify the PD so it would pass and assign audit_check as EDITED\n"
        "- provide a short edit_reason (<20 words)\n"
        "3) Select a recommended PD (RPD) from your final edited list, based on the HIERARCHY provided, an supply the remaining PDs as alternatives (APDs).\n\n"
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
        "</STAGE_III>\n"
        "OUTPUT FORMAT:\n"
        "Double-check you have followed the templates and have fulfilled all rules. Return JSON using "
        "the following schema:\n"
        "[JSON_SCHEMA]\n\n",
"UserPrompts": [
            "Here is the paper text: \n [PAPER_TEXT] \n"
            "The gene of interest is: [GENE]\n"
            "Follow your workflow and return parsable JSON as per the OUTPUT FORMAT schema supplied."
        ],
        "ValidationSchema": {
                  "type": "object",
                  "required": ["gene_id","RPD", "APDs"],
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
#### STAGE 1.1 - OPTIONAL: summary verification - more ocmplete summary but lowers performance with PDs, so not recommended.
    "verifyGeneSummary": {
        "SystemPrompt": "ROLE: You are a senior gene-curation expert with extensive experience in molecular biology and genomics literature review.\n"
                         "BACKGROUND: You will receive two inputs:\n"
                         "1. The full text of a scientific publication\n"
                         "2. A gene-specific summary drafted by an intern curator\n"
                         "TASK: Conduct a thorough review of the intern's gene summary, verifying accuracy and completeness against the source publication.\n"
                         "WORKFLOW: Follow this systematic approach:\n"
                         "STEP 1: Bullet-by-bullet verification\n"
                         "For EACH bullet point in the draft summary:\n"
                         "- Carefully assess whether the claim is fully supported by evidence in the paper\n"
                         "- If SUPPORTED AND evidence_location AND supporting_quotes are corrct, mark status as \"PASS\"\n"
                         "- If SUPPORTED but evidence_location or supporting_quotes are incorrect/incomplete:\n"
                         "  → Retain the bullet point but correct the evidence fields and mark status as \"WARN\".\n"
                         "- If NOT SUPPORTED or inaccurate:\n"
                         "  → Mark as \"FAIL\" and provide a brief explanation (≤20 words)\n"
                         "STEP 2: Completeness check\n"
                         "Conduct a comprehensive scan of the paper to identify any significant information about the "
                         "gene that was omitted from the summary. Focus on:\n"
                         "- Functional annotations\n"
                         "- Pathways and interactions\n"
                         "- Disease associations\n"
                         "- Expression patterns\n"
                         "- Regulatory mechanisms\n"
                         "- Domain conservation\n"
                         "- make sure all names for the gene used by the authors are listed one at least one point.\n"
                         "For each important missing fact, create a new bullet point with:\n"
                         "- Clear, concise statement\n"
                         "- Accurate evidence_location\n"
                         "- Relevant supporting_quotes\n"
                         "- Status marked as \"NEW\"\n"
                         "STEP 3: Quality assurance\n"
                         "Ensure all evidence_location references are precise and supporting_quotes are verbatim excerpts "
                        "from the paper.\n"
                         "OUTPUT FORMAT:\n"
                         "Structure your response as valid JSON following this exact schema:\n"
                         "[JSON_SCHEMA]\n\n"
                         "IMPORTANT NOTES:\n"
                         "- Prioritize accuracy over completeness\n"
                         "- Use direct quotes from the paper, not paraphrases\n"
                         "- Be conservative in your interpretations\n"
                         "- Focus on gene-specific information, not general biological concepts\n",

        "UserPrompts": [
            "Do not respond to this message. Here is the paper text: \n [PAPER_TEXT]",
            "Here is the draft summary for [GENE]:\n[SUMMARY]\n\nScrutinise it as per the system instructions and output only valid JSON."
            ],

        "ValidationSchema": {
             "type": "object",
             "required": [
               "type",
               "VerifiedSummary"
             ],
             "properties": {
               "type": {
                 "type": "string",
                 "enum": [
                   "array"
                 ]
               },
               "VerifiedSummary": {
                 "type": "array",
                 "items": {
                   "type": "object",
                   "required": [
                     "bullet_point",
                     "evidence_location",
                     "supporting_quotes",
                     "verification_status",
                     "reason"
                   ],
                   "properties": {
                     "bullet_point": {
                       "type": "string"
                     },
                     "evidence_location": {
                       "type": "string"
                     },
                     "supporting_quotes": {
                       "type": "array",
                       "items": {
                         "type": "string"
                       },
                       "maxItems": N_QUOTES
                     },
                     "verification_status": {
                       "type": "string",
                       "enum": [
                         "PASS",
                         "FAIL",
                         "WARN",
                         "NEW"
                       ]
                     },
                     "reason": {
                       "type": "string"
                     }
                   },
                   "additionalProperties": False
                 }
               }
             },
             "additionalProperties": False
           }
    },
#### STAGE 4 (NOT TESTED MUCH YET) : Merge all gene summaries from multiple papers for a gene to generate lit-wide summary
"mergeGeneSummaries": {
        "SystemPrompt":
            "ROLE: You are a senior curator synthesising findings for ONE gene across multiple publications.\n"
            "GOAL: Merge the paper-derived gene summaries into a single, literature-wide summary.\n"
            "BACKGROUND: Another expert curator has been extracting gene-specific information from publications. "
            "You will receive one summary for the gene of interest per publication. Each summary will be split into bullet "
            "point statements. You will have to consider all the information and combine it into a clear, concise summary "
            "in bullet point statements, tracking which original paper contributed to each bullet statement. Think step "
            "by step using the following WORKFLOW:"
            "<WORKFLOW>\n"
            "1) Deduplicate semantically similar bullets; prefer the clearest/most direct evidence phrasing.\n"
            "2) Track support across papers: attach a list of supporting PMIDs per bullet.\n"
            "3) Prioritise experimentally supported claims over inferences.\n"
            "4) Flag conflicts/contradictions between PMID summaries succinctly.\n"
            "5) Keep to gene-specific facts; If the gene has been studied in multiple species, highlight any differences"
            "in function. Note that absence of information in one species DOES NOT mean the gene function is not conserved,"
            "it might simply indicate that it was not tested in that species.\n"
            "6) Label statements as 'consensus' if backed by multiple papers with no contradicting statements, as 'unique'"
            "if only present in one paper, and as 'contradiction' if there is conflicting information.\n"
            "7) Output JSON strictly matching the SCHEMA below:\n"
            "<\WORKFLOW>\n"
            "<SCHEMA>\n"
            "[JSON_SCHEMA]\n"
            "<\SCHEMA>\n",
    "UserPrompts": [
            "You are given per-paper summaries for [GENE_ID]. Merge them into a literature-wide summary.\n"
            "<CONTENT_FOR_SUMMARY>\n"
            "[ALL_SUMMARIES_BLOCK]\n"
            "<\CONTENT_FOR_SUMMARY>\n"
        ],
    "ValidationSchema": {
            "type": "object",
            "properties": {
                "GeneID": {"type": "string"},
                "ShortSummary": {
                    "type": "string",
                    "description": "2–4 sentence overview synthesizing the merged evidence."
                },
                "LiteratureSummary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bullet_point": {"type": "string"},
                            "supporting_pmids": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "statement_type": {
                                "type": "string",
                                "enum": ["consensus", "unique", "contradiction"]
                            }
                        },
                        "required": ["bullet_point", "supporting_pmids", "statement_type"]
                    }
                },
                "Notes": {
                    "type": "string",
                    "description": "Optional editor notes (conflicts, caveats)."
                }
            },
            "required": ["GeneID", "ShortSummary", "LiteratureSummary"],
            "additionalProperties": False
        }
    },
}


# helper to replace portions of a prompt

# helper to replace portions of a prompt
def get_prompt_and_replace(stage_key, replacements, prompt_type = "SystemPrompt"):
    """
    Retrieves a specific prompt text from the global_prompts dictionary and replace [] placeholders with provided values.

    Args:
        stage_key (str): The key for the prompt in the global_prompts dictionary.
        replacements (dict): Keys are placeholder strings in the prompt and values are the text to replace them with.
        prompt_type (str): Whether replacements are made in "UserPrompt" or "SystemPrompt".

    Returns:
        str: The prompt text with placeholders replaced by the corresponding values from replacements.
    """
    # two sub-helpers to deal with numerics and when a json needs to be a string for replacemtn
    def _stringify(value):
        """Convert non-strings (dicts, ints, etc.) to pretty JSON strings."""
        return value if isinstance(value, str) else json.dumps(value, indent=2)

    def _replace_in_text(text):
        for ph, val in replacements.items():
            text = text.replace(f"[{ph}]", _stringify(val))
        return text

    raw = global_prompts_and_schema[stage_key][prompt_type]

    # If we have a list of prompts as in the case of user prompts, process each element; otherwise process the single string.
    if isinstance(raw, list):
        return [_replace_in_text(txt) for txt in raw]
    else:
        return _replace_in_text(raw)



def _get_http_timeout(default: int = 180) -> int:
    """
    Return the HTTP read-timeout in seconds. Override with OPENAI_HTTP_TIMEOUT env var.
    """
    import os
    try:
        return int(os.getenv("OPENAI_HTTP_TIMEOUT", default))
    except Exception:
        return default
def _is_openai_responses_model(model: str) -> bool:
    """
    Return True if this OpenAI model expects the Responses API and
    'max_completion_tokens' instead of Chat Completions 'max_tokens'.
    """
    prefixes = ("gpt-5", "o4", "gpt-4.1", "gpt-4o-")
    return any(model.startswith(p) for p in prefixes)

def call_prompt(provider, model, user_prompts, system_prompt, prefill_text = "{"):
    """
    Calls the LLM API with a list of strings and a system prompt.
    Args:
        provider (str): one of "anthropic", "openai", "openrouter" (via OpenAI SDK)
        model (str): the model for use with the API, e.g. "claude-sonnet-4-20250514"
        user_prompts (list of str): A list of strings to be sent to the API. Each will be treated as a separate user message.
        Currently designed so any prompt that needs to be cached is always the first message.
        For processing multiple genes of the same paper it is recommended to cache the paper text.
        system_prompt (str): The system prompt to be used.

    Returns:
        str: The response from the LLM API.
    """

    messages = []
    system_message = None
    local_max_tokens = max_tokens
    if model == "claude-3-5-haiku-20241022":
        local_max_tokens = 8192
    if system_prompt:
        system_message = system_prompt  # Store system prompt separately for Claude

    if len(user_prompts) > 1:
        cache_prompt = user_prompts[0]
        prompt = user_prompts[1]
        if provider == "anthropic":
            # Anthropic supports caching
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cache_prompt,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            # OpenRouter/OpenAI: concatenate
            messages.append({"role": "user", "content": cache_prompt + "\n\n" + prompt})
    else:
        prompt = user_prompts[0]
        messages.append({"role": "user", "content": prompt})


    # Defaults to get key from environment variable
    if provider == "anthropic":
        client = Anthropic()
        # Add prefill if specified (only for Claude)
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})
        # Create the request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "max_tokens": local_max_tokens,
            "temperature": model_temp,
        }

        # Add system prompt if it exists
        if system_message:
            request_params["system"] = system_message

        start = time.perf_counter()  # time the call
        response = client.messages.create(**request_params)
        elapsed = time.perf_counter() - start  # end time


        result = response.content[0].text if response.content else ""
        # If we used prefill, prepend it to the response
        if prefill_text:
            result = prefill_text + result

        usage = _extract_usage("anthropic", response)
        return result, usage, elapsed  # return usage and time too




    elif provider == "openrouter":

        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        openai_messages.extend(messages)

        # Create OpenAI client with OpenRouter configuration - simplified

        from openai import OpenAI as OpenAIClient

        client = OpenAIClient(

            base_url="https://openrouter.ai/api/v1",

            api_key=OPENROUTER_API_KEY,

            default_headers={

                "HTTP-Referer": "https://github.com/yourusername",  # Optional but recommended

                "X-Title": "Gene Curation Pipeline",  # Optional

            }

        )

        start = time.perf_counter()

        request_params = {

            "model": model,

            "messages": openai_messages,

            "max_tokens": local_max_tokens,

            "temperature": model_temp,

            "timeout": HTTP_TIMEOUT,  # Add explicit timeout

        }

        # Try with reasoning first, fallback if not supported

        try:

            # First attempt: with reasoning/thinking enabled

            response = client.chat.completions.create(

                **request_params,

                extra_body={

                    "provider": {

                        "allow_fallbacks": True,  # Allow OpenRouter to fallback if primary fails

                    },

                }

            )

        except Exception as e:

            error_msg = str(e).lower()

            # If reasoning not supported or other API error, retry without extras

            if "reasoning" in error_msg or "extra_body" in error_msg or "provider" in error_msg:

                print(f"  ⚠️  Retrying without extended params: {e}")

                response = client.chat.completions.create(**request_params)

            else:

                raise

        elapsed = time.perf_counter() - start

        result = response.choices[0].message.content

        # Extract usage from response

        usage_obj = getattr(response, 'usage', None)

        if usage_obj:

            usage = {

                "input": getattr(usage_obj, 'prompt_tokens', 0),

                "output": getattr(usage_obj, 'completion_tokens', 0),

                "total": getattr(usage_obj, 'total_tokens', 0)

            }

        else:

            usage = {"input": 0, "output": 0, "total": 0}

        return result, usage, elapsed

    elif (provider == "openai"):

        # Build chat-style messages first

        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        openai_messages.extend(messages)

        client = OpenAI()

        start = time.perf_counter()

        if _is_openai_responses_model(model):

            # Responses API

            resp_input = _to_responses_input(openai_messages)

            request_params = {

                "model": model,

                "input": resp_input,

                "instructions": system_prompt or "",

                "max_output_tokens": min(local_max_tokens, RESPONSES_MAX_OUTPUT_TOKENS),

                "timeout_s": _get_http_timeout(),

                # Correct placement for verbosity: under "text"

                "text": {

                    "format": {"type": "text"},

                    **({"verbosity": VERBOSITY} if VERBOSITY else {}),

                },

            }

            # Reasoning control stays top-level

            if REASONING_EFFORT:
                request_params["reasoning"] = {"effort": REASONING_EFFORT}

            # Keep temperature only if supported

            if _responses_supports_temperature(model) and (model_temp is not None):
                request_params["temperature"] = model_temp

            try:

                response = _openai_responses_create(client, **request_params)

            except RuntimeError as e:

                msg = str(e)

                # If API rejects verbosity, retry once without it

                if "Unknown parameter" in msg and "verbosity" in msg:

                    request_params.pop("text", None)  # drop verbosity + format block

                    # Minimal safe text config without verbosity

                    request_params["text"] = {"format": {"type": "text"}}

                    response = _openai_responses_create(client, **request_params)

                else:

                    raise

            elapsed = time.perf_counter() - start

            result = getattr(response, "output_text", "") or ""

            if not result:

                parts = []

                for item in getattr(response, "output", []) or []:

                    for c in item.get("content", []) or []:

                        t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)

                        if t:
                            parts.append(t)

                result = "".join(parts)

            usage = _extract_usage("openai-responses", response)


        else:

            # Chat Completions API

            try:

                response = client.chat.completions.create(

                    model=model,

                    messages=openai_messages,

                    max_tokens=local_max_tokens,

                    temperature=model_temp,

                    timeout=_get_http_timeout(),

                )

            except TypeError:

                response = client.chat.completions.create(

                    model=model,

                    messages=openai_messages,

                    max_tokens=local_max_tokens,

                    temperature=model_temp,

                )

            elapsed = time.perf_counter() - start

            result = response.choices[0].message.content

            usage = _extract_usage("openai-chat", response)

        return result, usage, elapsed
    else:
        raise ValueError("unsupported LLM provider, please use 'anthropic', 'openrouter', or 'openai'")

