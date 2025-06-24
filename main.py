import sys
import json
import requests
import re

from openai import OpenAI

# Constants required for API usage -  PubMed, PlasmoDB, OpenAI.

# The Base URL for PubMed API to fetch BioC JSON format.
pubmed_base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"

# The URL and project for getting aliases from PlasmoDB.
# This is specific to PlasmoDB, but could be adapted for other databases.
get_alias_url = "https://plasmodb.org/plasmo/service/record-types/gene/records"
get_alias_project = "PlasmoDB"

#  The sections of PubMed documents that are relevant for gene curation.
pubmed_sections = ['FIG', 'TABLE', 'RESULTS', 'CONCL', 'DISCUSSION', 'SUPPL'] # removed title, abstract, intro and added supplementary

open_ai_model = "gpt-4o"
max_tokens = 16384
model_temp = 0


# The prompts to use in the workflow, using [gene] as a placeholder for the gene ID and synonyms:
#  1) "extract": Extract all information related to the gene from the text, used for later stages
#  2) "summary": Summarise the information in a structured way (may be too long for display)
#  3) "short_summary": Provide a short summary for display,
#  4) "title": Provide a title for display.

# NB: prompts subject to change depending on final testing
defaultSystem = "You are a systematic gene curation assistant for scientific publications.  Your output will be used verbatim. Do not include any commentary, explanations, apologies, or disclaimers. Only return the final result as plain text."

global_prompts = {"extract": "From the text given, extract and quote all of the information which is related to [gene]. Quote all specific results, data, inferences or conclusions that are relevant to this specific gene.  But do not infer activity based on other genes, focus only on this specific gene product. ",

"summary" :"ROLE: You are a scientist preparing a literature review making a study of the of the gene known as [gene] GOAL: Your purpose is to systematically review the text and summarise. Think step-by-step using the following workflow: \n 1) Include any experiments conducted and their results, as well as all conclusions to do with the activity, location, domain or expression of this gene.\n 2) Include anything else that may be relevant to a scientist studying this gene. \n 3) Provide the key findings from your review in bullet point format. \n 4) Consider if each bullet point is based on direct evidence from a statement made in the text, or based on inferences you made from the text.\n 5) This gene is present in the text.  If it is only mentioned in passing, or without any conclusion, then include the context of where it is mentioned and supply direct quotes. \n 6) Classify each bullet point as ‘Direct’ or ‘Inferred’ in your response. \n Respond objectively.  Add no other commmentary. Do not refer to the gene by name or id as this is already included in the user output.",

"short_summary": " Give a one-sentence overview summary for [gene] in the previous text. If the evidence is limited or uncertain do not give a misleading summary by making statements that do not have clear support; you must include any and all limitations of the evidence such as putative or hypothetical etc.   Add no other commmentary.  Do not refer to the gene by name or id as this is already included in the user output. ",

"title": " Give a short title describing the role of [gene] in the previous text. If the evidence is limited do not give a misleading title by making statements that do not have clear support; you must include any  limitations of the evidence such as putative or hypothetical etc. Add no other commmentary.  Do not refer to the gene by name or id as this is already included in the user output." }


def gene_to_prompt(gene,genes):
    """
    Converts a gene ID and a list of synonyms into a readable string for prompts.

    Args:
        gene (str): The gene ID.
        genes (list of str): A list of synonyms for the gene.

    Returns:
        str: A formatted string that includes the gene ID and its synonyms.
    """

    if len(genes) ==0:
        return gene
    else:
        return gene + " ( also known as "+ " or ".join(genes) +" )"


def call_prompt(strings, system=defaultSystem):
    """
    Calls the OpenAI API with a list of strings and a system prompt.
    Args:
        strings (list of str): A list of strings to be sent to the API. Each will be treated as a separate user message.
        system (str): The system prompt to be used.

    Returns:
        str: The response from the OpenAI API.
    """

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for string in strings:
        messages.append({"role": "user", "content": string})

    # Defaults to get key from environment variable

    client = OpenAI()

    response = client.chat.completions.create(
        model=open_ai_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=model_temp,

    )

    return response.choices[0].message.content

def count_substrings(paper, gene_string):
    """
    Counts how many times a gene ID appears in a given text,
    ensuring it's not embedded within alphanumeric characters. Regular expressions courtesy of ChatGPT.

    Args:
        paper (str): The input text to search within.
        gene_string (str): The gene ID to search for.

    Returns:
        int: The count of non-embedded occurrences of the gene ID.

    Raises:
        ValueError: If the gene_string is empty.
    """

    # Check if it's a letters+digits pattern (e.g., EBA181).
    match = re.fullmatch(r'([a-zA-Z]+)([0-9]+)', gene_string)

    if match:
        # Support an optional hyphen between letters and numbers.
        part1, part2 = map(re.escape, match.groups())
        core = f"{part1}-?{part2}"
    else:
        core = re.escape(gene_string)

    # Pattern: substring not embedded in alphanumerics (whitespace is OK)

    pattern = rf'(?<![a-zA-Z0-9]){core}(?![a-zA-Z0-9])'
    matches = re.findall(pattern, paper, re.IGNORECASE)
    return len(matches)


def get_vpdb_alias(gene_id):
    """
    Fetches aliases for a given gene ID from database (currently hard-coded as PlasmoDB.)

    Args:
        gene_id (str): The gene ID to search for.

    Returns:
        list: The possible aliases for this gene
        If no aliases are found, returns empty list

    """

    url = get_alias_url
    project = get_alias_project

    headers = {
        "content-type": "application/json"
    }

    data = {
        "attributes": [],
        "primaryKey": [
            {"name": "source_id", "value": gene_id},
            {"name": "project_id", "value": project}
        ],
        "tables": ["AllProducts", "Alias"]
    }

    # Make the POST request to the (PlasmoDB) API.
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return []

    alias_set = set()

    # Parse the response to extract aliases from the Alias table.
    if "tables" in response.json():
        table = response.json()["tables"]

        aliases = table.get("Alias", [])
        for row in aliases:
            alias_set.add(row["alias"])

    alias_list = list(alias_set)
    return alias_list

def clean_text_output(text):
    """
    Cleans up the text by removing extra whitespace and brackets.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with extra whitespace and newlines removed.
    """

    text = text.strip()

    # Remove surrounding quotes (if they match and enclose the whole string)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    return text


def parse_pubmed_json(pubmed_json):
    """
    Parses a PubMed JSON response to extract and concatenate text from relevant sections.

    Args:
        pubmed_json (list of dict): The JSON response from the PubMed API,
                                    typically a list of (usually 1) documents.

    Returns:
        str: A single string containing the concatenated text from required sections, preserving the original ordering.

    Notes:
        Uses the global variable `pubmed_sections` to determine which sections to include.
    """

    document_text = ""

    for doc in pubmed_json:
        for document in doc.get("documents", []):
            for passage in document.get("passages", []):
                infons = passage.get("infons", {})
                section_type = infons.get("section_type", "")
                if section_type.upper() in {s.upper() for s in pubmed_sections}:
                    if "text" in passage:
                        document_text += passage["text"] + "\n"

    return document_text

def get_pubmed_json(pubmed_id):
    """
    Fetches the PubMed JSON for a given PubMed ID.
    Args:
        pubmed_id (str): The PubMed ID to fetch.
    Raises:
        ValueError: If the PubMed ID is not found or if the response is not in JSON format.
    Returns:
        dict: The JSON response from the PubMed API.
    Notes:
        The PubMed ID should be a valid identifier, and the function constructs the URL
        to fetch the JSON data using the global pubmed_base_url
    """

    url = pubmed_base_url + str(pubmed_id)
    response = requests.get(url)

    if response.status_code == 200:
        # Checks if there is a json response - the API returns html if the paper is not found.
        if not response.headers.get('Content-Type', '').startswith('application/json'):
            raise ValueError("Paper not found")

        return response.json()

    raise ValueError(f"Paper fetch status code: {response.status_code}")


def get_gene_synonyms(gene_id, paper):
    """
    Retrieves synonyms for a given gene ID from the (e.g. PlasmoDB) database and counts their occurrences in a given paper text to get those that are used.

    It returns the three most common aliases found in the paper.

    Args:
        gene_id (str): The gene ID for which to retrieve synonyms.
        paper (str): The text of the paper in which to search for synonyms.

    Returns:
        list: A list of the three most common aliases found in the paper

    Notes:
        This function uses the count_substrings function which does additional regex parsing to add hyphens and ignore substrings if not delimited

    """
    aliases = get_vpdb_alias(gene_id)

    # Remove the gene id from the aliases; it is used separately.
    aliases = [item for item in aliases if item != gene_id]

    # Count how often each alias appears in the paper.
    alias_count = {}
    for alias in aliases:
        count = count_substrings(paper, alias)
        if count > 0:
            alias_count[alias] = count

    # Return just the 3 most common.
    sorted_aliases = sorted(alias_count, key=alias_count.get, reverse=True)
    return sorted_aliases[:3]


def get_prompt_and_replace(key, replacements):
    """
    Retrieves a specific prompt text from the global_prompts dictionary and replaces [] placeholders with provided values (currently only [gene]).

    Args:
        key (str): The key for the prompt in the global_prompts dictionary.
        replacements (dict): Keys are placeholder strings in the prompt and values are the text to replace them with.

    Returns:
        str: The prompt text with placeholders replaced by the corresponding values from replacements.
    :
    """
    
    full_text = global_prompts[key]
    for key, value in replacements.items():
        full_text = full_text.replace(f"[{key}]", value)
    return full_text



def get_summary(gene_id, pubmed_id):
    """
    Retrieves a summary of a gene from a PubMed paper, including extracting relevant information and generating summaries.

    Args:
        gene_id: The ID of the gene to summarize.
        pubmed_id: The PubMed ID of the paper to summarize.
    Returns:
        dict: A dictionary containing the summary, extract, title, short summary, and synonyms related to the gene.

    """

    # Get the PubMed JSON for the given ID.
    pubmed_json = get_pubmed_json(pubmed_id)
    # Parse the PubMed JSON to get the text of the required sections.
    pubmed_text = parse_pubmed_json(pubmed_json)

    # Get the synonyms for the gene (e.g. from PlasmoDB).
    synonyms = get_gene_synonyms(gene_id, pubmed_text)
    # Converts the list into a 'nice' string, use it to replace the placeholders.
    gene_text = gene_to_prompt(gene_id, synonyms)
    replacements = {"gene" : gene_text}

    # Extract the relevant information from the PubMed text.
    extract_prompt = get_prompt_and_replace("extract", replacements)
    extract = call_prompt([pubmed_text,extract_prompt])

    # Summarise the gene information for this paper.
    summary_prompt = get_prompt_and_replace("summary", replacements)
    summary = clean_text_output(call_prompt([extract,summary_prompt]))


    # Create a short summary and title for the gene.
    short_summary_prompt = get_prompt_and_replace("short_summary", replacements)
    short_summary = clean_text_output(call_prompt([summary,short_summary_prompt]))

    title_prompt = get_prompt_and_replace("title", replacements)
    title = clean_text_output(call_prompt([extract, title_prompt]))

    # Return everything that might be used.

    result = {"code":0, "message":"OK", "title": title, "short_summary": short_summary, "summary": summary, "extract": extract, "gene_id": gene_id, "pubmed_id": pubmed_id, "synonyms": synonyms, "paper_text": pubmed_text}

    return result


def process_paper(gene_id, pubmed_id):
    """
    Processes a PubMed paper for a given gene ID and PubMed ID, extracting relevant information and generating summaries.

    Args:
        gene_id (str): The ID of the gene to process.
        pubmed_id (str): The PubMed ID of the paper to process.

    Returns:
        dict: A dictionary containing the result of the processing (with code 0), or  error (code 1).
    """

    try:
        result = get_summary(gene_id, pubmed_id)
    except ValueError as e:
        result = {"code": 1, "message": str(e), "gene_id": gene_id, "pubmed_id": pubmed_id}

    return result



def test_example():
    """
    Runs a test example with a specific gene ID and PubMed ID.
    This is a hard-coded example for testing purposes.
    Returns:
        dict: The result of processing the example paper.
    """
    return(process_paper("PF3D7_1133400", "27128092"))


def main():
    """
    Main function to run the script from the command line.
    It checks for command-line arguments and processes the paper accordingly.
    """

    if len(sys.argv) < 3:
        result = {"code": 1, "message": "Please provide a gene ID and a PubMed ID."}
        result = test_example()

    else:
        gene_id = sys.argv[1]
        pubmed_id = sys.argv[2]
        result = process_paper(gene_id, pubmed_id)
    return result


if __name__ == '__main__':
    # Print output for testing purposes.
    result = main()
    json_result = json.dumps(result, indent=4, ensure_ascii=False)
    print(json_result)





