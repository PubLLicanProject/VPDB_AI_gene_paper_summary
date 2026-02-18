from pipeline.prompts import *
from pipeline.utils import *

def getGeneSummary(pubmed_text, gene_text, key = "getGeneSummary"):
    """
        Converts a gene ID and a list of synonyms into a readable string for prompts.

        Args:
            pubmed_text (str): paper to process
            gene_text (str): The database identifer for the gene and its aliases found in the text
            key (str): the dictionary key name for the specified stage

        Returns:
            dict: a JSON object with summary bullet points, each supported by quotes and a specified evidence location
        """

    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # Set up replacements inside user and system prompts.
    replacements = {"N_QUOTES": N_QUOTES,
                      "GENE": gene_text,
                      "JSON_SCHEMA":  JSON_SCHEMA,
                      "PAPER_TEXT": pubmed_text}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{get_summary_system}")
    user= get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{get_summary_user}")
    # print(f"Generating summary...")
    try:
        # Call LLM
        LLM_summary, usage, secs = call_prompt(provider=summary_llm[0],
                                  model=summary_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        # print(f"LLM_summary: {LLM_summary}")
        parsed_summary = extract_json(LLM_summary)
        # print(f"Parsed summary: {parsed_summary}")
        return parsed_summary, usage, secs

    except Exception as e:
         # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None

def verifyGeneSummary(gene_text, pubmed_text, summary_json, key="verifyGeneSummary"):
    """
        Double-checks the initial gene summary for completenes and correctness.

        Args:
            gene_text (str): PMID and its aliases found in the paper text
            pubmed_text (str): paper to refer back to
            summary_json (str): The original summary json to correct.
            key (str): the dictionary key name for the specified stage

        Returns:
            dict: a JSON object with summary bullet points, each supported by quotes and a specified evidence location
            plus verification status and reason for it.
        """

    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # Set up replacements inside user and system prompts.
    replacements = {"SUMMARY": summary_json,
                    "GENE": gene_text,
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "PAPER_TEXT": pubmed_text}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements,
                                                prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{get_summary_system}")
    user = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{get_summary_user}")
    # print(f"Verifying summary...")
    try:
        # Call LLM
        LLM_summary, usage, secs = call_prompt(provider=summary_QC_llm[0],
                                  model=summary_QC_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        # print(f"LLM_summary: {LLM_summary}")
        parsed_summary = extract_json(LLM_summary)
        # print(f"Parsed summary: {parsed_summary}")

        return parsed_summary, usage, secs

    except Exception as e:
        # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None



def generatePDs(summary_json, gene_text, n_pds = N_PDs, key="generatePDs"):
    """
TO BE ADDED
    """
    # Abort if previous stage failed
    if not isinstance(summary_json, (dict, list)):
        return {"error": "Upstream summary not JSON"}, {}, None
    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # extract the summary bullet points only.
    bullets = collect_bullets(summary_json)
    print(f"BULLETS: {bullets}")
    summary_for_pd = "\n".join(bullets)


    # Set up replacements inside user and system prompts.
    replacements = {"N_PDs": n_pds,
                    "GENE": gene_text,
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "SUMMARY": summary_for_pd}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements,
                                                prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{get_summary_system}")
    user = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{get_summary_user}")
    # print(f"Verifying summary...")
    try:
        # Call LLM
        LLM_PDs, usage, secs = call_prompt(provider=PD_generator_llm[0],
                                  model=PD_generator_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        parsed_PDs = extract_json(LLM_PDs)

        return parsed_PDs, usage, secs

    except Exception as e:
        # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None


def selectPD(suggestedPDs, summary_for_pd, key="selectPD"):
    """
TO BE ADDED
    """
    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # 2) Convert the list of PD dicts to a neat numbered block
    pd_lines = [
        f"{i + 1}. {pd['description']}  "
        f"[evidence_code: {pd['evidence_code']}]"
        for i, pd in enumerate(suggestedPDs.get("PDs", []))
    ]
    PDs = "\n".join(pd_lines)
    print(f"input PDs: {PDs}")

    # Set up replacements inside user and system prompts.
    replacements = {"EXAMPLES": PD_EXAMPLES, # global
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "SUMMARY": summary_for_pd,
                    "PDs": PDs}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements,prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{system}")
    user = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{user}")
    # print(f"Verifying summary...")
    try:
        # Call LLM
        top_PD, usage, secs = call_prompt(provider=PD_picker_llm[0],
                                  model=PD_picker_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        top_PD = extract_json(top_PD)
        # print(top_PD)

        return top_PD, usage, secs

    except Exception as e:
        # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None

def verifyPDs(suggestedPDs, paper_text, gene_text, key="verifyPDs"):
    """
TO BE ADDED
    """
    # Get the Schema for the specific stage
    JSON_SCHEMA = global_prompts_and_schema[key]["ValidationSchema"]
    # print(f"Json schema retriever: \n {JSON_SCHEMA}")

    # 2) Convert the list of PD dicts to a neat numbered block
    pd_lines = [
        f"{i + 1}. {pd['description']}  "
        f"[evidence_code: {pd['evidence_code']}]"
        for i, pd in enumerate(suggestedPDs.get("PDs", []))
    ]
    PDs = "\n".join(pd_lines)
    # print(f"input PDs: {PDs}")

    # Set up replacements inside user and system prompts.
    replacements = {"GENE": gene_text, # global
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "PAPER_TEXT": paper_text,
                    "PDs": PDs}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements,prompt_type="SystemPrompt")
    # print(f"system prompt fetched: \n{system}")
    user = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="UserPrompts")
    # print(f"user prompt fetched: \n{user}")
    # print(f"Verifying summary...")
    try:
        # Call LLM
        verified_PDs, usage, secs = call_prompt(provider=PD_QC_llm[0],
                                  model=PD_QC_llm[1],
                                  system_prompt=system,
                                  user_prompts=user,
                                  prefill_text="{")
        # make sure not a string for saving as json
        verified_PDs = extract_json(verified_PDs)
        # print(top_PD)

        return verified_PDs, usage, secs

    except Exception as e:
        # print the error or log it
        print(f"LLM call failed: {e}")
        return {"error": str(e)}, {}, None


