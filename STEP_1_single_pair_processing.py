import argparse
import sys
from pathlib import Path
from openai import OpenAI  # if processing using GPT
from anthropic import Anthropic  # if processing suing Claude
from dotenv import load_dotenv


from config.global_settings_single import *
from pipeline.prompts import *
from pipeline.pubmed_helpers import *
from pipeline.vpdb_helpers import *
from pipeline.utils import *
from pipeline.utils import (_verified_to_select_candidates,
    _unverified_to_select_candidates,
    _extract_usage,
    _to_responses_input,
    _responses_supports_temperature)
from pipeline.pipeline_steps import *

# Load environment variables from .env file
load_dotenv() #- include API KEys



def save_status(pubmed_id, out_dir, step_key, gene_id, model_name, data, success=True, usage=None, seconds=None):
    """
    Save or merge LLM results for a PubMed ID + gene + workflow stage into a JSON file.

    Args:
        pubmed_id (str): The PubMed ID.
        out_dir (Path): Pathlib.Path object for the output directory.
        step_key (str): The workflow step, e.g. "getGeneSummary".
        gene_id (str): The gene locus tag.
        model_name (str): The model used for this step.
        data (any): The response data from the LLM.
        success (bool): Whether the LLM call succeeded.
        usage (dict): breakdown of input/output/total tokens used
        seconds (): time it took to complete pipeline step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    filepath = out_dir / f"{pubmed_id}.json"

    # Load existing data if present
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError: # NB this shouldn't happen if formatted correctly as per next if block upon saving.
            print(f"Warning: {filepath} is invalid JSON, overwriting.")

            existing_data = {}
    else:
        existing_data = {}


    retry_count = 1
    error_message = "Initial attempt failed to produce valid JSON." # default
    while not isinstance(data, (dict, list)) and retry_count <= max_retry:
        try:
            # Call LLM
            print(f"⚙️  Retrying with formatter model, attempt {retry_count}/{max_retry}")

            data_formatted, formatter_usage, formatter_secs = call_prompt(provider=formatter_llm[0],
                                                   model=formatter_llm[1],
                                                   system_prompt="Convert the supplied string to parsable JSON. respong with the corrected JSON ONLY and nothing else.",
                                                   user_prompts=[f"I tried parsing this JSON: <JSON> {data} </JSON> \n "
                                                                f"The following error message popped up: {error_message}"],
                                                   prefill_text="{")
            # print(f"Formatter answer: {data_formatted}")
            # make sure not a string for saving as json
            data = extract_json(data_formatted)
            # print(f"formatter_parsed data: {data}")

        except Exception as e:
            # print the error or log it
            error_message = f"parsing JSON failed: {e}"
            print(f"LLM call failed: {e}")
        finally:
            retry_count +=1

    if not isinstance(data, (dict, list)):
        print(f"❌ Failed to recover valid JSON after {max_retry} attempts.")
        return {"error": "Max retries exceeded"}, {}, None

    #
    # Merge new data into the existing file structure

    if step_key not in existing_data:
        existing_data[step_key] = {}

    # Create or update the gene entry
    if gene_id not in existing_data[step_key]:
        existing_data[step_key][gene_id] = {"gene_ID": gene_id}

    # Add or overwrite the model-specific block; success is True if we get this far
    existing_data[step_key][gene_id][model_name] = {
        "model": model_name,
        "success": True,
        "data": data,
        "usage": usage or {},
        "seconds": seconds
    }

    # ─────────────────────────────────────────
    # Save merged JSON
    # ─────────────────────────────────────────
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)

    # Return the final parsed data + token usage + time (for logging/debug)
    return data, usage or {}, seconds


def load_status(pubmed_id, out_dir, step_key, gene_id, llm):
    filepath = out_dir / f"{pubmed_id}.json"

    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"⚠️ Could not read {filepath} – invalid or unreadable JSON.")
        return None

        # Safely walk down the nesting
    step_data = data.get(step_key, {})
    gene_data = step_data.get(gene_id, {})
    model_entry = gene_data.get(llm)

    return model_entry  # will be None if any level missing

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

def generatePDs(summary_json, gene_text, n_pds=N_PDs, key="generatePDs"):
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
    replacements = {"EXAMPLES": PD_EXAMPLES,  # global
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "SUMMARY": summary_for_pd,
                    "PDs": PDs}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="SystemPrompt")
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
    replacements = {"GENE": gene_text,  # global
                    "JSON_SCHEMA": JSON_SCHEMA,
                    "PAPER_TEXT": paper_text,
                    "PDs": PDs}

    system = get_prompt_and_replace(stage_key=key, replacements=replacements, prompt_type="SystemPrompt")
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


def process_paper_gene_pair(pubmed_id = "31795916", gene_id = "Tb927.10.4200", host_db = "TriTrypDB",
                            save = True, verify_summary = False, generate_PD = True, verify_PDs = True, select_PD = False):
    """
    TODO: add description

    """
    print(f"Fetching paper text from PMID: {pubmed_id}...")
    # Get the PubMed JSON for the given ID.
    pubmed_json = get_pubmed_json(pubmed_id)
    # Parse the PubMed JSON to get the text of the required sections.
    pubmed_text = parse_pubmed_json(pubmed_json)

    print(f"Checking {gene_id} or its aliases are mentioned in the available text...")
    # Get the synonyms for the gene (e.g. from PlasmoDB).
    synonyms = get_gene_synonyms(gene_id, pubmed_text, host_db)
    gene_text = gene_to_prompt(gene_id, synonyms)
    #### STAGE 1 - summary gen --------------------------------------------------------------------------------------
    key = "getGeneSummary"
    # check if summary already exists; if so load from status file and skip LLM call
    summary_data = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id, llm=summary_llm[1])
    # print(f"SUMMARY DATA LOADED:{summary_data}")

    if check_if_ok(summary_data):
        print(f"✔️ Found existing summary for {gene_id} in {pubmed_id}, skipping generation.")
        parsed_summary = summary_data["data"]
    else:
        print(f"Generating gene summary for {gene_id} from {pubmed_id}...")
        parsed_summary, usage, secs = getGeneSummary(pubmed_text, gene_text)
        # else get the data for the next step
        if save:
            # Save result
            parsed_summary, _, _ = save_status(pubmed_id=pubmed_id, ## update with formatter here for next steps
                                    out_dir=OUT_DIR,
                                    step_key=key,
                                    gene_id=gene_id,
                                    model_name=summary_llm[1],
                                    data=parsed_summary,
                                    success=isinstance(parsed_summary, (dict, list)), # only success if saved correctly
                                    usage=usage,
                                    seconds=secs)
            # print(f"Summary successfuly generated and saved in {OUT_DIR}")

    # Check if gene is only mentioned in passing
    if check_if_in_passing(parsed_summary):
        print(f"⚠️ {gene_id} only mentioned in passing in {pubmed_id}. Skipping additional pipeline steps")
        return

    summary_for_PD = parsed_summary

    #### STAGE2 - optional summary verification ------- tends to make summary longer/more detailed which could be good
    if verify_summary:
        key = "verifyGeneSummary"
        # check if summary already verified; if so load from status file and skip LLM call
        summary_data = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id, llm=summary_QC_llm[1])
        # check if it is a dict, not just if it exists here:

        if check_if_ok(summary_data):
            print(f"✔️ Found verified summary for {gene_id} in {pubmed_id}, skipping verification.")
            verified_summary = summary_data["data"]

        else:
            print(f"Verifying gene summary for {gene_id} from {pubmed_id}...")
            verified_summary, usage, secs = verifyGeneSummary(summary_json= parsed_summary, pubmed_text=pubmed_text, gene_text=gene_text)
            # else get the data for the next step
            if save:
                # Save result
                verified_summary, _,_ = save_status(pubmed_id=pubmed_id,
                                                    out_dir=OUT_DIR,
                                                    step_key=key,
                                                    gene_id=gene_id,
                                                    model_name=summary_QC_llm[1],
                                                    data=verified_summary,
                                                    success=isinstance(verified_summary, (dict, list)), # only success if saved correctly
                                                    usage=usage,
                                                    seconds=secs)
                # print(f"Summary successfuly verified and saved in {OUT_DIR}")
        summary_for_PD = verified_summary

    # terminate function if we are only using it to generate summary and not for PDs
    if not generate_PD:
        return

    #### STAGE 3: process only summary bullet points without quotes for the PD generation step ---------------------
    key = "generatePDs"
    suggested_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                                  llm=PD_generator_llm[1])

    if check_if_ok(suggested_entry):
        print(f"✔️ Found suggested PDs for {gene_id} in {pubmed_id}, skipping generation.")
        suggested_PDs = suggested_entry["data"]
    else:
        print(f"Generating suggested PDs for {gene_id} from {pubmed_id}...")
        suggested_PDs, usage, secs = generatePDs(summary_json=summary_for_PD, gene_text=gene_text)
        if save:
            suggested_PDs, _, _ = save_status(pubmed_id=pubmed_id,
                                              out_dir=OUT_DIR,
                                              step_key=key,
                                              gene_id=gene_id,
                                              model_name=PD_generator_llm[1],
                                              data=suggested_PDs,
                                              success=isinstance(suggested_PDs, (dict, list)),
                                              usage=usage,
                                              seconds=secs)

    #### STAGE 4: verify PDs against paper text (NEW ORDER: before selection)
    verified_PDs = None # initialise for later check

    if verified_PDs:

        key = "verifyPDs"
        verified_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                                     llm=PD_QC_llm[1])

        if check_if_ok(verified_entry):
            print(f"✔️ Found verified PDs for {gene_id} in {pubmed_id}, skipping verification.")
            verified_PDs = verified_entry["data"]
        else:
            print(f"Verifying suggested PDs for {gene_id} from {pubmed_id}...")
            verified_PDs, usage, secs = verifyPDs(paper_text=pubmed_text, gene_text=gene_text,
                                                  suggestedPDs=suggested_PDs)
            if save:
                save_status(pubmed_id=pubmed_id,
                            out_dir=OUT_DIR,
                            step_key=key,
                            gene_id=gene_id,
                            model_name=PD_QC_llm[1],
                            data=verified_PDs,
                            usage=usage,
                            seconds=secs)

    #### STAGE 5: select best PD FROM VERIFIED CANDIDATES -----------------------------------------------------------

    if select_PD:
        key = "selectPD"
        select_entry = load_status(pubmed_id=pubmed_id, out_dir=OUT_DIR, step_key=key, gene_id=gene_id,
                                   llm=PD_picker_llm[1])

        if check_if_ok(select_entry):
            print(f"✔️ Found selected PD for {gene_id} in {pubmed_id}, skipping selection.")
            top_PD = select_entry["data"]
        else:
            print(f"Selecting best PD for {gene_id} from {pubmed_id}...")

            # Determine which PDs to use for selection
            if verify_PDs and verified_PDs is not None:
                # Use verified PDs if verification was run
                print(f"  → Using verified PDs for selection")
                candidates = _verified_to_select_candidates(verified_PDs)
                # Fallback: if verification produced nothing usable, use suggested_PDs
                if not candidates.get("PDs"):
                    print(f"  ⚠️ No verified candidates available, falling back to suggested PDs")
                    candidates = _unverified_to_select_candidates(suggested_PDs)
            else:
                # Use suggested PDs directly (no verification)
                print(f"  → Using unverified PDs for selection")
                candidates = _unverified_to_select_candidates(suggested_PDs)

            # Final check - do we have any candidates?
            if not candidates.get("PDs"):
                print(f"  ⚠️ No PD candidates available for selection, skipping")
            else:
                top_PD, usage, secs = selectPD(summary_for_pd=summary_for_PD, suggestedPDs=candidates)
                if save:
                    top_PD, _, _ = save_status(pubmed_id=pubmed_id,
                                               out_dir=OUT_DIR,
                                               step_key=key,
                                               gene_id=gene_id,
                                               model_name=PD_picker_llm[1],
                                               data=top_PD,
                                               success=isinstance(top_PD, (dict, list)),
                                               usage=usage,
                                               seconds=secs)

# run test
# process_paper_gene_pair()
def main():
    parser = argparse.ArgumentParser(
        description="Process a single PubMed ID / gene ID pair through the curation pipeline."
    )
    parser.add_argument(
        "--pubmed_id",
        type=str,
        default="31795916",
        help="PubMed ID of the paper to process (default: 31795916)"
    )
    parser.add_argument(
        "--gene_id",
        type=str,
        default="Tb927.10.4200",
        help="Gene locus tag to process (default: Tb927.10.4200)"
    )
    parser.add_argument(
        "--host_db",
        type=str,
        required=True,
        help="VEuPathDB database identifier. Valid options: plasmodb, toxodb, tritrypdb, "
             "cryptodb, fungidb, giardiadb, piroplasmadb, amoebadb, microsporidiadb, "
             "trichdb, hostdb, vectorbase, microbiomedb, clinepidb, orthomcl"
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Disable saving intermediate results to disk"
    )
    parser.add_argument(
        "--verify-summary",
        dest="verify_summary",
        action="store_true",  # Changed: now opt-in rather than opt-out
        help="Enable the summary verification stage (Stage 2) - skipped by default"
    )
    parser.add_argument(
        "--no-generate-pd",
        dest="generate_pd",
        action="store_false",
        help="Stop after summary generation, skipping PD stages (Stages 3–5)"
    )
    parser.set_defaults(save=True, verify_summary=False, generate_pd=True)  # verify_summary now False

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Pipeline: Single Pair Processing")
    print(f"  PubMed ID : {args.pubmed_id}")
    print(f"  Gene ID   : {args.gene_id}")
    print(f"  Host DB   : {args.host_db}")
    print(f"  Save      : {args.save}")
    print(f"  Verify summary : {args.verify_summary}")
    print(f"  Generate PDs   : {args.generate_pd}")
    print(f"{'='*60}\n")

    try:
        process_paper_gene_pair(
            pubmed_id=args.pubmed_id,
            gene_id=args.gene_id,
            host_db=args.host_db,
            save=args.save,
            verify_summary=args.verify_summary,
            generate_PD=args.generate_pd
        )
        print(f"\n✅ Pipeline completed for {args.gene_id} / PMID {args.pubmed_id}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


import sys

# Set the command-line arguments
sys.argv = ['STEP_1_single_pair_processing.py', '--host_db', 'PlasmoDB', '--pubmed_id', '25011111', '--gene_id', 'PF3D7_1355700']

# Now call main()
main()