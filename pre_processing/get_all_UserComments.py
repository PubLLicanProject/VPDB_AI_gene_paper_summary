import requests
import time
import os
from typing import Dict, List, Set
import csv

# All VEuPathDB project base URLs
VEUPATHDB_PROJECTS = {
    "AmoebaDB": "https://amoebadb.org/amoeba",
    "CryptoDB": "https://cryptodb.org/cryptodb",
    "FungiDB": "https://fungidb.org/fungidb",
    "GiardiaDB": "https://giardiadb.org/giardiadb",
    "HostDB": "https://hostdb.org/hostdb",
    "MicrosporidiaDB": "https://microsporidiadb.org/micro",
    "PiroplasmaDB": "https://piroplasmadb.org/piro",
    "PlasmoDB": "https://plasmodb.org/plasmo",
    "ToxoDB": "https://toxodb.org/toxo",
    "TrichDB": "https://trichdb.org/trichdb",
    "TriTrypDB": "https://tritrypdb.org/tritrypdb",
    "VectorBase": "https://vectorbase.org/vectorbase"
}


def load_organisms_from_file(db_name: str) -> List[str]:
    """
    Load organisms from the pre-downloaded organism files.

    Args:
        db_name: Name of the database (e.g., 'MicrosporidiaDB')

    Returns:
        List of organism names
    """
    file_path = f"./curated_data/organisms_by_DB/{db_name}_organisms.txt"

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    organisms = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if 'Organism' in row:
                    organisms.append(row['Organism'])

        print(f"Loaded {len(organisms)} organisms from {file_path}")
        return organisms

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def load_processed_organisms(progress_file: str) -> Set[str]:
    """
    Load the set of organisms that have already been processed.

    Args:
        progress_file: Path to the progress tracking file

    Returns:
        Set of organism names that have been processed
    """
    if not os.path.exists(progress_file):
        return set()

    processed = set()
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                organism = line.strip()
                if organism:
                    processed.add(organism)
        print(f"Found {len(processed)} already processed organisms")
        return processed
    except Exception as e:
        print(f"Error reading progress file: {e}")
        return set()


def mark_organism_processed(progress_file: str, organism: str):
    """
    Mark an organism as processed by appending to the progress file.

    Args:
        progress_file: Path to the progress tracking file
        organism: Organism name to mark as processed
    """
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{organism}\n")
    except Exception as e:
        print(f"Error writing to progress file: {e}")


def log_error(error_log_file: str, organism: str, status_code: int, error_msg: str = ""):
    """
    Log errors to a separate file for later analysis.

    Args:
        error_log_file: Path to the error log file
        organism: Organism name that failed
        status_code: HTTP status code
        error_msg: Additional error message
    """
    try:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{organism}\t{status_code}\t{error_msg}\n")
    except Exception as e:
        print(f"Error writing to error log: {e}")


def append_comments_to_file(output_file: str, response_text: str, organism: str,
                            is_first: bool = False):
    """
    Append user comments to the combined output file.

    Args:
        output_file: Path to the output TSV file
        response_text: Response text from API containing comments
        organism: Organism name (for adding a column)
        is_first: Whether this is the first write (include header)
    """
    lines = response_text.strip().split('\n')

    if len(lines) <= 1:
        # No data, only header or empty
        return False

    try:
        mode = 'w' if is_first else 'a'
        with open(output_file, mode, encoding='utf-8') as f:
            for idx, line in enumerate(lines):
                if idx == 0:
                    # Header line
                    if is_first:
                        # Add 'Organism' column to header
                        f.write(f"Organism\t{line}\n")
                else:
                    # Data line - prepend organism name
                    f.write(f"{organism}\t{line}\n")
        return True
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return False


def fetch_user_comments(db_name: str, db_base_url: str, organisms: List[str],
                        output_dir: str = "user_comments", sleep_time: int = 5):
    """
    Fetch user comments for a list of organisms from a specific database.
    Results are appended to a single file per database.

    Args:
        db_name: Name of the database (e.g., 'PlasmoDB')
        db_base_url: Base URL for the database API
        organisms: List of organism names
        output_dir: Directory to save output files
        sleep_time: Seconds to wait between requests
    """
    endpoint = f"{db_base_url}/service/record-types/transcript/searches/GenesByTaxon/reports/tableTabular"

    # Make directory for output if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    output_file = os.path.join(output_dir, f"{db_name}_all_comments.tsv")
    progress_file = os.path.join(output_dir, f"{db_name}_progress.txt")
    error_log_file = os.path.join(output_dir, f"{db_name}_errors.log")

    # Initialize error log with header if it doesn't exist
    if not os.path.exists(error_log_file):
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("Organism\tStatus_Code\tError_Message\n")

    # Load already processed organisms
    processed_organisms = load_processed_organisms(progress_file)

    # Check if output file exists (to determine if we need header)
    is_first_write = not os.path.exists(output_file)

    success_count = 0
    no_comments_count = 0
    error_count = 0
    error_422_count = 0
    skipped_count = len(processed_organisms)

    # Filter out already processed organisms
    organisms_to_process = [org for org in organisms if org not in processed_organisms]

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed organisms")

    if not organisms_to_process:
        print(f"All organisms already processed for {db_name}!")
        return success_count, no_comments_count, error_count, skipped_count, error_422_count

    print(f"Processing {len(organisms_to_process)} remaining organisms\n")

    for idx, organism in enumerate(organisms_to_process, 1):
        params = {
            "organism": f'["{organism}"]',
            "reportConfig": '{"tables":["UserComments"],"includeHeader":true}'
        }

        print(f"[{idx}/{len(organisms_to_process)}] Fetching comments for {organism}")

        try:
            response = requests.get(endpoint, params=params, timeout=60)

            if response.status_code == 200:
                # Check if there's actual data (not just header)
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    # Append to the combined file
                    if append_comments_to_file(output_file, response.text, organism,
                                               is_first=is_first_write):
                        print(f"  ✓ Appended {len(lines) - 1} comments")
                        success_count += 1
                        is_first_write = False  # No longer first write
                    else:
                        print(f"  ✗ Failed to write comments")
                        error_count += 1
                        log_error(error_log_file, organism, response.status_code, "Failed to write to file")
                        continue  # Don't mark as processed if write failed
                else:
                    print(f"  ○ No comments found")
                    no_comments_count += 1
            elif response.status_code == 422:
                error_msg = "Unprocessable Entity - likely organism name mismatch or invalid"
                print(f"  ✗ 422 Error: {error_msg}")
                print(f"     Organism string: '{organism}'")
                # Try to get response body for more details
                try:
                    error_detail = response.text[:200] if response.text else "No error detail"
                    print(f"     Response: {error_detail}")
                    log_error(error_log_file, organism, 422, error_detail)
                except:
                    log_error(error_log_file, organism, 422, error_msg)
                error_422_count += 1
                error_count += 1
                # Still mark as processed so we don't retry repeatedly
                mark_organism_processed(progress_file, organism)
                time.sleep(sleep_time)
                continue
            else:
                error_msg = f"HTTP {response.status_code}"
                print(f"  ✗ Failed. Status: {response.status_code}")
                try:
                    error_detail = response.text[:200] if response.text else "No error detail"
                    print(f"     Response: {error_detail}")
                    log_error(error_log_file, organism, response.status_code, error_detail)
                except:
                    log_error(error_log_file, organism, response.status_code, error_msg)
                error_count += 1
                # Don't mark as processed if request failed (might be temporary)
                time.sleep(sleep_time)
                continue

            # Mark organism as processed (whether it had comments or not)
            mark_organism_processed(progress_file, organism)

        except requests.exceptions.Timeout:
            print(f"  ✗ Request timed out")
            log_error(error_log_file, organism, 0, "Timeout")
            error_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            log_error(error_log_file, organism, 0, str(e))
            error_count += 1

        # Sleep to avoid overwhelming the server
        time.sleep(sleep_time)

    # Print summary for this database
    print(f"\n{db_name} Summary:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ○ No comments: {no_comments_count}")
    print(f"  ✗ Errors: {error_count} (422 errors: {error_422_count})")
    print(f"  ⊘ Skipped (already processed): {skipped_count}")
    print(f"  Total: {len(organisms)}")
    if error_422_count > 0:
        print(f"  ⚠ Check {error_log_file} for 422 error details\n")

    return success_count, no_comments_count, error_count, skipped_count, error_422_count


def main():
    """
    Main function to fetch all user comments from all VEuPathDB projects.
    """
    print("=" * 60)
    print("Fetching user comments from VEuPathDB projects")
    print("=" * 60)

    total_success = 0
    total_no_comments = 0
    total_errors = 0
    total_skipped = 0
    total_422_errors = 0
    total_organisms = 0

    for db_name, db_url in VEUPATHDB_PROJECTS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {db_name}")
        print(f"{'=' * 60}")

        # Load organisms from file
        organisms = load_organisms_from_file(db_name)

        if not organisms:
            print(f"No organisms found for {db_name}, skipping...\n")
            continue

        total_organisms += len(organisms)

        # Fetch user comments
        success, no_comments, errors, skipped, error_422 = fetch_user_comments(
            db_name, db_url, organisms, sleep_time=5
        )

        total_success += success
        total_no_comments += no_comments
        total_errors += errors
        total_skipped += skipped
        total_422_errors += error_422

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total organisms: {total_organisms}")
    print(f"  ✓ With comments: {total_success}")
    print(f"  ○ No comments: {total_no_comments}")
    print(f"  ✗ Errors: {total_errors} (422 errors: {total_422_errors})")
    print(f"  ⊘ Skipped (already processed): {total_skipped}")
    print("=" * 60)
    if total_422_errors > 0:
        print(f"\n⚠ Found {total_422_errors} organisms with 422 errors")
        print("Check the *_errors.log files in user_comments/ for details")
        print("These are likely organism name mismatches or deprecated organisms")


if __name__ == "__main__":
    main()