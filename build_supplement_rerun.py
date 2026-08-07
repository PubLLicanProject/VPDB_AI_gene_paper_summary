"""
build_supplement_rerun.py

Assembles the input for a supplement-aware batch re-run: every gene-paper pair where supplements
help = UNLOCKED (was alias_in_text FALSE, now found in supplement) UNION BOOSTED (already TRUE, gains
supplement mentions). Writes the input CSV + prints size / cost estimate. Does NOT launch anything.

Output: out/VPDB_PD_supplementary_rerun/input.csv  (+ input_annotated.csv with the 'source' column)
"""
import os
import pandas as pd

UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
BOOST = 'curated_data/all_PDs_with_PMID_2026_supplementary_mention_boost.csv'
OUTDIR = 'out/VPDB_PD_supplementary_rerun'
COST_PER_PAIR = 0.012  # rough $/pair (summary+PDs+verify), from the task-9 pilot


def _b(s):
    return s.astype(str).str.upper() == 'TRUE'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    u = pd.read_csv(UNLOCK)
    b = pd.read_csv(BOOST)

    unlocked = u[_b(u['unlocked'])][['Gene ID', 'pmid_CLEAN', 'Database']].copy()
    unlocked['source'] = 'unlock'
    boosted = b[_b(b['boosted'])][['Gene ID', 'pmid_CLEAN', 'Database']].copy()
    boosted['source'] = 'boost'

    both = pd.concat([unlocked, boosted], ignore_index=True)
    both = both.drop_duplicates(['Gene ID', 'pmid_CLEAN'])

    both[['Gene ID', 'pmid_CLEAN', 'Database']].to_csv(os.path.join(OUTDIR, 'input.csv'), index=False)
    both.to_csv(os.path.join(OUTDIR, 'input_annotated.csv'), index=False)

    n = len(both)
    papers = both['pmid_CLEAN'].nunique()
    print("=" * 64)
    print("SUPPLEMENT-AWARE BATCH RE-RUN — input assembled")
    print("=" * 64)
    print(f"total pairs : {n}   (unlock {len(unlocked)} + boost {len(boosted)}, deduped)")
    print(f"papers      : {papers}")
    print("\nby database:")
    print(both.groupby('Database').size().to_string())
    print("\nby source:")
    print(both.groupby('source').size().to_string())
    print(f"\nrough cost estimate @ ${COST_PER_PAIR:.3f}/pair : ${n * COST_PER_PAIR:,.0f}")
    print(f"input CSV -> {OUTDIR}/input.csv")


if __name__ == '__main__':
    main()
