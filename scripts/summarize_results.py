"""
Summarize result files for many basins into one CSV, with every result that
requires review listed first.

Usage: python scripts/summarize_results.py <results directory> [<output.csv>]
"""
import sys

from model_selection.io.summary import summarize_result_files, write_summary_csv

if __name__ == "__main__":
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results_summary.csv"
    rows = summarize_result_files(root)
    write_summary_csv(rows, out)
    flagged = [r for r in rows if r["review_required"]]
    print(f"{len(rows)} results, {len(flagged)} require review -> {out}")
    for r in flagged:
        print(f"  REVIEW  {r['gage']}  {r['pathway']}: {r['flag_codes']}")
