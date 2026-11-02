"""Summarizing result files across many basins, with results that require
review listed first."""
from __future__ import annotations

import csv
import json
import pathlib

SUMMARY_COLUMNS = ("review_required", "gage", "pathway", "flag_codes", "tier", "selected", "stopped_reason",
                   "flag_messages", "result_file")


def summarize_result_files(root: str) -> list[dict]:
    """One row per pooled or high-flow result JSON found under `root`
    (recursively), rows requiring review first, then by gage and pathway."""
    rows = []
    for path in sorted(pathlib.Path(root).rglob("*_result.json")):
        name = path.name
        if name.endswith("_pooled_result.json"):
            pathway, prefix = "pooled", "pooled_"
        elif name.endswith("_high_flow_result.json"):
            pathway, prefix = "high_flow", "high_flow_"
        else:
            continue
        data = json.loads(path.read_text())
        flags = data.get("review_flags", [])
        rows.append({
            "review_required": bool(data.get("review_required", bool(flags))),
            "gage": data.get("gage"),
            "pathway": pathway,
            "flag_codes": ";".join(f["code"] for f in flags),
            "tier": data.get("committed_tier") or data.get("measure"),
            "selected": ";".join(data.get(f"{prefix}selected") or []),
            "stopped_reason": data.get(f"{prefix}stopped_reason") or "",
            "flag_messages": " | ".join(f["message"] for f in flags),
            "result_file": str(path),
        })
    rows.sort(key=lambda r: (not r["review_required"], str(r["gage"]), r["pathway"]))
    return rows


def write_summary_csv(rows: list[dict], path: str) -> str:
    """Write summary rows to CSV; returns the path."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
