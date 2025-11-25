#!/usr/bin/env python
"""
Analyze PBO debug logs for concept-weight behavior.

What it does for each debug JSON:
- Prints session + stage metadata
- Lists concepts (id + label)
- For each categorization event ("round"):
    - Shows which image was last selected
    - Shows which concepts that image boosted
    - Prints top-5 concepts by weight
- Writes a wide CSV with weight trajectories over rounds:
    round, event_index, image_selected, boosted_concepts, concept_0, concept_1, ...

Expected JSON schema (matches current logs):
{
  "session_id": ...,
  "stage": "impression" | "spatial" | ...,
  "events": [
    {
      "timestamp": ...,
      "event_type": "initialization" | "image_selection" | "categorization",
      "data": { ... }
    },
    ...
  ],
  "final_state": ...
}
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


EPS = 1e-8


def normalize_simplex(w):
    """Just here if you decide to extend the analysis later."""
    import numpy as np
    w = np.asarray(w, dtype=float)
    w = w.clip(min=0.0)
    s = w.sum()
    if s <= EPS:
        return np.ones_like(w) / len(w)
    return w / s


def analyze_debug_file(path: Path, out_dir: Optional[Path] = None) -> None:
    print("=" * 80)
    print(f"Analyzing file: {path}")
    data = json.loads(path.read_text())

    session_id = data.get("session_id", "<unknown>")
    stage = data.get("stage", "<unknown>")
    events = data.get("events", [])
    print(f"Session: {session_id} | Stage: {stage} | #events: {len(events)}")

    # ------------------------------------------------------------------
    # 1) Initialization → concepts list
    # ------------------------------------------------------------------
    init = next((e for e in events if e["event_type"] == "initialization"), None)
    if init is None:
        print("  ! No initialization event found. Skipping.")
        return

    init_data = init["data"]
    concepts = init_data.get("concepts", [])
    concept_ids = [c["id"] for c in concepts]
    concept_labels = {c["id"]: c.get("label", c["id"]) for c in concepts}

    print(f"  Total concepts: {len(concepts)}")
    print("  First 5 concepts:")
    for c in concepts[:5]:
        print(f"    {c['id']:>10} : {c['label']}")

    # ------------------------------------------------------------------
    # 2) Collect categorization events (these contain per-round weights)
    # ------------------------------------------------------------------
    cat_events = [e for e in events if e["event_type"] == "categorization"]
    print(f"  #categorization events: {len(cat_events)}")

    if not cat_events:
        print("  ! No categorization events; nothing to track.")
        return

    # For mapping: "what image selection happened before this categorization?"
    image_selection_indices = [
        (i, e) for i, e in enumerate(events) if e["event_type"] == "image_selection"
    ]

    def last_image_before(ev_index: int) -> Optional[Dict[str, Any]]:
        last = None
        for idx, ev in image_selection_indices:
            if idx < ev_index:
                last = ev
            else:
                break
        return last

    # ------------------------------------------------------------------
    # 3) Build time-series (wide format: one row per round, columns per concept)
    # ------------------------------------------------------------------
    rows_wide: List[Dict[str, Any]] = []

    for round_idx, cat_ev in enumerate(cat_events, start=1):
        # Position of this categorization in the full event stream
        ev_index = events.index(cat_ev)
        cat_data = cat_ev["data"]
        details = cat_data.get("concept_details", [])

        # Map concept id → weight for this round
        w_by_id = {d["id"]: d["w"] for d in details}

        # Associate with last selected image (if any)
        img_ev = last_image_before(ev_index)
        img_id = None
        boosted_concepts: List[str] = []
        if img_ev is not None:
            img_data = img_ev["data"]
            img_id = img_data.get("image_id")
            boosted_concepts = img_data.get("concepts_boosted", [])

        # Row for CSV
        row: Dict[str, Any] = {
            "round": round_idx,
            "event_index": ev_index,
            "image_selected": img_id,
            "boosted_concepts": ";".join(boosted_concepts),
        }
        for cid in concept_ids:
            row[cid] = w_by_id.get(cid, float("nan"))
        rows_wide.append(row)

        # ------------------------------------------------------------------
        # 4) Console debug: what actually happened in this round?
        # ------------------------------------------------------------------
        print(f"\n  Round {round_idx}:")
        if img_id:
            print(f"    ↳ last image selected: {img_id}")
            if boosted_concepts:
                label_list = [concept_labels.get(c, c) for c in boosted_concepts]
                print(f"    ↳ boosted concepts: {', '.join(label_list)}")

        # top-5 concepts by weight
        top = sorted(
            [(cid, w_by_id.get(cid, 0.0)) for cid in concept_ids],
            key=lambda x: -x[1],
        )[:5]
        for cid, w in top:
            print(f"    {cid:>10} ({concept_labels[cid]:25s}) w = {w:.4f}")

    # ------------------------------------------------------------------
    # 5) Write CSV file with trajectories
    # ------------------------------------------------------------------
    if out_dir is None:
        out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + "_weights_wide.csv")

    fieldnames = ["round", "event_index", "image_selected", "boosted_concepts"] + concept_ids
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_wide:
            writer.writerow(row)

    print(f"\n  → Wrote wide weight trajectory CSV to: {out_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PBO debug JSON logs (concept weights over rounds)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more debug JSON files or directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory for CSVs (defaults to same dir as input file).",
    )
    args = parser.parse_args()

    # Collect all JSON files
    all_files: List[Path] = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            all_files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            all_files.append(p)
        else:
            print(f"Warning: {p} does not exist, skipping.")

    if not all_files:
        print("No JSON files found. Exiting.")
        return

    out_dir = Path(args.out_dir) if args.out_dir is not None else None

    for path in all_files:
        analyze_debug_file(path, out_dir=out_dir)


if __name__ == "__main__":
    main()
