"""
Parse all eval session folders into normalized CSV tables.

Outputs (in outputs/):
  - location_scores.csv
  - session_summary.csv
  - gp_diagnostics.csv
  - preference_drift.csv
  - gp_warnings.csv
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

SESSION_LOGS_DIR = Path(__file__).resolve().parent.parent / "session_logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _parse_timestamp(ts_str: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def _classify_method(image_name: str) -> str:
    if image_name == "sd_baseline_text.png":
        return "baseline_text"
    if image_name == "sd_style_transfer.png":
        return "style_transfer"
    if image_name == "user_customized.png":
        return "user_customized"
    if image_name.startswith("eval_alpha"):
        return "gp_refined"
    return "unknown"


def _extract_participant(session_name: str) -> str:
    m = re.match(r"eval_(P\d+)-\d+_", session_name)
    return m.group(1) if m else session_name


def _extract_preset(session_name: str) -> str:
    m = re.match(r"eval_P\d+-\d+_(.+?)(?:_Sample)?_\d{4}-", session_name)
    return m.group(1).replace("_", " ") if m else ""


def _preset_target_room(preset: str) -> str:
    """Map preset name to its primary target room."""
    mapping = {
        "Calm Home Office": "Home Office",
        "Cozy Bedroom": "Bedroom",
        "Inviting Livingroom": "Livingroom",
        "Inspiring Reading Nook": "Reading Nook",
        "Contemplative Bedroom": "Bedroom",
    }
    return mapping.get(preset, "")


def discover_sessions() -> list[Path]:
    return sorted(
        p for p in SESSION_LOGS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("eval_")
    )


# ---------------------------------------------------------------------------
# location_scores
# ---------------------------------------------------------------------------
def parse_location_scores(sessions: list[Path]) -> pd.DataFrame:
    rows = []
    for sess_dir in sessions:
        rank_file = sess_dir / "rank_order.json"
        if not rank_file.exists():
            continue
        with open(rank_file) as f:
            data = json.load(f)

        session_id = sess_dir.name
        participant = _extract_participant(session_id)
        preset = _extract_preset(session_id)
        target_room = _preset_target_room(preset)

        for location, rankings in data.get("rankings", {}).items():
            for rank_pos, entry in rankings.items():
                method = _classify_method(entry["image"])
                rows.append({
                    "participant": participant,
                    "session_id": session_id,
                    "preset": preset,
                    "location": location,
                    "method": method,
                    "rank_position": int(rank_pos),
                    "score": entry["score"],
                    "is_target_room": location == target_room,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------
def parse_session_summary(sessions: list[Path]) -> pd.DataFrame:
    rows = []
    for sess_dir in sessions:
        log_file = sess_dir / "eval_log.json"
        if not log_file.exists():
            continue
        with open(log_file) as f:
            log = json.load(f)

        session_id = sess_dir.name
        participant = _extract_participant(session_id)
        preset = _extract_preset(session_id)
        events = log.get("events", [])
        if not events:
            continue

        ts_start = _parse_timestamp(events[0]["timestamp"])
        ts_end = _parse_timestamp(events[-1]["timestamp"])
        total_duration = (ts_end - ts_start).total_seconds()

        hitl_init_ts = hitl_final_ts = None
        n_rounds = 0
        n_locations = 0
        gen_durations = []
        gen_start_ts = None
        converged = False

        for ev in events:
            t = ev["type"]
            ts = _parse_timestamp(ev["timestamp"])

            if t == "hitl_initialize":
                hitl_init_ts = ts
            elif t == "hitl_finalize":
                hitl_final_ts = ts
                n_rounds = ev["data"].get("rounds_completed", 0)
                converged = False
            elif t == "hitl_ranking":
                if ev["data"].get("is_converged"):
                    converged = True
            elif t == "slider_generation_start":
                gen_start_ts = ts
            elif t == "slider_generation_complete":
                if gen_start_ts:
                    gen_durations.append((ts - gen_start_ts).total_seconds())
                    gen_start_ts = None
                n_locations += 1

        hitl_dur = (
            (hitl_final_ts - hitl_init_ts).total_seconds()
            if hitl_init_ts and hitl_final_ts
            else None
        )
        avg_gen = sum(gen_durations) / len(gen_durations) if gen_durations else None

        n_tags = 0
        refined_file = sess_dir / "refined_preferences_v2.json"
        if refined_file.exists():
            with open(refined_file) as f:
                rp = json.load(f)
            n_tags = len(rp.get("tags", []))

        rows.append({
            "participant": participant,
            "session_id": session_id,
            "preset": preset,
            "n_rounds": n_rounds,
            "n_tags": n_tags,
            "converged": converged,
            "total_duration_s": round(total_duration, 1),
            "hitl_duration_s": round(hitl_dur, 1) if hitl_dur else None,
            "avg_generation_s": round(avg_gen, 1) if avg_gen else None,
            "n_locations_rated": n_locations,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gp_diagnostics
# ---------------------------------------------------------------------------
def parse_gp_diagnostics(sessions: list[Path]) -> pd.DataFrame:
    rows = []
    for sess_dir in sessions:
        diag_file = sess_dir / "gp_round_diagnostics_v2.jsonl"
        if not diag_file.exists():
            continue
        session_id = sess_dir.name
        with open(diag_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                metrics = rec.get("metrics", {})
                top_tags = metrics.get("top_5_tags", [])
                top_tag = top_tags[0]["tag"] if top_tags else None
                top_mu = top_tags[0]["mu"] if top_tags else None
                rows.append({
                    "session_id": session_id,
                    "round": rec.get("round"),
                    "pairwise_accuracy": metrics.get("pairwise_accuracy_before_update"),
                    "spearman_rho": metrics.get("spearman_rank_corr_before_update"),
                    "kendall_tau": metrics.get("kendall_pair_agreement_before_update"),
                    "image_variance": metrics.get("image_variance"),
                    "beta": metrics.get("beta"),
                    "tags_updated": metrics.get("tags_updated"),
                    "log_likelihood": metrics.get("ranking_log_likelihood_before_update"),
                    "top_tag": top_tag,
                    "top_mu": top_mu,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# preference_drift
# ---------------------------------------------------------------------------
def parse_preference_drift(sessions: list[Path]) -> pd.DataFrame:
    rows = []
    for sess_dir in sessions:
        session_id = sess_dir.name

        initial_file = sess_dir / "impression" / "user_manual_weights.json"
        refined_file = sess_dir / "refined_preferences_v2.json"
        if not initial_file.exists() or not refined_file.exists():
            continue

        with open(initial_file) as f:
            init_data = json.load(f)
        with open(refined_file) as f:
            ref_data = json.load(f)

        initial_weights = init_data.get("weights", {})
        final_weights = ref_data.get("weights", {})

        all_tags_detail = {
            t["text"]: t for t in ref_data.get("all_tag_details", [])
        }

        all_tags = set(initial_weights.keys()) | set(final_weights.keys())
        for tag in all_tags:
            iw = initial_weights.get(tag)
            fw = final_weights.get(tag)
            detail = all_tags_detail.get(tag, {})
            rows.append({
                "session_id": session_id,
                "tag": tag,
                "initial_weight": iw,
                "final_weight": fw,
                "delta_weight": (fw - iw) if (fw is not None and iw is not None) else None,
                "final_mu": detail.get("final_mu"),
                "final_sigma": detail.get("final_sigma"),
                "win_rate": detail.get("win_rate"),
                "times_shown": detail.get("times_shown"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gp_warnings
# ---------------------------------------------------------------------------
def parse_gp_warnings(sessions: list[Path]) -> pd.DataFrame:
    rows = []
    warning_re = re.compile(
        r"\[(\w+)\]\s*(Round\s+(\d+):)?\s*(.*)"
    )
    for sess_dir in sessions:
        summary_file = sess_dir / "gp_refinement" / "gp_refinement_summary.txt"
        if not summary_file.exists():
            continue
        session_id = sess_dir.name
        in_warnings = False
        past_warning_header = False
        with open(summary_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("WARNINGS"):
                    in_warnings = True
                    past_warning_header = False
                    continue
                if in_warnings and not past_warning_header and stripped.startswith("="):
                    past_warning_header = True
                    continue
                if in_warnings and past_warning_header:
                    if not stripped:
                        continue
                    m = warning_re.match(stripped)
                    if m:
                        rows.append({
                            "session_id": session_id,
                            "warning_type": m.group(1),
                            "round": int(m.group(3)) if m.group(3) else None,
                            "details": m.group(4).strip(),
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions()
    print(f"Found {len(sessions)} session(s)")

    print("  Parsing location scores...")
    df_scores = parse_location_scores(sessions)
    df_scores.to_csv(OUTPUT_DIR / "location_scores.csv", index=False)
    print(f"    -> {len(df_scores)} rows")

    print("  Parsing session summaries...")
    df_summary = parse_session_summary(sessions)
    df_summary.to_csv(OUTPUT_DIR / "session_summary.csv", index=False)
    print(f"    -> {len(df_summary)} rows")

    print("  Parsing GP diagnostics...")
    df_gp = parse_gp_diagnostics(sessions)
    df_gp.to_csv(OUTPUT_DIR / "gp_diagnostics.csv", index=False)
    print(f"    -> {len(df_gp)} rows")

    print("  Parsing preference drift...")
    df_drift = parse_preference_drift(sessions)
    df_drift.to_csv(OUTPUT_DIR / "preference_drift.csv", index=False)
    print(f"    -> {len(df_drift)} rows")

    print("  Parsing GP warnings...")
    df_warn = parse_gp_warnings(sessions)
    df_warn.to_csv(OUTPUT_DIR / "gp_warnings.csv", index=False)
    print(f"    -> {len(df_warn)} rows")

    print("Done. CSVs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
