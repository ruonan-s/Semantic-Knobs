#!/usr/bin/env python3
"""
Summarize HITL GP round diagnostics JSONL files.

Usage:
  python eval/utils/summarize_gp_round_diagnostics.py \
      --session-folder eval/session_logs/<session_id>

  # Or pass the JSONL file directly
  python eval/utils/summarize_gp_round_diagnostics.py \
      --jsonl eval/session_logs/<session_id>/gp_round_diagnostics_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RoundMetrics:
    round_num: int
    pairwise_acc: float
    log_lik: float
    mean_margin: float
    image_variance: float
    learning_rate: float
    predicted_top: int
    actual_top: int
    spearman: float
    kendall_agree: float
    top_tag: str
    top_tag_mu: float
    top_tag_sigma: float


def _resolve_jsonl_path(session_folder: Optional[str], jsonl_path: Optional[str]) -> str:
    if jsonl_path:
        return jsonl_path
    if not session_folder:
        raise ValueError("Either --session-folder or --jsonl must be provided.")
    return os.path.join(session_folder, "gp_round_diagnostics_v2.jsonl")


def _load_records(jsonl_path: str) -> List[Dict]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Diagnostics file not found: {jsonl_path}")

    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping malformed JSONL line: {e}")
    records.sort(key=lambda r: int(r.get("round", 0)))
    return records


def _to_round_metrics(records: List[Dict]) -> List[RoundMetrics]:
    out: List[RoundMetrics] = []
    for r in records:
        m = r.get("metrics", {})
        top5 = m.get("top_5_tags", []) or []
        top = top5[0] if top5 else {}
        out.append(
            RoundMetrics(
                round_num=int(r.get("round", m.get("round", 0))),
                pairwise_acc=float(m.get("pairwise_accuracy_before_update", 0.0)),
                log_lik=float(m.get("ranking_log_likelihood_before_update", 0.0)),
                mean_margin=float(m.get("mean_pair_margin_before_update", 0.0)),
                image_variance=float(m.get("image_variance", 0.0)),
                learning_rate=float(m.get("learning_rate", 0.0)),
                predicted_top=int(m.get("predicted_top_option_before_update", -1)),
                actual_top=int(m.get("actual_top_option", -1)),
                spearman=float(m.get("spearman_rank_corr_before_update", 0.0)),
                kendall_agree=float(m.get("kendall_pair_agreement_before_update", 0.0)),
                top_tag=str(top.get("tag", "")),
                top_tag_mu=float(top.get("mu", 0.0)),
                top_tag_sigma=float(top.get("sigma", 0.0)),
            )
        )
    return out


def _slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    p = np.polyfit(x, y, 1)
    return float(p[0])


def _print_summary(metrics: List[RoundMetrics]) -> None:
    n = len(metrics)
    if n == 0:
        print("No valid rounds found.")
        return

    acc = [m.pairwise_acc for m in metrics]
    ll = [m.log_lik for m in metrics]
    mg = [m.mean_margin for m in metrics]
    iv = [m.image_variance for m in metrics]
    lr = [m.learning_rate for m in metrics]
    sig = [m.top_tag_sigma for m in metrics]
    top_match = [1.0 if m.predicted_top == m.actual_top else 0.0 for m in metrics]
    spearman_vals = [m.spearman for m in metrics]
    kendall_vals = [m.kendall_agree for m in metrics]

    print("\n=== GP Round Diagnostics Summary ===")
    print(f"Rounds: {n}")
    print(f"Top-option prediction match rate: {sum(top_match) / n:.3f}")
    print(f"Spearman rank corr mean: {np.mean(spearman_vals):.4f} | slope/round: {_slope(spearman_vals):+.4f}")
    print(f"Kendall pair agreement mean: {np.mean(kendall_vals):.4f} | slope/round: {_slope(kendall_vals):+.4f}")
    print(f"Pairwise accuracy mean: {np.mean(acc):.4f} | slope/round: {_slope(acc):+.4f}")
    print(f"Ranking log-likelihood mean: {np.mean(ll):.4f} | slope/round: {_slope(ll):+.4f}")
    print(f"Mean pair margin mean: {np.mean(mg):.4f} | slope/round: {_slope(mg):+.4f}")
    print(f"Image variance mean: {np.mean(iv):.4f} | slope/round: {_slope(iv):+.4f}")
    print(f"Learning rate: first={lr[0]:.4f}, last={lr[-1]:.4f}")
    print(f"Top-tag sigma: first={sig[0]:.4f}, last={sig[-1]:.4f}, slope/round={_slope(sig):+.4f}")

    print("\nPer-round:")
    print(
        "round | acc    | sprmn  | kendall | loglik   | margin  | img_var | pred_top->actual | "
        "top_tag (mu,sigma)"
    )
    for m in metrics:
        print(
            f"{m.round_num:>5} | "
            f"{m.pairwise_acc:>6.3f} | "
            f"{m.spearman:>6.3f} | "
            f"{m.kendall_agree:>7.3f} | "
            f"{m.log_lik:>8.3f} | "
            f"{m.mean_margin:>7.3f} | "
            f"{m.image_variance:>7.3f} | "
            f"{m.predicted_top:>2}->{m.actual_top:<2} | "
            f"{m.top_tag[:24]:24s} ({m.top_tag_mu:.3f},{m.top_tag_sigma:.3f})"
        )


def _save_plot(metrics: List[RoundMetrics], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; skipping plot.")
        return

    if not metrics:
        return

    rounds = [m.round_num for m in metrics]
    acc = [m.pairwise_acc for m in metrics]
    ll = [m.log_lik for m in metrics]
    iv = [m.image_variance for m in metrics]
    sig = [m.top_tag_sigma for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(rounds, acc, marker="o", color="#4F46E5")
    axes[0, 0].set_title("Pairwise Accuracy")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.0, 1.0)

    axes[0, 1].plot(rounds, ll, marker="o", color="#F7567C")
    axes[0, 1].set_title("Ranking Log-Likelihood")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Log-Lik")

    axes[1, 0].plot(rounds, iv, marker="o", color="#5D576B")
    axes[1, 0].set_title("Image Variance")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Variance")

    axes[1, 1].plot(rounds, sig, marker="o", color="#22C55E")
    axes[1, 1].set_title("Top Tag Sigma")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Sigma")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GP round diagnostics JSONL.")
    parser.add_argument("--session-folder", type=str, default=None, help="Session folder path.")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to gp_round_diagnostics_v2.jsonl.")
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Output plot path (default: <session>/gp_round_diagnostics_summary.png).",
    )
    args = parser.parse_args()

    jsonl_path = _resolve_jsonl_path(args.session_folder, args.jsonl)
    records = _load_records(jsonl_path)
    metrics = _to_round_metrics(records)

    _print_summary(metrics)

    out_plot = args.out_plot
    if out_plot is None:
        out_plot = os.path.join(os.path.dirname(jsonl_path), "gp_round_diagnostics_summary.png")
    _save_plot(metrics, out_plot)


if __name__ == "__main__":
    main()

