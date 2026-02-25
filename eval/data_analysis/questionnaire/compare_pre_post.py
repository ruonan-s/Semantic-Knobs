"""
Generate pre/post questionnaire comparison plots for the 4 shared questions.

For each question, save one figure with three panels:
1) Overall pre vs post
2) Per participant (paired lines)
3) Per descriptor (grouped bars)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats


BASE_DIR = Path(__file__).resolve().parent
PRE_FILE = BASE_DIR / "pre_session.csv"
POST_FILE = BASE_DIR / "post_session.csv"
FIG_DIR = BASE_DIR.parent / "figures"

ID_COL = "Your participant ID"
DESC_COL = "Descriptor"

QUESTION_COLS = [
    "I have a clear sense of what [descriptor] means to me in terms of a physical space.",
    "I could describe my ideal [descriptor] space to an interior designer in concrete terms.",
    "I can mentally picture what a [descriptor] environment would look like for me.",
    "I could explain how my interpretation of [descriptor] differs from someone else's. ",
]

QUESTION_SHORT = {
    QUESTION_COLS[0]: "Q1 Clear sense",
    QUESTION_COLS[1]: "Q2 Concrete description",
    QUESTION_COLS[2]: "Q3 Mental picture",
    QUESTION_COLS[3]: "Q4 Explain differences",
}

PHASE_COLORS = {"pre": "#78909C", "post": "#E91E63"}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype(str).str.strip().str.upper()
    out[DESC_COL] = out[DESC_COL].astype(str).str.strip().str.title()
    return out


def _prepare_paired(pre: pd.DataFrame, post: pd.DataFrame, q: str) -> pd.DataFrame:
    pre_q = pre[[ID_COL, DESC_COL, q]].rename(columns={q: "pre"})
    post_q = post[[ID_COL, DESC_COL, q]].rename(columns={q: "post"})

    for c in ("pre", "post"):
        if c == "pre":
            pre_q[c] = pd.to_numeric(pre_q[c], errors="coerce")
        else:
            post_q[c] = pd.to_numeric(post_q[c], errors="coerce")

    # Some participant+descriptor rows may appear more than once.
    pre_q = pre_q.groupby([ID_COL, DESC_COL], as_index=False)["pre"].mean()
    post_q = post_q.groupby([ID_COL, DESC_COL], as_index=False)["post"].mean()

    paired = pre_q.merge(post_q, on=[ID_COL, DESC_COL], how="inner")
    paired = paired.dropna(subset=["pre", "post"])
    return paired


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _overall_significance(paired: pd.DataFrame) -> tuple[float | None, str]:
    if len(paired) < 5:
        return None, "n/a"
    try:
        res = scipy_stats.wilcoxon(paired["pre"], paired["post"], alternative="two-sided")
        return float(res.pvalue), _p_to_stars(float(res.pvalue))
    except ValueError:
        return None, "n/a"


def _plot_overall(paired: pd.DataFrame, question: str, out_stem: str) -> None:
    question_name = QUESTION_SHORT[question]

    overall = pd.DataFrame(
        {
            "phase": ["pre", "post"],
            "score": [paired["pre"].mean(), paired["post"].mean()],
            "sem": [
                paired["pre"].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
                paired["post"].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
            ],
        }
    )
    p_val, stars = _overall_significance(paired)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    bars = ax.bar(
        x,
        overall["score"].values,
        yerr=1.96 * overall["sem"].values,
        color=[PHASE_COLORS["pre"], PHASE_COLORS["post"]],
        edgecolor="white",
        capsize=6,
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean score (1-7)")
    ax.set_title(f"{question_name}: Overall")
    ax.set_ylim(0, 7.4)
    ax.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, overall["score"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add significance bracket for paired pre/post comparison.
    y_top = max(
        overall["score"].values[0] + 1.96 * overall["sem"].values[0],
        overall["score"].values[1] + 1.96 * overall["sem"].values[1],
    ) + 0.28
    ax.plot([0, 0, 1, 1], [y_top - 0.06, y_top, y_top, y_top - 0.06], color="#333333", linewidth=1.2)
    if p_val is None:
        sig_text = f"{stars}"
    else:
        sig_text = f"{stars} (p={p_val:.3g})"
    ax.text(0.5, y_top + 0.03, sig_text, ha="center", va="bottom", fontsize=9)
    ax.text(0.99, 0.02, f"Paired N={len(paired)} (Wilcoxon)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, alpha=0.7)

    fig.tight_layout()
    out_name = f"{out_stem}_overall.png"
    fig.savefig(FIG_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")


def _plot_per_participant(paired: pd.DataFrame, question: str, out_stem: str) -> None:
    question_name = QUESTION_SHORT[question]
    participant = paired.groupby(ID_COL)[["pre", "post"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    p_sorted = participant.sort_values(ID_COL)
    for _, row in p_sorted.iterrows():
        ax.plot([0, 1], [row["pre"], row["post"]], color="#B0BEC5", linewidth=1, alpha=0.9)
        ax.scatter([0, 1], [row["pre"], row["post"]],
                   color=[PHASE_COLORS["pre"], PHASE_COLORS["post"]], s=35, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Participant mean score")
    ax.set_title(f"{question_name}: Per Participant (paired)")
    ax.set_ylim(0, 7.3)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.02, 0.03, f"N participants: {len(p_sorted)}",
            transform=ax.transAxes, fontsize=8, alpha=0.7)

    fig.tight_layout()
    out_name = f"{out_stem}_per_participant.png"
    fig.savefig(FIG_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")


def _plot_per_descriptor(paired: pd.DataFrame, question: str, out_stem: str) -> None:
    question_name = QUESTION_SHORT[question]
    descriptor = paired.groupby(DESC_COL)[["pre", "post"]].mean().reset_index()
    descriptor = descriptor.sort_values(DESC_COL)
    fig, ax = plt.subplots(figsize=(8, 5))
    d_long = descriptor.melt(id_vars=[DESC_COL], value_vars=["pre", "post"],
                             var_name="phase", value_name="score")
    sns.barplot(
        data=d_long,
        x=DESC_COL,
        y="score",
        hue="phase",
        palette=PHASE_COLORS,
        ax=ax,
        edgecolor="white",
    )
    ax.set_title(f"{question_name}: Per Descriptor")
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 7.3)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="", loc="upper left")

    fig.tight_layout()
    out_name = f"{out_stem}_per_descriptor.png"
    fig.savefig(FIG_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pre = _clean(pd.read_csv(PRE_FILE))
    post = _clean(pd.read_csv(POST_FILE))

    for q in QUESTION_COLS:
        paired = _prepare_paired(pre, post, q)
        if paired.empty:
            print(f"Skipping {QUESTION_SHORT[q]} (no paired rows)")
            continue
        out_stem = f"questionnaire_{QUESTION_SHORT[q].lower().replace(' ', '_')}"
        _plot_overall(paired, q, out_stem)
        _plot_per_participant(paired, q, out_stem)
        _plot_per_descriptor(paired, q, out_stem)

    print(f"Done. Questionnaire figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
