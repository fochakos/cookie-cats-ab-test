"""Draw the README charts from the data and the outputs of analysis.py.

Usage (from the project root, after analysis.py):
    python src/charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import CONTROL, DATA, REPORTS, TREATMENT  # noqa: E402
from stats_tools import mde_two_prop  # noqa: E402

FIG = REPORTS / "figures"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COLORS = {CONTROL: BLUE, TREATMENT: ORANGE}
LABELS = {CONTROL: "Gate at level 30 (current)", TREATMENT: "Gate at level 40 (test)"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 24,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def tidy(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left" if axis == "y" else "bottom"].set_visible(False)
    ax.tick_params(axis=axis, length=0)


def retention_ci(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [(axes[0], "retention_1", "Played again 1 day after install"),
                              (axes[1], "retention_7", "Played again 7 days after install")]:
        r = res["retention"][metric]
        for i, (arm, p, n) in enumerate([(CONTROL, r["p_a"], r["n_control"]),
                                         (TREATMENT, r["p_b"], r["n_treatment"])]):
            half = 1.96 * np.sqrt(p * (1 - p) / n)
            ax.errorbar([i], [p], yerr=[[half], [half]], fmt="o", color=COLORS[arm], markersize=9,
                        markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=2, capsize=6, capthick=2)
            ax.text(i + 0.12, p, f"{p:.1%}", va="center", color=INK, fontsize=10)
        ax.set_xticks([0, 1], [LABELS[CONTROL], LABELS[TREATMENT]])
        ax.set_xlim(-0.5, 1.6)
        mid = (r["p_a"] + r["p_b"]) / 2
        ax.set_ylim(mid - 0.017, mid + 0.017)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        tidy(ax)
        verdict = "significant" if r["p_value_holm"] < 0.05 else "not significant"
        ax.set_title(title, fontsize=11, pad=8)
        ax.text(0.5, 0.04, f"difference {100 * r['diff']:+.2f} pp, p = {r['p_value']:.3f} ({verdict})",
                transform=ax.transAxes, ha="center", color=INK_2, fontsize=9.5)
    fig.suptitle("Moving the gate to level 40 lowered 7-day retention",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.955, f"Share of players retained, with 95% confidence intervals "
             f"({res['players']:,} players). The y-axes are zoomed in to show the gap.",
             color=INK_2, fontsize=9.5)
    fig.savefig(FIG / "01_retention_ci.png")
    plt.close(fig)


def bootstrap(res):
    boot = 100 * np.load(REPORTS / "bootstrap_retention_7.npy")
    r = res["retention"]["retention_7"]
    lo, hi = 100 * r["boot_ci_low"], 100 * r["boot_ci_high"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    counts, edges, patches = ax.hist(boot, bins=70, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    for patch, left in zip(patches, edges[:-1]):
        if not lo <= left <= hi:
            patch.set_alpha(0.35)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.text(0.03, counts.max() * 0.95, "no difference", color=INK, fontsize=9.5)
    ax.text((lo + hi) / 2, counts.max() * 1.04, f"95% interval {lo:.2f} to {hi:.2f} pp",
            ha="center", color=INK, fontsize=9.5)
    ax.set_ylim(top=counts.max() * 1.12)
    ax.set_title(f"{100 * r['boot_share_below_zero']:.1f}% of bootstrap samples favour the level-30 gate")
    subtitle(ax, "7-day retention, gate 40 minus gate 30, across 20,000 bootstrap resamples")
    ax.set_xlabel("Difference in 7-day retention (percentage points)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fig.savefig(FIG / "02_bootstrap_retention_7.png")
    plt.close(fig)


def rounds_survival():
    df = pd.read_csv(DATA)
    ks = np.arange(0, 81)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for arm in [CONTROL, TREATMENT]:
        r = df.loc[df.version == arm, "sum_gamerounds"].to_numpy()
        share = [(r >= k).mean() for k in ks]
        ax.plot(ks, share, color=COLORS[arm], linewidth=2, label=LABELS[arm])
    for gate in (30, 40):
        ax.axvline(gate, color=MUTED, linewidth=1, linestyle="--")
        ax.text(gate + 0.8, 0.93, f"{gate} rounds", color=MUTED, fontsize=9)
    reach30 = (df.sum_gamerounds >= 30).mean()
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 80)
    tidy(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88))
    ax.set_title(f"At most {reach30:.0%} of players could have reached level 30")
    subtitle(ax, "Share of players who played at least this many rounds in their first 14 days. "
             "Replays count as rounds, so reaching level 30 takes 30+ rounds.")
    ax.set_xlabel("Game rounds played")
    fig.savefig(FIG / "03_rounds_played.png")
    plt.close(fig)


def power_curve(res):
    r = res["retention"]["retention_7"]
    n_used = res["power"]["n_per_arm_used"]
    ns = np.linspace(5_000, 150_000, 300)
    mde = [100 * mde_two_prop(r["p_a"], n) for n in ns]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ns / 1000, mde, color=BLUE, linewidth=2)
    m_used = 100 * mde_two_prop(r["p_a"], n_used)
    ax.scatter([n_used / 1000], [m_used], s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"this test: {n_used / 1000:.1f}k per group\ndetects {m_used:.2f} pp",
                (n_used / 1000, m_used), xytext=(12, 14), textcoords="offset points", color=INK, fontsize=9.5)
    ax.axhline(abs(100 * r["diff"]), color=ORANGE, linewidth=1.2, linestyle="--")
    ax.text(150, abs(100 * r["diff"]) + 0.06, f"observed gap {abs(100 * r['diff']):.2f} pp",
            ha="right", color=INK_2, fontsize=9.5)
    ax.set_ylim(0, max(mde) * 1.05)
    tidy(ax)
    ax.set_title("The test was big enough to detect the gap it found")
    subtitle(ax, "Smallest 7-day retention difference detectable with 80% power at α = 0.05")
    ax.set_xlabel("Players per group (thousands)")
    ax.set_ylabel("Detectable difference (pp)")
    fig.savefig(FIG / "04_power_curve.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPORTS / "results.json").read_text())
    retention_ci(res)
    bootstrap(res)
    rounds_survival()
    power_curve(res)
    print(f"Saved charts to {FIG}")


if __name__ == "__main__":
    main()
