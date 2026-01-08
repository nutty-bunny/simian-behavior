import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import matplotlib.gridspec as gridspec

plt.rcParams["svg.fonttype"] = "none"

figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

# Load all datasets
# Row 1: Full datasets
rhesus_table_full = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo.csv")
tonkean_table_full = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo.csv")

# Row 2: Min attempts datasets
rhesus_table_min = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo-min-attempts.csv")
tonkean_table_min = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo-min-attempts.csv")

# Row 3: Top-13 datasets (these are tonkean only)
tonkean_table_full_13 = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/13-individual/tonkean-elo-13.csv")
tonkean_table_min_13 = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/13-individual/tonkean-elo-min-attempts-13.csv")

# Process all datasets
for table in [rhesus_table_full, tonkean_table_full, rhesus_table_min, tonkean_table_min, tonkean_table_full_13, tonkean_table_min_13]:
    table["start_date"] = pd.to_datetime(table["start_date"])
    table["end_date"] = pd.to_datetime(table["end_date"])
    table["task_midpoint"] = table["start_date"] + (table["end_date"] - table["start_date"]) / 2
    table["age_at_task"] = table.apply(
        lambda row: round(row["age"] + (row["task_midpoint"] - row["start_date"]).days / 365.25, 2)
        if pd.notna(row["age"]) and pd.notna(row["task_midpoint"]) else pd.NA,
        axis=1
    )

plt.rcParams['axes.axisbelow'] = False

# Color scheme for species comparison
colors_species = {"tonkean": "purple", "rhesus": "steelblue"}

# Color scheme for tonkean comparison
colors_comparison = {
    "full": "seagreen",
    "full_edge": "seagreen", 
    "full_line": "seagreen",
    "restricted": "navy",
    "restricted_edge": "navy",
    "restricted_line": "navy"
}

# Function for species comparison plot - ROW 1 (Full datasets with quadratic fits)
def p_lifespan_row1(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel, show_title
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=100,
        linewidth=1.5,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["rhesus"],
        s=100,
        alpha=0.7,
        edgecolor="steelblue",
        linewidth=1.5,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=100,
        linewidth=1.5,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["tonkean"],
        s=100,
        alpha=0.7,
        edgecolor="purple",
        linewidth=1.5,
    )

    # Row 1: Full datasets - quadratic for Hits, Errors, Premature; linear for Omissions
    if metric == "p_success":
        # Hits: inverted-U (peak) at 9.6 years
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=2,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )
    elif metric == "p_error":
        # Errors: U-shaped (valley) at 13.1 years
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=2,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )
    elif metric == "p_premature":
        # Premature: U-shaped (valley) at 5.9 years
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=2,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )
    elif metric == "p_omission":
        # Omissions: decreases linearly with age
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )

    if show_title:
        ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"], fontsize=16)
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7, labelsize=16)

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=18)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=18)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

# Function for species comparison plot - ROW 2 (Min attempts datasets)
def p_lifespan_row2(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel, show_title
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=100,
        linewidth=1.5,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["rhesus"],
        s=100,
        alpha=0.7,
        edgecolor="steelblue",
        linewidth=1.5,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=100,
        linewidth=1.5,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["tonkean"],
        s=100,
        alpha=0.7,
        edgecolor="purple",
        linewidth=1.5,
    )

    # Row 2: Min attempts - quadratic for Hits; linear for Premature and Omissions; no fit for Errors
    if metric == "p_success":
        # Hits: inverted-U (peak) at 4.0 years
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=2,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )
    elif metric == "p_error":
        # Errors: no significant age effect
        pass
    elif metric == "p_premature":
        # Premature: increases linearly with age
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )
    elif metric == "p_omission":
        # Omissions: decreases linearly with age
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2.0}
        )

    if show_title:
        ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"], fontsize=16)
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7, labelsize=16)

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=18)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=18)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

# Function for tonkean comparison plot - ROW 3 (Top-13 datasets, all linear fits)
def p_lifespan_combined(
    ax, metric, title, tonkean_full, tonkean_min, colors, show_xlabel, show_ylabel, show_title
):
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["full_edge"],
        s=100,
        linewidth=1.5,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["full"],
        s=100,
        alpha=0.7,
        edgecolor=colors["full_edge"],
        linewidth=1.5,
    )
    
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["restricted_edge"],
        s=100,
        linewidth=1.5,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["restricted"],
        s=100,
        alpha=0.7,
        edgecolor=colors["restricted_edge"],
        linewidth=1.5,
    )

    # Row 3: Top-13 datasets - all linear fits (no quadratic improvements)
    # Linear fits for all metrics
    sns.regplot(
        ax=ax,
        data=tonkean_full,
        x="age_at_task",
        y=metric,
        color=colors["full_line"], 
        order=1,
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2.0, "alpha": 1.0}
    )
    sns.regplot(
        ax=ax,
        data=tonkean_min,
        x="age_at_task",
        y=metric,
        color=colors["restricted_line"], 
        order=1,
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2.0, "alpha": 0.6}
    )

    if show_title:
        ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"], fontsize=16)
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7, labelsize=16)

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=18)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=18)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

# Create combined 3x4 plot with slightly wider aspect
sns.set(style="whitegrid", font_scale=1.2)

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]

# First row: Full datasets (with titles, quadratic fits)
for i, (ax, metric, title) in enumerate(zip(axes[0], metrics, titles)):
    show_xlabel = False
    show_ylabel = (i == 0)
    show_title = True
    p_lifespan_row1(
        ax, metric, title, rhesus_table_full, tonkean_table_full, colors_species, show_xlabel, show_ylabel, show_title
    )

# Second row: Min attempts datasets (no titles, mixed fits)
for i, (ax, metric, title) in enumerate(zip(axes[1], metrics, titles)):
    show_xlabel = False
    show_ylabel = (i == 0)
    show_title = False
    p_lifespan_row2(
        ax, metric, title, rhesus_table_min, tonkean_table_min, colors_species, show_xlabel, show_ylabel, show_title
    )

# Third row: Top-13 tonkean comparison (no titles, with age labels, linear fits only)
for i, (ax, metric, title) in enumerate(zip(axes[2], metrics, titles)):
    show_xlabel = True
    show_ylabel = (i == 0)
    show_title = False
    p_lifespan_combined(
        ax, metric, title, tonkean_table_full_13, tonkean_table_min_13, colors_comparison, show_xlabel, show_ylabel, show_title
    )

# Create complete legend with all combinations
legend_handles = [
    # Row 1 & 2: M. mulatta
    plt.Line2D([], [], color=colors_species["rhesus"], marker="o", linestyle="None", 
               markersize=10, label=r"$\it{M.\ mulatta}$ (f)", alpha=0.7, markeredgecolor="steelblue", markeredgewidth=1.5),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="steelblue",
               linestyle="None", markersize=10, markeredgewidth=1.5, label=r"$\it{M.\ mulatta}$ (m)"),
    # Row 1 & 2: M. tonkeana
    plt.Line2D([], [], color=colors_species["tonkean"], marker="o", linestyle="None", 
               markersize=10, label=r"$\it{M.\ tonkeana}$ (f)", alpha=0.7, markeredgecolor="purple", markeredgewidth=1.5),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors_species["tonkean"],
               linestyle="None", markersize=10, markeredgewidth=1.5, label=r"$\it{M.\ tonkeana}$ (m)"),
    # Row 3: Top-13 Restricted
    plt.Line2D([], [], color=colors_comparison["restricted"], marker="o", linestyle="None", 
               markersize=10, label="Top-13 Restricted (f)", alpha=0.7, markeredgecolor=colors_comparison["restricted_edge"], markeredgewidth=1.5),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors_comparison["restricted_edge"],
               linestyle="None", markersize=10, markeredgewidth=1.5, label="Top-13 Restricted (m)"),
    # Row 3: Top-13 Unrestricted
    plt.Line2D([], [], color=colors_comparison["full"], marker="o", linestyle="None", 
               markersize=10, label="Top-13 Unrestricted (f)", alpha=0.7, markeredgecolor=colors_comparison["full_edge"], markeredgewidth=1.5),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors_comparison["full_edge"],
               linestyle="None", markersize=10, markeredgewidth=1.5, label="Top-13 Unrestricted (m)"),
]

fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), 
           ncol=4, fontsize=13, frameon=True, columnspacing=1.5)

plt.tight_layout(rect=[0, 0.04, 1, 1])

filename_base = "combined-proportion-plots-3rows"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
plt.savefig(save_path_png, format="png", dpi=300, bbox_inches='tight')
plt.show()