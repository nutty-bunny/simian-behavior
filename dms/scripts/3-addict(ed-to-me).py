import os
import pickle
import warnings  # they annoy me
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

plt.rcParams["svg.fonttype"] = "none"

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/dms/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

prime_de = pd.read_csv(
    "/Users/similovesyou/Desktop/qts/simian-behavior/data/prime-de/prime-de.csv"
)

dmt_rick = os.path.join(directory, "dms.pickle")
with open(dmt_rick, "rb") as handle:
    data = pickle.load(handle)

# Replace your current data loading with:
rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/rhesus-elo-min-attempts.csv")
tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/tonkean-elo-min-attempts.csv")

# just some session stats idk
"""
# For Rhesus macaques
rhesus_sessions_range = {
    "min_sessions": rhesus_table["#_sessions"].min(),
    "max_sessions": rhesus_table["#_sessions"].max(),
    "mean_sessions": rhesus_table["#_sessions"].mean(),
}

rhesus_attempts_range = {
    "min_attempts": rhesus_table["#_attempts"].min(),
    "max_attempts": rhesus_table["#_attempts"].max(),
    "mean_attempts": rhesus_table["#_attempts"].mean(),
}

# For Tonkean macaques
tonkean_sessions_range = {
    "min_sessions": tonkean_table["#_sessions"].min(),
    "max_sessions": tonkean_table["#_sessions"].max(),
    "mean_sessions": tonkean_table["#_sessions"].mean(),
}

tonkean_attempts_range = {
    "min_attempts": tonkean_table["#_attempts"].min(),
    "max_attempts": tonkean_table["#_attempts"].max(),
    "mean_attempts": tonkean_table["#_attempts"].mean(),
}
"""

# tables now summarize device engagement, task performance & demographics
print()
print("tonkean task engagement")
print(tonkean_table)
print()
print("rhesus task engagement")
print(rhesus_table)
print()

plt.rcParams["font.family"] = (
    "DejaVu Sans"  # set global font to be used across all plots ^^ **change later**
)


# plotting proportions across the lifespan
def p_lifespan(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.6,
        edgecolor="steelblue",
        linewidth=1,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.6,
        edgecolor="purple",
        linewidth=1,
    )

    sns.regplot(
        ax=ax,
        data=rhesus_table,
        x="age",
        y=metric,
        color=colors["rhesus"],
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )
    sns.regplot(
        ax=ax,
        data=tonkean_table,
        x="age",
        y=metric,
        color=colors["tonkean"],
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )
    sns.regplot(
        ax=ax,
        data=pd.concat([rhesus_table, tonkean_table]),
        x="age",
        y=metric,
        color="black",
        order=2,
        scatter=False,
        ci=None,
    )

    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    # Remove bold x and y axes but keep thick tick marks
    ax.spines["bottom"].set_linewidth(1)  # Default thickness
    ax.spines["left"].set_linewidth(1)  # Default thickness
    ax.tick_params(
        axis="both", which="major", width=2.5, length=7
    )  # Thicker and longer ticks

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")


sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "DejaVu Sans"})

fig, axes = plt.subplots(1, 4, figsize=(20, 6))  # Keep in a row

# Adjusted colors
colors = {"tonkean": "purple", "rhesus": "steelblue"}

metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes, metrics, titles, [True] * 4, [True] + [False] * 3
):
    p_lifespan(
        ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
    )

handles = [
    plt.Line2D(
        [],
        [],
        color=colors["rhesus"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="steelblue",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (m)",
    ),
    plt.Line2D(
        [],
        [],
        color=colors["tonkean"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="purple",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (m)",
    ),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

fig.suptitle(
    "Task Performance Across the Lifespan in Macaque Species",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="purple",
)

plt.tight_layout(rect=[0, 0.02, 1, 0.93])

filename_base = "outcome-proportions"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
plt.show()

def rt_lifespan(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
):
    # Adjust very low values to sit slightly above the x-axis
    rhesus_table[metric] = np.where(
        rhesus_table[metric] < 0.02, 0.02, rhesus_table[metric]
    )
    tonkean_table[metric] = np.where(
        tonkean_table[metric] < 0.02, 0.02, tonkean_table[metric]
    )

    # Scatter plot for rhesus monkeys
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.6,
        edgecolor="steelblue",
        linewidth=1,
    )

    # Scatter plot for tonkean monkeys
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.6,
        edgecolor="purple",
        linewidth=1,
    )

    # Regression lines
    sns.regplot(
        ax=ax,
        data=rhesus_table,
        x="age",
        y=metric,
        color=colors["rhesus"],
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )
    sns.regplot(
        ax=ax,
        data=tonkean_table,
        x="age",
        y=metric,
        color=colors["tonkean"],
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )
    sns.regplot(
        ax=ax,
        data=pd.concat([rhesus_table, tonkean_table]),
        x="age",
        y=metric,
        color="black",
        order=2,
        scatter=False,
        ci=None,
    )

    # Titles and labels
    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans")
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("RTs (ms)", fontsize=14)
    else:
        ax.set_ylabel("")


# Set plot style
sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "DejaVu Sans"})

# Create subplots with the fixed height (16x7)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# **Adjusted Colors**
colors = {"tonkean": "purple", "rhesus": "steelblue"}

metrics = ["rt_success", "rt_error"]
titles = ["Hits", "Error"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes.flatten(), metrics, titles, [True, True], [True, False]
):
    rt_lifespan(
        ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
    )

# Legend (Reusing handles)
handles = [
    plt.Line2D(
        [],
        [],
        color=colors["rhesus"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="steelblue",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (m)",
    ),
    plt.Line2D(
        [],
        [],
        color=colors["tonkean"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="purple",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (m)",
    ),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

# Main title
fig.suptitle(
    "Reaction Time Profiles Across the Lifespan in Macaque Species",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="purple",
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

filename_base = "reaction-times"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
plt.show()

# plotting session frequencies (mean attempt # per session)
combined_df = pd.concat(
    [  # combine both species into one df
        rhesus_table.assign(species="Rhesus"),
        tonkean_table.assign(species="Tonkean"),
    ]
).drop_duplicates(subset="name", ignore_index=True)

# ffs these warnings r gonna end me

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    plt.figure(figsize=(10, 8))
    jitter = 0.1
    colors = {"Rhesus": "steelblue", "Tonkean": "purple"}  # Adjusted colors

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]  # males hollow
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]  # females filled

        sns.stripplot(
            x="species",
            y="age",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.6,
            size=10,
        )
        sns.stripplot(
            x="species",
            y="age",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
        )

    for species in [
        "Rhesus",
        "Tonkean",
    ]:  # Loop to plot error bars for each species (irrespective of gender)
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age"].mean()
        std_age = subset["age"].std()
        plt.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
        )

    plt.ylabel("Age", fontsize=14)  # Increased y-label size
    plt.title(
        "Demographics",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
    )  # Updated title

    plt.xticks(
        ticks=[0, 1],
        labels=[
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )  # Increased x-tick label size

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}")
    )  # Remove decimals from age
    plt.ylim(0, 25)
    plt.xlim(-0.4, 1.4)  # Adjust x-lims to move the plots together

    plt.xlabel("")  # Removed x-axis label

    filename_base = "demographics"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Create figure with original dimensions
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- LEFT PLOT (EXACT original reproduction) ---
    jitter = 0.1
    colors = {"Rhesus": "steelblue", "Tonkean": "purple", "mulatta": "steelblue"}

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.6,
            size=10,
            ax=ax1,
        )
        sns.stripplot(
            x="species",
            y="age",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax1,
        )

    for species in ["Rhesus", "Tonkean"]:
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age"].mean()
        std_age = subset["age"].std()
        ax1.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax1.set_ylabel("Age", fontsize=14)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )

    # Add SILABE subtitle
    ax1.text(
        0.5,
        1.05,
        "SILABE",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- RIGHT PLOT (PRIME-DE) ---
    prime_de["gender"] = prime_de["sex"].map({"M": 1, "F": 2})
    prime_de["dummy_x"] = 0

    # Plot females (filled)
    sns.stripplot(
        x="dummy_x",
        y="age_years",
        data=prime_de[prime_de["gender"] == 2],
        jitter=jitter,
        marker="o",
        color=colors["mulatta"],
        alpha=0.6,
        size=10,
        ax=ax2,
    )

    # Plot males (hollow)
    sns.stripplot(
        x="dummy_x",
        y="age_years",
        data=prime_de[prime_de["gender"] == 1],
        jitter=jitter,
        marker="o",
        facecolors="none",
        edgecolor=colors["mulatta"],
        size=10,
        linewidth=1.5,
        ax=ax2,
    )

    # Add mean/std
    mean_age = prime_de["age_years"].mean()
    std_age = prime_de["age_years"].std()
    ax2.errorbar(
        x=[0],
        y=[mean_age],
        yerr=[std_age],
        fmt="none",
        ecolor=colors["mulatta"],
        capsize=5,
        elinewidth=2,
        zorder=10,
    )

    # Format x-axis identically to left plot
    ax2.set_xticks([0])
    ax2.set_xticklabels(
        [f"$\\it{{Macaca\\ mulatta}}$\n(n={len(prime_de)})"], fontsize=14
    )

    # Add PRIME-DE subtitle
    ax2.text(
        0.5,
        1.05,
        "PRIME-DE",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- CRITICAL: ORIGINAL SPINE STYLING ---
    # Left plot - exactly as in your original (default matplotlib spines)
    ax1.spines["top"].set_visible(False)  # Only top spine removed

    # Right plot - match original styling
    ax2.spines["top"].set_visible(False)  # Only top spine removed
    ax2.set_ylabel("")
    ax2.set_yticklabels([])  # Remove y-axis numbers but keep spine

    # --- Shared formatting ---
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_ylim(0, 25)
        ax.set_xlabel("")

    # Adjust x-limits
    ax1.set_xlim(-0.5, 1.5)
    ax2.set_xlim(-0.5, 0.5)

    # Main title (DejaVu Sans only for title)
    fig.suptitle(
        "Demographics",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
        y=1.02,
    )

    plt.tight_layout()

    filename_base = "demographics-prime-de"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()
# awake vs anaes.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 2]}
    )

    jitter = 0.1
    colors = {"Rhesus": "steelblue", "Tonkean": "purple", "mulatta": "steelblue"}

    # -------- LEFT PLOT (SILABE) --------
    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.6,
            size=10,
            ax=ax1,
        )
        sns.stripplot(
            x="species",
            y="age",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax1,
        )

        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age"].mean()
        std_age = subset["age"].std()
        ax1.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax1.set_ylabel("Age", fontsize=14)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )
    ax1.text(
        0.5,
        1.05,
        "SILABE",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax1.spines["top"].set_visible(False)

    # -------- RIGHT PLOT (PRIME-DE: Awake vs Anesthetized) --------
    prime_de["gender"] = prime_de["sex"].map({"M": 1, "F": 2})
    prime_de["awake"] = prime_de["awake"].map({"Y": "Awake", "N": "Anesthetized"})

    for i, status in enumerate(["Awake", "Anesthetized"]):
        group = prime_de[prime_de["awake"] == status].copy()
        group["dummy_x"] = i

        # Plot females (filled)
        sns.stripplot(
            x="dummy_x",
            y="age_years",
            data=group[group["gender"] == 2],
            jitter=jitter,
            marker="o",
            color=colors["mulatta"],
            alpha=0.6,
            size=10,
            ax=ax2,
        )

        # Plot males (hollow)
        sns.stripplot(
            x="dummy_x",
            y="age_years",
            data=group[group["gender"] == 1],
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors["mulatta"],
            size=10,
            linewidth=1.5,
            ax=ax2,
        )

        # Error bars
        mean_age = group["age_years"].mean()
        std_age = group["age_years"].std()
        ax2.errorbar(
            x=[i],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors["mulatta"],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    # Set ticks with group-specific counts
    awake_n = prime_de[prime_de["awake"] == "Awake"].shape[0]
    anes_n = prime_de[prime_de["awake"] == "Anesthetized"].shape[0]
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f"$\\it{{Anesthetized}}$\n(n={anes_n})", f"$\\it{{Awake}}$\n(n={awake_n})"],
        fontsize=14,
    )

    ax2.text(
        0.5,
        1.05,
        "PRIME-DE",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2.spines["top"].set_visible(False)

    # -------- Shared formatting --------
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 25)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    ax1.set_xlim(-0.5, 1.5)
    ax2.set_xlim(-0.5, 1.5)

    # -------- Title & Save --------
    fig.suptitle(
        "Demographics",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
        y=1.02,
    )

    plt.tight_layout()
    filename_base = "demographics-awake"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()

gender_counts = combined_df.groupby(["species", "gender"]).size().unstack()
gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
gender_percentages.rename(columns={1: "Male (%)", 2: "Female (%)"}, inplace=True)
gender_percentages

def tossNaN(df, columns):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace inf with NaN
    return df.dropna(subset=columns)  # drop NaNs

columns = ["mean_attempts_per_session", "gender"]

# apply the function to toss NaNs
rhesus_table = tossNaN(rhesus_table, columns)
tonkean_table = tossNaN(tonkean_table, columns)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    jitter = 0.1
    colors = {"Rhesus": "steelblue", "Tonkean": "purple"}
    titles = ["Species", "Gender", "Age", "Social Status (Elo-Rating)"]

    for i, ax in enumerate(axes):
        if i == 0:
            for species in ["Rhesus", "Tonkean"]:
                females = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] == 2)
                ]
                males = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] != 2)
                ]

                sns.stripplot(
                    x="species",
                    y="mean_attempts_per_session",
                    data=females,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],
                    alpha=0.6,
                    size=10,
                    ax=ax,
                )
                sns.stripplot(
                    x="species",
                    y="mean_attempts_per_session",
                    data=males,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],
                    size=10,
                    linewidth=1.5,
                    ax=ax,
                )

            for species in ["Rhesus", "Tonkean"]:
                subset = combined_df[combined_df["species"] == species]
                mean = subset["mean_attempts_per_session"].mean()
                std = subset["mean_attempts_per_session"].std()
                ax.errorbar(
                    x=[species],
                    y=[mean],
                    yerr=[std],
                    fmt="none",
                    ecolor=colors[species],
                    capsize=5,
                    elinewidth=2,
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [
                    f'$\\it{{Macaca\\ mulatta}}$\n(n = {combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
                    f'$\\it{{Macaca\\ tonkeana}}$\n(n = {combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
                ],
                fontsize=14,
            )
            ax.set_xlim(-0.4, 1.4)

        elif i == 1:
            for species in ["Rhesus", "Tonkean"]:
                females = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] == 2)
                ]
                males = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] == 1)
                ]

                sns.stripplot(
                    x="gender",
                    y="mean_attempts_per_session",
                    data=females,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],
                    alpha=0.6,
                    size=10,
                    ax=ax,
                )
                sns.stripplot(
                    x="gender",
                    y="mean_attempts_per_session",
                    data=males,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],
                    size=10,
                    linewidth=1.5,
                    ax=ax,
                )

            for gender in [1, 2]:
                subset = combined_df[combined_df["gender"] == gender]
                mean = subset["mean_attempts_per_session"].mean()
                std = subset["mean_attempts_per_session"].std()
                ax.errorbar(
                    x=[gender - 1],
                    y=[mean],
                    yerr=[std],
                    fmt="none",
                    ecolor="black",
                    capsize=5,
                    elinewidth=2,
                )

            female_count = combined_df[combined_df["gender"] == 2].shape[0]
            male_count = combined_df[combined_df["gender"] == 1].shape[0]

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"Females\n(n = {female_count})", f"Males\n(n = {male_count})"],
                fontsize=14,
            )
            ax.set_xlim(-0.4, 1.4)

        elif i == 2:
            for species in ["Rhesus", "Tonkean"]:
                females = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] == 2)
                ]
                males = combined_df[
                    (combined_df["species"] == species) & (combined_df["gender"] != 2)
                ]

                ax.scatter(
                    females["age"],
                    females["mean_attempts_per_session"],
                    color=colors[species],
                    alpha=0.6,
                    s=100,
                )
                ax.scatter(
                    males["age"],
                    males["mean_attempts_per_session"],
                    edgecolor=colors[species],
                    facecolor="none",
                    s=100,
                    linewidth=1.5,
                )

            sns.regplot(
                x="age",
                y="mean_attempts_per_session",
                data=combined_df,
                order=2,
                scatter=False,
                ax=ax,
                color="black",
            )
            ax.set_xticks(range(0, 22, 5))

        elif i == 3:
        # Plot for both species
            for species in ["Rhesus", "Tonkean"]:
                species_data = combined_df[combined_df["species"] == species]
                females = species_data[species_data["gender"] == 2]
                males = species_data[species_data["gender"] == 1]
                
                ax.scatter(
                    females["elo_mean"],
                    females["mean_attempts_per_session"],
                    color=colors[species],
                    alpha=0.6,
                    s=100,
                )
                ax.scatter(
                    males["elo_mean"],
                    males["mean_attempts_per_session"],
                    edgecolor=colors[species],
                    facecolor="none",
                    s=100,
                    linewidth=1.5,
                )
                
                # Add regression line for each species
                sns.regplot(
                    x="elo_mean",
                    y="mean_attempts_per_session",
                    data=species_data,
                    scatter=False,
                    ax=ax,
                    color=colors[species],
                )
            
            ax.set_xlabel("Social Status (Elo-Rating)", fontsize=14)

        ax.set_title(titles[i], fontsize=18, fontweight="bold", fontname="DejaVu Sans")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_ylim(0, 33)
        ax.set_yticks(range(0, 33, 10))

        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        ax.set_xlabel("")
        if i in [0, 2]:
            ax.set_ylabel("Mean Count", fontsize=15)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(
        "Multivariate Predictors of Task Engagement",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
    )

    filename_base = "task-engagement"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()

    # Perform species comparison (as done previously)
    rhesus_data = combined_df[combined_df["species"] == "Rhesus"]["mean_attempts_per_session"]
    tonkean_data = combined_df[combined_df["species"] == "Tonkean"]["mean_attempts_per_session"]

    t_stat, p_value = stats.ttest_ind(rhesus_data, tonkean_data, equal_var=False)

    # Descriptive statistics for Rhesus
    rhesus_mean = rhesus_data.mean()
    rhesus_std = rhesus_data.std()
    rhesus_n = rhesus_data.count()
    rhesus_ci = stats.t.interval(
        0.95, rhesus_n - 1, loc=rhesus_mean, scale=rhesus_std / np.sqrt(rhesus_n)
    )

    # Descriptive statistics for Tonkean
    tonkean_mean = tonkean_data.mean()
    tonkean_std = tonkean_data.std()
    tonkean_n = tonkean_data.count()
    tonkean_ci = stats.t.interval(
        0.95, tonkean_n - 1, loc=tonkean_mean, scale=tonkean_std / np.sqrt(tonkean_n)
    )

    print(
        f"Rhesus: Mean attempts = {rhesus_mean:.2f}, SD = {rhesus_std:.2f}, n = {rhesus_n}, 95% CI = ({rhesus_ci[0]:.2f}, {rhesus_ci[1]:.2f})"
    )
    print(
        f"Tonkean: Mean attempts = {tonkean_mean:.2f}, SD = {tonkean_std:.2f}, n = {tonkean_n}, 95% CI = ({tonkean_ci[0]:.2f}, {tonkean_ci[1]:.2f})"
    )

    print(f"\nT-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("There is a significant difference between the species.")
    else:
        print("There is no significant difference between the species.")

    # Gender testing across species
    print("\nGender comparison across species:")

    rhesus_females = combined_df[
        (combined_df["species"] == "Rhesus") & (combined_df["gender"] == 2)
    ]["mean_attempts_per_session"]
    rhesus_males = combined_df[
        (combined_df["species"] == "Rhesus") & (combined_df["gender"] == 1)
    ]["mean_attempts_per_session"]
    tonkean_females = combined_df[
        (combined_df["species"] == "Tonkean") & (combined_df["gender"] == 2)
    ]["mean_attempts_per_session"]
    tonkean_males = combined_df[
        (combined_df["species"] == "Tonkean") & (combined_df["gender"] == 1)
    ]["mean_attempts_per_session"]

    # Perform t-tests for gender within species
    print("\nWithin species gender comparison:")

    print("Rhesus: ")
    t_stat_rhesus, p_value_rhesus = stats.ttest_ind(
        rhesus_females, rhesus_males, equal_var=False
    )
    print(f"T-statistic: {t_stat_rhesus:.4f}, P-value: {p_value_rhesus:.4f}")

    print("Tonkean: ")
    t_stat_tonkean, p_value_tonkean = stats.ttest_ind(
        tonkean_females, tonkean_males, equal_var=False
    )
    print(f"T-statistic: {t_stat_tonkean:.4f}, P-value: {p_value_tonkean:.4f}")

    # Gender comparison across species (combined)
    print("\nAcross species gender comparison:")
    all_females = combined_df[combined_df["gender"] == 2]["mean_attempts_per_session"]
    all_males = combined_df[combined_df["gender"] == 1]["mean_attempts_per_session"]
    t_stat_gender, p_value_gender = stats.ttest_ind(
        all_females, all_males, equal_var=False
    )

    print(f"\nT-statistic: {t_stat_gender:.4f}, P-value: {p_value_gender:.4f}")
    if p_value_gender < 0.05:
        print("There is a significant gender difference across species.")
    else:
        print("There is no significant gender difference across species.")

    # Pearson correlation across species
    pearson_corr_all, pearson_p_all = stats.pearsonr(
        combined_df["age"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"Overall Pearson correlation between age and mean attempts: r = {pearson_corr_all:.3f}, p = {pearson_p_all:.3f}"
    )

    # Spearman correlation across species (non-parametric)
    spearman_corr_all, spearman_p_all = stats.spearmanr(
        combined_df["age"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"Overall Spearman correlation between age and mean attempts: r = {spearman_corr_all:.3f}, p = {spearman_p_all:.3f}"
    )

    # Within species - Rhesus
    rhesus_data = combined_df[combined_df["species"] == "Rhesus"]
    pearson_corr_rhesus, pearson_p_rhesus = stats.pearsonr(
        rhesus_data["age"], rhesus_data["mean_attempts_per_session"]
    )
    spearman_corr_rhesus, spearman_p_rhesus = stats.spearmanr(
        rhesus_data["age"], rhesus_data["mean_attempts_per_session"]
    )
    print(
        f"Rhesus - Pearson: r = {pearson_corr_rhesus:.3f}, p = {pearson_p_rhesus:.3f}"
    )
    print(
        f"Rhesus - Spearman: r = {spearman_corr_rhesus:.3f}, p = {spearman_p_rhesus:.3f}"
    )

    # Within species - Tonkean
    tonkean_data = combined_df[combined_df["species"] == "Tonkean"]
    pearson_corr_tonkean, pearson_p_tonkean = stats.pearsonr(
        tonkean_data["age"], tonkean_data["mean_attempts_per_session"]
    )
    spearman_corr_tonkean, spearman_p_tonkean = stats.spearmanr(
        tonkean_data["age"], tonkean_data["mean_attempts_per_session"]
    )
    print(
        f"Tonkean - Pearson: r = {pearson_corr_tonkean:.3f}, p = {pearson_p_tonkean:.3f}"
    )
    print(
        f"Tonkean - Spearman: r = {spearman_corr_tonkean:.3f}, p = {spearman_p_tonkean:.3f}"
    )
    # Pearson correlation between Elo-Rating and mean attempts across species
    pearson_corr_elo_all, pearson_p_elo_all = stats.pearsonr(
        combined_df["elo_mean"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"\nOverall Pearson correlation between Elo-Rating and mean attempts: r = {pearson_corr_elo_all:.3f}, p = {pearson_p_elo_all:.3f}"
    )

    # Spearman correlation (non-parametric)
    spearman_corr_elo_all, spearman_p_elo_all = stats.spearmanr(
        combined_df["elo_mean"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"Overall Spearman correlation between Elo-Rating and mean attempts: r = {spearman_corr_elo_all:.3f}, p = {spearman_p_elo_all:.3f}"
    )

    # Within species - Rhesus
    pearson_corr_elo_rhesus, pearson_p_elo_rhesus = stats.pearsonr(
        rhesus_data["elo_mean"], rhesus_data["mean_attempts_per_session"]
    )
    spearman_corr_elo_rhesus, spearman_p_elo_rhesus = stats.spearmanr(
        rhesus_data["elo_mean"], rhesus_data["mean_attempts_per_session"]
    )
    print(
        f"\nRhesus - Pearson correlation (Elo): r = {pearson_corr_elo_rhesus:.3f}, p = {pearson_p_elo_rhesus:.3f}"
    )
    print(
        f"Rhesus - Spearman correlation (Elo): r = {spearman_corr_elo_rhesus:.3f}, p = {spearman_p_elo_rhesus:.3f}"
    )

    # Within species - Tonkean
    pearson_corr_elo_tonkean, pearson_p_elo_tonkean = stats.pearsonr(
        tonkean_data["elo_mean"], tonkean_data["mean_attempts_per_session"]
    )
    spearman_corr_elo_tonkean, spearman_p_elo_tonkean = stats.spearmanr(
        tonkean_data["elo_mean"], tonkean_data["mean_attempts_per_session"]
    )
    print(
        f"\nTonkean - Pearson correlation (Elo): r = {pearson_corr_elo_tonkean:.3f}, p = {pearson_p_elo_tonkean:.3f}"
    )
    print(
        f"Tonkean - Spearman correlation (Elo): r = {spearman_corr_elo_tonkean:.3f}, p = {spearman_p_elo_tonkean:.3f}"
    )
    
# I have no clue what follows, ignore
"""
# Statistical Tests
def run_stat_tests(group1, group2, group1_name, group2_name, metric_name):
   
    #Runs appropriate statistical test (t-test or Mann-Whitney U) for two groups.
    #Prints a clear summary of the results.
    
    # Check normality
    normal_group1 = shapiro(group1).pvalue > 0.05
    normal_group2 = shapiro(group2).pvalue > 0.05

    # Choose test based on normality
    if normal_group1 and normal_group2:
        test_name = "t-test"
        test_result = ttest_ind(group1, group2, equal_var=False)
    else:
        test_name = "Mann-Whitney U"
        test_result = mannwhitneyu(group1, group2)

    # Print results
    print(f"\n{metric_name}: {group1_name} vs {group2_name} ({test_name})")
    print(f"  Test Statistic: {test_result.statistic:.3f}")
    print(f"  p-value: {test_result.pvalue:.3f}")


metrics = {
    "Mean Attempts Per Session": ("mean_attempts_per_session", rhesus_table, tonkean_table),
    "Total Number of Attempts": ("#_attempts", rhesus_table, tonkean_table),
    "Proportion of Successful Attempts": ("p_success", rhesus_table, tonkean_table),
    "Proportion of Erroneous Attempts": ("p_error", rhesus_table, tonkean_table),
    "Proportion of Omitted Attempts": ("p_omission", rhesus_table, tonkean_table),
    "Proportion of Premature Attempts": ("p_premature", rhesus_table, tonkean_table),
    "Age of Subjects": ("age", rhesus_table, tonkean_table),
    "Mean Reaction Time (Successful Attempts)": (
        "rt_success",
        rhesus_table,
        tonkean_table,
    ),
    "Mean Reaction Time (Erroneous Attempts)": (
        "rt_error",
        rhesus_table,
        tonkean_table,
    ),
}

# Species-wise comparisons
for metric_name, (metric, rhesus_data, tonkean_data) in metrics.items():
    run_stat_tests(
        rhesus_data[metric], tonkean_data[metric], "Rhesus", "Tonkean", metric_name
    )

# Gender-wise comparisons
for metric_name, (metric, _, _) in metrics.items():
    male_data = combined_df[combined_df["gender"] == 1][metric]
    female_data = combined_df[combined_df["gender"] == 2][metric]
    run_stat_tests(male_data, female_data, "Male", "Female", metric_name)

species_list = ["Rhesus", "Tonkean"]

for species in species_list:
    for metric_name, (metric, _, _) in metrics.items():
        species_data = combined_df[combined_df["species"] == species]

        male_data_species = species_data[species_data["gender"] == 1][metric]
        female_data_species = species_data[species_data["gender"] == 2][metric]

        run_stat_tests(
            male_data_species,
            female_data_species,
            f"{species} Male",
            f"{species} Female",
            f"{metric_name} ({species})",
        )


proportions = {}

for species, monkeys in data.items():
    proportions[species] = {}

    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")

        if attempts is not None:
            # Calculate proportions of each result type
            result_counts = attempts["result"].value_counts(normalize=True)

            proportions[species][name] = {
                "p_success": result_counts.get("success", 0.0),
                "p_error": result_counts.get("error", 0.0),
                "p_omission": result_counts.get("stepomission", 0.0),
                "p_premature": result_counts.get("premature", 0.0),
            }

# Convert to a DataFrame for better visualization
proportions_df = []
for species, species_data in proportions.items():
    for name, stats in species_data.items():
        stats["species"] = species
        stats["name"] = name
        proportions_df.append(stats)

proportions_df = pd.DataFrame(proportions_df)

for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        print(
            f"{name}: {monkey_data['attempts']['result'].value_counts(normalize=True)}"
        )


# Function to run statistical tests and post hoc tests
def run_stat_tests(group1, group2, group1_name, group2_name, metric_name):
    
    #Runs appropriate statistical test (t-test or Mann-Whitney U) for two groups.
    #If significant, applies post hoc Bonferroni correction for multiple testing.
    #Prints a clear summary of the results.
    
    # Check normality
    normal_group1 = shapiro(group1).pvalue > 0.05
    normal_group2 = shapiro(group2).pvalue > 0.05

    # Choose test based on normality
    if normal_group1 and normal_group2:
        test_name = "t-test"
        test_result = ttest_ind(group1, group2, equal_var=False)
    else:
        test_name = "Mann-Whitney U"
        test_result = mannwhitneyu(group1, group2)

    # Print results
    print(f"\n{metric_name}: {group1_name} vs {group2_name} ({test_name})")
    print(f"  Test Statistic: {test_result.statistic:.3f}")
    print(f"  p-value: {test_result.pvalue:.3f}")

    # Apply post hoc test if significant
    if test_result.pvalue < 0.05:
        print(
            "  Significant difference found! Applying Bonferroni correction for post hoc tests."
        )
        pvals = [test_result.pvalue]
        corrected_pvals = smm.multipletests(pvals, method="bonferroni")[1]
        print(f"  Bonferroni corrected p-value: {corrected_pvals[0]:.3f}")


# Metrics to be tested
metrics = {
    "Mean Attempts Per Session": ("mean_attempts_per_session", rhesus_table, tonkean_table),
    "Total Number of Attempts": ("#_attempts", rhesus_table, tonkean_table),
    "Proportion of Successful Attempts": ("p_success", rhesus_table, tonkean_table),
    "Proportion of Erroneous Attempts": ("p_error", rhesus_table, tonkean_table),
    "Proportion of Omitted Attempts": ("p_omission", rhesus_table, tonkean_table),
    "Proportion of Premature Attempts": ("p_premature", rhesus_table, tonkean_table),
    "Age of Subjects": ("age", rhesus_table, tonkean_table),
    "Mean Reaction Time (Successful Attempts)": (
        "rt_success",
        rhesus_table,
        tonkean_table,
    ),
    "Mean Reaction Time (Erroneous Attempts)": (
        "rt_error",
        rhesus_table,
        tonkean_table,
    ),
}

# Species-wise comparisons
for metric_name, (metric, rhesus_data, tonkean_data) in metrics.items():
    run_stat_tests(
        rhesus_data[metric], tonkean_data[metric], "Rhesus", "Tonkean", metric_name
    )

# Gender-wise comparisons
for metric_name, (metric, _, _) in metrics.items():
    male_data = combined_df[combined_df["gender"] == 1][metric]
    female_data = combined_df[combined_df["gender"] == 2][metric]
    run_stat_tests(male_data, female_data, "Male", "Female", metric_name)

species_list = ["Rhesus", "Tonkean"]

for species in species_list:
    for metric_name, (metric, _, _) in metrics.items():
        species_data = combined_df[combined_df["species"] == species]

        male_data_species = species_data[species_data["gender"] == 1][metric]
        female_data_species = species_data[species_data["gender"] == 2][metric]

        run_stat_tests(
            male_data_species,
            female_data_species,
            f"{species} Male",
            f"{species} Female",
            f"{metric_name} ({species})",
        )

###

# Calculate proportions
proportions = {}

for species, monkeys in data.items():
    proportions[species] = {}

    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")

        if attempts is not None:
            # Calculate proportions of each result type
            result_counts = attempts["result"].value_counts(normalize=True)

            proportions[species][name] = {
                "p_success": result_counts.get("success", 0.0),
                "p_error": result_counts.get("error", 0.0),
                "p_omission": result_counts.get("stepomission", 0.0),
                "p_premature": result_counts.get("premature", 0.0),
            }

# Convert to a DataFrame for better visualization
proportions_df = []
for species, species_data in proportions.items():
    for name, stats in species_data.items():
        stats["species"] = species
        stats["name"] = name
        proportions_df.append(stats)

proportions_df = pd.DataFrame(proportions_df)

for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        print(
            f"{name}: {monkey_data['attempts']['result'].value_counts(normalize=True)}"
        )
"""