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

plt.rcParams["svg.fonttype"] = "none"

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

prime_de = pd.read_csv(
    "/Users/similovesyou/Desktop/qts/simian-behavior/data/prime-de/prime-de.csv"
)

pickle_rick = os.path.join(directory, "data.pickle")
with open(pickle_rick, "rb") as handle:
    data = pickle.load(handle)

# Replace your current data loading with:
rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo.csv")
tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo.csv")

# Load demographics and merge birthdate
mkid_rhesus = pd.read_csv(os.path.join(directory, "mkid_rhesus.csv"))
mkid_tonkean = pd.read_csv(os.path.join(directory, "mkid_tonkean.csv"))

# Standardize names for merging
mkid_rhesus['name'] = mkid_rhesus['name'].str.lower().str.strip()
mkid_tonkean['name'] = mkid_tonkean['name'].str.lower().str.strip()

# Merge birthdate into tables
rhesus_table = rhesus_table.merge(
    mkid_rhesus[['name', 'birthdate']],
    on='name',
    how='left'
)

tonkean_table = tonkean_table.merge(
    mkid_tonkean[['name', 'birthdate']],
    on='name',
    how='left'
)

# ===== ADD SPECIES INDICATOR =====
rhesus_table['species'] = 0  # M. mulatta = 0
tonkean_table['species'] = 1  # M. tonkeana = 1
# ==================================


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

# start_date and end_date objects?
rhesus_table["start_date"] = pd.to_datetime(rhesus_table["start_date"])
rhesus_table["end_date"] = pd.to_datetime(rhesus_table["end_date"])
tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])
tonkean_table["end_date"] = pd.to_datetime(tonkean_table["end_date"])

# Compute midpoint of task period
rhesus_table["task_midpoint"] = rhesus_table["start_date"] + (rhesus_table["end_date"] - rhesus_table["start_date"]) / 2
tonkean_table["task_midpoint"] = tonkean_table["start_date"] + (tonkean_table["end_date"] - tonkean_table["start_date"]) / 2

# Convert birthdate to datetime
rhesus_table["birthdate"] = pd.to_datetime(rhesus_table["birthdate"])
tonkean_table["birthdate"] = pd.to_datetime(tonkean_table["birthdate"])

# Calculate age at task midpoint from birthdate
rhesus_table["age_at_task"] = (
    (rhesus_table["task_midpoint"] - rhesus_table["birthdate"]).dt.days / 365.25
)

tonkean_table["age_at_task"] = (
    (tonkean_table["task_midpoint"] - tonkean_table["birthdate"]).dt.days / 365.25
)

# Let's see how the ages differ
summary_cols = ["name", "age", "age_at_task"]
summary_rhesus_table = rhesus_table[summary_cols].copy()
summary_tonkean_table = tonkean_table[summary_cols].copy()

summary_rhesus_table.sort_values(by="name", inplace=True)
summary_tonkean_table.sort_values(by="name", inplace=True)

print(summary_rhesus_table.to_string(index=False))
print(summary_tonkean_table.to_string(index=False))

plt.rcParams['axes.axisbelow'] = False  # Forces axes/grid below all elements
# plotting proportions across the lifespan
def p_lifespan(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
        
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.7,
        edgecolor="steelblue",
        linewidth=1,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.7,
        edgecolor="purple",
        linewidth=1,
    )

#     sns.regplot(
#         ax=ax,
#         data=rhesus_table,
#         x="age_at_task",
#         y=metric,
#         color=colors["rhesus"],
#         order=2,
#         scatter=False,
#         ci=None,
#         line_kws={"linestyle": "--"},
#     )
#     sns.regplot(
#         ax=ax,
#         data=tonkean_table,
#         x="age_at_task",
#         y=metric,
#         color=colors["tonkean"],
#         order=2,
#         scatter=False,
#         ci=None,
#         line_kws={"linestyle": "--"},
#     )
#     sns.regplot(
#         ax=ax,
#         data=pd.concat([rhesus_table, tonkean_table]),
#         x="age_at_task",
#         y=metric,
#         color="black",
#         order=2,
#         scatter=False,
#         ci=None,
#     )
    if metric == "p_success":
        # Only plot quadratic fits for these significant metrics
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=2,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5}
        )
    # For errors and omissions, don't plot any regression lines
    elif metric == "p_error":
        pass

    elif metric == "p_omission" or metric == "p_premature":
        # Plot linear fit for omissions
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=1,  # linear
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5}
        )

    ax.set_title(title, fontsize=20, fontweight="bold", fontname="Times New Roman")
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


sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "Times New Roman"})

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
    fontname="Times New Roman",
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
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.7,
        edgecolor="steelblue",
        linewidth=1,
    )

    # Scatter plot for tonkean monkeys
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.7,
        edgecolor="purple",
        linewidth=1,
    )

    # Regression lines
#     sns.regplot(
#         ax=ax,
#         data=rhesus_table,
#         x="age_at_task",
#         y=metric,
#         color=colors["rhesus"],
#         order=2,
#         scatter=False,
#         ci=None,
#         line_kws={"linestyle": "--"},
#     )
#     sns.regplot(
#         ax=ax,
#         data=tonkean_table,
#         x="age_at_task",
#         y=metric,
#         color=colors["tonkean"],
#         order=2,
#         scatter=False,
#         ci=None,
#         line_kws={"linestyle": "--"},
#     )
#     sns.regplot(
#         ax=ax,
#         data=pd.concat([rhesus_table, tonkean_table]),
#         x="age_at_task",
#         y=metric,
#         color="black",
#         order=2,
#         scatter=False,
#         ci=None,
#     )
    
    combined_data = pd.concat([rhesus_table, tonkean_table])

    if metric == "rt_success":
        pass  # No fits for non-significant results
    
    elif metric == "rt_error":
        sns.regplot(
            ax=ax,
            data=pd.concat([rhesus_table, tonkean_table]),
            x="age_at_task",
            y=metric,
            color="black", 
            order=1,  # linear
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5}
        ) 
        
    # Titles and labels
    ax.set_title(title, fontsize=20, fontweight="bold", fontname="Times New Roman")
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
sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "Times New Roman"})

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
    fontname="Times New Roman",
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
    
    plt.figure(figsize=(9, 8))
    jitter = 0.15
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
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=10,
        )
        sns.stripplot(
            x="species",
            y="age_at_task",
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
        mean_age = subset["age_at_task"].mean()
        sem_age = subset["age_at_task"].std() / np.sqrt(len(subset))  # Standard Error of Mean
        ci_95 = 1.96 * sem_age  # 95% Confidence Interval
        plt.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[ci_95],
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
        fontname="Times New Roman",
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
    plt.xlim(-0.6, 1.6)  # Adjust x-lims to move the plots together
    
    plt.xlabel("")  # Removed x-axis label
    
    filename_base = "demographics"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")
    
    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()
gender_counts = combined_df.groupby(["species", "gender"]).size().unstack()
gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
gender_percentages.rename(columns={1: "Male (%)", 2: "Female (%)"}, inplace=True)
gender_percentages


# Print comprehensive statistics
print("=" * 70)
print("DEMOGRAPHICS SUMMARY")
print("=" * 70)

for species in ["Rhesus", "Tonkean"]:
    subset = combined_df[combined_df["species"] == species]
    males = subset[subset["gender"] != 2]
    females = subset[subset["gender"] == 2]
    
    print(f"\n{species.upper()} (Macaca {'mulatta' if species == 'Rhesus' else 'tonkeana'})")
    print("-" * 70)
    print(f"Total n: {len(subset)}")
    print(f"Males: {len(males)} ({len(males)/len(subset)*100:.1f}%)")
    print(f"Females: {len(females)} ({len(females)/len(subset)*100:.1f}%)")
    
    print(f"\nAge at Task (years):")
    print(f"  Overall: {subset['age_at_task'].mean():.2f} ± {subset['age_at_task'].std():.2f} "
          f"(range: {subset['age_at_task'].min():.2f} - {subset['age_at_task'].max():.2f})")
    print(f"  Males: {males['age_at_task'].mean():.2f} ± {males['age_at_task'].std():.2f} "
          f"(range: {males['age_at_task'].min():.2f} - {males['age_at_task'].max():.2f})")
    print(f"  Females: {females['age_at_task'].mean():.2f} ± {females['age_at_task'].std():.2f} "
          f"(range: {females['age_at_task'].min():.2f} - {females['age_at_task'].max():.2f})")
    
    sem = subset['age_at_task'].std() / np.sqrt(len(subset))
    ci_95 = 1.96 * sem
    print(f"  SEM: {sem:.2f}")
    print(f"  95% CI: [{subset['age_at_task'].mean() - ci_95:.2f}, "
          f"{subset['age_at_task'].mean() + ci_95:.2f}]")

print("\n" + "=" * 70)
print("GENDER DISTRIBUTION")
print("=" * 70)

gender_counts = combined_df.groupby(["species", "gender"]).size().unstack()
gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
gender_percentages.rename(columns={1: "Male (%)", 2: "Female (%)"}, inplace=True)
print(gender_percentages)

print("\n" + "=" * 70)

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
                    alpha=0.7,
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
                n = len(subset)
                se = std / np.sqrt(n)  # Standard error
                ax.errorbar(
                    x=[species],
                    y=[mean],
                    yerr=[se],
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
                    alpha=0.7,
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
                se = std / np.sqrt(n)
                ax.errorbar(
                    x=[gender - 1],
                    y=[mean],
                    yerr=[se],
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
                    females["age_at_task"],
                    females["mean_attempts_per_session"],
                    color=colors[species],
                    alpha=0.7,
                    s=100,
                )
                ax.scatter(
                    males["age_at_task"],
                    males["mean_attempts_per_session"],
                    edgecolor=colors[species],
                    facecolor="none",
                    s=100,
                    linewidth=1.5,
                )

            sns.regplot(
                x="age_at_task",
                y="mean_attempts_per_session",
                data=combined_df,
                order=1,  # Changed from 2 to 1 (linear)
                scatter=False,
                ax=ax,
                color="black",
            )
            ax.set_xticks(range(0, 22, 5))

        elif i == 3:
            # Plot for both species - NO REGRESSION LINES
            for species in ["Rhesus", "Tonkean"]:
                species_data = combined_df[combined_df["species"] == species]
                females = species_data[species_data["gender"] == 2]
                males = species_data[species_data["gender"] == 1]
                
                ax.scatter(
                    females["elo_mean"],
                    females["mean_attempts_per_session"],
                    color=colors[species],
                    alpha=0.7,
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
                
                # Remove all regression lines since no significant relationships
            
            ax.set_xlabel("Social Status (Elo-Rating)", fontsize=14)
        ax.set_title(titles[i], fontsize=18, fontweight="bold", fontname="Times New Roman")
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
        fontname="Times New Roman",
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

 # Function to calculate and print descriptive stats for a group
    def print_group_stats(name, data):
        mean = data.mean()
        std = data.std()
        n = data.count()
        if n > 1:
            ci = stats.t.interval(
                0.95, n - 1, loc=mean, scale=std / np.sqrt(n)
            )
            print(f"{name}: Mean = {mean:.2f}, SD = {std:.2f}, n = {n}, 95% CI = ({ci[0]:.2f}, {ci[1]:.2f})")
        else:
            print(f"{name}: Mean = {mean:.2f}, SD = {std:.2f}, n = {n}, 95% CI = N/A (sample size too small)")

    # Print descriptive stats for gender groups within species
    print("\nWithin species gender descriptive statistics:")

    print_group_stats("Rhesus females", rhesus_females)
    print_group_stats("Rhesus males", rhesus_males)
    print_group_stats("Tonkean females", tonkean_females)
    print_group_stats("Tonkean males", tonkean_males)

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
    print("\nAcross species gender descriptive statistics:")

    all_females = combined_df[combined_df["gender"] == 2]["mean_attempts_per_session"]
    all_males = combined_df[combined_df["gender"] == 1]["mean_attempts_per_session"]

    print_group_stats("All females", all_females)
    print_group_stats("All males", all_males)

    print("\nAcross species gender comparison:")

    t_stat_gender, p_value_gender = stats.ttest_ind(
        all_females, all_males, equal_var=False
    )

    print(f"\nT-statistic: {t_stat_gender:.4f}, P-value: {p_value_gender:.4f}")
    if p_value_gender < 0.05:
        print("There is a significant gender difference across species.")
    else:
        print("There is no significant gender difference across species.")


import matplotlib.gridspec as gridspec

# Set style
sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "Times New Roman"})

# Create figure with custom gridspec (2 rows, 3 cols, right col wider)
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(
    2, 3, 
    width_ratios=[1, 1, 1.2],  # tighter props, wider RTs
    hspace=0.35, 
    wspace=0.15                   # reduce horizontal spacing
)

# Colors
colors = {"tonkean": "purple", "rhesus": "steelblue"}

# ---- Proportions (left 2×2) ----
p_metrics = ["p_success", "p_error", "p_premature", "p_omission"]
p_titles = ["Hits", "Errors", "Premature", "Omissions"]

p_axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    p_axes, p_metrics, p_titles,
    [False, False, True, True],   # X labels only on bottom row
    [True, False, True, False]    # Y labels only on left column
):
    p_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

# ---- Reaction times (stacked right) ----
rt_metrics = ["rt_success", "rt_error"]
rt_titles = ["Hits", "Errors"]

rt_axes = [
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 2]),
]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    rt_axes, rt_metrics, rt_titles,
    [False, True],   # X label only on bottom
    [True, True]     # Both should probably have y labels
):
    rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

# ---- Shared legend ----
handles = [
    plt.Line2D([], [], color=colors["rhesus"], marker="o", linestyle="None", markersize=8,
               label=r"$\it{Macaca\ mulatta}$ (f)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="steelblue",
               linestyle="None", markersize=8, label=r"$\it{Macaca\ mulatta}$ (m)"),
    plt.Line2D([], [], color=colors["tonkean"], marker="o", linestyle="None", markersize=8,
               label=r"$\it{Macaca\ tonkeana}$ (f)"),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="purple",
               linestyle="None", markersize=8, label=r"$\it{Macaca\ tonkeana}$ (m)"),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4)

# ---- Super title ----
fig.suptitle(
    "Reaction Times and Task Performance Across the Lifespan in Macaque Species",
    fontsize=28, fontweight="bold", fontname="Times New Roman", color="purple"
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save
filename_base = "combined-grid-performance"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()


# General performance metrics analysis - Quadratic patterns across lifespan
print("\nQUADRATIC PATTERNS ACROSS LIFESPAN ANALYSIS (WITH SPECIES COVARIATE):")

metrics = {
    "p_success": "Hits",
    "p_error": "Errors",
    "p_premature": "Premature",
    "p_omission": "Omissions",
    "rt_success": "RT Hits",
    "rt_error": "RT Errors",
}

# Combine all data
all_data = pd.concat([rhesus_table, tonkean_table])

# Create results table
regression_results = []

for metric, metric_name in metrics.items():
    # Filter out missing values
    clean_data = all_data.dropna(subset=[metric, "age_at_task", "species"])

    if len(clean_data) < 4:  # Need at least 4 points for quadratic + species
        continue

    # ===== LINEAR MODEL WITH SPECIES =====
    # Performance = β₀ + β₁(age) + β₂(species) + ε
    X_linear = sm.add_constant(clean_data[["age_at_task", "species"]])
    y = clean_data[metric]
    linear_model = sm.OLS(y, X_linear).fit()

    # ===== QUADRATIC MODEL WITH SPECIES =====
    # Performance = β₀ + β₁(age) + β₂(age²) + β₃(species) + ε
    X_quad = sm.add_constant(
        pd.DataFrame({
            "age": clean_data["age_at_task"], 
            "age_sq": clean_data["age_at_task"] ** 2,
            "species": clean_data["species"]
        })
    )
    quad_model = sm.OLS(y, X_quad).fit()

    # Calculate AIC values
    linear_aic = linear_model.aic
    quad_aic = quad_model.aic
    delta_aic = linear_aic - quad_aic  # Positive = quadratic is better

    # Calculate peak/valley age if quadratic term is significant
    peak_age = None
    peak_type = None
    if quad_model.pvalues["age_sq"] < 0.05:
        peak_age = -quad_model.params["age"] / (2 * quad_model.params["age_sq"])
        peak_type = "Peak" if quad_model.params["age_sq"] < 0 else "Valley"

    # Store results
    regression_results.append(
        {
            "Metric": metric_name,
            "N": len(clean_data),
            "Mean": clean_data[metric].mean(),
            "SD": clean_data[metric].std(),
            "Range": f"{clean_data[metric].min():.3f}-{clean_data[metric].max():.3f}",
            "Linear_R2": linear_model.rsquared,
            "Linear_p": linear_model.f_pvalue,
            "Linear_AIC": linear_aic,
            "Linear_age_coef": linear_model.params["age_at_task"],
            "Linear_age_p": linear_model.pvalues["age_at_task"],
            "Linear_species_coef": linear_model.params["species"],
            "Linear_species_p": linear_model.pvalues["species"],
            "Quad_R2": quad_model.rsquared,
            "Quad_p": quad_model.f_pvalue,
            "Quad_AIC": quad_aic,
            "Quad_age_coef": quad_model.params["age"],
            "Quad_age2_coef": quad_model.params["age_sq"],
            "Quad_species_coef": quad_model.params["species"],
            "Quad_age_p": quad_model.pvalues["age"],
            "Quad_age2_p": quad_model.pvalues["age_sq"],
            "Quad_species_p": quad_model.pvalues["species"],
            "Delta_AIC": delta_aic,
            "AIC_better": "Quadratic" if delta_aic > 2 else "Linear" if delta_aic < -2 else "Similar",
            "R2_improvement": quad_model.rsquared - linear_model.rsquared,
            "Peak_age": peak_age,
            "Peak_type": peak_type,
        }
    )

# Create comprehensive output table
print("\n" + "=" * 180)
print("LIFESPAN DEVELOPMENTAL PATTERNS - LINEAR vs QUADRATIC MODELS (CONTROLLING FOR SPECIES)")
print("=" * 180)

results_df = pd.DataFrame(regression_results)

# Main table with key results
print(
    f"{'Metric':<12} {'N':<4} {'Mean±SD':<15} {'Range':<20} {'Lin R²':<7} {'Quad R²':<8} {'ΔR²':<6} {'ΔAIC':<8} {'Better':<10} {'Peak/Valley':<12}"
)
print("-" * 180)

for _, row in results_df.iterrows():
    mean_sd = f"{row['Mean']:.3f}±{row['SD']:.3f}"
    delta_r2 = row["R2_improvement"]

    if pd.notna(row["Peak_age"]) and row["Peak_type"]:
        peak_str = f"{row['Peak_type']} {row['Peak_age']:.1f}y"
    else:
        peak_str = "None"

    print(
        f"{row['Metric']:<12} {int(row['N']):<4} {mean_sd:<15} {row['Range']:<20} {row['Linear_R2']:<7.3f} "
        f"{row['Quad_R2']:<8.3f} {delta_r2:<6.3f} {row['Delta_AIC']:<8.1f} {row['AIC_better']:<10} {peak_str:<12}"
    )

# Detailed statistical table with species coefficients
print("\n" + "=" * 160)
print("DETAILED STATISTICAL RESULTS (INCLUDING SPECIES EFFECTS)")
print("=" * 160)
print(
    f"{'Metric':<12} {'Lin β_age':<10} {'β_age p':<8} {'Lin β_sp':<10} {'β_sp p':<8} "
    f"{'Quad β₁':<10} {'Quad β₂':<10} {'Quad β_sp':<10} {'β₁ p':<8} {'β₂ p':<8} {'β_sp p':<8}"
)
print("-" * 160)

for _, row in results_df.iterrows():
    print(
        f"{row['Metric']:<12} {row['Linear_age_coef']:<10.3f} {row['Linear_age_p']:<8.3f} "
        f"{row['Linear_species_coef']:<10.3f} {row['Linear_species_p']:<8.3f} "
        f"{row['Quad_age_coef']:<10.3f} {row['Quad_age2_coef']:<10.3f} {row['Quad_species_coef']:<10.3f} "
        f"{row['Quad_age_p']:<8.3f} {row['Quad_age2_p']:<8.3f} {row['Quad_species_p']:<8.3f}"
    )

# Summary of quadratic patterns
print("\n" + "=" * 80)
print("SUMMARY OF DEVELOPMENTAL PATTERNS")
print("=" * 80)

quadratic_metrics = results_df[results_df["AIC_better"] == "Quadratic"]
if not quadratic_metrics.empty:
    print("Metrics best explained by quadratic patterns (controlling for species):")
    for _, row in quadratic_metrics.iterrows():
        pattern = (
            "inverted-U (peak)" if row["Peak_type"] == "Peak" else "U-shaped (valley)"
        )
        print(
            f"• {row['Metric']}: {pattern} at {row['Peak_age']:.1f} years (ΔR² = +{row['R2_improvement']:.3f}, ΔAIC = {row['Delta_AIC']:.1f})"
        )
else:
    print("No metrics showed quadratic patterns as preferred by AIC.")

linear_only = results_df[results_df["AIC_better"] == "Linear"]
if not linear_only.empty:
    print(f"\nMetrics best explained by linear age effects (controlling for species):")
    for _, row in linear_only.iterrows():
        direction = "increases" if row["Linear_age_coef"] > 0 else "decreases"
        if row["Linear_age_p"] < 0.05:
            print(
                f"• {row['Metric']}: {direction} linearly with age (R² = {row['Linear_R2']:.3f}, ΔAIC = {row['Delta_AIC']:.1f})"
            )
        else:
            print(
                f"• {row['Metric']}: no significant age effect (R² = {row['Linear_R2']:.3f}, ΔAIC = {row['Delta_AIC']:.1f})"
            )

similar_metrics = results_df[results_df["AIC_better"] == "Similar"]
if not similar_metrics.empty:
    print(f"\nMetrics with similar model fits:")
    for _, row in similar_metrics.iterrows():
        print(
            f"• {row['Metric']}: linear and quadratic models perform similarly (ΔAIC = {row['Delta_AIC']:.1f})"
        )

# Add summary of species effects
print("\n" + "=" * 80)
print("SPECIES EFFECTS SUMMARY")
print("=" * 80)
print("Positive β_species = M. tonkeana performs higher than M. mulatta")
print("Negative β_species = M. tonkeana performs lower than M. mulatta\n")

for _, row in results_df.iterrows():
    if row["Quad_species_p"] < 0.05:
        direction = "higher" if row["Quad_species_coef"] > 0 else "lower"
        print(
            f"• {row['Metric']}: M. tonkeana {direction} by {abs(row['Quad_species_coef']):.3f} "
            f"(p = {row['Quad_species_p']:.3f})"
        )

print("\nLegend:")
print("ΔR² = R² improvement from linear to quadratic model")
print("ΔAIC = AIC difference (Linear AIC - Quadratic AIC); positive values favor quadratic")
print("  ΔAIC > 2: substantial evidence for quadratic model")
print("  ΔAIC < -2: substantial evidence for linear model")
print("  |ΔAIC| ≤ 2: models perform similarly")
print("β₁ = linear age coefficient, β₂ = quadratic age coefficient, β_sp = species coefficient")
print("Peak = maximum performance age, Valley = minimum performance age")
print("Better = preferred model based on AIC comparison")
print("Species coded as: 0 = M. mulatta, 1 = M. tonkeana")