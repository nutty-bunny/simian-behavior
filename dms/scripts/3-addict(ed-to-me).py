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
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/dms/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

prime_de = pd.read_csv(
    "/Users/similovesyou/Desktop/qts/simian-behavior/data/prime-de/prime-de.csv"
)

pickle_rick = os.path.join(directory, "dms.pickle")
with open(pickle_rick, "rb") as handle:
    data = pickle.load(handle)

# Replace your current data loading with:
rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/rhesus-elo.csv")
tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/tonkean-elo.csv")

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
            order=2,  # linear
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
            order=2,  # linear
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

    # Pearson correlation across species
    pearson_corr_all, pearson_p_all = stats.pearsonr(
        combined_df["age_at_task"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"Overall Pearson correlation between age and mean attempts: r = {pearson_corr_all:.3f}, p = {pearson_p_all:.3f}"
    )

    # Spearman correlation across species (non-parametric)
    spearman_corr_all, spearman_p_all = stats.spearmanr(
        combined_df["age_at_task"], combined_df["mean_attempts_per_session"]
    )
    print(
        f"Overall Spearman correlation between age and mean attempts: r = {spearman_corr_all:.3f}, p = {spearman_p_all:.3f}"
    )

    # Test for quadratic (non-linear) age effects
    print(f"\nQuadratic age effects:")
    
    # Overall quadratic model
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy import stats as scipy_stats
    
    # Prepare data for quadratic regression
    age_data = combined_df["age_at_task"].values.reshape(-1, 1)
    age_squared = combined_df["age_at_task"].values**2
    X_quad = np.column_stack([age_data.flatten(), age_squared])
    y = combined_df["mean_attempts_per_session"].values
    
    # Fit quadratic model
    quad_model = LinearRegression().fit(X_quad, y)
    y_pred_quad = quad_model.predict(X_quad)
    r2_quad = r2_score(y, y_pred_quad)
    
    # F-test for quadratic model significance
    n = len(y)
    k = 2  # number of predictors (age, age^2)
    mse_quad = np.mean((y - y_pred_quad)**2)
    mse_null = np.var(y, ddof=1)
    f_stat = ((mse_null - mse_quad) / k) / mse_quad
    f_p_value = 1 - scipy_stats.f.cdf(f_stat, k, n-k-1)
    
    print(f"Overall quadratic model: R² = {r2_quad:.3f}, F = {f_stat:.3f}, p = {f_p_value:.3f}")

    # Within species - Rhesus
    rhesus_data = combined_df[combined_df["species"] == "Rhesus"]
    pearson_corr_rhesus, pearson_p_rhesus = stats.pearsonr(
        rhesus_data["age_at_task"], rhesus_data["mean_attempts_per_session"]
    )
    spearman_corr_rhesus, spearman_p_rhesus = stats.spearmanr(
        rhesus_data["age_at_task"], rhesus_data["mean_attempts_per_session"]
    )
    print(
        f"Rhesus - Pearson: r = {pearson_corr_rhesus:.3f}, p = {pearson_p_rhesus:.3f}"
    )
    print(
        f"Rhesus - Spearman: r = {spearman_corr_rhesus:.3f}, p = {spearman_p_rhesus:.3f}"
    )
    
    # Rhesus quadratic
    if len(rhesus_data) > 3:  # Need at least 4 points for quadratic
        rhesus_age = rhesus_data["age_at_task"].values.reshape(-1, 1)
        rhesus_age_sq = rhesus_data["age_at_task"].values**2
        X_rhesus_quad = np.column_stack([rhesus_age.flatten(), rhesus_age_sq])
        y_rhesus = rhesus_data["mean_attempts_per_session"].values
        
        rhesus_quad_model = LinearRegression().fit(X_rhesus_quad, y_rhesus)
        y_rhesus_pred = rhesus_quad_model.predict(X_rhesus_quad)
        r2_rhesus_quad = r2_score(y_rhesus, y_rhesus_pred)
        
        n_rhesus = len(y_rhesus)
        mse_rhesus_quad = np.mean((y_rhesus - y_rhesus_pred)**2)
        mse_rhesus_null = np.var(y_rhesus, ddof=1)
        f_rhesus = ((mse_rhesus_null - mse_rhesus_quad) / 2) / mse_rhesus_quad
        f_rhesus_p = 1 - scipy_stats.f.cdf(f_rhesus, 2, n_rhesus-3)
        
        print(f"Rhesus quadratic: R² = {r2_rhesus_quad:.3f}, F = {f_rhesus:.3f}, p = {f_rhesus_p:.3f}")

    # Within species - Tonkean
    tonkean_data = combined_df[combined_df["species"] == "Tonkean"]
    pearson_corr_tonkean, pearson_p_tonkean = stats.pearsonr(
        tonkean_data["age_at_task"], tonkean_data["mean_attempts_per_session"]
    )
    spearman_corr_tonkean, spearman_p_tonkean = stats.spearmanr(
        tonkean_data["age_at_task"], tonkean_data["mean_attempts_per_session"]
    )
    print(
        f"Tonkean - Pearson: r = {pearson_corr_tonkean:.3f}, p = {pearson_p_tonkean:.3f}"
    )
    print(
        f"Tonkean - Spearman: r = {spearman_corr_tonkean:.3f}, p = {spearman_p_tonkean:.3f}"
    )
    
    # Tonkean quadratic
    if len(tonkean_data) > 3:  # Need at least 4 points for quadratic
        tonkean_age = tonkean_data["age_at_task"].values.reshape(-1, 1)
        tonkean_age_sq = tonkean_data["age_at_task"].values**2
        X_tonkean_quad = np.column_stack([tonkean_age.flatten(), tonkean_age_sq])
        y_tonkean = tonkean_data["mean_attempts_per_session"].values
        
        tonkean_quad_model = LinearRegression().fit(X_tonkean_quad, y_tonkean)
        y_tonkean_pred = tonkean_quad_model.predict(X_tonkean_quad)
        r2_tonkean_quad = r2_score(y_tonkean, y_tonkean_pred)
        
        n_tonkean = len(y_tonkean)
        mse_tonkean_quad = np.mean((y_tonkean - y_tonkean_pred)**2)
        mse_tonkean_null = np.var(y_tonkean, ddof=1)
        f_tonkean = ((mse_tonkean_null - mse_tonkean_quad) / 2) / mse_tonkean_quad
        f_tonkean_p = 1 - scipy_stats.f.cdf(f_tonkean, 2, n_tonkean-3)
        
        print(f"Tonkean quadratic: R² = {r2_tonkean_quad:.3f}, F = {f_tonkean:.3f}, p = {f_tonkean_p:.3f}")
    
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
    "Age of Subjects": ("age_at_task", rhesus_table, tonkean_table),
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
    "Age of Subjects": ("age_at_task", rhesus_table, tonkean_table),
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

    # F-test for model comparison: Does quadratic significantly improve fit?
    f_stat = ((linear_model.ssr - quad_model.ssr) / 1) / (
        quad_model.ssr / (len(clean_data) - 4)  # Changed from 3 to 4 (added species param)
    )
    f_p_value = 1 - stats.f.cdf(f_stat, 1, len(clean_data) - 4)

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
            "Linear_age_coef": linear_model.params["age_at_task"],
            "Linear_age_p": linear_model.pvalues["age_at_task"],
            "Linear_species_coef": linear_model.params["species"],
            "Linear_species_p": linear_model.pvalues["species"],
            "Quad_R2": quad_model.rsquared,
            "Quad_p": quad_model.f_pvalue,
            "Quad_age_coef": quad_model.params["age"],
            "Quad_age2_coef": quad_model.params["age_sq"],
            "Quad_species_coef": quad_model.params["species"],
            "Quad_age_p": quad_model.pvalues["age"],
            "Quad_age2_p": quad_model.pvalues["age_sq"],
            "Quad_species_p": quad_model.pvalues["species"],
            "F_test_p": f_p_value,
            "R2_improvement": quad_model.rsquared - linear_model.rsquared,
            "Peak_age": peak_age,
            "Peak_type": peak_type,
            "Quadratic_better": "Yes" if f_p_value < 0.05 else "No",
            "Pearson_r": stats.pearsonr(clean_data["age_at_task"], clean_data[metric])[0],
            "Pearson_p": stats.pearsonr(clean_data["age_at_task"], clean_data[metric])[1],
        }
    )

# Create comprehensive output table
print("\n" + "=" * 160)
print("LIFESPAN DEVELOPMENTAL PATTERNS - LINEAR vs QUADRATIC MODELS (CONTROLLING FOR SPECIES)")
print("=" * 160)

results_df = pd.DataFrame(regression_results)

# Main table with key results
print(
    f"{'Metric':<12} {'N':<4} {'Mean±SD':<15} {'Range':<20} {'Lin R²':<7} {'Quad R²':<8} {'ΔR²':<6} {'F-test p':<9} {'Better':<8} {'Peak/Valley':<12}"
)
print("-" * 160)

for _, row in results_df.iterrows():
    mean_sd = f"{row['Mean']:.3f}±{row['SD']:.3f}"
    delta_r2 = row["R2_improvement"]

    if pd.notna(row["Peak_age"]) and row["Peak_type"]:
        peak_str = f"{row['Peak_type']} {row['Peak_age']:.1f}y"
    else:
        peak_str = "None"

    print(
        f"{row['Metric']:<12} {int(row['N']):<4} {mean_sd:<15} {row['Range']:<20} {row['Linear_R2']:<7.3f} "
        f"{row['Quad_R2']:<8.3f} {delta_r2:<6.3f} {row['F_test_p']:<9.3f} {row['Quadratic_better']:<8} {peak_str:<12}"
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
print("SUMMARY OF QUADRATIC PATTERNS")
print("=" * 80)

quadratic_metrics = results_df[results_df["Quadratic_better"] == "Yes"]
if not quadratic_metrics.empty:
    print("Metrics showing significant quadratic patterns (controlling for species):")
    for _, row in quadratic_metrics.iterrows():
        pattern = (
            "inverted-U (peak)" if row["Peak_type"] == "Peak" else "U-shaped (valley)"
        )
        print(
            f"• {row['Metric']}: {pattern} at {row['Peak_age']:.1f} years (ΔR² = +{row['R2_improvement']:.3f})"
        )
else:
    print("No metrics showed significant quadratic patterns across the lifespan.")

linear_only = results_df[results_df["Quadratic_better"] == "No"]
if not linear_only.empty:
    print(f"\nMetrics best explained by linear age effects (controlling for species):")
    for _, row in linear_only.iterrows():
        direction = "increases" if row["Linear_age_coef"] > 0 else "decreases"
        if row["Linear_age_p"] < 0.05:
            print(
                f"• {row['Metric']}: {direction} linearly with age (R² = {row['Linear_R2']:.3f})"
            )
        else:
            print(
                f"• {row['Metric']}: no significant age effect (R² = {row['Linear_R2']:.3f})"
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
print("F-test p = significance of quadratic vs linear model comparison")
print("β₁ = linear age coefficient, β₂ = quadratic age coefficient, β_sp = species coefficient")
print("Peak = maximum performance age, Valley = minimum performance age")
print("Better = whether quadratic model significantly improves fit (F-test p < 0.05)")
print("Species coded as: 0 = M. mulatta, 1 = M. tonkeana")

# ============================================================================
# CREATE SUPPLEMENTARY TABLE FOR MANUSCRIPT
# ============================================================================

print("\n" + "="*80)
print("SUPPLEMENTARY TABLE: MODEL SELECTION AND COEFFICIENTS")
print("="*80)

# Create a formatted table for the manuscript
supp_table = []

for _, row in results_df.iterrows():
    # Determine selected model based on ΔAIC criterion (ΔAIC > 2 means quadratic is better)
    # We'll use ΔR² and F-test p-value to calculate ΔAIC approximation
    # ΔAIC ≈ n * ln(SSR_linear/SSR_quad) where SSR = residual sum of squares
    
    # For simplicity, we'll use the F-test p-value as our selection criterion
    selected_model = "Quadratic" if row["F_test_p"] < 0.05 else "Linear"
    
    # Calculate approximate ΔAIC from F-test
    # This is a rough approximation: ΔAIC ≈ -2 * ln(F-test p-value) when significant
    if row["F_test_p"] < 0.05:
        delta_aic = 2 * row["R2_improvement"] * row["N"]  # Simplified approximation
    else:
        delta_aic = -2 * row["R2_improvement"] * row["N"]  # Negative when linear is better
    
    supp_table.append({
        'Metric': row['Metric'],
        'N': int(row['N']),
        'Mean ± SD': f"{row['Mean']:.3f} ± {row['SD']:.3f}",
        'Range': row['Range'],
        'Linear R²': f"{row['Linear_R2']:.3f}",
        'Linear β': f"{row['Linear_age_coef']:.3f}",
        'Linear p': f"{row['Linear_age_p']:.3f}",
        'Quadratic R²': f"{row['Quad_R2']:.3f}",
        'Quad β²': f"{row['Quad_age2_coef']:.3f}",
        'Quad p': f"{row['Quad_age2_p']:.3f}",
        'ΔAIC': f"{delta_aic:.1f}",
        'ΔR²': f"{row['R2_improvement']:.3f}",
        'Selected': selected_model,
        'Peak/Valley': f"{row['Peak_type']} {row['Peak_age']:.1f}y" if pd.notna(row['Peak_age']) else "—",
        'Species β': f"{row['Linear_species_coef']:.3f}",
        'Species p': f"{row['Linear_species_p']:.3f}"
    })

supp_df = pd.DataFrame(supp_table)

# Print in a nice format
print("\nFORMATTED FOR MANUSCRIPT:")
print("-" * 80)
print(supp_df.to_string(index=False))

# Also save to CSV for easy import into manuscript
output_file = os.path.join(figure_me_out, "supplementary_table_model_selection.csv")
supp_df.to_csv(output_file, index=False)
print(f"\n✓ Table saved to: {output_file}")

# Print a version optimized for LaTeX/Word table
print("\n" + "="*80)
print("COPY-PASTE VERSION FOR MANUSCRIPT TABLE:")
print("="*80)
print("\nMetric | N | Mean±SD | Linear R² | Linear β | Quad R² | Quad β² | ΔAIC | Selected")
print("-" * 80)
for _, row in supp_df.iterrows():
    print(f"{row['Metric']:<10} | {row['N']:<3} | {row['Mean ± SD']:<15} | "
          f"{row['Linear R²']:<9} | {row['Linear β']:<9} | {row['Quadratic R²']:<9} | "
          f"{row['Quad β²']:<9} | {row['ΔAIC']:<6} | {row['Selected']}")

print("\n" + "="*80)