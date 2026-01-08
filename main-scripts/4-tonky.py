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
import matplotlib.gridspec as gridspec

plt.rcParams["svg.fonttype"] = "none"

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/13-individual"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

# Load both datasets
tonkean_table_full = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/13-individual/tonkean-elo-13.csv")
tonkean_table_min = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/13-individual/tonkean-elo-min-attempts-13.csv")

# Load demographics and merge birthdate
mkid_tonkean = pd.read_csv(os.path.join(directory, "mkid_tonkean.csv"))

# Standardize names for merging
mkid_tonkean['name'] = mkid_tonkean['name'].str.lower().str.strip()

# Merge birthdate into both tables
tonkean_table_full = tonkean_table_full.merge(
    mkid_tonkean[['name', 'birthdate']],
    on='name',
    how='left'
)

tonkean_table_min = tonkean_table_min.merge(
    mkid_tonkean[['name', 'birthdate']],
    on='name',
    how='left'
)

# tables now summarize device engagement, task performance & demographics
print()
print("tonkean task engagement - full dataset")
print(tonkean_table_full)
print()
print("tonkean task engagement - restricted dataset")
print(tonkean_table_min)
print()

# Process both datasets
for table_name, tonkean_table in [("full", tonkean_table_full), ("restricted", tonkean_table_min)]:
    # start_date and end_date objects?
    tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])
    tonkean_table["end_date"] = pd.to_datetime(tonkean_table["end_date"])
    
    # Compute midpoint of task period
    tonkean_table["task_midpoint"] = tonkean_table["start_date"] + (tonkean_table["end_date"] - tonkean_table["start_date"]) / 2
    
    # Convert birthdate to datetime
    tonkean_table["birthdate"] = pd.to_datetime(tonkean_table["birthdate"])
    
    # Calculate age at task midpoint from birthdate
    tonkean_table["age_at_task"] = (
        (tonkean_table["task_midpoint"] - tonkean_table["birthdate"]).dt.days / 365.25
    )

# Let's see how the ages differ
summary_cols = ["name", "age", "age_at_task"]
summary_full = tonkean_table_full[summary_cols].copy().sort_values(by="name")
summary_min = tonkean_table_min[summary_cols].copy().sort_values(by="name")

print(summary_full.to_string(index=False))
print(summary_min.to_string(index=False))

print("Full dataset age summary:")
print(summary_full.to_string(index=False))
print("\nRestricted dataset age summary:")
print(summary_min.to_string(index=False))

plt.rcParams['axes.axisbelow'] = False  # Forces axes/grid below all elements

# Updated color scheme - using forest green and orange
colors = {
    "full": "forestgreen",
    "full_edge": "forestgreen", 
    "full_line": "forestgreen",
    "restricted": "darkorange",
    "restricted_edge": "darkorange",
    "restricted_line": "darkorange"
}

# plotting proportions across the lifespan - COMBINED VERSION
def p_lifespan_combined(
    ax, metric, title, tonkean_full, tonkean_min, colors, show_xlabel, show_ylabel
):
    # Full dataset - normal alpha
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["full_edge"],
        s=80,
        linewidth=1,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["full"],
        s=80,
        alpha=0.7,
        edgecolor=colors["full_edge"],
        linewidth=1,
    )
    
    # Restricted dataset - same alpha values
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["restricted_edge"],
        s=80,
        linewidth=1,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["restricted"],
        s=80,
        alpha=0.7,
        edgecolor=colors["restricted_edge"],
        linewidth=1,
    )

    # Regression lines based on statistical significance
    if metric == "p_success":
        # Both datasets show significant linear decrease with age
        sns.regplot(
            ax=ax,
            data=tonkean_full,
            x="age_at_task",
            y=metric,
            color=colors["full_line"], 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5, "alpha": 1.0}
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
            line_kws={"linewidth": 1.5, "alpha": 0.6}
        )
    elif metric == "p_error":
        # No significant relationships for either dataset
        pass
    elif metric == "p_premature":
        # Both datasets show significant linear increase with age
        sns.regplot(
            ax=ax,
            data=tonkean_full,
            x="age_at_task",
            y=metric,
            color=colors["full_line"], 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5, "alpha": 1.0}
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
            line_kws={"linewidth": 1.5, "alpha": 0.6}
        )
    elif metric == "p_omission":
        # Both datasets show significant linear decrease with age
        sns.regplot(
            ax=ax,
            data=tonkean_full,
            x="age_at_task",
            y=metric,
            color=colors["full_line"], 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5, "alpha": 1.0}
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
            line_kws={"linewidth": 1.5, "alpha": 0.6}
        )

    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7)

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

# Combined reaction times function
def rt_lifespan_combined(
    ax, metric, title, tonkean_full, tonkean_min, colors, show_xlabel, show_ylabel
):
    # Full dataset scatter plots
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["full_edge"],
        s=80,
        linewidth=1,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_full[tonkean_full["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["full"],
        s=80,
        alpha=0.7,
        edgecolor=colors["full_edge"],
        linewidth=1,
    )
    
    # Restricted dataset scatter plots - same alpha values
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 1],
        x="age_at_task",
        y=metric,
        facecolors="none",
        edgecolor=colors["restricted_edge"],
        s=80,
        linewidth=1,
        alpha=1.0
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_min[tonkean_min["gender"] == 2],
        x="age_at_task",
        y=metric,
        color=colors["restricted"],
        s=80,
        alpha=0.7,
        edgecolor=colors["restricted_edge"],
        linewidth=1,
    )

    # Regression lines based on statistical significance
    if metric == "rt_success":
        # No significant relationships for either dataset
        pass
    elif metric == "rt_error":
        # Both datasets show significant linear decrease with age
        sns.regplot(
            ax=ax,
            data=tonkean_full,
            x="age_at_task",
            y=metric,
            color=colors["full_line"], 
            order=1,
            scatter=False,
            ci=None,
            line_kws={"linewidth": 1.5, "alpha": 1.0}
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
            line_kws={"linewidth": 1.5, "alpha": 0.6}
        )

    ax.set_title(title, fontsize=20, fontweight="bold")
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

# Combined proportions plot
sns.set(style="whitegrid", font_scale=1.2)

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes, metrics, titles, [True] * 4, [True] + [False] * 3
):
    p_lifespan_combined(
        ax, metric, title, tonkean_table_full, tonkean_table_min, colors, show_xlabel, show_ylabel
    )

# Updated legend for combined plot with new colors
handles = [
    plt.Line2D([], [], color=colors["full"], marker="o", linestyle="None", markersize=8,
               label=r"Full Dataset (f)", alpha=0.7),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors["full_edge"],
               linestyle="None", markersize=8, label=r"Full Dataset (m)"),
    plt.Line2D([], [], color=colors["restricted"], marker="o", linestyle="None", markersize=8,
               label=r"Restricted Dataset (f)", alpha=0.7),
    plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors["restricted_edge"],
               linestyle="None", markersize=8, label=r"Restricted Dataset (m)"),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4)

fig.suptitle(
    "Task Performance Across the Lifespan in Macaca tonkeana (Full vs Restricted)",
    fontsize=22, fontweight="bold", color="black"
)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])

filename_base = "outcome-proportions-tonkean-combined"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

# Combined reaction times plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
metrics = ["rt_success", "rt_error"]
titles = ["Hits", "Error"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes.flatten(), metrics, titles, [True, True], [True, False]
):
    rt_lifespan_combined(
        ax, metric, title, tonkean_table_full, tonkean_table_min, colors, show_xlabel, show_ylabel
    )

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

fig.suptitle(
    "Reaction Time Profiles Across the Lifespan in Macaca tonkeana (Full vs Restricted)",
    fontsize=22, fontweight="bold", color="black"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

filename_base = "reaction-times-tonkean-combined"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

# Updated demographics plot - RESTRICTED ON LEFT, FULL ON RIGHT
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    
    # Create combined dataframe for demographics - ORDER CHANGED
    combined_demo_df = pd.concat([
        tonkean_table_min.assign(species="Restricted", dataset_type="Restricted"),
        tonkean_table_full.assign(species="Full", dataset_type="Full")
    ])
    
    plt.figure(figsize=(12, 8))
    jitter = 0.15
    
    # Updated order: Restricted first (left), then Full (right)
    for species in ["Restricted", "Full"]:
        males = combined_demo_df[
            (combined_demo_df["species"] == species) & (combined_demo_df["gender"] != 2)
        ]  # males hollow
        females = combined_demo_df[
            (combined_demo_df["species"] == species) & (combined_demo_df["gender"] == 2)
        ]  # females filled
        
        alpha_val = 0.7  # Same alpha for both datasets
        
        sns.stripplot(
            x="species",
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species.lower()],
            alpha=alpha_val,
            size=10,
        )
        sns.stripplot(
            x="species",
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species.lower()],
            size=10,
            linewidth=1.5,
            alpha=1.0  # Same alpha for both datasets
        )
    
    # Error bars for each dataset
    for species in ["Restricted", "Full"]:
        subset = combined_demo_df[combined_demo_df["species"] == species]
        mean_age = subset["age_at_task"].mean()
        sem_age = subset["age_at_task"].std() / np.sqrt(len(subset))
        ci_95 = 1.96 * sem_age
        alpha_val = 1.0  # Same alpha for both datasets
        plt.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[ci_95],
            fmt="none",
            ecolor=colors[species.lower()],
            capsize=5,
            elinewidth=2,
            alpha=alpha_val
        )
    
    plt.ylabel("Age", fontsize=14)
    plt.title(
        "Demographics",
        fontsize=25,
        fontweight="bold",
        color="black",
    )
    
    # Updated x-axis labels with new order
    plt.xticks(
        ticks=[0, 1],
        labels=[
            f'$\\it{{Restricted\\ Dataset}}$\n(n={len(tonkean_table_min)})',
            f'$\\it{{Full\\ Dataset}}$\n(n={len(tonkean_table_full)})',
        ],
        fontsize=14,
    )
    
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}")
    )
    plt.ylim(0, 25)
    plt.xlim(-0.6, 1.6)
    plt.xlabel("")
    
    filename_base = "demographics-tonkean-comparison"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")
    
    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()

# Gender statistics for both datasets
print("\nGender distribution comparison:")
gender_counts_full = tonkean_table_full.groupby("gender").size()
gender_percentages_full = (gender_counts_full / len(tonkean_table_full)) * 100
print("Full dataset:")
print(f"  Males (gender=1): {gender_counts_full.get(1, 0)} ({gender_percentages_full.get(1, 0):.1f}%)")
print(f"  Females (gender=2): {gender_counts_full.get(2, 0)} ({gender_percentages_full.get(2, 0):.1f}%)")

gender_counts_min = tonkean_table_min.groupby("gender").size()
gender_percentages_min = (gender_counts_min / len(tonkean_table_min)) * 100
print("Restricted dataset:")
print(f"  Males (gender=1): {gender_counts_min.get(1, 0)} ({gender_percentages_min.get(1, 0):.1f}%)")
print(f"  Females (gender=2): {gender_counts_min.get(2, 0)} ({gender_percentages_min.get(2, 0):.1f}%)")

def tossNaN(df, columns):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=columns)

columns = ["mean_attempts_per_session", "gender"]
tonkean_table_full = tossNaN(tonkean_table_full, columns)
tonkean_table_min = tossNaN(tonkean_table_min, columns)

# Combined grid plot
sns.set(style="whitegrid", font_scale=1.2)
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.15)

p_metrics = ["p_success", "p_error", "p_premature", "p_omission"]
p_titles = ["Hits", "Errors", "Premature", "Omissions"]

p_axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    p_axes, p_metrics, p_titles,
    [False, False, True, True], [True, False, True, False]
):
    p_lifespan_combined(ax, metric, title, tonkean_table_full, tonkean_table_min, colors, show_xlabel, show_ylabel)

rt_metrics = ["rt_success", "rt_error"]
rt_titles = ["Hits", "Errors"]
rt_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    rt_axes, rt_metrics, rt_titles, [False, True], [True, True]
):
    rt_lifespan_combined(ax, metric, title, tonkean_table_full, tonkean_table_min, colors, show_xlabel, show_ylabel)

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4)

fig.suptitle(
    "Reaction Times and Task Performance Across the Lifespan in Macaca tonkeana (Full vs Restricted)",
    fontsize=24, fontweight="bold", color="black"
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

filename_base = "combined-grid-performance-tonkean-comparison"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All plots saved to:")
print(figure_me_out)
def individual_changes_plot(tonkean_full, tonkean_min, colors, metrics, titles):
    """
    Create slope plots showing individual performance changes between full and restricted datasets
    """
    # Merge datasets on individual name to get paired data
    merged_data = pd.merge(
        tonkean_full[['name'] + metrics + ['gender']], 
        tonkean_min[['name'] + metrics + ['gender']], 
        on=['name', 'gender'], 
        suffixes=('_full', '_restricted')
    )
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # X positions (spread farther apart)
    x_restricted, x_full = 0.25, 0.75
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        full_col = f"{metric}_full"
        restricted_col = f"{metric}_restricted"
        
        # Plot lines connecting restricted to full for each individual
        for _, row in merged_data.iterrows():
            ax.plot([x_restricted, x_full], [row[restricted_col], row[full_col]], 
                   color="gray", alpha=0.4, linewidth=0.8)
            
            # Restricted dataset point
            if row['gender'] == 1:  # Male = hollow
                ax.scatter(x_restricted, row[restricted_col], facecolors="none", 
                           edgecolor=colors['restricted'], s=60, linewidth=1.5, zorder=3)
            else:  # Female = filled
                ax.scatter(x_restricted, row[restricted_col], color=colors['restricted'], 
                           s=60, alpha=0.7, zorder=3)
            
            # Full dataset point
            if row['gender'] == 1:  # Male = hollow
                ax.scatter(x_full, row[full_col], facecolors="none", 
                           edgecolor=colors['full'], s=60, linewidth=1.5, zorder=3)
            else:  # Female = filled
                ax.scatter(x_full, row[full_col], color=colors['full'], 
                           s=60, alpha=0.7, zorder=3)
        
        # Mean line
        mean_restricted = merged_data[restricted_col].mean()
        mean_full = merged_data[full_col].mean()
        
        ax.plot([x_restricted, x_full], [mean_restricted, mean_full], 
                color="darkgrey", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter([x_restricted, x_full], [mean_restricted, mean_full], 
                   color="darkgrey", s=140, zorder=5, marker="*")  # slightly bigger stars
        
        # Formatting
        ax.set_xlim(0, 1)  
        ax.set_xticks([x_restricted, x_full])
        ax.set_xticklabels(['Restricted Dataset', 'Full Dataset'], fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel('Proportion', fontsize=14)
            # Keep only 0 and 1 on y-axis
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"])
        else:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        # Horizontal grid lines every 0.2
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    handles = [
        plt.Line2D([], [], color=colors["restricted"], marker="o", linestyle="None", markersize=8,
                   label="Restricted Dataset (f)", alpha=0.7),
        plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors["restricted"],
                   linestyle="None", markersize=8, label="Restricted Dataset (m)"),
        plt.Line2D([], [], color=colors["full"], marker="o", linestyle="None", markersize=8,
                   label="Full Dataset (f)", alpha=0.7),
        plt.Line2D([], [], color="none", marker="o", markeredgecolor=colors["full"],
                   linestyle="None", markersize=8, label="Full Dataset (m)"),
        plt.Line2D([], [], color="darkgrey", marker="*", linestyle="-", markersize=12,
                   label="Mean", linewidth=2, alpha=0.9)
    ]
    
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5)
    
    plt.suptitle('Individual Performance Changes: Restricted vs Full Dataset', 
                fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    
    return fig



# Create both plots
colors = {
    "full": "forestgreen",
    "restricted": "darkorange"
}

metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]

# Slope plot (updated)
fig1 = individual_changes_plot(tonkean_table_full, tonkean_table_min, colors, metrics, titles)

filename_base = "individual-performance-changes-slope"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

# Quadratic patterns analysis - Restricted vs Full datasets comparison
print("\n" + "="*80)
print("QUADRATIC PATTERNS ACROSS LIFESPAN - RESTRICTED VS FULL DATASETS")
print("="*80)

metrics = {
    "p_success": "Hits",
    "p_error": "Errors", 
    "p_premature": "Premature",
    "p_omission": "Omissions",
    "rt_success": "RT Hits",
    "rt_error": "RT Errors",
}

# Create results table for both datasets
regression_results = []

for dataset_name, dataset in [("Restricted", tonkean_table_min), ("Full", tonkean_table_full)]:
    for metric, metric_name in metrics.items():
        # Filter out missing values
        clean_data = dataset.dropna(subset=[metric, "age_at_task"])
        
        if len(clean_data) < 3:  # Need at least 3 points for quadratic
            continue
        
        # Linear regression (baseline model)
        X_linear = sm.add_constant(clean_data["age_at_task"])
        y = clean_data[metric]
        linear_model = sm.OLS(y, X_linear).fit()
        
        # Quadratic regression
        X_quad = sm.add_constant(
            pd.DataFrame({
                "age": clean_data["age_at_task"],
                "age_sq": clean_data["age_at_task"] ** 2
            })
        )
        quad_model = sm.OLS(y, X_quad).fit()
        
        # F-test for model comparison
        f_stat = ((linear_model.ssr - quad_model.ssr) / 1) / (
            quad_model.ssr / (len(clean_data) - 3)
        )
        f_p_value = 1 - stats.f.cdf(f_stat, 1, len(clean_data) - 3)
        
        # Calculate peak/valley age if quadratic term is significant
        peak_age = None
        peak_type = None
        if quad_model.pvalues["age_sq"] < 0.05:
            peak_age = -quad_model.params["age"] / (2 * quad_model.params["age_sq"])
            peak_type = "Peak" if quad_model.params["age_sq"] < 0 else "Valley"
        
        # Store results
        regression_results.append({
            "Dataset": dataset_name,
            "Metric": metric_name,
            "N": len(clean_data),
            "Mean": clean_data[metric].mean(),
            "SD": clean_data[metric].std(),
            "Range": f"{clean_data[metric].min():.3f}-{clean_data[metric].max():.3f}",
            "Linear_R2": linear_model.rsquared,
            "Linear_p": linear_model.f_pvalue,
            "Linear_coef": linear_model.params["age_at_task"],
            "Linear_coef_p": linear_model.pvalues["age_at_task"],
            "Quad_R2": quad_model.rsquared,
            "Quad_p": quad_model.f_pvalue,
            "Quad_age_coef": quad_model.params["age"],
            "Quad_age2_coef": quad_model.params["age_sq"],
            "Quad_age_p": quad_model.pvalues["age"],
            "Quad_age2_p": quad_model.pvalues["age_sq"],
            "F_test_p": f_p_value,
            "R2_improvement": quad_model.rsquared - linear_model.rsquared,
            "Peak_age": peak_age,
            "Peak_type": peak_type,
            "Quadratic_better": "Yes" if f_p_value < 0.05 else "No",
            "Pearson_r": stats.pearsonr(clean_data["age_at_task"], clean_data[metric])[0],
            "Pearson_p": stats.pearsonr(clean_data["age_at_task"], clean_data[metric])[1],
        })

# Create comprehensive output table
results_df = pd.DataFrame(regression_results)

# Main table with key results
print(f"\n{'Dataset':<12} {'Metric':<12} {'N':<4} {'Mean±SD':<15} {'Lin R²':<7} {'Quad R²':<8} {'ΔR²':<6} {'F-test p':<9} {'Better':<8} {'Peak/Valley':<12}")
print("-" * 160)

for _, row in results_df.iterrows():
    mean_sd = f"{row['Mean']:.3f}±{row['SD']:.3f}"
    delta_r2 = row["R2_improvement"]
    
    if pd.notna(row["Peak_age"]) and row["Peak_type"]:
        peak_str = f"{row['Peak_type']} {row['Peak_age']:.1f}y"
    else:
        peak_str = "None"
    
    print(f"{row['Dataset']:<12} {row['Metric']:<12} {int(row['N']):<4} {mean_sd:<15} {row['Linear_R2']:<7.3f} "
          f"{row['Quad_R2']:<8.3f} {delta_r2:<6.3f} {row['F_test_p']:<9.3f} {row['Quadratic_better']:<8} {peak_str:<12}")

# Detailed statistical table
print("\n" + "="*160)
print("DETAILED STATISTICAL RESULTS")
print("="*160)
print(f"{'Dataset':<12} {'Metric':<12} {'Linear β':<10} {'Lin p':<8} {'Quad β₁':<10} {'Quad β₂':<10} {'β₁ p':<8} {'β₂ p':<8} {'Pearson r':<10} {'r p':<8}")
print("-" * 160)

for _, row in results_df.iterrows():
    print(f"{row['Dataset']:<12} {row['Metric']:<12} {row['Linear_coef']:<10.3f} {row['Linear_coef_p']:<8.3f} {row['Quad_age_coef']:<10.3f} "
          f"{row['Quad_age2_coef']:<10.3f} {row['Quad_age_p']:<8.3f} {row['Quad_age2_p']:<8.3f} "
          f"{row['Pearson_r']:<10.3f} {row['Pearson_p']:<8.3f}")

# Summary comparison between datasets
print("\n" + "="*80)
print("COMPARISON SUMMARY: RESTRICTED VS FULL DATASETS")
print("="*80)

for metric_name in metrics.values():
    restricted_row = results_df[(results_df["Dataset"] == "Restricted") & (results_df["Metric"] == metric_name)]
    full_row = results_df[(results_df["Dataset"] == "Full") & (results_df["Metric"] == metric_name)]
    
    if not restricted_row.empty and not full_row.empty:
        print(f"\n{metric_name}:")
        print(f"  Restricted: N={int(restricted_row.iloc[0]['N'])}, Linear R²={restricted_row.iloc[0]['Linear_R2']:.3f}, "
              f"Quad R²={restricted_row.iloc[0]['Quad_R2']:.3f}, Quad better: {restricted_row.iloc[0]['Quadratic_better']}")
        print(f"  Full:       N={int(full_row.iloc[0]['N'])}, Linear R²={full_row.iloc[0]['Linear_R2']:.3f}, "
              f"Quad R²={full_row.iloc[0]['Quad_R2']:.3f}, Quad better: {full_row.iloc[0]['Quadratic_better']}")

print("\nLegend:")
print("ΔR² = R² improvement from linear to quadratic model")
print("F-test p = significance of quadratic vs linear model comparison")
print("β₁ = linear age coefficient, β₂ = quadratic age coefficient")
print("Peak = maximum performance age, Valley = minimum performance age")
print("Better = whether quadratic model significantly improves fit (F-test p < 0.05)")