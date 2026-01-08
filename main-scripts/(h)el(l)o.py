import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

### PRODUCE FIGURES THAT ARE RESTRICTED TO THE SAME PERIOD FOR ALL, including: 
    # Aligned elo periods + same number of sessions (attempts?)
    # Same as above just for other plots 

# re-run script after cleaning pickle rick
directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-behavior/visualisations"

pickle_rick = os.path.join(directory, "data.pickle")
with open(pickle_rick, "rb") as handle:
    data = pickle.load(handle)

# load tonkean_table_elo that's been aligned with elo-rating period (see social_status_bby.py)
tonkean_table_elo = pd.read_excel(
    os.path.join(directory, "hierarchy", "tonkean_table_elo.xlsx")
)

# note 2 simi -- proportions differ bcuz the task period has been shortened
print()
print("macaca tonkeana (n = 35) incl. elo")
print(tonkean_table_elo)

# let's append demographics
# 1 = male, 2 = female

mkid_tonkean = directory + "/mkid_tonkean.csv"
demographics = pd.read_csv(mkid_tonkean)
demographics["name"] = demographics["name"].str.lower()
demographics.rename(
    columns={"Age": "age"}, inplace=True
)  # switch names 2 lowercase cuz im 2 ocd

print()
print("tonkean demographics (n = 35)")
print(demographics)

# merge demographics
tonkean_table_elo = tonkean_table_elo.join(
    demographics.set_index("name")[["id", "gender", "birthdate", "age"]],
    on="name",
    how="left",
)
# summarize device engagement, task performance & demographics
print()
print("tonkean task engagement (incl. elo)")
print(tonkean_table_elo)

plt.rcParams["font.family"] = (
    "DejaVu Sans"  # set global font to be used across all plots ^^ **change later**
)
sns.set(
    style="whitegrid", font_scale=1.2, rc={"font.family": "DejaVu Sans"}
)  # ** modify font selection here! **


# plotting proportions across the lifespan
def p_lifespan(ax, metric, title, data, colors, show_xlabel, show_ylabel):
    sns.scatterplot(
        ax=ax,
        data=data[data["gender"] == 1],
        x="age",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=100,
        linewidth=1,
    )  # Modified color
    sns.scatterplot(
        ax=ax,
        data=data[data["gender"] == 2],
        x="age",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.6,
        edgecolor="purple",
        linewidth=1,
    )  # Modified size

    sns.regplot(
        ax=ax,
        data=data,
        x="age",
        y=metric,
        color=colors["tonkean"],
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )  # Quadratic fit

    ax.set_title(
        title, fontsize=20, fontweight="bold", fontname="DejaVu Sans"
    )  # Subtitle styling
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"])  # Only retain ticks 0 and 1
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")
    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)  # Updated font size
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=14)  # Updated font size
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = {"tonkean": "purple"}
metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]  # Updated subtitles

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes.flatten(),
    metrics,
    titles,
    [False, False, True, True],
    [True, False, True, False],
):
    p_lifespan(ax, metric, title, tonkean_table_elo, colors, show_xlabel, show_ylabel)

# Legend
handles = [
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
    "Task Performance Across the Lifespan in Macaques: Elo-Rating Included",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="purple",
)  # Updated main title

# Adjust placements and save as png
plt.tight_layout(rect=[0, 0.02, 1, 0.93])  # Reduced top margin
# plt.savefig('/Users/similovesyou/Downloads/Simian_Success.png', format='png')  # Save as png
plt.show()


def p_lifespan(ax, metric, title, data, show_xlabel, show_ylabel):
    # Scale elo_mean to use as point sizes
    min_size = 50
    max_size = 300
    sizes = (
        (data["elo_mean"] - data["elo_mean"].min())
        / (data["elo_mean"].max() - data["elo_mean"].min())
    ) * (max_size - min_size) + min_size

    # Even softer colormap: starts more pale
    custom_purple = LinearSegmentedColormap.from_list(
        "custom_purple", ["#F0E6F6", "#D8BFD8", "#800080", "#9932CC"]
    )
    norm = plt.Normalize(data["elo_mean"].min(), data["elo_mean"].max())

    # Split by gender
    male_data = data[data["gender"] == 1]
    female_data = data[data["gender"] == 2]

    scatter_male = ax.scatter(
        male_data["age"],
        male_data[metric],
        c="none",
        edgecolors=custom_purple(norm(male_data["elo_mean"])),
        s=sizes[data["gender"] == 1],
        linewidth=1.5,
    )
    scatter_female = ax.scatter(
        female_data["age"],
        female_data[metric],
        c=female_data["elo_mean"],
        cmap=custom_purple,
        s=sizes[data["gender"] == 2],
        linewidth=1.5,
        edgecolors="face",
    )

    # Add regression line using the SAME purple as before
    sns.regplot(
        ax=ax,
        data=data,
        x="age",
        y=metric,
        color="#800080",
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )

    # Set title and limits
    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    # Keep thick tick marks but default axis thickness
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7)

    # Show or hide x and y labels
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


# Create subplots and plot data
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
metrics = ["p_success", "p_error", "p_premature", "p_omission"]
titles = ["Hits", "Errors", "Premature", "Omissions"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes, metrics, titles, [True] * 4, [True] + [False] * 3
):
    p_lifespan(ax, metric, title, tonkean_table_elo, show_xlabel, show_ylabel)

# Legend
handles = [
    plt.Line2D(
        [],
        [],
        color="#800080",
        marker="o",
        linestyle="None",
        markersize=10,
        label=r"$\it{Macaca\ tonkeana}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="#800080",
        linestyle="None",
        markersize=10,
        label=r"$\it{Macaca\ tonkeana}$ (m)",
    ),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)

fig.suptitle(
    "Reaction Time Profiles Across the Lifespan in Tonkean Macaques: Elo-Rating Included",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="#800080",
)

# Adjust layout and save as SVG
plt.tight_layout(rect=[0, 0.02, 1, 0.93])

filename_base = "outcome-proportions-hierarchy"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
plt.show()

def rt_hierarchy(ax, metric, title, data, show_xlabel, show_ylabel, show_colorbar):
    # Scale elo_mean to use as point sizes
    min_size = 50
    max_size = 300
    sizes = (
        (data["elo_mean"] - data["elo_mean"].min())
        / (data["elo_mean"].max() - data["elo_mean"].min())
    ) * (max_size - min_size) + min_size

    # Custom purple colormap (lighter to darker purple)
    custom_purple = LinearSegmentedColormap.from_list(
        "custom_purple", ["#F0E6F6", "#D8BFD8", "#800080", "#9932CC"]
    )
    norm = plt.Normalize(data["elo_mean"].min(), data["elo_mean"].max())

    # Adjust very low values to slightly sit on the x-axis
    y_values = np.where(data[metric] < 0.02, 0.02, data[metric])

    # Split by gender
    male_data = data[data["gender"] == 1]
    female_data = data[data["gender"] == 2]

    ax.scatter(
        male_data["age"],
        y_values[data["gender"] == 1],
        c="none",
        edgecolors=custom_purple(norm(male_data["elo_mean"])),
        s=sizes[data["gender"] == 1],
        linewidth=1.5,
        label="Male",
    )
    ax.scatter(
        female_data["age"],
        y_values[data["gender"] == 2],
        c=female_data["elo_mean"],
        cmap=custom_purple,
        s=sizes[data["gender"] == 2],
        edgecolors="face",
        label="Female",
    )

    # Regression line
    sns.regplot(
        ax=ax,
        data=data,
        x="age",
        y=metric,
        color="#800080",
        order=2,
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--"},
    )

    # Titles and formatting
    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans")
    ax.set_xlim(0, 25)
    ax.grid(True, linestyle="--")

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("RTs", fontsize=14)
    else:
        ax.set_ylabel("")


# Create subplots and plot data
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
metrics = ["rt_success", "rt_error"]
titles = ["Hits", "Errors"]

for ax, metric, title, show_xlabel, show_ylabel, show_colorbar in zip(
    axes.flatten(), metrics, titles, [True, True], [True, True], [False, True]
):  # Only the right plot gets the colorbar
    rt_hierarchy(
        ax, metric, title, tonkean_table_elo, show_xlabel, show_ylabel, show_colorbar
    )

fig.suptitle(
    "How Fast Are They? (Incl. Elo-Score)",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="#800080",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

filename_base = "reaction-times-hierarchy"
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

plt.savefig(save_path_svg, format="svg")
plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
plt.show()


output_excel_path = os.path.join(directory, "hierarchy/tonkean_task_demographics.xlsx")
tonkean_table_elo.to_excel(output_excel_path, index=False)
print(f"Table saved to {output_excel_path}")

# i dont think this is necessary if im doing mvap

# # the following function mimics p_lifespan hence for comments refer there
# def rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel):

#     sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 1], x='age', y=metric, facecolors='none', edgecolor='steelblue', s=60, linewidth=1)
#     sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 2], x='age', y=metric, color=colors['rhesus'], s=60, alpha=0.6, edgecolor='steelblue', linewidth=1)

#     sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 1], x='age', y=metric, facecolors='none', edgecolor='navy', s=60, linewidth=1)
#     sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 2], x='age', y=metric, color=colors['tonkean'], s=60, alpha=0.6, edgecolor='navy', linewidth=1)

#     sns.regplot(ax=ax, data=rhesus_table, x='age', y=metric, color=colors['rhesus'], order=2, scatter=False, ci=None, line_kws={'linestyle':'--'})
#     sns.regplot(ax=ax, data=tonkean_table, x='age', y=metric, color=colors['tonkean'], order=2, scatter=False, ci=None, line_kws={'linestyle':'--'})
#     sns.regplot(ax=ax, data=pd.concat([rhesus_table, tonkean_table]), x='age', y=metric, color='black', order=2, scatter=False, ci=None)

#     ax.set_title(title)
#     ax.set_xlim(0, 25)
#     ax.set_xticks(range(0, 26, 5))
#     ax.grid(True, which='both', linestyle='--')
#     ax.grid(True, which='major', axis='x')
#     if show_xlabel:
#         ax.set_xlabel('Age')
#     else:
#         ax.set_xticklabels([])
#         ax.set_xlabel('')
#     if show_ylabel:
#         ax.set_ylabel('Reaction Time (ms)')
#     else:
#         ax.set_ylabel('')

# sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "DejaVu Sans"})

# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# # **change colors**
# colors = {'tonkean': 'darkblue', 'rhesus': 'lightblue'}

# metrics = ['rt_success', 'rt_error']
# titles = ['Mean Reaction Time for Successful Attempts Across Age', 'Mean Reaction Time for Erroneous Attempts Across Age']

# for ax, metric, title, show_xlabel, show_ylabel in zip(axes.flatten(), metrics, titles, [True, True], [True, False]):
#     rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

# fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4) # reusing handles from before to avoid redundancy

# fig.suptitle('Mean Reaction Time Across Age with Respect to Response Type', fontsize=20, fontweight='bold')

# # hey sim! ** adjust placements **
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# # plotting session frequencies (mean attempt # per session)
# combined_df = pd.concat([  # combine both species into one df
#     rhesus_table.assign(species='Rhesus'),
#     tonkean_table.assign(species='Tonkean')
# ]).drop_duplicates(subset='name', ignore_index=True)

# # ffs these warnings r gonna end me
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=FutureWarning)

#     plt.figure(figsize=(10, 8))
#     jitter = 0.1
#     colors = {'Rhesus': 'blue', 'Tonkean': 'green'}  # adjust colors

#     for species in ['Rhesus', 'Tonkean']:
#         males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]  # males hollow
#         females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]  # females filled

#         sns.stripplot(x='species', y='age', data=females, jitter=jitter, marker='o', color=colors[species], alpha=0.6, size=10)
#         sns.stripplot(x='species', y='age', data=males, jitter=jitter, marker='o', facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5)

#     for species in ['Rhesus', 'Tonkean']: # loop to plot error bars for each species (irrespective of gender)
#         subset = combined_df[combined_df['species'] == species]
#         mean_age = subset['age'].mean()
#         std_age = subset['age'].std()
#         plt.errorbar(x=[species], y=[mean_age], yerr=[std_age], fmt='none', ecolor=colors[species], capsize=5, elinewidth=2)

#     plt.xlabel('Species', fontsize=12, fontstyle='italic')
#     plt.ylabel('Age')
#     plt.title('Age Distribution of Monkeys by Species')

#     plt.xticks(ticks=[0, 1],
#            labels=[f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
#                    f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})'],
#            fontsize=12)

#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}')) # remove decimals when plotting age
#     plt.ylim(0, 25)
#     plt.xlim(-0.4, 1.4) # adjust x-lims to move the plots together (to be used in following plots)
#     plt.show()

# def tossNaN(df, columns):
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace inf with NaN
#     return df.dropna(subset=columns)  # drop NaNs

# columns = ['mean_att#_/sess', 'gender']

# Find the minimum number of attempts across all monkeys
min_attempts = float("inf")

for species_data in data.values():
    for monkey_data in species_data.values():
        n_attempts = monkey_data["attempts"].shape[0]
        if n_attempts < min_attempts:
            min_attempts = n_attempts

print(f"Minimum number of attempts across all monkeys: {min_attempts}")

# Apply trimming: keep only the earliest min_attempts for each monkey
for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        attempts = monkey_data["attempts"]
        # Make sure instant_begin is sorted (earliest first)
        attempts_sorted = attempts.sort_values(by="instant_begin")
        # Trim to minimum count
        trimmed_attempts = attempts_sorted.head(min_attempts)
        # Update the monkey_data with trimmed attempts
        monkey_data["attempts_trimmed"] = trimmed_attempts

print("All monkey DataFrames now have a trimmed 'attempts_trimmed' version.")

### Redo this trimmed stuff for elo (might overlap more)
### Export all the dataframes based on which the plots are done with representative names
