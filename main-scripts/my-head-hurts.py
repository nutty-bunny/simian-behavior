import os
import pickle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats  
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings

# ** re-run this script once pickle rick is finalised! **
directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
base = "/Users/similovesyou/Desktop/qts/simian-behavior"
pickle_rick = os.path.join(directory, "data.pickle")

with open(pickle_rick, "rb") as handle:
    raw = pickle.load(handle)

# Name mappings for both species
tonkean_abrv_2_fn = {
    "abr": "abricot",
    "ala": "alaryc",
    "alv": "alvin",
    "anu": "anubis",
    "bar": "barnabe",
    "ber": "berenice",
    "ces": "cesar",
    "dor": "dory",
    "eri": "eric",
    "jea": "jeanne",
    "lad": "lady",
    "las": "lassa",
    "nem": "nema",
    "nen": "nenno",
    "ner": "nereis",
    "ola": "olaf",
    "olg": "olga",
    "oli": "olli",
    "pac": "patchouli",
    "pat": "patsy",
    "wal": "wallace",
    "wat": "walt",
    "wot": "wotan",
    "yan": "yang",
    "yak": "yannick",
    "yin": "yin",
    "yoh": "yoh",
    "ult": "ulysse",
    "fic": "ficelle",
    "gan": "gandhi",
    "hav": "havanna",
    "con": "controle",
    "gai": "gaia",
    "her": "hercules",
    "han": "hanouk",
    "hor": "horus",
    "imo": "imoen",
    "ind": "indigo",
    "iro": "iron",
    "isi": "isis",
    "joy": "joy",
    "jip": "jipsy",
}

rhesus_abrv_2_fn = {
    "the": "theoden",
    "any": "anyanka",
    "djo": "djocko",
    "vla": "vladimir",
    "yel": "yelena",
    "bor": "boromir",
    "far": "faramir",
    "yva": "yvan",
    "baa": "baal",
    "spl": "spliff",
    "ol": "ol",
    "nat": "natasha",
    "arw": "arwen",
    "eow": "eowyn",
    "mar": "mar"
}

# Create a species mapping dictionary
species_mapping = {}
for abrv, name in tonkean_abrv_2_fn.items():
    species_mapping[name] = 'Tonkean'
for abrv, name in rhesus_abrv_2_fn.items():
    species_mapping[name] = 'Rhesus'

# Load both hierarchy files
tonkean_hierarchy_file = os.path.join(directory, "hierarchy/tonkean/elo_matrix.xlsx")
rhesus_hierarchy_file = os.path.join(directory, "hierarchy/rhesus/elo_matrix.xlsx")

tonkean_hierarchy = pd.read_excel(tonkean_hierarchy_file)
rhesus_hierarchy = pd.read_excel(rhesus_hierarchy_file)

# Rename columns in both dataframes
tonkean_hierarchy = tonkean_hierarchy.rename(columns=tonkean_abrv_2_fn)
rhesus_hierarchy = rhesus_hierarchy.rename(columns=rhesus_abrv_2_fn)

# Combine elo values from both species
tonkean_elo = tonkean_hierarchy.iloc[:, 1:].values.flatten()
rhesus_elo = rhesus_hierarchy.iloc[:, 1:].values.flatten()

all_elo = np.concatenate([tonkean_elo, rhesus_elo])
all_elo = all_elo[~pd.isnull(all_elo)]  # remove NaNs

print("Tonkean hierarchy:")
print(tonkean_hierarchy)
print("\nRhesus hierarchy:")
print(rhesus_hierarchy)

tonkean = []
rhesus = []
elo = {}  # stores filtered hierarchy data

# Create a combined dictionary for hierarchy data
hierarchies = {
    "tonkean": tonkean_hierarchy,
    "rhesus": rhesus_hierarchy
}

# Remove first two months for each monkey - elo score will have stbailised
for species, hierarchy in hierarchies.items():
    hierarchy["Date"] = pd.to_datetime(hierarchy["Date"])
    for name in hierarchy.columns[1:]:  # skip the Date column
        if name in hierarchy:
            monkey_elo = hierarchy[["Date", name]].dropna()
            if not monkey_elo.empty:
                # Get the first date with ELO data
                first_date = monkey_elo["Date"].min()
                # Calculate date after 2 months
                cutoff_date = first_date + pd.DateOffset(months=2)
                # Filter to keep only data after 2 months
                filtered_elo = monkey_elo[monkey_elo["Date"] > cutoff_date]
                elo[name] = filtered_elo

# Create a reverse mapping from full name to species
fullname_to_species = {}
for abrv, name in tonkean_abrv_2_fn.items():
    fullname_to_species[name] = "tonkean"
for abrv, name in rhesus_abrv_2_fn.items():
    fullname_to_species[name] = "rhesus"

for species, species_data in raw.items():
    count = len(species_data)

    for name, monkey_data in species_data.items():
        attempts = monkey_data["attempts"]

        attempts["instant_begin"] = pd.to_datetime(
            attempts["instant_begin"], unit="ms", errors="coerce"
        )
        attempts["instant_end"] = pd.to_datetime(
            attempts["instant_end"], unit="ms", errors="coerce"
        )

        # Convert to datetime64[ns] for consistent comparison
        first_attempt = pd.to_datetime(attempts["instant_begin"].min()).normalize()
        last_attempt = pd.to_datetime(attempts["instant_end"].max()).normalize()

        # Determine which hierarchy to use based on the species
        hierarchy = hierarchies.get(species)
        if hierarchy is None:
            print(f"No hierarchy data available for species: {species}")
            continue

        # Ensure hierarchy Date column is datetime64[ns]
        hierarchy["Date"] = pd.to_datetime(hierarchy["Date"])

        if name in hierarchy.columns:
            filtered_hierarchy = hierarchy[
                (hierarchy["Date"] >= first_attempt)
                & (hierarchy["Date"] <= last_attempt)
            ]  # filter hierarchy scores based on task period
            elo[name] = filtered_hierarchy[
                ["Date", name]
            ].dropna()  # drop NaNs where there is no hierarchy assessment

        if (
            name not in elo or elo[name].empty
        ):  # skip monkeys with no hierarchy data
            print(
                f"Monkey {name} has no ELO data. Task period: {first_attempt.date()} to {last_attempt.date()}. No ELO data available."
            )
            continue

        first_elo = elo[name]["Date"].min()
        last_elo = elo[name]["Date"].max()

        new_start_date = max(
            first_attempt, first_elo
        )  # overall onset that aligns with available elo
        new_end_date = min(last_attempt, last_elo)  # overall offset that aligns too

        if (
            new_start_date > new_end_date
        ):  # Check if there's no overlap between the periods
            print(
                f"Monkey {name} has no overlapping periods. Task period: {first_attempt.date()} to {last_attempt.date()}. ELO period: {first_elo.date()} to {last_elo.date()}."
            )
            continue

        # only include attempts that fall within the elo timeframe
        filtered_attempts = attempts[
            (attempts["instant_begin"] >= new_start_date)
            & (attempts["instant_end"] <= new_end_date)
        ]

        # everything that follows is performed on the restricted df
        attempt_count = filtered_attempts["id"].nunique()
        session_count = filtered_attempts["session"].nunique()
        outcome = filtered_attempts["result"].value_counts(normalize=True)
        rt_success = filtered_attempts[filtered_attempts["result"] == "success"][
            "reaction_time"
        ].mean()
        rt_error = filtered_attempts[filtered_attempts["result"] == "error"][
            "reaction_time"
        ].mean()

        monkey_data = {
            "name": name,
            "#_attempts": attempt_count,
            "#_sessions": session_count,
            "start_date": new_start_date.date(),
            "end_date": new_end_date.date(),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "mean_sess_duration": filtered_attempts["session_duration"].mean(),
            "mean_att#_/sess": filtered_attempts.groupby("session")["id"]
            .count()
            .mean(),
        }

        # Add to the appropriate list based on species
        if species == "tonkean":
            tonkean.append(monkey_data)
        elif species == "rhesus":
            rhesus.append(monkey_data)
        else:
            print(f"Unknown species: {species}")

for monkey in tonkean:
    monkey['species'] = 'tonkean'
for monkey in rhesus:
    monkey['species'] = 'rhesus'

# Convert to DataFrames
tonkean_df = pd.DataFrame(tonkean)
rhesus_df = pd.DataFrame(rhesus)

print("Tonkean monkeys data:")
print(tonkean_df)
print("\nRhesus monkeys data:")
print(rhesus_df)

# Convert hierarchy Date to datetime and set as index
tonkean_hierarchy['Date'] = pd.to_datetime(tonkean_hierarchy['Date'])
rhesus_hierarchy['Date'] = pd.to_datetime(rhesus_hierarchy['Date'])

# Create DataFrames from our collected data
tonkean_df = pd.DataFrame(tonkean)
rhesus_df = pd.DataFrame(rhesus)

# Sanity check to see if periods align
print("Checking period alignment:")
for species_data, species_name in [(tonkean, "Tonkean"), (rhesus, "Rhesus")]:
    for item in species_data:
        name = item["name"]
        print(f"\n{species_name} monkey: {name}")
        print(f"Task frame: {item['start_date']} to {item['end_date']}")
        if name in elo and not elo[name].empty:
            print(f"ELO frame: {elo[name]['Date'].min().date()} to {elo[name]['Date'].max().date()}")
        else:
            print("No hierarchy data available for this period")

# Calculate ELO means for both species
elo_mean = {}
for name, data in elo.items():
    if not data.empty:
        mean = round(data[name].mean())  # Rounding since ELO scores are integers
        elo_mean[name] = mean
    else:
        elo_mean[name] = None

# Append ELO mean scores to tables
for df, species in [(tonkean_df, 'tonkean'), (rhesus_df, 'rhesus')]:
    if 'elo_mean' not in df.columns:
        df['elo_mean'] = None
    
    for index, row in df.iterrows():
        name = row["name"]
        if name in elo_mean:
            df.loc[index, "elo_mean"] = elo_mean[name]
        else:
            df.loc[index, "elo_mean"] = None

# Load demographic data
mkid_rhesus = os.path.join(directory, "mkid_rhesus.csv")
mkid_tonkean = os.path.join(directory, "mkid_tonkean.csv")

demographics = {
    "mkid_rhesus": pd.read_csv(mkid_rhesus),
    "mkid_tonkean": pd.read_csv(mkid_tonkean),
}

# switch names to lowercase
demographics["mkid_tonkean"]["name"] = demographics["mkid_tonkean"]["name"].str.lower()
demographics["mkid_rhesus"]["name"] = demographics["mkid_rhesus"]["name"].str.lower()

# standardize column names
demographics["mkid_tonkean"].rename(columns={"Age": "age"}, inplace=True)
demographics["mkid_rhesus"].rename(columns={"Age": "age"}, inplace=True)

print("\nRhesus demographics (n = 16)")
print(demographics["mkid_rhesus"])
print("\nTonkean demographics (n = 35)")
print(demographics["mkid_tonkean"])

# Enhanced merge function with debugging
def safe_merge(behavior_df, demo_df, species_name):
    # Create clean name columns for merging
    behavior_df['merge_name'] = behavior_df['name'].str.lower().str.strip()
    demo_df['merge_name'] = demo_df['name'].str.lower().str.strip()
    
    # Print debug info
    print(f"\nDebugging {species_name} merge:")
    print(f"Behavior names sample: {behavior_df['merge_name'].head(3).tolist()}")
    print(f"Demographic names sample: {demo_df['merge_name'].head(3).tolist()}")
    
    # Perform merge
    merged = behavior_df.merge(
        demo_df[['merge_name', 'gender', 'age']],
        on='merge_name',
        how='left'
    )
    
    # Clean up
    merged = merged.drop(columns='merge_name')
    print(f"Merged gender values: {merged['gender'].unique()}")
    print(f"Missing genders: {merged['gender'].isna().sum()}/{len(merged)}")
    
    return merged

# Apply to both species with validation
if "mkid_tonkean" in demographics:
    print("\nMerging Tonkean demographics...")
    tonkean_df = safe_merge(tonkean_df, demographics["mkid_tonkean"], "Tonkean")
else:
    print("Warning: 'mkid_tonkean' not found in demographics")

if "mkid_rhesus" in demographics:
    print("\nMerging Rhesus demographics...")
    rhesus_df = safe_merge(rhesus_df, demographics["mkid_rhesus"], "Rhesus")
else:
    print("Warning: 'mkid_rhesus' not found in demographics")

# Verify merge results
print("\n=== MERGE VERIFICATION ===")
print("Tonkean gender distribution:")
print(tonkean_df['gender'].value_counts(dropna=False))
print("\nRhesus gender distribution:")
print(rhesus_df['gender'].value_counts(dropna=False))

# Initialize lists for both species
tonkean_proportions = []
rhesus_proportions = []

# Process both species
for species in ['tonkean', 'rhesus']:
    if species not in raw:
        print(f"Warning: {species} not found in data")
        continue
        
    for name, monkey_info in raw[species].items():
        # Debug print
        print(f"\nProcessing {name} ({species})")
        
        # Check data structure
        if not isinstance(monkey_info, dict) or 'attempts' not in monkey_info:
            print(f"Skipping {name} - invalid data structure")
            continue
            
        attempts = monkey_info["attempts"]
        
        # Debug print
        print(f"Total attempts before filtering: {len(attempts)}")

        # Ensure datetime conversion
        try:
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
        except Exception as e:
            print(f"Datetime conversion error for {name}: {e}")
            continue

        # Filter hierarchy data
        if name not in elo or elo[name].empty:
            print(f"No ELO data for {name}")
            continue
            
        # Get date ranges
        try:
            first_attempt = attempts["instant_begin"].min().date()
            last_attempt = attempts["instant_end"].max().date()
            first_elo = elo[name]["Date"].min().date()
            last_elo = elo[name]["Date"].max().date()
        except Exception as e:
            print(f"Date range error for {name}: {e}")
            continue

        new_start_date = max(first_attempt, first_elo)
        new_end_date = min(last_attempt, last_elo)
        
        # Debug print
        print(f"Date ranges - Attempts: {first_attempt} to {last_attempt}")
        print(f"Date ranges - ELO: {first_elo} to {last_elo}")
        print(f"Filtering range: {new_start_date} to {new_end_date}")

        # Filter attempts
        mask = (
            (attempts["instant_begin"].dt.date >= new_start_date) & 
            (attempts["instant_end"].dt.date <= new_end_date))
        filtered_attempts = attempts[mask]
        
        # Debug print
        print(f"Attempts after filtering: {len(filtered_attempts)}")

        if len(filtered_attempts) == 0:
            print(f"No attempts in filtered period for {name}")
            continue

        # Group by session
        sessions_grouped = filtered_attempts.groupby("session")
        
        for session, session_data in sessions_grouped:
            total_attempts = len(session_data)
            if total_attempts == 0:
                continue
                
            outcome_counts = session_data["result"].value_counts().to_dict()
            
            proportions = {
                "monkey": name,
                "species": species,
                "session": session,
                "total_attempts": total_attempts,
                "p_success": outcome_counts.get("success", 0.0) / total_attempts,
                "p_error": outcome_counts.get("error", 0.0) / total_attempts,
                "p_omission": outcome_counts.get("stepomission", 0.0) / total_attempts,
                "p_premature": outcome_counts.get("prematured", 0.0) / total_attempts,
            }

            if species == 'tonkean':
                tonkean_proportions.append(proportions)
            else:
                rhesus_proportions.append(proportions)

# Create DataFrames
tonkean_proportions_df = pd.DataFrame(tonkean_proportions) if tonkean_proportions else pd.DataFrame()
rhesus_proportions_df = pd.DataFrame(rhesus_proportions) if rhesus_proportions else pd.DataFrame()

# Dictionary of species data
species_data = {
    "Tonkean": {
        "proportions_df": tonkean_proportions_df,
        "summary_df": tonkean_df,
        "color": "#5386b6"
    },
    "Rhesus": {
        "proportions_df": rhesus_proportions_df,
        "summary_df": rhesus_df,
        "color": "#d57459"
    }
}

for species, content in species_data.items():
    proportions_df = content["proportions_df"]
    summary_df = content["summary_df"]

    if proportions_df.empty:
        print(f"No data for {species}")
        continue

    unique_monkeys = proportions_df["monkey"].unique()

    for monkey in unique_monkeys:
        monkey_df = proportions_df[proportions_df["monkey"] == monkey]
        monkey_df = monkey_df.sort_values("session").reset_index(drop=True)

        # Raw session-wise proportions
        stacked_df = monkey_df[
            ["p_success", "p_error", "p_premature", "p_omission"]
        ].dropna()
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

        # Hierarchy data
        if monkey not in elo:
            print(f"[{species}] No ELO data for {monkey}")
            continue

        hierarchy_monkey_df = elo[monkey]
        elo_series = hierarchy_monkey_df[monkey]

        # Combined plot
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # Top plot: Stacked proportions
        ax1 = fig.add_subplot(gs[0])
        ax1.stackplot(
            range(1, len(stacked_df) + 1),
            stacked_df["p_success"],
            stacked_df["p_error"],
            stacked_df["p_premature"],
            stacked_df["p_omission"],
            colors=["#5386b6", "#d57459", "#e8bc60", "#8d5993"],
            labels=["Success", "Error", "Premature", "Stepomission"],
        )

        # Dashed line for overall success
        try:
            overall_p_success = summary_df.loc[
                summary_df["name"] == monkey, "p_success"
            ].values[0]
        except IndexError:
            overall_p_success = None

        if overall_p_success is not None:
            ax1.axhline(y=overall_p_success, color="black", linestyle=(0, (5, 10)), linewidth=1.5)
            ax1.text(
                len(stacked_df) + 7,
                overall_p_success,
                f"{overall_p_success:.2f}",
                color="black",
                fontsize=14,
                fontname="Times New Roman",
                verticalalignment="center",
            )

        ax1.set_ylabel("Stacked Proportion", fontsize=14, fontname="Times New Roman")
        ax1.set_xlim(1, len(stacked_df))
        ax1.set_ylim(0, 1)
        ax1.set_yticks([0, 1])
        ax1.legend(loc="upper left", fontsize=12, prop={"family": "Times New Roman"})
        ax1.set_title(
            f"{monkey.capitalize()} ({species})",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman",
        )
        ax1.set_xlabel("Session Count", fontsize=14, fontname="Times New Roman")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Bottom plot: Elo (no smoothing)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(hierarchy_monkey_df["Date"], elo_series, color="#FFD700", linewidth=2)
        ax2.set_title("Hierarchy", fontsize=16, fontweight="bold", fontname="Times New Roman")

        # Dynamic y-limits
        y_mid = 1000
        y_range = max(abs(elo_series.max() - y_mid), abs(elo_series.min() - y_mid)) + 20
        ax2.set_ylim(y_mid - y_range, y_mid + y_range)
        ax2.set_yticks([700, 1000, 1300])

        # Dynamic x-limits
        ax2.set_xlim(hierarchy_monkey_df["Date"].min(), hierarchy_monkey_df["Date"].max())
        date_range_days = (hierarchy_monkey_df["Date"].max() - hierarchy_monkey_df["Date"].min()).days
        if date_range_days > 365:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))

        ax2.set_ylabel("Elo", fontsize=14, fontname="Times New Roman")
        ax2.set_xlabel("Date", fontsize=14, fontname="Times New Roman")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Uncomment to save figures
        # fig.savefig(base / "plots" / f"{monkey}_{species.lower()}.svg", format="svg")
        # fig.savefig(base / "plots" / f"{monkey}_{species.lower()}.png", format="png", dpi=300)

        plt.show()
        plt.close(fig)

# Configuration
CHUNK_SIZE = 200  # trials per chunk
OUTCOME_COLORS = {
    'success': '#4daf4a',
    'error': '#e41a1c',
    'premature': '#ff7f00',
    'omission': '#984ea3'
}

# --------------------------
# Plotting functions
def plot_subject_performance(df, name, species):
    """Plot individual subject performance vs hierarchy"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Performance plot
    for outcome in ['success', 'error', 'premature', 'omission']:
        if f'p_{outcome}' in df.columns:
            ax1.plot(df['chunk'], df[f'p_{outcome}'], 'o-',
                    color=OUTCOME_COLORS[outcome], label=outcome.capitalize(),
                    markersize=5, linewidth=1)
    
    ax1.set_ylabel('Proportion')
    ax1.set_ylim(0, 1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_title(f"{name} ({species}) - Performance")
    ax1.grid(True, alpha=0.3)
    
    # Hierarchy plot with gaps
    x_vals = df['chunk'].values
    elo_vals = df['mean_elo'].values
    mask = ~df['mean_elo'].isna()
    
    # Only plot if we have at least some ELO data
    if any(mask):
        # Plot line segments only between available points
        for i in range(len(x_vals)-1):
            if mask[i] and mask[i+1]:
                ax2.plot(x_vals[i:i+2], elo_vals[i:i+2], 'r-', linewidth=1.5)
            if mask[i]:
                ax2.plot(x_vals[i], elo_vals[i], 'ro', 
                        markersize=6, markerfacecolor='white')
        
        # Mark missing data
        if any(~mask):
            min_elo = np.nanmin(elo_vals[mask])
            ax2.plot(x_vals[~mask], (min_elo-50)*np.ones(sum(~mask)),
                    'rx', markersize=8, label='Missing ELO')
        
        ax2.set_ylabel('ELO Score')
    else:
        ax2.text(0.5, 0.5, 'No ELO data available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('100-Trial Chunks')
    ax2.set_title("Hierarchy (gaps show missing data)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_summary_results(df):
    """Plot aggregated results"""
    plt.figure(figsize=(10, 6))
    
    # Success vs ELO scatter
    sns.lmplot(data=df, x='mean_elo', y='p_success', 
              hue='species', height=6, aspect=1.5,
              ci=95, scatter_kws={'alpha':0.6, 's':80})
    
    plt.xlabel('Mean ELO Score')
    plt.ylabel('Success Rate')
    plt.title('Relationship Between Hierarchy and Performance\n(All Subjects)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------
# Main analysis
all_results = []

for species in raw.keys():
    for name in raw[species].keys():
        print(f"\nProcessing {name} ({species})...")
        
        # Prepare attempt data
        attempts = raw[species][name]["attempts"].copy()
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms")
        attempts = attempts.sort_values("instant_begin").reset_index(drop=True)
        
        # Create chunks (keep all chunks)
        attempts["chunk"] = (attempts.index // CHUNK_SIZE) + 1
        chunk_data = []
        
        for chunk_num, chunk_df in attempts.groupby("chunk"):
            chunk_start = chunk_df["instant_begin"].min()
            chunk_end = chunk_df["instant_begin"].max()
            
            # Get ELO data if available
            elo_subset = elo.get(name, pd.DataFrame())
            if not elo_subset.empty:
                elo_subset = elo_subset[
                    (elo_subset["Date"] >= chunk_start) & 
                    (elo_subset["Date"] <= chunk_end)]
            
            # Calculate all performance metrics
            outcomes = chunk_df["result"].value_counts(normalize=True)
            
            chunk_data.append({
                "subject": name,
                "species": species,
                "chunk": chunk_num,
                "start_date": chunk_start,
                "p_success": outcomes.get("success", 0),
                "p_error": outcomes.get("error", 0),
                "p_premature": outcomes.get("prematured", 0),
                "p_omission": outcomes.get("stepomission", 0),
                "mean_elo": elo_subset[name].mean() if not elo_subset.empty else np.nan,
                "has_elo": not elo_subset.empty
            })
        
        # Store results
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            all_results.append(df)
            plot_subject_performance(df, name, species)

# Combine all results
full_df = pd.concat(all_results).dropna(subset=['mean_elo'])

# Statistical Analysis
print("\n=== STATISTICAL ANALYSIS ===")

# 1. Prepare performance metrics
performance_metrics = ['p_success', 'p_error', 'p_premature', 'p_omission']

# 2. Mixed models for each performance metric
for metric in performance_metrics:
    print(f"\n--- {metric.upper().replace('_', ' ')} ---")
    model = ols(f'{metric} ~ mean_elo + C(species)', data=full_df).fit()
    print(model.summary())
    
    # Calculate partial correlations controlling for species
    grouped = full_df.groupby('species')[['mean_elo', metric]].corr().iloc[0::2,1]
    print("\nSpecies-specific partial correlations:")
    print(grouped)

# 3. Multivariate analysis
print("\n=== MULTIVARIATE ANALYSIS ===")
# Convert to subject-level averages
subject_df = full_df.groupby(['subject', 'species'])[
    ['mean_elo'] + performance_metrics].mean().reset_index()

# Pairwise correlations
corr_matrix = subject_df[performance_metrics + ['mean_elo']].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# Visualize relationships
plot_summary_results(subject_df)

# Step 1: Find minimum number of attempts across all monkeys
min_attempts = float("inf")

for species_data in raw.values():
    for monkey_data in species_data.values():
        n_attempts = len(monkey_data["attempts"])
        if n_attempts < min_attempts:
            min_attempts = n_attempts

print(f"Minimum number of attempts across all monkeys: {min_attempts}")

# Step 2: Trim all monkeys to min_attempts (earliest attempts)
for species, species_data in raw.items():
    for name, monkey_data in species_data.items():
        attempts = monkey_data["attempts"]
        # Convert to datetime if not already
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms")
        # Sort by time and take first min_attempts
        trimmed_attempts = attempts.sort_values("instant_begin").head(min_attempts)
        monkey_data["attempts_trimmed"] = trimmed_attempts

# Step 3: Rebuild summary tables using trimmed attempts
rhesus_trimmed = []
tonkean_trimmed = []

for species, species_data in raw.items():
    for name, monkey_data in species_data.items():
        attempts = monkey_data["attempts_trimmed"]
        
        # Calculate metrics
        outcome = attempts["result"].value_counts(normalize=True)
        rt_success = attempts[attempts["result"] == "success"]["reaction_time"].mean()
        rt_error = attempts[attempts["result"] == "error"]["reaction_time"].mean()
        
        # Get ELO score for trimmed period
        if name in elo and not elo[name].empty:
            first_date = attempts["instant_begin"].min()
            last_date = attempts["instant_begin"].max()
            elo_subset = elo[name][
                (elo[name]["Date"] >= first_date) & 
                (elo[name]["Date"] <= last_date)]
            mean_elo = elo_subset[name].mean() if not elo_subset.empty else np.nan
        else:
            mean_elo = np.nan
        
        # Get demographics
        demo_info = None
        if species == "rhesus" and "mkid_rhesus" in demographics:
            demo_info = demographics["mkid_rhesus"][demographics["mkid_rhesus"]["name"] == name].iloc[0]
        elif species == "tonkean" and "mkid_tonkean" in demographics:
            demo_info = demographics["mkid_tonkean"][demographics["mkid_tonkean"]["name"] == name].iloc[0]
        
        entry = {
            "name": name,
            "species": species,
            "#_attempts": len(attempts),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "elo_mean": mean_elo,
            "age": demo_info["age"] if demo_info is not None else np.nan,
            "gender": demo_info["gender"] if demo_info is not None else np.nan
        }

        if species == "rhesus":
            rhesus_trimmed.append(entry)
        elif species == "tonkean":
            tonkean_trimmed.append(entry)

for monkey in rhesus_trimmed:
    monkey['species'] = 'rhesus'
for monkey in tonkean_trimmed:
    monkey['species'] = 'tonkean'
# Create DataFrames
rhesus_df_trimmed = pd.DataFrame(rhesus_trimmed)
tonkean_df_trimmed = pd.DataFrame(tonkean_trimmed)
combined_df_trimmed = pd.concat([rhesus_df_trimmed, tonkean_df_trimmed])

print("\nTrimmed Rhesus data:")
print(rhesus_df_trimmed.head())
print("\nTrimmed Tonkean data:")
print(tonkean_df_trimmed.head())

# Save trimmed data as new pickle file
pickle_rick_trimmed = os.path.join(directory, "data-trimmed.pickle")

# Prepare data structure matching original format but with trimmed attempts
trimmed_data = {
    "rhesus": {},
    "tonkean": {}
}

for species, species_data in raw.items():
    for name, monkey_data in species_data.items():
        trimmed_data[species][name] = {
            "attempts": monkey_data.get("attempts_trimmed", pd.DataFrame()),
            # Include any other keys from original data you want to preserve
        }

# Save to pickle
with open(pickle_rick_trimmed, "wb") as handle:
    pickle.dump(trimmed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved trimmed data to: {pickle_rick_trimmed}")
print(f"Original data remains at: {pickle_rick}")


# First, let's create a function to handle NaN values similar to the tossNaN function you referenced
def tossNaN(df, columns):
    """Remove rows with NaN values in specified columns"""
    return df.dropna(subset=columns)

# Apply the function to both species' data
columns = ['elo_mean', 'mean_att#_/sess']  # Columns we need for analysis
rhesus_table_elo = tossNaN(rhesus_df, columns)
tonkean_table_elo = tossNaN(tonkean_df, columns)

# Combine both species for some analyses
combined_df = pd.concat([rhesus_df, tonkean_df])

# Create legend handles for the plot
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Rhesus'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Tonkean'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markeredgecolor='black', markersize=10, label='Male'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Female')
]

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
                    y="mean_att#_/sess",
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
                    y="mean_att#_/sess",
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
                mean = subset["mean_att#_/sess"].mean()
                std = subset["mean_att#_/sess"].std()
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
                    y="mean_att#_/sess",
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
                    y="mean_att#_/sess",
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
                mean = subset["mean_att#_/sess"].mean()
                std = subset["mean_att#_/sess"].std()
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
                    females["mean_att#_/sess"],
                    color=colors[species],
                    alpha=0.6,
                    s=100,
                )
                ax.scatter(
                    males["age"],
                    males["mean_att#_/sess"],
                    edgecolor=colors[species],
                    facecolor="none",
                    s=100,
                    linewidth=1.5,
                )

            sns.regplot(
                x="age",
                y="mean_att#_/sess",
                data=combined_df,
                order=2,
                scatter=False,
                ax=ax,
                color="black",
            )
            ax.set_xticks(range(0, 22, 5))

        elif i == 3:
            # Modified to include both species' ELO data
            for species in ["Rhesus", "Tonkean"]:
                species_df = rhesus_table_elo if species == "Rhesus" else tonkean_table_elo
                
                # Ensure numeric data types and drop NaN values
                species_df = species_df.copy()
                species_df["elo_mean"] = pd.to_numeric(species_df["elo_mean"], errors="coerce")
                species_df["mean_att#_/sess"] = pd.to_numeric(species_df["mean_att#_/sess"], errors="coerce")
                species_df = species_df.dropna(subset=["elo_mean", "mean_att#_/sess"])
                
                females = species_df[species_df["gender"] == 2]
                males = species_df[species_df["gender"] == 1]

                ax.scatter(
                    females["elo_mean"],
                    females["mean_att#_/sess"],
                    color=colors[species],
                    alpha=0.6,
                    s=100,
                    label=f"{species} Females" if i == 3 else ""
                )
                ax.scatter(
                    males["elo_mean"],
                    males["mean_att#_/sess"],
                    edgecolor=colors[species],
                    facecolor="none",
                    s=100,
                    linewidth=1.5,
                    label=f"{species} Males" if i == 3 else ""
                )

                # Only plot regression if we have enough data points
                if len(species_df) > 1:
                    sns.regplot(
                        x="elo_mean",
                        y="mean_att#_/sess",
                        data=species_df,
                        scatter=False,
                        ax=ax,
                        color=colors[species],
                    )

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
    save_path_svg = os.path.join(directory, "plots", f"{filename_base}.svg")
    save_path_png = os.path.join(directory, "plots", f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)  # Optional: increase DPI for PNG
    plt.show()


