import os
import pickle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.generalized_estimating_equations import GEE

import pymc as pm
import arviz as az

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
base = "/Users/similovesyou/Desktop/qts/simian-behavior/dms"
dmt_rick = os.path.join(directory, 'dms.pickle')
with open(dmt_rick, 'rb') as handle:
    data = pickle.load(handle)

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

tonkean_hierarchy_file = os.path.join(directory, "hierarchy/tonkean/elo_matrix.xlsx")
rhesus_hierarchy_file = os.path.join(directory, "hierarchy/rhesus/elo_matrix.xlsx")

tonkean_hierarchy = pd.read_excel(tonkean_hierarchy_file)
rhesus_hierarchy = pd.read_excel(rhesus_hierarchy_file)

tonkean_hierarchy = tonkean_hierarchy.rename(columns=tonkean_abrv_2_fn)
rhesus_hierarchy = rhesus_hierarchy.rename(columns=rhesus_abrv_2_fn)

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

for species, species_data in data.items():
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
            "mean_attempts_per_session": filtered_attempts.groupby("session")["id"]
            .count()
            .mean(),
        }

        if species == "tonkean":
            tonkean.append(monkey_data)
        elif species == "rhesus":
            rhesus.append(monkey_data)
        else:
            print(f"Unknown species: {species}")

tonkean_df = pd.DataFrame(tonkean)
rhesus_df = pd.DataFrame(rhesus)

print("Tonkean monkeys data:")
print(tonkean_df)
print("\nRhesus monkeys data:")
print(rhesus_df)


"""
# What if also restrict to the same number of attempts? 
all_attempt_counts = []
for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        if name in elo and not elo[name].empty:
            # Get ELO-filtered attempts (from your original code)
            attempts = monkey_data["attempts"].copy()
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
            
            # Apply ELO date filtering
            filtered_attempts = attempts[
                (attempts["instant_begin"] >= elo[name]["Date"].min()) &
                (attempts["instant_end"] <= elo[name]["Date"].max())
            ]
            all_attempt_counts.append(len(filtered_attempts))

min_attempts = min(all_attempt_counts)
print(f"\nMinimum attempts across monkeys (post-ELO filtering): {min_attempts}")
def build_restricted_df(original_df, data_dict, species):
    restricted_data = []
    
    for _, row in original_df.iterrows():
        name = row["name"]
        if name not in elo or elo[name].empty:
            continue  # Skip monkeys without ELO data
        
        monkey_data = data_dict[species][name]
        attempts = monkey_data["attempts"].copy()
        
        # Apply ELO filtering
        attempts = attempts[
            (attempts["instant_begin"] >= elo[name]["Date"].min()) &
            (attempts["instant_end"] <= elo[name]["Date"].max())
        ]
        
        # Restrict to first `min_attempts` (chronological order)
        attempts = attempts.sort_values("instant_begin").head(min_attempts)
        
        # Recompute metrics (identical to your original)
        attempt_count = len(attempts)
        session_count = attempts["session"].nunique()
        outcome = attempts["result"].value_counts(normalize=True)
        rt_success = attempts[attempts["result"] == "success"]["reaction_time"].mean()
        rt_error = attempts[attempts["result"] == "error"]["reaction_time"].mean()
        
        restricted_data.append({
            "name": name,
            "#_attempts": attempt_count,
            "#_sessions": session_count,
            "start_date": pd.to_datetime(attempts["instant_begin"].min()).date(),
            "end_date": pd.to_datetime(attempts["instant_end"].max()).date(),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "mean_sess_duration": attempts["session_duration"].mean(),
            "mean_attempts_per_session": attempts.groupby("session")["id"].count().mean(),
        })
    
    return pd.DataFrame(restricted_data)

# Apply to both species
tonkean_df_min = build_restricted_df(tonkean_df, data, "tonkean")
rhesus_df_min = build_restricted_df(rhesus_df, data, "rhesus")

print("\nTonkean monkeys data (min # attempts):")
print(tonkean_df_min.head())
print("\nRhesus monkeys data (min # attempts):")
print(rhesus_df_min.head())
"""
# Specify the number of attempts you want to consider
DESIRED_ATTEMPTS = 448  # Change this to whatever number you want

# Optional: Check how many monkeys have at least this many attempts
all_attempt_counts = []
monkeys_with_sufficient_data = 0

for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        if name in elo and not elo[name].empty:
            # Get ELO-filtered attempts
            attempts = monkey_data["attempts"].copy()
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
            
            # Apply ELO date filtering
            filtered_attempts = attempts[
                (attempts["instant_begin"] >= elo[name]["Date"].min()) &
                (attempts["instant_end"] <= elo[name]["Date"].max())
            ]
            
            attempt_count = len(filtered_attempts)
            all_attempt_counts.append(attempt_count)
            
            if attempt_count >= DESIRED_ATTEMPTS:
                monkeys_with_sufficient_data += 1

print(f"Desired attempts per monkey: {DESIRED_ATTEMPTS}")
print(f"Monkeys with at least {DESIRED_ATTEMPTS} attempts: {monkeys_with_sufficient_data}")
print(f"Total monkeys with ELO data: {len(all_attempt_counts)}")
print(f"Range of available attempts: {min(all_attempt_counts)} - {max(all_attempt_counts)}")

def build_restricted_df(original_df, data_dict, species, num_attempts):
    restricted_data = []
    skipped_monkeys = []
    
    for _, row in original_df.iterrows():
        name = row["name"]
        if name not in elo or elo[name].empty:
            continue  # Skip monkeys without ELO data
        
        monkey_data = data_dict[species][name]
        attempts = monkey_data["attempts"].copy()
        
        # Apply ELO filtering
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
        attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
        
        attempts = attempts[
            (attempts["instant_begin"] >= elo[name]["Date"].min()) &
            (attempts["instant_end"] <= elo[name]["Date"].max())
        ]
        
        # Check if monkey has enough attempts
        if len(attempts) < num_attempts:
            skipped_monkeys.append(f"{name} (has {len(attempts)} attempts)")
            continue
        
        # Restrict to first `num_attempts` (chronological order)
        attempts = attempts.sort_values("instant_begin").head(num_attempts)
        
        # Recompute metrics
        attempt_count = len(attempts)
        session_count = attempts["session"].nunique()
        outcome = attempts["result"].value_counts(normalize=True)
        rt_success = attempts[attempts["result"] == "success"]["reaction_time"].mean()
        rt_error = attempts[attempts["result"] == "error"]["reaction_time"].mean()
        
        restricted_data.append({
            "name": name,
            "#_attempts": attempt_count,
            "#_sessions": session_count,
            "start_date": pd.to_datetime(attempts["instant_begin"].min()).date(),
            "end_date": pd.to_datetime(attempts["instant_end"].max()).date(),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "mean_sess_duration": attempts["session_duration"].mean(),
            "mean_attempts_per_session": attempts.groupby("session")["id"].count().mean(),
        })
    
    if skipped_monkeys:
        print(f"\n{species.capitalize()} monkeys skipped (insufficient attempts):")
        for monkey in skipped_monkeys:
            print(f"  - {monkey}")
    
    return pd.DataFrame(restricted_data)

# Apply to both species
tonkean_df_min = build_restricted_df(tonkean_df, data, "tonkean", DESIRED_ATTEMPTS)
rhesus_df_min = build_restricted_df(rhesus_df, data, "rhesus", DESIRED_ATTEMPTS)

print(f"\nFinal sample sizes:")
print(f"Tonkean monkeys: {len(tonkean_df_min)}")
print(f"Rhesus monkeys: {len(rhesus_df_min)}")
print(f"Total monkeys: {len(tonkean_df_min) + len(rhesus_df_min)}")

# Verify all monkeys have exactly the desired number of attempts
print(f"\nVerification - all monkeys should have exactly {DESIRED_ATTEMPTS} attempts:")
print("Tonkean:", tonkean_df_min["#_attempts"].unique())
print("Rhesus:", rhesus_df_min["#_attempts"].unique())

tonkean_hierarchy['Date'] = pd.to_datetime(tonkean_hierarchy['Date'])
rhesus_hierarchy['Date'] = pd.to_datetime(rhesus_hierarchy['Date'])

print("Checking period alignment:")
for df, species_name in [(tonkean_df, "Tonkean (Original)"), 
                         (tonkean_df_min, "Tonkean (Restricted)"),
                         (rhesus_df, "Rhesus (Original)"),
                         (rhesus_df_min, "Rhesus (Restricted)")]:
    print(f"\n\n=== {species_name} ===")
    for _, row in df.iterrows():
        name = row["name"]
        print(f"\nMonkey: {name}")
        print(f"Task frame: {row['start_date']} to {row['end_date']}")
        if name in elo and not elo[name].empty:
            print(f"ELO frame: {elo[name]['Date'].min().date()} to {elo[name]['Date'].max().date()}")
        else:
            print("No hierarchy data available for this period")

# Function to calculate ELO mean for a specific date range
def calculate_elo_mean(name, start_date, end_date):
    if name not in elo or elo[name].empty:
        return None
    # Convert to datetime for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter ELO scores within the given date range
    mask = (elo[name]['Date'] >= start_date) & (elo[name]['Date'] <= end_date)
    filtered_elo = elo[name][mask]
    
    return round(filtered_elo[name].mean()) if not filtered_elo.empty else None

# For original DataFrame
for df in [tonkean_df, rhesus_df]:
    df['elo_mean'] = df.apply(
        lambda row: calculate_elo_mean(row['name'], row['start_date'], row['end_date']),
        axis=1
    )

# For restricted DataFrame
for df in [tonkean_df_min, rhesus_df_min]:
    df['elo_mean'] = df.apply(
        lambda row: calculate_elo_mean(row['name'], row['start_date'], row['end_date']),
        axis=1
    )
# Verify results
print("\nElo-aligned tonkean:")
print(tonkean_df[['name', '#_attempts', 'elo_mean']].head())
print("\nElo-aligned and min attempt count tonkean:")
print(tonkean_df_min[['name', '#_attempts', 'elo_mean']].head())

# Function to build a restricted DataFrame with a fixed number of attempts, ELO-aligned
def build_fixed_attempts_df(original_df, data_dict, species, n_attempts=100):
    restricted_data = []
    
    for _, row in original_df.iterrows():
        name = row["name"]
        if name not in elo or elo[name].empty:
            continue  # Skip monkeys without ELO data
        
        monkey_data = data_dict[species][name]
        attempts = monkey_data["attempts"].copy()
        
        # Convert times to datetime if not already
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
        attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
        
        # Apply ELO filtering
        attempts = attempts[
            (attempts["instant_begin"] >= elo[name]["Date"].min()) &
            (attempts["instant_end"] <= elo[name]["Date"].max())
        ]
        
        # Restrict to first `n_attempts` (chronological order)
        attempts = attempts.sort_values("instant_begin").head(n_attempts)
        
        if attempts.empty:
            continue
        
        # Recompute metrics
        attempt_count = len(attempts)
        session_count = attempts["session"].nunique()
        outcome = attempts["result"].value_counts(normalize=True)
        rt_success = attempts[attempts["result"] == "success"]["reaction_time"].mean()
        rt_error = attempts[attempts["result"] == "error"]["reaction_time"].mean()
        
        restricted_data.append({
            "name": name,
            "#_attempts": attempt_count,
            "#_sessions": session_count,
            "start_date": pd.to_datetime(attempts["instant_begin"].min()).date(),
            "end_date": pd.to_datetime(attempts["instant_end"].max()).date(),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "mean_sess_duration": attempts["session_duration"].mean(),
            "mean_attempts_per_session": attempts.groupby("session")["id"].count().mean(),
        })
    
    df_fixed = pd.DataFrame(restricted_data)
    
    # Add ELO mean
    df_fixed['elo_mean'] = df_fixed.apply(
        lambda row: calculate_elo_mean(row['name'], row['start_date'], row['end_date']),
        axis=1
    )
    
    return df_fixed

# Build 100-attempts ELO-aligned DataFrames for both species
tonkean_df_100 = build_fixed_attempts_df(tonkean_df, data, "tonkean", n_attempts=100)
rhesus_df_100 = build_fixed_attempts_df(rhesus_df, data, "rhesus", n_attempts=100)

# Add species column
tonkean_df_100['species'] = 'Tonkean'
rhesus_df_100['species'] = 'Rhesus'

# Combine both species
combined_100_df = pd.concat([tonkean_df_100, rhesus_df_100], ignore_index=True)

# Save to CSV
output_file = os.path.join(base, "derivatives", "elo-100-attempts.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

combined_100_df.to_csv(output_file, index=False)

print(f"Saved combined Elo-aligned 100-attempts table to {output_file}")


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

print("\nRhesus demographics")
print(demographics["mkid_rhesus"])
print("\nTonkean demographics")
print(demographics["mkid_tonkean"])

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

if "mkid_tonkean" in demographics:
    print("\nMerging Tonkean demographics...")
    tonkean_df = safe_merge(tonkean_df, demographics["mkid_tonkean"], "Tonkean (original)")
    tonkean_df_min = safe_merge(tonkean_df_min, demographics["mkid_tonkean"], "Tonkean (trimmed)")
else:
    print("Warning: 'mkid_tonkean' not found in demographics")

if "mkid_rhesus" in demographics:
    print("\nMerging Rhesus demographics...")
    rhesus_df = safe_merge(rhesus_df, demographics["mkid_rhesus"], "Rhesus (original)")
    rhesus_df_min = safe_merge(rhesus_df_min, demographics["mkid_rhesus"], "Rhesus (trimmed)")
else:
    print("Warning: 'mkid_rhesus' not found in demographics")

# save for future processing
derivatives_dir = Path(base) / "derivatives"
derivatives_dir.mkdir(parents=True, exist_ok=True)

# Save all four tables
tables = {
    "tonkean-elo": tonkean_df,
    "tonkean-elo-min-attempts": tonkean_df_min,
    "rhesus-elo": rhesus_df,
    "rhesus-elo-min-attempts": rhesus_df_min
}

for name, df in tables.items():
    df.to_csv(derivatives_dir / f"{name}.csv", index=False)
    df.to_pickle(derivatives_dir / f"{name}.pkl")

print(f"Saved files to {derivatives_dir}:")
print(*[f"- {name}.csv/.pkl" for name in tables.keys()], sep="\n")

# Initialize lists for both species
tonkean_proportions = []
rhesus_proportions = []

# Process both species
for species in ['tonkean', 'rhesus']:
    if species not in data:
        print(f"Warning: {species} not found in data")
        continue
        
    for name, monkey_info in data[species].items():
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
        # print(f"Date ranges - Attempts: {first_attempt} to {last_attempt}")
        # print(f"Date ranges - ELO: {first_elo} to {last_elo}")
        print(f"Filtering range: {new_start_date} to {new_end_date}")

        mask = (
            (attempts["instant_begin"].dt.date >= new_start_date) & 
            (attempts["instant_end"].dt.date <= new_end_date))
        filtered_attempts = attempts[mask]
        
        print(f"Attempts after filtering: {len(filtered_attempts)}")

        if len(filtered_attempts) == 0:
            print(f"No attempts in filtered period for {name}")
            continue

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

### GLM MODELING on raw values 
# NOTE-2-simi: what about standardizing the scores?

if not isinstance(all_elo, pd.DataFrame):
    # Reconstruct ELO data if needed
    elo_list = []
    for species in ["tonkean", "rhesus"]:
        hierarchy = tonkean_hierarchy if species == "tonkean" else rhesus_hierarchy
        for col in hierarchy.columns[1:]:  # Skip Date column
            temp = hierarchy[['Date', col]].dropna()
            temp['name'] = col
            temp = temp.rename(columns={col: 'elo_score'})
            elo_list.append(temp)
    all_elo = pd.concat(elo_list)

attempt_counts = []
for species in ["tonkean", "rhesus"]:
    for name, monkey_data in data[species].items():
        if name in all_elo['name'].unique():
            attempts = monkey_data["attempts"].copy()
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms")
            
            monkey_elo = all_elo[all_elo['name'] == name]
            first_elo = monkey_elo['Date'].min()
            last_elo = monkey_elo['Date'].max()
            
            mask = (attempts['instant_begin'] >= first_elo) & (attempts['instant_end'] <= last_elo)
            attempt_counts.append(len(attempts[mask]))

min_attempts = min(attempt_counts)
print(f"Minimum attempts across all monkeys: {min_attempts}")

attempts_list = []

for species in ["tonkean", "rhesus"]:
    for name, monkey_data in data[species].items():
        attempts = monkey_data["attempts"].copy()
        attempts["species"] = species
        attempts["name"] = name
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms")
        attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms")
        
        if name not in all_elo['name'].unique():
            continue
            
        monkey_elo = all_elo[all_elo['name'] == name]
        first_elo = monkey_elo['Date'].min()
        last_elo = monkey_elo['Date'].max()
        
        mask = (attempts['instant_begin'] >= first_elo) & (attempts['instant_end'] <= last_elo)
        filtered = attempts[mask].sort_values('instant_begin').head(min_attempts)
        
        if len(filtered) == min_attempts:
            attempts_list.append(filtered)

final_data = pd.concat(attempts_list)
final_data = pd.merge_asof(
    final_data.sort_values('instant_begin'),
    all_elo.sort_values('Date')[['name', 'Date', 'elo_score']],
    left_on='instant_begin',
    right_on='Date',
    by='name',
    direction='nearest'
)

# Merge with demographic data
demographics = pd.concat([
    demographics["mkid_tonkean"].assign(species="tonkean"),
    demographics["mkid_rhesus"].assign(species="rhesus")
])
final_data = final_data.merge(
    demographics[['name', 'age', 'gender']],
    on=['name'],
    how='left'
)

# Convert result types to binary columns
result_types = ['success', 'error', 'premature', 'stepomission']
for res in result_types:
    final_data[res] = (final_data['result'] == res).astype(int)

print(f"Final dataset ready for modeling with {len(final_data)} attempts")
print(final_data[['name', 'species', 'age', 'elo_score', 'result']].head())

# First ensure all categorical variables are properly encoded
final_data['species'] = pd.Categorical(final_data['species'])
final_data['gender'] = pd.Categorical(final_data['gender'])

# Model each outcome type using GEE (Generalized Estimating Equations)
# This accounts for within-monkey correlation

results = {}
for outcome in result_types:
    print(f"\n~~~ MODELING: {outcome.upper()} ~~~")
    
    try:
        model = GEE.from_formula(
            f"{outcome} ~ elo_score + species + age + gender",
            groups=final_data["name"],
            data=final_data,
            family=families.Binomial()
        ).fit()
        
        print(model.summary())
        results[outcome] = model
        
    except Exception as e:
        print(f"Failed to fit {outcome}: {str(e)}")

# Save the modeling data
output_path = Path(base) / "derivatives" / "raw-modeling-data-elo-min-attempts.csv"
final_data.to_csv(output_path, index=False)

print(f"Saved modeling data to: {output_path}")

### PLOTTING OF ROLLING AVERAGES
tonkean_proportions_df = pd.DataFrame(tonkean_proportions) if tonkean_proportions else pd.DataFrame()
rhesus_proportions_df = pd.DataFrame(rhesus_proportions) if rhesus_proportions else pd.DataFrame()

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
        output_dir = os.path.join(base, "plots", "roll-all-over-me")
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.svg"), format="svg")
        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.png"), format="png", dpi=300)

        plt.show()
        plt.close(fig)

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

        # Create figure (now single plot)
        fig = plt.figure(figsize=(10, 8))
        
        # Stacked proportions plot
        ax = fig.add_subplot(111)
        ax.stackplot(
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
            ax.axhline(y=overall_p_success, color="black", linestyle=(0, (5, 10)), linewidth=1.5)
            ax.text(
                len(stacked_df) + 7,
                overall_p_success,
                f"{overall_p_success:.2f}",
                color="black",
                fontsize=14,
                fontname="Times New Roman",
                verticalalignment="center",
            )

        ax.set_ylabel("Stacked Proportion", fontsize=14, fontname="Times New Roman")
        ax.set_xlabel("Session Count", fontsize=14, fontname="Times New Roman")
        ax.set_xlim(1, len(stacked_df))
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.legend(loc="upper left", fontsize=12, prop={"family": "Times New Roman"})
        ax.set_title(
            f"{monkey.capitalize()} ({species})",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman",
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        output_dir = os.path.join(base, "plots", "roll-all-over-me-no-hierarchy")
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.svg"), format="svg")
        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.png"), format="png", dpi=300)

        plt.show()
        plt.close(fig)

# GLM TIME!!! 
combined_df = pd.concat([
    tonkean_df_min.assign(species='Tonkean'),
    rhesus_df_min.assign(species='Rhesus')
]).reset_index(drop=True)

combined_df['age'] = pd.to_numeric(combined_df['age'], errors='coerce')
combined_df['elo_mean'] = pd.to_numeric(combined_df['elo_mean'], errors='coerce')

combined_df = combined_df.dropna(subset=['age', 'elo_mean', 'p_success'])

combined_df['age_sq'] = combined_df['age']**2
combined_df['age_cubed'] = combined_df['age']**3

print(f"Final sample size: {len(combined_df)}")
print(combined_df[['name', 'species', 'age', 'elo_mean', 'p_success']].head())

# Function to calculate VIF
def calculate_vifs(data, predictors):
    X = sm.add_constant(data[predictors])
    vifs = pd.DataFrame()
    vifs["Variable"] = X.columns
    vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vifs

# Check VIF between age and ELO
vif_results = calculate_vifs(combined_df, ['age', 'elo_mean'])
print("\nVIF between age and ELO:")
print(vif_results)

# Set up spline parameters
knots = np.quantile(combined_df['age'], [0.25, 0.5, 0.75])
spline_formula = f"bs(age, knots={tuple(knots)}, degree=3, include_intercept=False)"

def analyze_metric(metric, df):
    """Run spline analysis for a given metric"""
    model = glm(f"{metric} ~ {spline_formula} + elo_mean + species", 
               data=df, 
               family=sm.families.Gaussian()).fit()
    
    # Generate predictions
    age_grid = np.linspace(df['age'].min(), df['age'].max(), 100)
    pred_data = pd.DataFrame({
        'age': age_grid,
        'elo_mean': df['elo_mean'].mean(),
        'species': 'Tonkean'
    })
    
    # Add spline terms
    spline_basis = dmatrix(spline_formula, pred_data, return_type='dataframe')
    pred_data = pd.concat([pred_data, spline_basis], axis=1)
    pred_data['pred'] = model.predict(pred_data)
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(age_grid, pred_data['pred'], 'r-', linewidth=2)
    plt.scatter(df['age'], df[metric], alpha=0.3)
    for knot in knots:
        plt.axvline(knot, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Age (years)')
    plt.ylabel(metric)
    plt.title(f'Spline Fit: {metric} vs Age')
    plt.show()
    
    return model

# Run analyses
metrics = ['p_success', 'p_error', 'p_premature', 'p_omission', 
           'rt_success', 'rt_error']
results = {metric: analyze_metric(metric, combined_df) for metric in metrics}

for metric, model in results.items():
    print(f"\n~~~ {metric.upper()} ~~~")
    print(model.summary())

# Bayesian GLM with splines? No idea how to interpret this
with pm.Model() as bayesian_model:
    σ = pm.HalfNormal('σ', sigma=1)  
    β_elo = pm.Normal('β_elo', mu=0, sigma=1)
    β_species = pm.Normal('β_species', mu=0, sigma=1)
    
    # Spline basis (same as before)
    age_spline = dmatrix(spline_formula, {'age': combined_df['age']}, return_type='dataframe')
    
    # Linear predictor
    η = (
        β_elo * combined_df['elo_mean'] + 
        β_species * (combined_df['species'] == 'Tonkean').astype(int) +
        pm.math.dot(age_spline, pm.Normal('β_spline', mu=0, sigma=1, shape=age_spline.shape[1]))
    )
    
    # Likelihood
    pm.Normal('y', mu=η, sigma=σ, observed=combined_df['p_success'])  
    
    # Sampling
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.95,  # Helps with convergence
        random_seed=42
    )

# Basic summary
print(az.summary(trace, round_to=2))



# Monkeys you want to plot in a 3x3 grid
selected_monkeys = ["barnabe", "olli", "alaryc", "olaf", "wotan", "lassa", "walt", "nereis", "olga"]

fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=False, sharey=False)
axes = axes.flatten()

for i, monkey in enumerate(selected_monkeys):
    ax = axes[i]
    
    # Loop through species to find monkey data
    for species, content in species_data.items():
        proportions_df = content["proportions_df"]
        summary_df = content["summary_df"]

        if proportions_df.empty or monkey not in proportions_df["monkey"].unique():
            continue

        monkey_df = proportions_df[proportions_df["monkey"] == monkey]
        monkey_df = monkey_df.sort_values("session").reset_index(drop=True)

        # Raw session-wise proportions
        stacked_df = monkey_df[["p_success", "p_error", "p_premature", "p_omission"]].dropna()
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

        # Apply rolling window of 10 sessions
        stacked_df = stacked_df.rolling(window=1, min_periods=1, center=True).mean()

        # Stacked proportions plot
        ax.stackplot(
            range(1, len(stacked_df) + 1),
            stacked_df["p_success"],
            stacked_df["p_error"],
            stacked_df["p_premature"],
            stacked_df["p_omission"],
            edgecolor="none",
            linewidth=0,
            baseline="zero",
            colors=["#5386b6", "#d57459", "#e8bc60", "#8d5993"],
            labels=["Success", "Error", "Premature", "Omission"],
        )

        # Dashed line for overall success (thin line, no text label)
        try:
            overall_p_success = summary_df.loc[summary_df["name"] == monkey, "p_success"].values[0]
        except IndexError:
            overall_p_success = None

        if overall_p_success is not None:
            ax.axhline(
                y=overall_p_success,
                color="black",
                linestyle=(0, (8, 4)),  # long dash pattern
                linewidth=1.0,
            )

        # Titles = just monkey name
        ax.set_title(
            f"{monkey.capitalize()}",
            fontsize=24,
            fontweight="bold",
            fontname="Times New Roman",
        )

        # Formatting
        ax.set_xlim(1, len(stacked_df))
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- X axis ---
        ax.tick_params(axis="x", labelsize=14)
        if i >= 6:  # last row only
            ax.set_xlabel("Session Count", fontsize=20, fontname="Times New Roman")
        else:
            ax.set_xlabel("")

        # --- Y axis ---
        if i % 3 == 0:  # first col only
            ax.set_ylabel("Stacked Proportion", fontsize=20, fontname="Times New Roman")
            ax.set_yticks([0, 1])  # show 0 and 1
            ax.tick_params(axis="y", labelsize=14)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])  # no ticks or labels on other columns

# Add legend once, at top
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           fontsize=18, prop={"family": "Times New Roman"})

plt.tight_layout(rect=[0.04, 0.04, 1, 0.95], w_pad=2, h_pad=2)
output_dir = os.path.join(base, "plots", "roll-all-over-me-no-hierarchy")
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "monkeys_3x3.svg"), format="svg")
fig.savefig(os.path.join(output_dir, "monkeys_3x3.png"), format="png", dpi=300)

plt.show()
plt.close(fig)

