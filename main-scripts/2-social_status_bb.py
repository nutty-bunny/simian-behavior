# TO BE FIXED -- there are now gaps in the rolling plots...
# UPDATE -- issue fixed, the hierarchy plot does not work though, needs to account for date differences (since sessions do not occur every day)
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod import families
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
import warnings
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF

# ** re-run this script once pickle rick is finalised! **
directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
base = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt"
pickle_rick = os.path.join(directory, "data.pickle")

plt.rcParams["svg.fonttype"] = "none"

with open(pickle_rick, "rb") as handle:
    data = pickle.load(handle)

with open(os.path.join(directory, "tav.pickle"), "rb") as handle:
    tav_dict = pickle.load(handle)

with open(os.path.join(directory, "dms.pickle"), "rb") as handle:
    dms_dict = pickle.load(handle)

# Remove monkeys with RFID issues globally
monkeys_to_remove = {"joy", "jipsy"}
for species in data:
    for name in list(data[species].keys()):
        if name in monkeys_to_remove:
            del data[species][name]

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

def plot_elo(hierarchy_df, title, smooth_window=10, n_individuals=None, max_days=None, highlight_monkeys=None):
    # Extract date and convert to days since first observation
    dates = pd.to_datetime(hierarchy_df["Date"])
    days = (dates - dates.iloc[0]).dt.days
    
    # If max_days is given, filter the DataFrame
    if max_days is not None:
        mask = days <= max_days
        hierarchy_df = hierarchy_df.loc[mask]
        days = days[mask]
    
    # Drop the Date column
    elo_df = hierarchy_df.drop(columns=["Date"])
    
    # Remove monkeys with no data
    elo_df = elo_df.loc[:, elo_df.notna().any()]
    
    # Select top N individuals with most data
    if n_individuals is not None:
        counts = elo_df.notna().sum().sort_values(ascending=False)
        top_monkeys = counts.head(n_individuals).index
        elo_df = elo_df[top_monkeys]
    
    # Apply rolling average smoothing
    elo_df_smooth = elo_df.rolling(window=smooth_window, min_periods=1).mean()
    
    # Ensure highlight_monkeys are valid
    if highlight_monkeys is not None:
        highlight_monkeys = [m for m in highlight_monkeys if m in elo_df_smooth.columns]
    else:
        highlight_monkeys = []
    
    plt.figure(figsize=(12, 7))
    for monkey in elo_df_smooth.columns:
        if elo_df_smooth[monkey].notna().sum() > 1:
            if monkey in highlight_monkeys:
                plt.plot(days, elo_df_smooth[monkey], alpha=1.0, linewidth=3)
            else:
                plt.plot(days, elo_df_smooth[monkey], alpha=0.7, linewidth=1.5)
    
    plt.xlabel("Days of Observation", fontsize=12)  # Reduced from 16
    plt.ylabel("Elo-rating", fontsize=12)          # Reduced from 16
    plt.title(title, fontsize=18, fontweight='bold', pad=20)  # Reduced from 24
    
    # Set custom x-axis ticks every 500 days
    plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12)  # Reduced from 16
    
    # Set y-axis ticks starting at 600, every 200 units
    y_max = plt.ylim()[1]
    y_ticks = range(600, int(y_max//200+1)*200+1, 200)
    plt.yticks(y_ticks, fontsize=12)  # Reduced from 16
    
    # Set y-axis lower bound to 500
    plt.ylim(bottom=500)
    
    # Custom grid - only vertical lines, thicker
    plt.grid(True, axis='x', alpha=0.5, linewidth=1.5)
    
    # Set x-axis range to 2000 days
    plt.xlim(0, 2000)
    
    plt.tight_layout()
    
    # Save as both SVG and PNG to the ELO folder
    save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots/ELO/"
    import os
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    
    filename = "hierarchy_dynamics_tonkean"
    plt.savefig(f"{save_path}{filename}.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}{filename}.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    

plot_elo(
    tonkean_hierarchy,
    "Hierarchy Dynamics in a Subset of Tonkean Macaques",
    smooth_window=100,
    n_individuals=12,
    highlight_monkeys=["lassa", "jeanne", "nereis", "berenice", "patchouli"],
    max_days=None
)

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

def describe_sessions(df, species_name):
    print(f"\nDescriptive statistics for {species_name}:")
    session_counts = df["#_sessions"]
    attempt_counts = df["#_attempts"]

    stats = {
        "sessions_min": session_counts.min(),
        "sessions_max": session_counts.max(),
        "sessions_mean": session_counts.mean(),
        "sessions_sd": session_counts.std(),

        "attempts_min": attempt_counts.min(),
        "attempts_max": attempt_counts.max(),
        "attempts_mean": attempt_counts.mean(),
        "attempts_sd": attempt_counts.std(),
    }

    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")

# Run for Tonkean
describe_sessions(tonkean_df, "Tonkean monkeys")

# Run for Rhesus
describe_sessions(rhesus_df, "Rhesus monkeys")

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

def _to_dt(series, unit_ms=True):
    # Convert to datetime safely
    if unit_ms:
        return pd.to_datetime(series, unit="ms", errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def compute_control_outcomes(control_dict, species, name, start_dt, end_dt):
    """
    Restrict control task attempts (TAV or DMS) to [start_dt, end_dt],
    then compute outcome proportions (like for 5-CSRT).
    Returns a dict with p_success, p_error, p_omission, p_premature.
    """
    try:
        m = control_dict.get(species, {}).get(name)
        if m is None or "attempts" not in m:
            return {"p_success": np.nan, "p_error": np.nan,
                    "p_omission": np.nan, "p_premature": np.nan}

        att = m["attempts"].copy()
        att["instant_begin"] = pd.to_datetime(att["instant_begin"], unit="ms", errors="coerce")
        att["instant_end"]   = pd.to_datetime(att["instant_end"],   unit="ms", errors="coerce")

        # Restrict to the same window
        win = att[(att["instant_begin"] >= start_dt) & (att["instant_end"] <= end_dt)]
        if win.empty:
            return {"p_success": np.nan, "p_error": np.nan,
                    "p_omission": np.nan, "p_premature": np.nan}

        outcome = win["result"].value_counts(normalize=True)
        return {
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),  # use the same column name as in data
        }
    except Exception:
        return {"p_success": np.nan, "p_error": np.nan,
                "p_omission": np.nan, "p_premature": np.nan}


def build_restricted_df(original_df, data_dict, species, tav_dict=None, dms_dict=None):
    restricted_data = []

    for _, row in original_df.iterrows():
        name = row["name"]
        if name not in elo or elo[name].empty:
            continue  # Skip monkeys without ELO data

        attempts = data_dict[species][name]["attempts"].copy()

        # ensure datetime for filtering
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
        attempts["instant_end"]   = pd.to_datetime(attempts["instant_end"],   unit="ms", errors="coerce")

        # Apply ELO window
        elo_start = elo[name]["Date"].min()
        elo_end   = elo[name]["Date"].max()
        attempts = attempts[
            (attempts["instant_begin"] >= elo_start) &
            (attempts["instant_end"]   <= elo_end)
        ]

        # Restrict to first `min_attempts` in chronological order
        attempts = attempts.sort_values("instant_begin").head(min_attempts)
        if attempts.empty:
            continue

        # Define the *restricted* time window for this monkey
        start_dt = attempts["instant_begin"].min().normalize()
        end_dt   = attempts["instant_end"].max().normalize()

        # Main-task metrics (5-CSRT)
        attempt_count = len(attempts)
        session_count = attempts["session"].nunique()
        outcome = attempts["result"].value_counts(normalize=True)
        rt_success = attempts.loc[attempts["result"] == "success", "reaction_time"].mean()
        rt_error   = attempts.loc[attempts["result"] == "error",   "reaction_time"].mean()

        # Control-task scores, aligned to the same restricted window
        # Control-task outcomes
        tav_out = {"p_success": np.nan, "p_error": np.nan,
                "p_omission": np.nan, "p_premature": np.nan}
        dms_out = {"p_success": np.nan, "p_error": np.nan,
                "p_omission": np.nan, "p_premature": np.nan}

        if tav_dict is not None:
            tav_out = compute_control_outcomes(tav_dict, species, name, start_dt, end_dt)

        if dms_dict is not None:
            dms_out = compute_control_outcomes(dms_dict, species, name, start_dt, end_dt)

        restricted_data.append({
            "name": name,
            "#_attempts": attempt_count,
            "#_sessions": session_count,
            "start_date": start_dt.date(),
            "end_date": end_dt.date(),
            "p_success": outcome.get("success", 0.0),
            "p_error": outcome.get("error", 0.0),
            "p_omission": outcome.get("stepomission", 0.0),
            "p_premature": outcome.get("prematured", 0.0),
            "rt_success": rt_success,
            "rt_error": rt_error,
            "mean_sess_duration": attempts["session_duration"].mean(),
            "mean_attempts_per_session": attempts.groupby("session")["id"].count().mean(),
            # append control outcomes
            #"tav_p_success": tav_out["p_success"],
            #"tav_p_error": tav_out["p_error"],
            #"tav_p_omission": tav_out["p_omission"],
            #"tav_p_premature": tav_out["p_premature"],
            #"dms_p_success": dms_out["p_success"],
            #"dms_p_error": dms_out["p_error"],
            #"dms_p_omission": dms_out["p_omission"],
            #"dms_p_premature": dms_out["p_premature"],
        })
    return pd.DataFrame(restricted_data)

# Apply to both species
#tonkean_df_min = build_restricted_df(tonkean_df, data, "tonkean", tav_dict=tav_dict, dms_dict=dms_dict)
#rhesus_df_min  = build_restricted_df(rhesus_df,  data, "rhesus",  tav_dict=tav_dict, dms_dict=dms_dict)

# Apply to both species
tonkean_df_min = build_restricted_df(tonkean_df, data, "tonkean")
rhesus_df_min  = build_restricted_df(rhesus_df,  data, "rhesus")


print("\nTonkean monkeys data (min # attempts):")
print(tonkean_df_min.head())
print("\nRhesus monkeys data (min # attempts):")
print(rhesus_df_min.head())

print("Descriptive stats on FULL datasets:")
describe_sessions(tonkean_df, "Tonkean monkeys")
describe_sessions(rhesus_df, "Rhesus monkeys")

print("\nDescriptive stats on RESTRICTED datasets:")
describe_sessions(tonkean_df_min, "Tonkean monkeys (restricted)")
describe_sessions(rhesus_df_min, "Rhesus monkeys (restricted)")

# the takeaway is that nothing is significant because there's no enough data per subject (at least for the restricted period)
"""
import pandas as pd
from scipy.stats import spearmanr

def test_task_control_correlations(df, control_cols_tav, control_cols_dms, main_cols=None, species_name="all"):
    if main_cols is None:
        main_cols = ['p_success', 'p_error', 'p_omission', 'p_premature']
    
    results = []

    # Loop over main-task metrics and TAV/DMS metrics
    for main_col in main_cols:
        for control_col in control_cols_tav + control_cols_dms:
            # Drop rows with NaN for either metric
            sub_df = df[[main_col, control_col]].dropna()
            if len(sub_df) < 2:
                # Not enough data to correlate
                continue
            
            stat, pval = spearmanr(sub_df[main_col], sub_df[control_col])
            results.append({
                'species': species_name,
                'main_metric': main_col,
                'control_metric': control_col,
                'spearman_r': stat,
                'p_value': pval,
                'n': len(sub_df)
            })
    
    return pd.DataFrame(results)

# Example usage:
control_tav_cols = ['tav_p_success', 'tav_p_error', 'tav_p_omission', 'tav_p_premature']
control_dms_cols = ['dms_p_success', 'dms_p_error', 'dms_p_omission', 'dms_p_premature']

# Tonkean
tonkean_corrs = test_task_control_correlations(tonkean_df_min, control_tav_cols, control_dms_cols, species_name="Tonkean")
# Rhesus
rhesus_corrs = test_task_control_correlations(rhesus_df_min, control_tav_cols, control_dms_cols, species_name="Rhesus")

# Combine all species
all_corrs = pd.concat([tonkean_corrs, rhesus_corrs], ignore_index=True)

print(all_corrs)

"""

# Add this code after the main loop that creates tonkean_df and rhesus_df
# This will add TAV and DMS performance scores aligned to the 5-CSRT timeframe

def add_control_task_scores(df, data_dict, tav_dict, dms_dict, species):
    """
    For each monkey in the dataframe, compute TAV and DMS performance scores
    within the same time window as their 5-CSRT data.
    """
    # Create lists to store control task scores
    tav_success = []
    tav_error = []
    tav_omission = []
    tav_premature = []
    tav_attempts = []
    
    dms_success = []
    dms_error = []
    dms_omission = []
    dms_premature = []
    dms_attempts = []
    
    for _, row in df.iterrows():
        name = row["name"]
        start_date = pd.to_datetime(row["start_date"])
        end_date = pd.to_datetime(row["end_date"])
        
        # --- TAV SCORES ---
        tav_data = tav_dict.get(species, {}).get(name)
        if tav_data is not None and "attempts" in tav_data:
            tav_attempts_df = tav_data["attempts"].copy()
            tav_attempts_df["instant_begin"] = pd.to_datetime(
                tav_attempts_df["instant_begin"], unit="ms", errors="coerce"
            )
            tav_attempts_df["instant_end"] = pd.to_datetime(
                tav_attempts_df["instant_end"], unit="ms", errors="coerce"
            )
            
            # Filter to 5-CSRT time window
            tav_filtered = tav_attempts_df[
                (tav_attempts_df["instant_begin"] >= start_date) &
                (tav_attempts_df["instant_end"] <= end_date)
            ]
            
            if not tav_filtered.empty:
                tav_outcome = tav_filtered["result"].value_counts(normalize=True)
                tav_success.append(tav_outcome.get("success", 0.0))
                tav_error.append(tav_outcome.get("error", 0.0))
                tav_omission.append(tav_outcome.get("stepomission", 0.0))
                tav_premature.append(tav_outcome.get("prematured", 0.0))
                tav_attempts.append(len(tav_filtered))
            else:
                tav_success.append(np.nan)
                tav_error.append(np.nan)
                tav_omission.append(np.nan)
                tav_premature.append(np.nan)
                tav_attempts.append(0)
        else:
            tav_success.append(np.nan)
            tav_error.append(np.nan)
            tav_omission.append(np.nan)
            tav_premature.append(np.nan)
            tav_attempts.append(0)
        
        # --- DMS SCORES ---
        dms_data = dms_dict.get(species, {}).get(name)
        if dms_data is not None and "attempts" in dms_data:
            dms_attempts_df = dms_data["attempts"].copy()
            dms_attempts_df["instant_begin"] = pd.to_datetime(
                dms_attempts_df["instant_begin"], unit="ms", errors="coerce"
            )
            dms_attempts_df["instant_end"] = pd.to_datetime(
                dms_attempts_df["instant_end"], unit="ms", errors="coerce"
            )
            
            # Filter to 5-CSRT time window
            dms_filtered = dms_attempts_df[
                (dms_attempts_df["instant_begin"] >= start_date) &
                (dms_attempts_df["instant_end"] <= end_date)
            ]
            
            if not dms_filtered.empty:
                dms_outcome = dms_filtered["result"].value_counts(normalize=True)
                dms_success.append(dms_outcome.get("success", 0.0))
                dms_error.append(dms_outcome.get("error", 0.0))
                dms_omission.append(dms_outcome.get("stepomission", 0.0))
                dms_premature.append(dms_outcome.get("prematured", 0.0))
                dms_attempts.append(len(dms_filtered))
            else:
                dms_success.append(np.nan)
                dms_error.append(np.nan)
                dms_omission.append(np.nan)
                dms_premature.append(np.nan)
                dms_attempts.append(0)
        else:
            dms_success.append(np.nan)
            dms_error.append(np.nan)
            dms_omission.append(np.nan)
            dms_premature.append(np.nan)
            dms_attempts.append(0)
    
    # Add all columns to dataframe
    df["tav_p_success"] = tav_success
    df["tav_p_error"] = tav_error
    df["tav_p_omission"] = tav_omission
    df["tav_p_premature"] = tav_premature
    df["tav_attempts"] = tav_attempts
    
    df["dms_p_success"] = dms_success
    df["dms_p_error"] = dms_error
    df["dms_p_omission"] = dms_omission
    df["dms_p_premature"] = dms_premature
    df["dms_attempts"] = dms_attempts
    
    return df


# Apply to both species dataframes
tonkean_df = add_control_task_scores(tonkean_df, data, tav_dict, dms_dict, "tonkean")
rhesus_df = add_control_task_scores(rhesus_df, data, tav_dict, dms_dict, "rhesus")

print("\nTonkean with TAV/DMS scores:")
print(tonkean_df[["name", "p_success", "tav_p_success", "dms_p_success", "tav_attempts", "dms_attempts"]])

print("\nRhesus with TAV/DMS scores:")
print(rhesus_df[["name", "p_success", "tav_p_success", "dms_p_success", "tav_attempts", "dms_attempts"]])

# Combine both species for correlation analysis
all_monkeys_df = pd.concat([tonkean_df, rhesus_df], ignore_index=True)

# Summary statistics for reporting
print("\n" + "="*70)
print("CONTROL TASK ANALYSIS: TEMPORAL ALIGNMENT")
print("="*70)

# Calculate overlap statistics
tav_overlaps = []
dms_overlaps = []

for _, row in all_monkeys_df.iterrows():
    if row['tav_attempts'] > 0:
        # Calculate days of overlap (end_date - start_date)
        overlap_days = (pd.to_datetime(row['end_date']) - pd.to_datetime(row['start_date'])).days
        tav_overlaps.append(overlap_days)
    
    if row['dms_attempts'] > 0:
        overlap_days = (pd.to_datetime(row['end_date']) - pd.to_datetime(row['start_date'])).days
        dms_overlaps.append(overlap_days)

# Print summary
n_tav = all_monkeys_df['tav_p_success'].notna().sum()
n_dms = all_monkeys_df['dms_p_success'].notna().sum()
n_total = len(all_monkeys_df)

print(f"\n5-CSRT vs TAV:")
print(f"  - Individuals with overlapping data: {n_tav}/{n_total}")
print(f"  - Mean overlap: {np.mean(tav_overlaps):.0f} days (SD: {np.std(tav_overlaps):.0f})")
print(f"  - Range: {np.min(tav_overlaps):.0f} - {np.max(tav_overlaps):.0f} days")
print(f"  - Total TAV attempts during overlap: {all_monkeys_df['tav_attempts'].sum():.0f}")

print(f"\n5-CSRT vs DMS:")
print(f"  - Individuals with overlapping data: {n_dms}/{n_total}")
print(f"  - Mean overlap: {np.mean(dms_overlaps):.0f} days (SD: {np.std(dms_overlaps):.0f})")
print(f"  - Range: {np.min(dms_overlaps):.0f} - {np.max(dms_overlaps):.0f} days")
print(f"  - Total DMS attempts during overlap: {all_monkeys_df['dms_attempts'].sum():.0f}")

# Check correlations between 5-CSRT and control tasks (across both species)
print("\n" + "="*70)
print("CORRELATION MATRICES")
print("="*70)

print("\n5-CSRT vs TAV (all performance metrics):")
print("-" * 70)
tav_cols = ["p_success", "p_error", "p_omission", "p_premature", 
            "tav_p_success", "tav_p_error", "tav_p_omission", "tav_p_premature"]
tav_corr = all_monkeys_df[tav_cols].corr()
print(tav_corr.to_string())

print("\n\n5-CSRT vs DMS (all performance metrics):")
print("-" * 70)
dms_cols = ["p_success", "p_error", "p_omission", "p_premature",
            "dms_p_success", "dms_p_error", "dms_p_omission", "dms_p_premature"]
dms_corr = all_monkeys_df[dms_cols].corr()
print(dms_corr.to_string())

# Highlight key cross-task correlations
print("\n" + "="*70)
print("KEY CROSS-TASK CORRELATIONS")
print("="*70)

from scipy.stats import pearsonr

# TAV correlations with stats
print("\n5-CSRT → TAV:")
for csrt_col, tav_col, label in [
    ('p_success', 'tav_p_success', 'Success'),
    ('p_error', 'tav_p_error', 'Error'),
    ('p_omission', 'tav_p_omission', 'Omission'),
    ('p_premature', 'tav_p_premature', 'Premature')
]:
    # Remove NaN pairs
    valid_data = all_monkeys_df[[csrt_col, tav_col]].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data[csrt_col], valid_data[tav_col])
        n = len(valid_data)
        print(f"  {label:10s}  r = {r:6.3f}, p = {p:.4f}, n = {n}")
    else:
        print(f"  {label:10s}  insufficient data")

# DMS correlations with stats
print("\n5-CSRT → DMS:")
for csrt_col, dms_col, label in [
    ('p_success', 'dms_p_success', 'Success'),
    ('p_error', 'dms_p_error', 'Error'),
    ('p_omission', 'dms_p_omission', 'Omission'),
    ('p_premature', 'dms_p_premature', 'Premature')
]:
    # Remove NaN pairs
    valid_data = all_monkeys_df[[csrt_col, dms_col]].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data[csrt_col], valid_data[dms_col])
        n = len(valid_data)
        print(f"  {label:10s}  r = {r:6.3f}, p = {p:.4f}, n = {n}")
    else:
        print(f"  {label:10s}  insufficient data")

print("\n" + "="*70)
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

for df in [tonkean_df, rhesus_df]:
    df['elo_mean'] = df.apply(
        lambda row: calculate_elo_mean(row['name'], row['start_date'], row['end_date']),
        axis=1
    )

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
    """Consistently merge demographics with behavior data"""
    # Handle empty DataFrames
    if behavior_df.empty:
        print(f"\nDebugging {species_name} merge:")
        print(f"Behavior DataFrame is empty - returning empty DataFrame")
        return behavior_df.copy()
    
    # Check if 'name' column exists
    if 'name' not in behavior_df.columns:
        print(f"\nDebugging {species_name} merge:")
        print(f"No 'name' column in behavior DataFrame - returning original DataFrame")
        return behavior_df.copy()
    
    # Check if demographic columns already exist
    has_gender = 'gender' in behavior_df.columns
    has_age = 'age' in behavior_df.columns
    
    if has_gender and has_age:
        print(f"\nDebugging {species_name} merge:")
        print(f"Demographics already present in behavior DataFrame - returning unchanged")
        return behavior_df.copy()
    
    # Create clean name columns for merging
    behavior_df_copy = behavior_df.copy()
    demo_df_copy = demo_df.copy()
    
    behavior_df_copy['merge_name'] = behavior_df_copy['name'].str.lower().str.strip()
    demo_df_copy['merge_name'] = demo_df_copy['name'].str.lower().str.strip()
    
    # Print debug info
    print(f"\nDebugging {species_name} merge:")
    print(f"Behavior DataFrame: {len(behavior_df_copy)} rows")
    print(f"Behavior names sample: {behavior_df_copy['merge_name'].head(3).tolist()}")
    print(f"Demographic names sample: {demo_df_copy['merge_name'].head(3).tolist()}")
    
    # Check which demographic columns are available and needed
    available_cols = ['merge_name']
    if 'gender' in demo_df_copy.columns and not has_gender:
        available_cols.append('gender')
    if 'age' in demo_df_copy.columns and not has_age:
        available_cols.append('age')
    
    print(f"Available demographic columns to add: {available_cols[1:] if len(available_cols) > 1 else 'none'}")
    
    # Perform merge only if demographic data exists and is needed
    if len(available_cols) > 1:
        # Check for overlapping names before merge
        behavior_names = set(behavior_df_copy['merge_name'])
        demo_names = set(demo_df_copy['merge_name'])
        overlap = behavior_names.intersection(demo_names)
        print(f"Names that will be matched: {len(overlap)}/{len(behavior_names)} ({list(overlap)[:5]}{'...' if len(overlap) > 5 else ''})")
        
        merged = behavior_df_copy.merge(
            demo_df_copy[available_cols],
            on='merge_name',
            how='left'
        )
        
        # Check if merge was successful
        if 'gender' in available_cols:
            if 'gender' in merged.columns:
                print(f"Merged gender values: {merged['gender'].unique()}")
                print(f"Missing genders: {merged['gender'].isna().sum()}/{len(merged)}")
            else:
                print("Error: Gender column not found after merge")
        
        if 'age' in available_cols:
            if 'age' in merged.columns:
                print(f"Age data: {merged['age'].notna().sum()}/{len(merged)} individuals have age data")
            else:
                print("Error: Age column not found after merge")
    else:
        merged = behavior_df_copy.copy()
        print("No demographic columns to merge or all already present")
    
    # Clean up
    if 'merge_name' in merged.columns:
        merged = merged.drop(columns='merge_name')
    
    return merged

def clean_duplicate_demo_columns(df):
    """Remove duplicate demographic columns created by multiple merges"""
    if df.empty:
        return df
    
    # Keep only the final gender and age columns, remove duplicates
    cols_to_drop = []
    for col in df.columns:
        if col in ['gender_x', 'gender_y', 'age_x', 'age_y']:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"  Removing duplicate columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df

# Clean and merge demographics for original datasets
if "mkid_tonkean" in demographics:
    print("\nMerging Tonkean demographics...")
    tonkean_df = clean_duplicate_demo_columns(tonkean_df)
    tonkean_df_min = clean_duplicate_demo_columns(tonkean_df_min)
    tonkean_df = safe_merge(tonkean_df, demographics["mkid_tonkean"], "Tonkean (original)")
    tonkean_df_min = safe_merge(tonkean_df_min, demographics["mkid_tonkean"], "Tonkean (trimmed)")
else:
    print("Warning: 'mkid_tonkean' not found in demographics")

if "mkid_rhesus" in demographics:
    print("\nMerging Rhesus demographics...")
    rhesus_df = clean_duplicate_demo_columns(rhesus_df)
    rhesus_df_min = clean_duplicate_demo_columns(rhesus_df_min)
    rhesus_df = safe_merge(rhesus_df, demographics["mkid_rhesus"], "Rhesus (original)")
    rhesus_df_min = safe_merge(rhesus_df_min, demographics["mkid_rhesus"], "Rhesus (trimmed)")
else:
    print("Warning: 'mkid_rhesus' not found in demographics")

# Save original datasets to standard derivatives directory
derivatives_dir = Path(base) / "derivatives"
derivatives_dir.mkdir(parents=True, exist_ok=True)

# Save original four tables
original_tables = {
    "tonkean-elo": tonkean_df,
    "tonkean-elo-min-attempts": tonkean_df_min,
    "rhesus-elo": rhesus_df,
    "rhesus-elo-min-attempts": rhesus_df_min
}

for name, df in original_tables.items():
    df.to_csv(derivatives_dir / f"{name}.csv", index=False)
    df.to_pickle(derivatives_dir / f"{name}.pkl")

print(f"Saved original files to {derivatives_dir}:")
print(*[f"- {name}.csv/.pkl" for name in original_tables.keys()], sep="\n")

# ============================================================================
# SEPARATE SECTION: CREATE TOP 13 DATASETS FROM COMBINED SAMPLE
# ============================================================================

print("\n" + "="*60)
print("CREATING TOP 13 INDIVIDUAL DATASETS (FROM COMBINED SAMPLE)")
print("="*60)

def create_top_13_from_combined():
    """Create datasets for top 13 individuals by attempt count from COMBINED sample"""
    
    # Combine both species datasets and add species identifier
    tonkean_with_species = tonkean_df.copy()
    tonkean_with_species['species'] = 'tonkean'
    
    rhesus_with_species = rhesus_df.copy()
    rhesus_with_species['species'] = 'rhesus'
    
    # Combine datasets
    combined_df = pd.concat([tonkean_with_species, rhesus_with_species], ignore_index=True)
    
    print(f"Combined dataset: {len(combined_df)} individuals")
    print(f"  - Tonkean: {len(tonkean_with_species)}")
    print(f"  - Rhesus: {len(rhesus_with_species)}")
    
    # Get top 13 from combined sample
    top_13_combined = combined_df.nlargest(13, '#_attempts')
    
    print(f"\nTop 13 individuals from combined sample:")
    for _, row in top_13_combined.iterrows():
        print(f"  {row['name']} ({row['species']}): {row['#_attempts']} attempts")
    
    # Split back into species
    top_13_tonkean = top_13_combined[top_13_combined['species'] == 'tonkean'].drop('species', axis=1)
    top_13_rhesus = top_13_combined[top_13_combined['species'] == 'rhesus'].drop('species', axis=1)
    
    print(f"\nAfter splitting:")
    print(f"  - Tonkean individuals in top 13: {len(top_13_tonkean)}")
    print(f"  - Rhesus individuals in top 13: {len(top_13_rhesus)}")
    
    # Get minimum attempts among the top 13 (from combined sample)
    min_attempts_top13 = top_13_combined['#_attempts'].min()
    print(f"\nMinimum attempts among top 13 (combined): {min_attempts_top13}")
    
    # Create min-attempts versions
    def create_min_attempts_version(species_df, species_name, species_key):
        min_data = []
        
        for _, row in species_df.iterrows():
            name = row["name"]
            if name not in elo or elo[name].empty:
                continue
                
            # Get attempts data
            attempts = data[species_key][name]["attempts"].copy()
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
            
            # Apply ELO window
            elo_start = elo[name]["Date"].min()
            elo_end = elo[name]["Date"].max()
            attempts = attempts[
                (attempts["instant_begin"] >= elo_start) &
                (attempts["instant_end"] <= elo_end)
            ]
            
            # Restrict to first min_attempts_top13 in chronological order
            attempts = attempts.sort_values("instant_begin").head(min_attempts_top13)
            if attempts.empty:
                continue
                
            # Calculate metrics
            start_dt = attempts["instant_begin"].min().normalize()
            end_dt = attempts["instant_end"].max().normalize()
            attempt_count = len(attempts)
            session_count = attempts["session"].nunique()
            outcome = attempts["result"].value_counts(normalize=True)
            rt_success = attempts.loc[attempts["result"] == "success", "reaction_time"].mean()
            rt_error = attempts.loc[attempts["result"] == "error", "reaction_time"].mean()
            
            # Calculate ELO mean for this restricted period
            elo_mean = calculate_elo_mean(name, start_dt.date(), end_dt.date())
            
            min_data.append({
                "name": name,
                "#_attempts": attempt_count,
                "#_sessions": session_count,
                "start_date": start_dt.date(),
                "end_date": end_dt.date(),
                "p_success": outcome.get("success", 0.0),
                "p_error": outcome.get("error", 0.0),
                "p_omission": outcome.get("stepomission", 0.0),
                "p_premature": outcome.get("prematured", 0.0),
                "rt_success": rt_success,
                "rt_error": rt_error,
                "mean_sess_duration": attempts["session_duration"].mean(),
                "mean_attempts_per_session": attempts.groupby("session")["id"].count().mean(),
                "elo_mean": elo_mean
            })
        
        return pd.DataFrame(min_data)
    
    # Create min-attempts versions for both species
    tonkean_top13_min = create_min_attempts_version(top_13_tonkean, "Tonkean", "tonkean")
    rhesus_top13_min = create_min_attempts_version(top_13_rhesus, "Rhesus", "rhesus")
    
    return top_13_tonkean, tonkean_top13_min, top_13_rhesus, rhesus_top13_min

# Create the datasets
tonkean_elo_13, tonkean_min_attempts_13, rhesus_elo_13, rhesus_min_attempts_13 = create_top_13_from_combined()

# Apply demographics consistently to ALL top 13 datasets
print(f"\n" + "="*50)
print("ADDING DEMOGRAPHICS TO TOP 13 DATASETS")
print("="*50)

def clean_duplicate_demo_columns(df):
    """Remove duplicate demographic columns created by multiple merges"""
    if df.empty:
        return df
    
    # Keep only the final gender and age columns, remove duplicates
    cols_to_drop = []
    for col in df.columns:
        if col in ['gender_x', 'gender_y', 'age_x', 'age_y']:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"  Removing duplicate columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df

# Clean and add demographics to ALL tonkean top 13 datasets
if "mkid_tonkean" in demographics:
    print("Processing Tonkean top 13 datasets...")
    
    # Clean any existing duplicate columns first
    tonkean_elo_13 = clean_duplicate_demo_columns(tonkean_elo_13)
    tonkean_min_attempts_13 = clean_duplicate_demo_columns(tonkean_min_attempts_13)
    
    # Apply demographics
    tonkean_elo_13 = safe_merge(tonkean_elo_13, demographics["mkid_tonkean"], "Tonkean top 13")
    tonkean_min_attempts_13 = safe_merge(tonkean_min_attempts_13, demographics["mkid_tonkean"], "Tonkean top 13 min-attempts")
else:
    print("Warning: 'mkid_tonkean' not found in demographics for top 13 datasets")

# Clean and add demographics to ALL rhesus top 13 datasets (even if empty)
if "mkid_rhesus" in demographics:
    print("Processing Rhesus top 13 datasets...")
    
    # Clean any existing duplicate columns first
    rhesus_elo_13 = clean_duplicate_demo_columns(rhesus_elo_13)
    rhesus_min_attempts_13 = clean_duplicate_demo_columns(rhesus_min_attempts_13)
    
    # Apply demographics (safe_merge handles empty dataframes)
    rhesus_elo_13 = safe_merge(rhesus_elo_13, demographics["mkid_rhesus"], "Rhesus top 13")
    rhesus_min_attempts_13 = safe_merge(rhesus_min_attempts_13, demographics["mkid_rhesus"], "Rhesus top 13 min-attempts")
else:
    print("Warning: 'mkid_rhesus' not found in demographics for top 13 datasets")

# Create separate directory for top 13 datasets
top13_dir = Path(base) / "derivatives" / "13-individual"
top13_dir.mkdir(parents=True, exist_ok=True)

# Create top 13 tables dictionary
top13_tables = {
    "tonkean-elo-13": tonkean_elo_13,
    "tonkean-elo-min-attempts-13": tonkean_min_attempts_13,
    "rhesus-elo-13": rhesus_elo_13,
    "rhesus-elo-min-attempts-13": rhesus_min_attempts_13
}

# Save top 13 datasets to separate directory
print(f"\n" + "="*50)
print("SAVING TOP 13 DATASETS")
print("="*50)

for name, df in top13_tables.items():
    df.to_csv(top13_dir / f"{name}.csv", index=False)
    df.to_pickle(top13_dir / f"{name}.pkl")
    print(f"✓ Saved {name}: {df.shape[0]} individuals, {df.shape[1]} variables")

print(f"\nTop 13 datasets saved to: {top13_dir}")

# Print comprehensive summary statistics
print(f"\n" + "="*60)
print("COMPLETE DATASET SUMMARY")
print("="*60)

print("\nORIGINAL DATASETS (saved to derivatives/):")
for name, df in original_tables.items():
    if df.empty:
        print(f"  {name}: EMPTY")
    else:
        attempts_info = f"attempts: {df['#_attempts'].min()}-{df['#_attempts'].max()}" if '#_attempts' in df.columns else "no attempt data"
        demo_info = ""
        if 'gender' in df.columns:
            demo_info += f", gender: {df['gender'].notna().sum()}/{len(df)}"
        if 'age' in df.columns:
            demo_info += f", age: {df['age'].notna().sum()}/{len(df)}"
        print(f"  {name}: {len(df)} individuals, {attempts_info}{demo_info}")

print("\nTOP 13 DATASETS (saved to derivatives/13-individual/):")
for name, df in top13_tables.items():
    if df.empty:
        print(f"  {name}: EMPTY")
    else:
        attempts_info = f"attempts: {df['#_attempts'].min()}-{df['#_attempts'].max()}" if '#_attempts' in df.columns else "no attempt data"
        demo_info = ""
        if 'gender' in df.columns:
            demo_info += f", gender: {df['gender'].notna().sum()}/{len(df)}"
        if 'age' in df.columns:
            demo_info += f", age: {df['age'].notna().sum()}/{len(df)}"
        print(f"  {name}: {len(df)} individuals, {attempts_info}{demo_info}")


"""
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
"""

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

# Step 1: Reconstruct `all_elo` if needed
if not isinstance(all_elo, pd.DataFrame):
    elo_list = []
    for species in ["tonkean", "rhesus"]:
        hierarchy = tonkean_hierarchy if species == "tonkean" else rhesus_hierarchy
        for col in hierarchy.columns[1:]:
            temp = hierarchy[['Date', col]].dropna()
            temp['name'] = col
            temp = temp.rename(columns={col: 'elo_score'})
            elo_list.append(temp)
    all_elo = pd.concat(elo_list)

# Step 2: Build dataset using all attempts within ELO time span (no minimum filtering)
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
        first_elo, last_elo = monkey_elo['Date'].min(), monkey_elo['Date'].max()

        mask = (attempts['instant_begin'] >= first_elo) & (attempts['instant_end'] <= last_elo)
        filtered = attempts[mask].sort_values('instant_begin')

        if len(filtered) > 0:
            attempts_list.append(filtered)

final_data = pd.concat(attempts_list)

# Step 3: Merge ELO score based on nearest timestamp
final_data = pd.merge_asof(
    final_data.sort_values('instant_begin'),
    all_elo.sort_values('Date')[['name', 'Date', 'elo_score']],
    left_on='instant_begin',
    right_on='Date',
    by='name',
    direction='nearest'
)

# Step 4: Merge demographic data
demographics = pd.concat([
    demographics["mkid_tonkean"].assign(species="tonkean"),
    demographics["mkid_rhesus"].assign(species="rhesus")
])
final_data = final_data.merge(
    demographics[['name', 'age', 'gender']],
    on='name',
    how='left'
)

# Step 5: Create binary columns for each result type
result_types = ['success', 'error', 'premature', 'stepomission']
for res in result_types:
    final_data[res] = (final_data['result'] == res).astype(int)

# Step 6 (optional): Standardize elo_score within species
final_data['elo_score_z'] = final_data.groupby('species')['elo_score'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Step 7: Prepare categorical variables
final_data['species'] = pd.Categorical(final_data['species'])
final_data['gender'] = pd.Categorical(final_data['gender'])

# --- New Step 7a: Create individual-day grouping variable for clustering ---
final_data['date_only'] = final_data['instant_begin'].dt.date
final_data['ind_day'] = final_data['name'].astype(str) + '_' + final_data['date_only'].astype(str)

### PLOTTING OF ROLLING AVERAGES
# --- Setup ---
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
"""
# with hierarchy
for species, content in species_data.items():
    proportions_df = content["proportions_df"]
    summary_df = content["summary_df"]

    if proportions_df.empty:
        print(f"No data for {species}")
        continue

    unique_monkeys = proportions_df["monkey"].unique()

    for monkey in unique_monkeys:
        monkey_df = proportions_df[proportions_df["monkey"] == monkey].sort_values("session").reset_index(drop=True)
        num_sessions = len(monkey_df)

        # Determine rolling window size
        if num_sessions > 5000:
            rolling_window = 1
        elif num_sessions < 500:
            rolling_window = 1
        else:
            rolling_window = 1

        # Apply rolling smoothing to each p_* column
        for col in ["p_success", "p_error", "p_omission", "p_premature"]:
            monkey_df[f"{col}_smoothed"] = monkey_df[col].rolling(rolling_window, center=True).mean()

        # Drop rows with any NaNs (edges from rolling)
        smoothed_df = monkey_df[[
            "p_success_smoothed", "p_error_smoothed", "p_premature_smoothed", "p_omission_smoothed"
        ]].dropna()

        # Normalize the smoothed proportions to sum to 1
        smoothed_df = smoothed_df.div(smoothed_df.sum(axis=1), axis=0)

        # Check for Elo data
        if monkey not in elo:
            print(f"[{species}] No ELO data for {monkey}")
            continue

        hierarchy_monkey_df = elo[monkey]
        elo_series = hierarchy_monkey_df[monkey].rolling(rolling_window, min_periods=1).mean()

        # --- Plotting ---
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # Top plot: Stacked smoothed proportions
        ax1 = fig.add_subplot(gs[0])
        ax1.stackplot(
            range(1, len(smoothed_df) + 1),
            smoothed_df["p_success_smoothed"],
            smoothed_df["p_error_smoothed"],
            smoothed_df["p_premature_smoothed"],
            smoothed_df["p_omission_smoothed"],
            colors=["#5386b6", "#d57459", "#e8bc60", "#8d5993"],
            edgecolor="none",  # Prevents border lines between stacks
            linewidth=0,  # No line width on stack borders
            baseline="zero",  # Stable baseline to avoid float rounding issues
            labels=["Success", "Error", "Premature", "Stepomission"],
            antialiased=False  # <-- This helps eliminate the white lines
        )

        # Dashed line for overall success
        try:
            overall_p_success = summary_df.loc[summary_df["name"] == monkey, "p_success"].values[0]
        except IndexError:
            overall_p_success = None

        if overall_p_success is not None:
            ax1.axhline(y=overall_p_success, color="black", linestyle=(0, (5, 10)), linewidth=1.5)
            ax1.text(
                len(smoothed_df) + 7,
                overall_p_success,
                f"{overall_p_success:.2f}",
                color="black",
                fontsize=14,
                fontname="Times New Roman",
                verticalalignment="center",
            )

        ax1.set_ylabel("Stacked Proportion", fontsize=14, fontname="Times New Roman")
        ax1.set_xlim(1, len(smoothed_df))
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

        # Bottom plot: Smoothed Elo
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

"""
"""
# Without hierarchy 
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
            edgecolor="none",  # Prevents border lines between stacks
            linewidth=0,  # No line width on stack borders
            baseline="zero",  # Stable baseline to avoid float rounding issues
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

        # Uncomment to save figures
        output_dir = os.path.join(base, "plots", "roll-all-over-me-no-hierarchy")
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.svg"), format="svg")
        fig.savefig(os.path.join(output_dir, f"{monkey}_{species.lower()}.png"), format="png", dpi=300)

        plt.show()
        plt.close(fig)
"""

# Monkeys you want to plot in a 3x3 grid
selected_monkeys = ["barnabe", "olli", "alaryc", "olaf", "wotan", "lassa", "walt", "nereis", "olga"]

# Configure rolling window size
rolling_window = 10

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
        
        # Apply rolling window (now configurable)
        stacked_df = stacked_df.rolling(window=rolling_window, min_periods=1, center=True).mean()
        
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
fig.savefig(os.path.join(output_dir, "rolling-plots-2x3.svg"), format="svg")
fig.savefig(os.path.join(output_dir, "rolling-plots-2x3.png"), format="png", dpi=300)

plt.show()
plt.close(fig)

# Two monkeys you want to compare
selected_monkeys = ["spliff", "olli"]  # Change these to your preferred monkeys
rolling_windows = [1, 10, 20, 30]

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)

for row, monkey in enumerate(selected_monkeys):
    # Loop through species to find monkey data
    for species, content in species_data.items():
        proportions_df = content["proportions_df"]
        summary_df = content["summary_df"]

        if proportions_df.empty or monkey not in proportions_df["monkey"].unique():
            continue

        monkey_df = proportions_df[proportions_df["monkey"] == monkey]
        monkey_df = monkey_df.sort_values("session").reset_index(drop=True)

        # Raw session-wise proportions
        stacked_df = monkey_df[
            ["p_success", "p_error", "p_premature", "p_omission"]
        ].dropna()
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

        for col, window in enumerate(rolling_windows):
            ax = axes[row, col]

            # Apply rolling window
            if window == 1:
                windowed_df = stacked_df  # No smoothing
            else:
                windowed_df = stacked_df.rolling(
                    window=window, min_periods=1, center=True
                ).mean()

            # Stacked proportions plot
            ax.stackplot(
                range(1, len(windowed_df) + 1),
                windowed_df["p_success"],
                windowed_df["p_error"],
                windowed_df["p_premature"],
                windowed_df["p_omission"],
                edgecolor="none",
                linewidth=0,
                baseline="zero",
                colors=["#5386b6", "#d57459", "#e8bc60", "#8d5993"],
                labels=["Success", "Error", "Premature", "Omission"],
            )

            # Dashed line for overall success
            try:
                overall_p_success = summary_df.loc[
                    summary_df["name"] == monkey, "p_success"
                ].values[0]
            except IndexError:
                overall_p_success = None

            if overall_p_success is not None:
                ax.axhline(
                    y=overall_p_success,
                    color="black",
                    linestyle=(0, (8, 4)),  # long dash pattern
                    linewidth=1.0,
                )

            # Titles
            if row == 0:  # Top row gets rolling window info
                ax.set_title(
                    f"Rolling Window: {window}",
                    fontsize=18,
                    fontweight="bold",
                    fontname="Times New Roman",
                )

            # Formatting
            ax.set_xlim(1, len(windowed_df))
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.5)

            # X axis - only bottom row
            if row == 1:  # bottom row only
                ax.set_xlabel("Session Count", fontsize=16, fontname="Times New Roman")
                ax.tick_params(axis="x", labelsize=12)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelsize=12)

            # Y axis - only first column gets labels
            if col == 0:  # first column only
                if row == 0:
                    ax.set_ylabel(
                        f"{monkey.capitalize()}\n\nStacked Proportion",
                        fontsize=16,
                        fontname="Times New Roman",
                    )
                else:
                    ax.set_ylabel(
                        f"{monkey.capitalize()}\n\nStacked Proportion",
                        fontsize=16,
                        fontname="Times New Roman",
                    )
                ax.set_yticks([0, 0.5, 1])
                ax.tick_params(axis="y", labelsize=12)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

# Add legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    fontsize=16,
    prop={"family": "Times New Roman"},
)

plt.tight_layout(rect=[0.04, 0.04, 1, 0.95], w_pad=1, h_pad=2)
plt.show()
plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_improved_data_retention_plots(combined_df, min_attempts_threshold=1651, show_plot=False):
    """
    Create improved data retention visualization with species-specific coloring
    Works with your actual data structure
    
    Parameters:
    - combined_df: DataFrame with your data
    - min_attempts_threshold: Minimum trial threshold (default 1651)
    - show_plot: Whether to display the plot (default False, only saves)
    """
    
    # Create output directory
    output_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots/effect-of-restriction"
    os.makedirs(output_dir, exist_ok=True)
    
    # Species-specific colors - your established palette
    tonkean_color = "#8B4A9C"  # Purple
    rhesus_color = "#4682B4"   # Steel blue
    lost_color = "#E8E8E8"     # Light gray for lost data
    
    # Create figure with 2x2 subplots - optimized spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # Main title with proper spacing
    fig.suptitle('Impact of Minimum Trial Requirements on Dataset Composition', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Sort by total attempts (using your column name)
    df_by_attempts = combined_df.sort_values('#_attempts')
    
    # Debug species mapping - let's see what's actually in the data
    print("Species values in data:", df_by_attempts['species'].unique())
    print("Species counts:", df_by_attempts['species'].value_counts())
    
    # Create species colors - simplified without alpha for debugging
    colors = []
    for idx, row in df_by_attempts.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:  # More flexible matching
            colors.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors.append(rhesus_color)
        else:
            colors.append('#808080')  # Gray for unknown
            print(f"Unknown species: {row['species']} for {row['name']}")
    
    print(f"Color assignments: {len([c for c in colors if c == tonkean_color])} tonkean, {len([c for c in colors if c == rhesus_color])} rhesus")
    
    # Plot 1: Before standardization (by attempts)
    bars1 = ax1.bar(range(len(df_by_attempts)), df_by_attempts['#_attempts'], 
                    color=colors, width=0.8, alpha=0.8)
    
    ax1.set_title('Before Standardization\n(Ordered by Total Trials)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Individual subjects (ordered by number of attempts)', fontsize=11)
    ax1.set_ylabel('Number of Trials', fontsize=11)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.locator_params(axis='y', nbins=6)
    
    # Add legend for species
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tonkean_color, label=r'$\mathit{M.\ tonkeana}$'),
                      Patch(facecolor=rhesus_color, label=r'$\mathit{M.\ mulatta}$')]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Add summary stats
    total_trials = df_by_attempts['#_attempts'].sum()
    n_subjects = len(df_by_attempts)
    ax1.text(0.02, 0.85, f'Total trials: {total_trials:,}\nSubjects: {n_subjects}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: After standardization (by attempts)
    kept_data = np.minimum(df_by_attempts['#_attempts'], min_attempts_threshold)
    lost_data = df_by_attempts['#_attempts'] - kept_data
    
    bars2 = ax2.bar(range(len(df_by_attempts)), kept_data, 
                    color=colors, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices = np.where(lost_data > 0)[0]
    if len(lost_indices) > 0:
        bars3 = ax2.bar(lost_indices, lost_data[lost_indices], 
                        bottom=kept_data[lost_indices],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax2.set_title(f'After Standardization to {min_attempts_threshold:,} Trials\n(Same Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Individual subjects (ordered by number of attempts)', fontsize=11)
    ax2.set_ylabel('Number of Trials', fontsize=11)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.locator_params(axis='y', nbins=6)
    
    # Legend combining species and retention status
    legend_elements2 = [Patch(facecolor=tonkean_color, label=r'$\mathit{M.\ tonkeana}$ (retained)'),
                       Patch(facecolor=rhesus_color, label=r'$\mathit{M.\ mulatta}$ (retained)'),
                       Patch(facecolor=lost_color, label='Excluded trials')]
    ax2.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Calculate retention stats
    total_kept = kept_data.sum()
    total_lost = lost_data.sum()
    retention_pct = (total_kept / total_trials) * 100
    
    ax2.text(0.02, 0.85, f'Retained: {total_kept:,} trials ({retention_pct:.1f}%)\nExcluded: {total_lost:,} trials', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom row: Ordered by age - same species coloring logic
    df_by_age = combined_df.dropna(subset=['age']).sort_values('age')
    
    # Create colors for age-ordered data
    colors_age = []
    for idx, row in df_by_age.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:
            colors_age.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors_age.append(rhesus_color)
        else:
            colors_age.append('#808080')  # Gray for unknown
    
    # Plot 3: Before standardization (by age)
    bars3 = ax3.bar(range(len(df_by_age)), df_by_age['#_attempts'], 
                    color=colors_age, width=0.8, alpha=0.8)
    
    ax3.set_title('Before Standardization\n(Ordered by Age)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Individual subjects (ordered by age)', fontsize=11)
    ax3.set_ylabel('Number of Trials', fontsize=11)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    ax3.locator_params(axis='y', nbins=6)
    
    # Age range info
    age_range = f"{df_by_age['age'].min():.1f} - {df_by_age['age'].max():.1f} years"
    ax3.text(0.02, 0.85, f'Age range: {age_range}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: After standardization (by age)
    kept_data_age = np.minimum(df_by_age['#_attempts'], min_attempts_threshold)
    lost_data_age = df_by_age['#_attempts'] - kept_data_age
    
    bars4 = ax4.bar(range(len(df_by_age)), kept_data_age, 
                    color=colors_age, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices_age = np.where(lost_data_age > 0)[0]
    if len(lost_indices_age) > 0:
        bars5 = ax4.bar(lost_indices_age, lost_data_age[lost_indices_age], 
                        bottom=kept_data_age[lost_indices_age],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax4.set_title(f'After Standardization to {min_attempts_threshold:,} Trials\n(Same Age Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Individual subjects (ordered by age)', fontsize=11)
    ax4.set_ylabel('Number of Trials', fontsize=11)
    ax4.tick_params(axis='x', labelbottom=False)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    ax4.locator_params(axis='y', nbins=6)
    
    # Age bias analysis
    young_third = len(df_by_age) // 3
    old_third = len(df_by_age) - young_third
    high_contrib_young = np.sum((df_by_age['#_attempts'].iloc[:young_third] > min_attempts_threshold))
    high_contrib_old = np.sum((df_by_age['#_attempts'].iloc[old_third:] > min_attempts_threshold))
    
    ax4.text(0.02, 0.85, f'High contributors:\nYoungest third: {high_contrib_young}\nOldest third: {high_contrib_old}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Optimize spacing to prevent title overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25)
    
    # Save plot and display
    save_path = os.path.join(output_dir, "data_retention_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")
    
    plt.show()  # Display the plot
    
    return fig

# Create your combined dataframe and call the function
combined_df = pd.concat([tonkean_df, rhesus_df], ignore_index=True)
combined_df['species'] = combined_df['name'].map(species_mapping)

# Call the function to create the plots
create_improved_data_retention_plots(combined_df, min_attempts_threshold=1651)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_improved_data_retention_plots(combined_df, min_attempts_threshold=1651):
    """
    Create improved data retention visualization with species-specific coloring
    Works with your actual data structure
    
    Parameters:
    - combined_df: DataFrame with your data
    - min_attempts_threshold: Minimum trial threshold (default 1651)
    """
    
    # Create output directory
    output_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots/effect-of-restriction"
    os.makedirs(output_dir, exist_ok=True)
    
    # Species-specific colors - your established palette
    tonkean_color = "#8B4A9C"  # Purple
    rhesus_color = "#4682B4"   # Steel blue
    lost_color = "#E8E8E8"     # Light gray for lost data
    
    # Create figure with 2x2 subplots - optimized spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # Main title with proper spacing
    fig.suptitle('Impact of Minimum Trial Requirements on Dataset Composition', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Sort by total attempts (using your column name)
    df_by_attempts = combined_df.sort_values('#_attempts')
    
    # Debug species mapping - let's see what's actually in the data
    print("Species values in data:", df_by_attempts['species'].unique())
    print("Species counts:", df_by_attempts['species'].value_counts())
    
    # Create species colors - simplified without alpha for debugging
    colors = []
    for idx, row in df_by_attempts.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:  # More flexible matching
            colors.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors.append(rhesus_color)
        else:
            colors.append('#808080')  # Gray for unknown
            print(f"Unknown species: {row['species']} for {row['name']}")
    
    print(f"Color assignments: {len([c for c in colors if c == tonkean_color])} tonkean, {len([c for c in colors if c == rhesus_color])} rhesus")
    
    # Plot 1: Before standardization (by attempts)
    bars1 = ax1.bar(range(len(df_by_attempts)), df_by_attempts['#_attempts'], 
                    color=colors, width=0.8, alpha=0.8)
    
    ax1.set_title('Before Standardization\n(Ordered by Total Trials)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Individual subjects (ordered by number of attempts)', fontsize=11)
    ax1.set_ylabel('Number of Trials', fontsize=11)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.locator_params(axis='y', nbins=6)
    
    # Add legend for species
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tonkean_color, label='M. tonkeana'),
                      Patch(facecolor=rhesus_color, label='M. mulatta')]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Add summary stats
    total_trials = df_by_attempts['#_attempts'].sum()
    n_subjects = len(df_by_attempts)
    ax1.text(0.02, 0.85, f'Total trials: {total_trials:,}\nSubjects: {n_subjects}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: After standardization (by attempts)
    kept_data = np.minimum(df_by_attempts['#_attempts'].values, min_attempts_threshold)
    lost_data = df_by_attempts['#_attempts'].values - kept_data
    
    bars2 = ax2.bar(range(len(df_by_attempts)), kept_data, 
                    color=colors, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices = np.where(lost_data > 0)[0]
    if len(lost_indices) > 0:
        bars3 = ax2.bar(lost_indices, lost_data[lost_indices], 
                        bottom=kept_data[lost_indices],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax2.set_title(f'After Standardization to {min_attempts_threshold:,} Trials\n(Same Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Individual subjects (ordered by number of attempts)', fontsize=11)
    ax2.set_ylabel('Number of Trials', fontsize=11)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.locator_params(axis='y', nbins=6)
    
    # Legend combining species and retention status
    legend_elements2 = [Patch(facecolor=tonkean_color, label='M. tonkeana (retained)'),
                       Patch(facecolor=rhesus_color, label='M. mulatta (retained)'),
                       Patch(facecolor=lost_color, label='Excluded trials')]
    ax2.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Calculate retention stats
    total_kept = kept_data.sum()
    total_lost = lost_data.sum()
    retention_pct = (total_kept / total_trials) * 100
    
    ax2.text(0.02, 0.85, f'Retained: {total_kept:,} trials ({retention_pct:.1f}%)\nExcluded: {total_lost:,} trials', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom row: Ordered by age - same species coloring logic
    df_by_age = combined_df.dropna(subset=['age']).sort_values('age')
    
    # Create colors for age-ordered data
    colors_age = []
    for idx, row in df_by_age.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:
            colors_age.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors_age.append(rhesus_color)
        else:
            colors_age.append('#808080')  # Gray for unknown
    
    # Plot 3: Before standardization (by age)
    bars3 = ax3.bar(range(len(df_by_age)), df_by_age['#_attempts'], 
                    color=colors_age, width=0.8, alpha=0.8)
    
    ax3.set_title('Before Standardization\n(Ordered by Age)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Individual subjects (ordered by age)', fontsize=11)
    ax3.set_ylabel('Number of Trials', fontsize=11)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    ax3.locator_params(axis='y', nbins=6)
    
    # Age range info
    age_range = f"{df_by_age['age'].min():.1f} - {df_by_age['age'].max():.1f} years"
    ax3.text(0.02, 0.85, f'Age range: {age_range}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: After standardization (by age)
    kept_data_age = np.minimum(df_by_age['#_attempts'].values, min_attempts_threshold)
    lost_data_age = df_by_age['#_attempts'].values - kept_data_age
    
    bars4 = ax4.bar(range(len(df_by_age)), kept_data_age, 
                    color=colors_age, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices_age = np.where(lost_data_age > 0)[0]
    if len(lost_indices_age) > 0:
        bars5 = ax4.bar(lost_indices_age, lost_data_age[lost_indices_age], 
                        bottom=kept_data_age[lost_indices_age],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax4.set_title(f'After Standardization to {min_attempts_threshold:,} Trials\n(Same Age Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Individual subjects (ordered by age)', fontsize=11)
    ax4.set_ylabel('Number of Trials', fontsize=11)
    ax4.tick_params(axis='x', labelbottom=False)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    ax4.locator_params(axis='y', nbins=6)
    
    # Age bias analysis
    young_third = len(df_by_age) // 3
    old_third = len(df_by_age) - young_third
    high_contrib_young = np.sum((df_by_age['#_attempts'].iloc[:young_third] > min_attempts_threshold))
    high_contrib_old = np.sum((df_by_age['#_attempts'].iloc[old_third:] > min_attempts_threshold))
    
    ax4.text(0.02, 0.85, f'High contributors:\nYoungest third: {high_contrib_young}\nOldest third: {high_contrib_old}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Optimize spacing to prevent title overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25)
    
    # Save plot and display
    save_path = os.path.join(output_dir, "data_retention_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")
    
    plt.show()  # Display the plot
    
    return fig


def create_top_13_data_retention_plots(combined_df, n_top=13):
    """
    Create data retention visualization for the top N individuals with highest attempts
    Works with your actual data structure
    
    Parameters:
    - combined_df: DataFrame with your data
    - n_top: Number of top individuals to analyze (default 13)
    """
    
    # Create output directory
    output_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots/effect-of-restriction"
    os.makedirs(output_dir, exist_ok=True)
    
    # SELECT TOP N INDIVIDUALS WITH HIGHEST ATTEMPTS
    top_n_df = combined_df.nlargest(n_top, '#_attempts')
    
    # Calculate the minimum attempts across the TOP N monkeys only
    min_attempts_top_n = top_n_df['#_attempts'].min()
    
    print(f"Minimum attempts across top {n_top} monkeys: {min_attempts_top_n}")
    print(f"Total top {n_top} monkeys: {len(top_n_df)}")
    
    # Species-specific colors - your established palette
    tonkean_color = "#8B4A9C"  # Purple
    rhesus_color = "#4682B4"   # Steel blue
    lost_color = "#E8E8E8"     # Light gray for lost data
    
    # Create figure with 2x2 subplots - optimized spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # Main title with proper spacing
    fig.suptitle(f'Impact of Minimum Trial Requirements on Top {n_top} Highest Contributing Subjects', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Sort by total attempts
    df_by_attempts = top_n_df.sort_values('#_attempts')
    
    # Create species colors
    colors = []
    for idx, row in df_by_attempts.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:
            colors.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors.append(rhesus_color)
        else:
            colors.append('#808080')  # Gray for unknown
    
    # Plot 1: Before standardization (by attempts) - TOP N ONLY
    bars1 = ax1.bar(range(len(df_by_attempts)), df_by_attempts['#_attempts'], 
                    color=colors, width=0.8, alpha=0.8)
    
    ax1.set_title(f'Before Standardization - Top {n_top} Contributors\n(Ordered by Total Trials)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel(f'Top {n_top} subjects (ordered by number of attempts)', fontsize=11)
    ax1.set_ylabel('Number of Trials', fontsize=11)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.locator_params(axis='y', nbins=6)
    
    # Add legend for species
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tonkean_color, label='M. tonkeana'),
                      Patch(facecolor=rhesus_color, label='M. mulatta')]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Add summary stats
    total_trials = df_by_attempts['#_attempts'].sum()
    n_subjects = len(df_by_attempts)
    ax1.text(0.02, 0.85, f'Total trials: {total_trials:,}\nTop subjects: {n_subjects}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: After standardization (by attempts) - TOP N ONLY
    kept_data = np.minimum(df_by_attempts['#_attempts'].values, min_attempts_top_n)
    lost_data = df_by_attempts['#_attempts'].values - kept_data
    
    bars2 = ax2.bar(range(len(df_by_attempts)), kept_data, 
                    color=colors, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices = np.where(lost_data > 0)[0]
    if len(lost_indices) > 0:
        bars3 = ax2.bar(lost_indices, lost_data[lost_indices], 
                        bottom=kept_data[lost_indices],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax2.set_title(f'After Standardization to {min_attempts_top_n:,} Trials\n(Same Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel(f'Top {n_top} subjects (ordered by number of attempts)', fontsize=11)
    ax2.set_ylabel('Number of Trials', fontsize=11)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.locator_params(axis='y', nbins=6)
    
    # Legend combining species and retention status
    legend_elements2 = [Patch(facecolor=tonkean_color, label='M. tonkeana (retained)'),
                       Patch(facecolor=rhesus_color, label='M. mulatta (retained)'),
                       Patch(facecolor=lost_color, label='Excluded trials')]
    ax2.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Calculate retention stats
    total_kept = kept_data.sum()
    total_lost = lost_data.sum()
    retention_pct = (total_kept / total_trials) * 100
    
    ax2.text(0.02, 0.85, f'Retained: {total_kept:,} trials ({retention_pct:.1f}%)\nExcluded: {total_lost:,} trials', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom row: Ordered by age - TOP N ONLY
    df_by_age = top_n_df.dropna(subset=['age']).sort_values('age')
    
    # Create colors for age-ordered data
    colors_age = []
    for idx, row in df_by_age.iterrows():
        species_val = str(row['species']).lower().strip()
        if 'tonk' in species_val:
            colors_age.append(tonkean_color)
        elif 'rhes' in species_val or 'mulat' in species_val:
            colors_age.append(rhesus_color)
        else:
            colors_age.append('#808080')  # Gray for unknown
    
    # Plot 3: Before standardization (by age) - TOP N ONLY
    bars3 = ax3.bar(range(len(df_by_age)), df_by_age['#_attempts'], 
                    color=colors_age, width=0.8, alpha=0.8)
    
    ax3.set_title(f'Before Standardization - Top {n_top} Contributors\n(Ordered by Age)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel(f'Top {n_top} subjects (ordered by age)', fontsize=11)
    ax3.set_ylabel('Number of Trials', fontsize=11)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)
    ax3.locator_params(axis='y', nbins=6)
    
    # Age range info
    if len(df_by_age) > 0:
        age_range = f"{df_by_age['age'].min():.1f} - {df_by_age['age'].max():.1f} years"
        ax3.text(0.02, 0.85, f'Age range: {age_range}\nWith age data: {len(df_by_age)}/{n_top}', 
                 transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: After standardization (by age) - TOP N ONLY
    kept_data_age = np.minimum(df_by_age['#_attempts'].values, min_attempts_top_n)
    lost_data_age = df_by_age['#_attempts'].values - kept_data_age
    
    bars4 = ax4.bar(range(len(df_by_age)), kept_data_age, 
                    color=colors_age, width=0.8, alpha=0.8)
    
    # Bars for lost data (uniform gray) - only where there's lost data
    lost_indices_age = np.where(lost_data_age > 0)[0]
    if len(lost_indices_age) > 0:
        bars5 = ax4.bar(lost_indices_age, lost_data_age[lost_indices_age], 
                        bottom=kept_data_age[lost_indices_age],
                        color=lost_color, alpha=0.7, width=0.8)
    
    ax4.set_title(f'After Standardization to {min_attempts_top_n:,} Trials\n(Same Age Ordering)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel(f'Top {n_top} subjects (ordered by age)', fontsize=11)
    ax4.set_ylabel('Number of Trials', fontsize=11)
    ax4.tick_params(axis='x', labelbottom=False)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.legend(handles=legend_elements2, loc='upper left', framealpha=0.9, fontsize=10)
    ax4.locator_params(axis='y', nbins=6)
    
    # Age bias analysis for top N
    if len(df_by_age) > 2:  # Need at least 3 subjects for thirds analysis
        young_third = len(df_by_age) // 3
        old_third = len(df_by_age) - young_third
        high_contrib_young = np.sum((df_by_age['#_attempts'].iloc[:young_third] > min_attempts_top_n))
        high_contrib_old = np.sum((df_by_age['#_attempts'].iloc[old_third:] > min_attempts_top_n))
        
        ax4.text(0.02, 0.85, f'Contributors above min:\nYoungest third: {high_contrib_young}\nOldest third: {high_contrib_old}', 
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Optimize spacing to prevent title overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25)
    save_path_png = os.path.join(output_dir, f"top_{n_top}_data_retention_analysis.png")
    save_path_svg = os.path.join(output_dir, f"top_{n_top}_data_retention_analysis.svg")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_svg, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path_png}")
    print(f"Plot saved to: {save_path_svg}")
    
    plt.show()  # Display the plot
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"{'TOP {n_top} DATA RESTRICTION ANALYSIS':^60}".format(n_top=n_top))
    print(f"{'='*60}")
    print(f"Total top {n_top} monkeys analyzed: {len(top_n_df)}")
    print(f"Total attempts across top {n_top}: {total_trials:,}")
    print(f"Minimum attempts threshold: {min_attempts_top_n:,}")
    print(f"Attempts kept after restriction: {total_kept:,}")
    print(f"Attempts lost: {total_lost:,}")
    print(f"Percentage of data lost: {retention_pct:.1f}% retained, {100-retention_pct:.1f}% lost")
    
    return fig

# Create your combined dataframe and call the functions
combined_df = pd.concat([tonkean_df, rhesus_df], ignore_index=True)
combined_df['species'] = combined_df['name'].map(species_mapping)

# Call the original function for all subjects
create_improved_data_retention_plots(combined_df, min_attempts_threshold=1651)

# Call the new function for top 13 subjects
create_top_13_data_retention_plots(combined_df, n_top=13)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prepare data for visualization
# Combine both species
all_monkeys_df = pd.concat([tonkean_df, rhesus_df], ignore_index=True)

# Add species column
all_monkeys_df['species'] = ['Tonkean'] * len(tonkean_df) + ['Rhesus'] * len(rhesus_df)

# Define metrics to plot
metrics = ['p_success', 'p_error', 'p_omission', 'p_premature']
metric_labels = ['Success Rate', 'Error Rate', 'Omission Rate', 'Premature Rate']

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Color scheme
rhesus_color = 'steelblue'
tonkean_color = 'purple'
individual_color = 'darkgray'  # More gray
individual_alpha = 0.2  # Even more faint!
individual_linewidth = 0.7
individual_markersize = 3.5  # Slightly smaller
mean_linewidth = 2.0

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    # Get the corresponding control task columns
    tav_col = f'tav_{metric}'
    dms_col = f'dms_{metric}'
    
    # Prepare data for each task (x-axis positions)
    tasks = ['TAV', 'DMS', '5-CSRT']
    x_positions = [0, 1, 2]
    
    # Track sample sizes for each task
    n_tav = 0
    n_dms = 0
    n_csrt = 0
    
    # Species-specific means
    rhesus_means = []
    tonkean_means = []
    
    # Plot individual lines for each monkey
    for _, monkey in all_monkeys_df.iterrows():
        values = []
        positions = []
        
        # TAV value
        if pd.notna(monkey[tav_col]):
            values.append(monkey[tav_col])
            positions.append(0)
        
        # DMS value
        if pd.notna(monkey[dms_col]):
            values.append(monkey[dms_col])
            positions.append(1)
        
        # 5-CSRT value (always present)
        values.append(monkey[metric])
        positions.append(2)
        
        # Plot line - thin connecting lines in species color, dots in gray
        if monkey['species'] == 'Rhesus':
            line_color = rhesus_color
        else:
            line_color = tonkean_color
        
        if len(values) > 1:
            # Plot connecting line first (faint, species color)
            ax.plot(positions, values, '-', color=line_color, alpha=individual_alpha, 
                   linewidth=individual_linewidth, zorder=1)
            # Plot dots on top (gray)
            ax.plot(positions, values, 'o', color=individual_color, alpha=0.6,
                   markersize=4, zorder=2)
    
    # Calculate and plot species means
    for task_idx, task in enumerate(['tav', 'dms', 'csrt']):
        if task == 'csrt':
            col = metric
        else:
            col = f'{task}_{metric}'
        
        # Rhesus mean
        rhesus_data = all_monkeys_df[all_monkeys_df['species'] == 'Rhesus'][col].dropna()
        if len(rhesus_data) > 0:
            rhesus_means.append((task_idx, rhesus_data.mean()))
            if task == 'tav':
                n_tav = max(n_tav, len(rhesus_data))
            elif task == 'dms':
                n_dms = max(n_dms, len(rhesus_data))
            else:
                n_csrt = max(n_csrt, len(rhesus_data))
        
        # Tonkean mean
        tonkean_data = all_monkeys_df[all_monkeys_df['species'] == 'Tonkean'][col].dropna()
        if len(tonkean_data) > 0:
            tonkean_means.append((task_idx, tonkean_data.mean()))
            if task == 'tav':
                n_tav = len(all_monkeys_df[col].dropna())
            elif task == 'dms':
                n_dms = len(all_monkeys_df[col].dropna())
            else:
                n_csrt = len(all_monkeys_df[col].dropna())
    
    # Plot mean lines with dashed style
    if len(rhesus_means) > 1:
        rhesus_x, rhesus_y = zip(*rhesus_means)
        ax.plot(rhesus_x, rhesus_y, 'o--', color=rhesus_color, linewidth=mean_linewidth, 
               markersize=7, label='Rhesus', zorder=10)
    
    if len(tonkean_means) > 1:
        tonkean_x, tonkean_y = zip(*tonkean_means)
        ax.plot(tonkean_x, tonkean_y, 'o--', color=tonkean_color, linewidth=mean_linewidth, 
               markersize=7, label='Tonkean', zorder=10)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'TAV\n(n={n_tav})', f'DMS\n(n={n_dms})', f'5-CSRT\n(n={n_csrt})'])
    
    # Only add y-axis label for left column (idx 0 and 2)
    if idx in [0, 2]:
        ax.set_ylabel('Proportion', fontsize=11)
    else:
        # Remove y-axis tick labels AND ticks for right column
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(label, fontsize=15, fontweight='bold', pad=10)
    ax.set_xlim(-0.2, 2.2)
    
# Create legend below all subplots
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=rhesus_color, linewidth=2, label='Rhesus'),
    Line2D([0], [0], color=tonkean_color, linewidth=2, label='Tonkean'),
    Line2D([0], [0], color='gray', linewidth=1, marker='o', markersize=4, label='Individual'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, 
          frameon=True, bbox_to_anchor=(0.5, -0.02))

# Main title
fig.suptitle('Performance Metrics Across Cognitive Tasks', 
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])

# Save figure
save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/plots/"
plt.savefig(f"{save_path}cross_task_performance.svg", format='svg', dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}cross_task_performance.png", format='png', dpi=300, bbox_inches='tight')

plt.show()

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Colors and styles
rhesus_color = 'steelblue'
tonkean_color = 'purple'
individual_color = 'darkgray'
individual_alpha = 0.2
individual_linewidth = 0.7
individual_markersize = 6
mean_linewidth = 2.2
mean_color = 'black'
mean_alpha = 0.85  # slightly transparent black
mean_markersize = 7.5  # slightly larger than individuals

# Adjusted x positions: widened spacing for better subplot balance
x_positions = [-0.1, 1.0, 2.1]
tasks = ['TAV', 'DMS', '5-CSRT']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    tav_col = f'tav_{metric}'
    dms_col = f'dms_{metric}'
    
    # Plot individual trajectories
    for _, monkey in all_monkeys_df.iterrows():
        values, positions = [], []
        if pd.notna(monkey[tav_col]):
            values.append(monkey[tav_col])
            positions.append(x_positions[0])
        if pd.notna(monkey[dms_col]):
            values.append(monkey[dms_col])
            positions.append(x_positions[1])
        values.append(monkey[metric])
        positions.append(x_positions[2])
        
        line_color = rhesus_color if monkey['species'] == 'Rhesus' else tonkean_color
        
        if len(values) > 1:
            ax.plot(positions, values, '-', color=line_color, alpha=individual_alpha,
                    linewidth=individual_linewidth, zorder=1)
            ax.plot(positions, values, 'o', color=individual_color, alpha=0.7,
                    markersize=individual_markersize, zorder=2)
    
    # Compute pooled sample mean
    pooled_means = []
    ns = []
    for pos, task in zip(x_positions, ['tav', 'dms', 'csrt']):
        col = metric if task == 'csrt' else f'{task}_{metric}'
        data = all_monkeys_df[col].dropna()
        if len(data) > 0:
            pooled_means.append((pos, data.mean()))
        ns.append(len(data))
    
    # Plot black dashed sample mean line (slightly transparent)
    if len(pooled_means) > 1:
        x_vals, y_vals = zip(*pooled_means)
        ax.plot(
            x_vals, y_vals, 'o--', color=mean_color, alpha=mean_alpha,
            linewidth=mean_linewidth, markersize=mean_markersize,
            label='Sample Mean', zorder=10
        )
    
    # Axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f'TAV\n(n={ns[0]})', f'DMS\n(n={ns[1]})', f'5-CSRT\n(n={ns[2]})'],
        fontsize=12
    )
    
    if idx in [0, 2]:
        ax.set_ylabel('Proportion', fontsize=13, labelpad=8)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Remove x-axis labels for top row (subplots 0 and 1)
    if idx in [0, 1]:
        ax.set_xticklabels([])  # hide labels
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, 2.3)
    ax.set_title(label, fontsize=17, fontweight='bold', pad=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=mean_color, alpha=mean_alpha, linestyle='--',
           linewidth=2.2, marker='o', markersize=mean_markersize,
           label='Sample Mean'),
    Line2D([0], [0], color=rhesus_color, linewidth=1, label=r'$M.\ mulatta$', alpha=0.5),
    Line2D([0], [0], color=tonkean_color, linewidth=1, label=r'$M.\ tonkeana$', alpha=0.5),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.03))

# Bigger main title
fig.suptitle('Performance Metrics Across Cognitive Tasks',
             fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save figure
plt.savefig(f"{save_path}cross_task_performance_sample_mean_softblack.svg", format='svg', dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}cross_task_performance_sample_mean_softblack.png", format='png', dpi=300, bbox_inches='tight')

plt.show()
# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Colors and styles
rhesus_color = 'steelblue'
tonkean_color = 'purple'
individual_color = 'darkgray'
individual_alpha = 0.2
individual_linewidth = 0.7
individual_markersize = 6
mean_linewidth = 2.2
mean_color = 'black'
mean_alpha = 0.85  # slightly transparent black
mean_markersize = 7.5

# Slightly less extreme x positions for wide but balanced spacing
x_positions = [-0.4, 1.0, 2.4]
tasks = ['TAV', 'DMS', '5-CSRT']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    tav_col = f'tav_{metric}'
    dms_col = f'dms_{metric}'
    
    # Plot individual trajectories
    for _, monkey in all_monkeys_df.iterrows():
        values, positions = [], []
        if pd.notna(monkey[tav_col]):
            values.append(monkey[tav_col])
            positions.append(x_positions[0])
        if pd.notna(monkey[dms_col]):
            values.append(monkey[dms_col])
            positions.append(x_positions[1])
        values.append(monkey[metric])
        positions.append(x_positions[2])
        
        line_color = rhesus_color if monkey['species'] == 'Rhesus' else tonkean_color
        
        if len(values) > 1:
            ax.plot(positions, values, '-', color=line_color, alpha=individual_alpha,
                    linewidth=individual_linewidth, zorder=1)
            ax.plot(positions, values, 'o', color=individual_color, alpha=0.7,
                    markersize=individual_markersize, zorder=2)
    
    # Compute species means + pooled mean
    pooled_means = []
    rhesus_means = []
    tonkean_means = []
    ns = []
    
    for pos, task in zip(x_positions, ['tav', 'dms', 'csrt']):
        col = metric if task == 'csrt' else f'{task}_{metric}'
        
        data_all = all_monkeys_df[col].dropna()
        data_rhesus = all_monkeys_df[all_monkeys_df['species'] == 'Rhesus'][col].dropna()
        data_tonkean = all_monkeys_df[all_monkeys_df['species'] == 'Tonkean'][col].dropna()
        
        if len(data_all) > 0:
            pooled_means.append((pos, data_all.mean()))
        if len(data_rhesus) > 0:
            rhesus_means.append((pos, data_rhesus.mean()))
        if len(data_tonkean) > 0:
            tonkean_means.append((pos, data_tonkean.mean()))
        ns.append(len(data_all))
    
    # Plot species-level dashed means
    if len(rhesus_means) > 1:
        x_r, y_r = zip(*rhesus_means)
        ax.plot(x_r, y_r, 'o--', color=rhesus_color, linewidth=mean_linewidth,
                markersize=mean_markersize, alpha=0.9, label=r'$M.\ mulatta$', zorder=10)
    if len(tonkean_means) > 1:
        x_t, y_t = zip(*tonkean_means)
        ax.plot(x_t, y_t, 'o--', color=tonkean_color, linewidth=mean_linewidth,
                markersize=mean_markersize, alpha=0.9, label=r'$M.\ tonkeana$', zorder=10)
    
    # Plot black dashed pooled mean line
    if len(pooled_means) > 1:
        x_vals, y_vals = zip(*pooled_means)
        ax.plot(x_vals, y_vals, 'o--', color=mean_color, alpha=mean_alpha,
                linewidth=mean_linewidth, markersize=mean_markersize,
                label='Sample Mean', zorder=12)
    
    # Axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f'TAV\n(n={ns[0]})', f'DMS\n(n={ns[1]})', f'5-CSRT\n(n={ns[2]})'],
        fontsize=12
    )
    
    if idx in [0, 2]:
        ax.set_ylabel('Proportion', fontsize=13, labelpad=8)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Remove x-axis labels for top row
    if idx in [0, 1]:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 2.5)  # slightly less extreme than previous wide
    ax.set_title(label, fontsize=17, fontweight='bold', pad=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', alpha=mean_alpha, linestyle='--',
           linewidth=2.2, marker='o', markersize=mean_markersize, label='Sample Mean'),
    Line2D([0], [0], color=rhesus_color, linestyle='--', linewidth=2.2,
           marker='o', markersize=mean_markersize, label=r'$M.\ mulatta$'),
    Line2D([0], [0], color=tonkean_color, linestyle='--', linewidth=2.2,
           marker='o', markersize=mean_markersize, label=r'$M.\ tonkeana$'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.035))

# Main title (slightly lowered)
fig.suptitle('Performance Metrics Across Cognitive Tasks',
             fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save figure
plt.savefig(f"{save_path}cross_task_performance_species_sample_mean_wide_adjusted.svg", format='svg', dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}cross_task_performance_species_sample_mean_wide_adjusted.png", format='png', dpi=300, bbox_inches='tight')

plt.show()

import scipy.stats as stats
from scipy.stats import f_oneway
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

# Prepare data for repeated measures ANOVA
# We need long format: one row per observation
results_summary = []

for metric in metrics:
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR: {metric.upper()}")
    print(f"{'='*60}\n")
    
    # Prepare long-format data for this metric
    long_data = []
    
    for idx, monkey in all_monkeys_df.iterrows():
        monkey_id = idx
        
        # Get values for each task
        tav_val = monkey[f'tav_{metric}']
        dms_val = monkey[f'dms_{metric}']
        csrt_val = monkey[metric]
        
        # Only include monkeys with data for at least 2 tasks
        available_tasks = []
        if pd.notna(tav_val):
            long_data.append({'subject': monkey_id, 'task': 'TAV', 'value': tav_val})
            available_tasks.append('TAV')
        if pd.notna(dms_val):
            long_data.append({'subject': monkey_id, 'task': 'DMS', 'value': dms_val})
            available_tasks.append('DMS')
        if pd.notna(csrt_val):
            long_data.append({'subject': monkey_id, 'task': '5-CSRT', 'value': csrt_val})
            available_tasks.append('5-CSRT')
    
    long_df = pd.DataFrame(long_data)
    
    # Count subjects per comparison
    subjects_tav_dms = len(set(long_df[long_df['task'].isin(['TAV', 'DMS'])]['subject']))
    subjects_tav_csrt = len(set(long_df[long_df['task'].isin(['TAV', '5-CSRT'])]['subject']))
    subjects_dms_csrt = len(set(long_df[long_df['task'].isin(['DMS', '5-CSRT'])]['subject']))
    subjects_all_three = len(all_monkeys_df[
        all_monkeys_df[f'tav_{metric}'].notna() & 
        all_monkeys_df[f'dms_{metric}'].notna() & 
        all_monkeys_df[metric].notna()
    ])
    
    print(f"Sample sizes:")
    print(f"  Subjects with TAV & DMS: {subjects_tav_dms}")
    print(f"  Subjects with TAV & 5-CSRT: {subjects_tav_csrt}")
    print(f"  Subjects with DMS & 5-CSRT: {subjects_dms_csrt}")
    print(f"  Subjects with all three tasks: {subjects_all_three}\n")
    
    # Repeated measures ANOVA (using subjects with all three tasks)
    complete_data = all_monkeys_df[
        all_monkeys_df[f'tav_{metric}'].notna() & 
        all_monkeys_df[f'dms_{metric}'].notna() & 
        all_monkeys_df[metric].notna()
    ].copy()
    
    if len(complete_data) >= 3:  # Need at least 3 subjects
        # Prepare data for repeated measures ANOVA
        rm_long_data = []
        for idx, monkey in complete_data.iterrows():
            rm_long_data.append({'subject': idx, 'task': 'TAV', 'value': monkey[f'tav_{metric}']})
            rm_long_data.append({'subject': idx, 'task': 'DMS', 'value': monkey[f'dms_{metric}']})
            rm_long_data.append({'subject': idx, 'task': '5-CSRT', 'value': monkey[metric]})
        
        rm_df = pd.DataFrame(rm_long_data)
        
        # Run repeated measures ANOVA
        try:
            aovrm = AnovaRM(rm_df, 'value', 'subject', within=['task'])
            res = aovrm.fit()
            
            print(f"Repeated Measures ANOVA (n={len(complete_data)} subjects):")
            print(res)
            
            # Extract p-value
            p_value = res.anova_table['Pr > F']['task']
            f_value = res.anova_table['F Value']['task']
            
            print(f"\nMain effect of task: F = {f_value:.3f}, p = {p_value:.4f}")
            
            # Only do post-hoc if significant
            if p_value < 0.05:
                print(f"\n*** Significant main effect found (p = {p_value:.4f}) ***")
                print("\nPost-hoc pairwise comparisons (Bonferroni corrected):")
                
                # Pairwise t-tests with Bonferroni correction
                from scipy.stats import ttest_rel
                
                tav_vals = complete_data[f'tav_{metric}'].values
                dms_vals = complete_data[f'dms_{metric}'].values
                csrt_vals = complete_data[metric].values
                
                # Three comparisons, so alpha = 0.05/3 = 0.0167
                alpha_corrected = 0.05 / 3
                
                # TAV vs DMS
                t_stat, p_val = ttest_rel(tav_vals, dms_vals)
                sig = "***" if p_val < alpha_corrected else "ns"
                print(f"  TAV vs DMS: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")
                
                # TAV vs 5-CSRT
                t_stat, p_val = ttest_rel(tav_vals, csrt_vals)
                sig = "***" if p_val < alpha_corrected else "ns"
                print(f"  TAV vs 5-CSRT: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")
                
                # DMS vs 5-CSRT
                t_stat, p_val = ttest_rel(dms_vals, csrt_vals)
                sig = "***" if p_val < alpha_corrected else "ns"
                print(f"  DMS vs 5-CSRT: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")
                
                print(f"\n  (Bonferroni corrected alpha = {alpha_corrected:.4f})")
            else:
                print(f"\nNo significant main effect (p = {p_value:.4f}). Post-hoc tests not performed.")
        
        except Exception as e:
            print(f"Error running repeated measures ANOVA: {e}")
    else:
        print(f"Insufficient subjects with complete data (n={len(complete_data)}). Need at least 3.")
    
    # Store results
    results_summary.append({
        'metric': metric,
        'n_complete': len(complete_data) if len(complete_data) >= 3 else 0
    })

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)