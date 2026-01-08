"""
Cross-Task Analysis - Matches prep.py preprocessing exactly
===========================================================
1. Age trajectories: Uses ALL ELO-aligned data per task (maximize power)
2. Cross-task correlations: Uses MATCHED EXPOSURE (first N attempts from each task)
3. Age calculation: Uses midpoint of ELO window (matches your prep.py exactly)
"""

import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ============================================================================
# LOAD DATA
# ============================================================================

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
base = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt"

with open(os.path.join(directory, "data.pickle"), "rb") as f:
    data = pickle.load(f)

with open(os.path.join(directory, "tav.pickle"), "rb") as f:
    tav_dict = pickle.load(f)

with open(os.path.join(directory, "dms.pickle"), "rb") as f:
    dms_dict = pickle.load(f)

# Remove monkeys with RFID issues
monkeys_to_remove = {"joy", "jipsy"}
for species in data:
    for name in list(data[species].keys()):
        if name in monkeys_to_remove:
            del data[species][name]

# Name mappings
tonkean_abrv_2_fn = {
    "abr": "abricot", "ala": "alaryc", "alv": "alvin", "anu": "anubis",
    "bar": "barnabe", "ber": "berenice", "ces": "cesar", "dor": "dory",
    "eri": "eric", "jea": "jeanne", "lad": "lady", "las": "lassa",
    "nem": "nema", "nen": "nenno", "ner": "nereis", "ola": "olaf",
    "olg": "olga", "oli": "olli", "pac": "patchouli", "pat": "patsy",
    "wal": "wallace", "wat": "walt", "wot": "wotan", "yan": "yang",
    "yak": "yannick", "yin": "yin", "yoh": "yoh", "ult": "ulysse",
    "fic": "ficelle", "gan": "gandhi", "hav": "havanna", "con": "controle",
    "gai": "gaia", "her": "hercules", "han": "hanouk", "hor": "horus",
    "imo": "imoen", "ind": "indigo", "iro": "iron", "isi": "isis",
}

rhesus_abrv_2_fn = {
    "the": "theoden", "any": "anyanka", "djo": "djocko", "vla": "vladimir",
    "yel": "yelena", "bor": "boromir", "far": "faramir", "yva": "yvan",
    "baa": "baal", "spl": "spliff", "ol": "ol", "nat": "natasha",
    "arw": "arwen", "eow": "eowyn", "mar": "mar"
}

# Load hierarchy
tonkean_hierarchy = pd.read_excel(os.path.join(directory, "hierarchy/tonkean/elo_matrix.xlsx"))
rhesus_hierarchy = pd.read_excel(os.path.join(directory, "hierarchy/rhesus/elo_matrix.xlsx"))

tonkean_hierarchy = tonkean_hierarchy.rename(columns=tonkean_abrv_2_fn)
rhesus_hierarchy = rhesus_hierarchy.rename(columns=rhesus_abrv_2_fn)

# Load demographics with birthdate
mkid_rhesus = pd.read_csv(os.path.join(directory, "mkid_rhesus.csv"))
mkid_tonkean = pd.read_csv(os.path.join(directory, "mkid_tonkean.csv"))

mkid_rhesus['name'] = mkid_rhesus['name'].str.lower().str.strip()
mkid_tonkean['name'] = mkid_tonkean['name'].str.lower().str.strip()
mkid_rhesus.rename(columns={"Age": "age"}, inplace=True)
mkid_tonkean.rename(columns={"Age": "age"}, inplace=True)

# Combine demographics
demographics = pd.concat([
    mkid_tonkean.assign(species="tonkean"),
    mkid_rhesus.assign(species="rhesus")
])

print(f"Demographics loaded: {len(demographics)} individuals")
print(f"  Tonkean: {len(mkid_tonkean)}")
print(f"  Rhesus: {len(mkid_rhesus)}")

# ============================================================================
# BUILD ELO DICTIONARY - MATCHES PREP.PY EXACTLY
# ============================================================================

elo = {}
hierarchies = {
    "tonkean": tonkean_hierarchy,
    "rhesus": rhesus_hierarchy
}

# Remove first two months for each monkey - matches prep.py exactly
for species, hierarchy in hierarchies.items():
    hierarchy["Date"] = pd.to_datetime(hierarchy["Date"])
    for name in hierarchy.columns[1:]:
        if name in hierarchy:
            monkey_elo = hierarchy[["Date", name]].dropna()
            if not monkey_elo.empty:
                first_date = monkey_elo["Date"].min()
                cutoff_date = first_date + pd.DateOffset(months=2)
                filtered_elo = monkey_elo[monkey_elo["Date"] > cutoff_date]
                if not filtered_elo.empty:
                    elo[name] = filtered_elo

print(f"\nELO data available for {len(elo)} individuals")

# Create output directory
output_dir = os.path.join(base, "derivatives", "cross_task")
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PART 1: AGE TRAJECTORY ANALYSIS (Uses ALL ELO-aligned data)
# ============================================================================

print("\n" + "="*60)
print("PART 1: AGE TRAJECTORY ANALYSIS (ALL ELO-ALIGNED DATA)")
print("="*60)

def build_task_summary_full(task_dict, task_name, elo_dict, demographics_df):
    """
    Build one row per monkey using ALL ELO-aligned data.
    Calculates age at MIDPOINT of ELO window (matches prep.py exactly).
    """
    results = []
    
    for species in task_dict.keys():
        for name in task_dict[species].keys():
            if name not in elo_dict or elo_dict[name].empty:
                continue
            
            attempts = task_dict[species][name]["attempts"].copy()
            attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
            attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
            
            # Get ELO window (after 2-month cutoff)
            elo_start = elo_dict[name]["Date"].min()
            elo_end = elo_dict[name]["Date"].max()
            
            # Filter attempts to ELO window
            attempts = attempts[
                (attempts["instant_begin"] >= elo_start) &
                (attempts["instant_end"] <= elo_end)
            ]
            
            if attempts.empty or len(attempts) < 100:
                continue
            
            demo = demographics_df[(demographics_df["name"] == name) & (demographics_df["species"] == species)]
            if demo.empty:
                continue
            
            # CRITICAL FIX: Calculate age at midpoint of ELO WINDOW, not attempt min/max
            # This matches your prep.py exactly
            task_midpoint = elo_start + (elo_end - elo_start) / 2
            
            birthdate = pd.to_datetime(demo["birthdate"].values[0])
            age_at_task = (task_midpoint - birthdate).days / 365.25
            
            gender = demo["gender"].values[0]
            outcome = attempts["result"].value_counts(normalize=True)
            
            results.append({
                "name": name,
                "species": species,
                "task": task_name,
                "gender": gender,
                "age": age_at_task,
                "n_attempts": len(attempts),
                "p_success": outcome.get("success", 0.0),
                "p_error": outcome.get("error", 0.0),
                "p_omission": outcome.get("stepomission", 0.0),
                "p_premature": outcome.get("prematured", 0.0),
            })
    
    return pd.DataFrame(results)

df_5csrt_full = build_task_summary_full(data, "5-CSRT", elo, demographics)
df_tav_full = build_task_summary_full(tav_dict, "TAV", elo, demographics)
df_dms_full = build_task_summary_full(dms_dict, "DMS", elo, demographics)

print(f"\n5-CSRT: {len(df_5csrt_full)} individuals")
print(f"TAV: {len(df_tav_full)} individuals")
print(f"DMS: {len(df_dms_full)} individuals")

# Save full summaries
df_5csrt_full.to_csv(os.path.join(output_dir, "5csrt_full_summary.csv"), index=False)
df_tav_full.to_csv(os.path.join(output_dir, "tav_full_summary.csv"), index=False)
df_dms_full.to_csv(os.path.join(output_dir, "dms_full_summary.csv"), index=False)

# ============================================================================
# FIT AGE TRAJECTORIES
# ============================================================================

def fit_age_trajectory(df, outcome="p_success"):
    """Fit linear and quadratic models"""
    data = df[[outcome, "age"]].dropna()
    if len(data) < 5:
        return None
    
    X = data[["age"]].values
    y = data[outcome].values
    
    # Linear
    model_lin = LinearRegression().fit(X, y)
    r2_lin = model_lin.score(X, y)
    beta_lin = model_lin.coef_[0]
    
    # Quadratic
    poly = PolynomialFeatures(degree=2)
    X_quad = poly.fit_transform(X)
    model_quad = LinearRegression().fit(X_quad, y)
    r2_quad = model_quad.score(X_quad, y)
    
    beta_0 = model_quad.coef_[0]
    beta_1 = model_quad.coef_[1]
    beta_2 = model_quad.coef_[2]
    
    # F-test
    n = len(y)
    ss_res_lin = np.sum((y - model_lin.predict(X)) ** 2)
    ss_res_quad = np.sum((y - model_quad.predict(X_quad)) ** 2)
    
    if ss_res_lin > ss_res_quad:
        f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n - 3))
        f_pval = 1 - stats.f.cdf(f_stat, 1, n - 3)
    else:
        f_stat, f_pval = 0, 1.0
    
    peak_age = None
    if f_pval < 0.05 and abs(beta_2) > 1e-10:
        peak_age = -beta_1 / (2 * beta_2)
        if peak_age < 0 or peak_age > 30:
            peak_age = None
    
    r_pearson, p_pearson = stats.pearsonr(X.flatten(), y)
    
    return {
        "n": n,
        "linear_r2": r2_lin,
        "linear_beta": beta_lin,
        "quadratic_r2": r2_quad,
        "quadratic_beta2": beta_2,
        "f_stat": f_stat,
        "f_pval": f_pval,
        "peak_age": peak_age,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
    }

print("\n" + "="*60)
print("AGE TRAJECTORY MODELS")
print("="*60)

outcomes = ["p_success", "p_error", "p_premature", "p_omission"]
trajectory_results = []

for task_name, task_df in [("5-CSRT", df_5csrt_full), ("TAV", df_tav_full), ("DMS", df_dms_full)]:
    if task_df.empty:
        continue
    
    print(f"\n{task_name}:")
    for outcome in outcomes:
        result = fit_age_trajectory(task_df, outcome)
        if result is None:
            continue
        
        sig_quad = "✓ QUADRATIC" if result['f_pval'] < 0.05 else "✗ Linear"
        print(f"  {outcome}: {sig_quad} (R²={result['quadratic_r2']:.3f}, F={result['f_stat']:.2f}, p={result['f_pval']:.4f})", end="")
        if result['peak_age']:
            print(f" → peak/valley at {result['peak_age']:.1f} years")
        else:
            print()
        
        trajectory_results.append({
            "task": task_name,
            "outcome": outcome,
            **result
        })

df_trajectories = pd.DataFrame(trajectory_results)
df_trajectories.to_csv(os.path.join(output_dir, "age_trajectories.csv"), index=False)

# ============================================================================
# TRAJECTORY SHAPE COMPARISONS
# ============================================================================

print("\n" + "="*60)
print("TRAJECTORY SHAPE COMPARISONS")
print("="*60)

def compare_trajectory_shapes(df_task1, df_task2, outcome, task1_name, task2_name):
    """Test for different trajectory shapes using interaction terms"""
    df1 = df_task1[[outcome, "age"]].copy()
    df1["task"] = 0
    df2 = df_task2[[outcome, "age"]].copy()
    df2["task"] = 1
    
    combined = pd.concat([df1, df2], ignore_index=True).dropna()
    if len(combined) < 10:
        return None
    
    X = combined[["age", "task"]].values
    y = combined[outcome].values
    
    age = X[:, 0]
    task = X[:, 1]
    
    # Full model with interactions
    X_full = np.column_stack([
        np.ones(len(age)),
        age,
        age ** 2,
        task,
        age * task,
        (age ** 2) * task
    ])
    
    # Reduced model without interactions
    X_reduced = X_full[:, [0, 1, 2, 3]]
    
    model_full = LinearRegression().fit(X_full, y)
    model_reduced = LinearRegression().fit(X_reduced, y)
    
    ss_res_full = np.sum((y - model_full.predict(X_full)) ** 2)
    ss_res_reduced = np.sum((y - model_reduced.predict(X_reduced)) ** 2)
    
    n = len(y)
    if ss_res_reduced > ss_res_full:
        f_stat = ((ss_res_reduced - ss_res_full) / 2) / (ss_res_full / (n - 6))
        f_pval = 1 - stats.f.cdf(f_stat, 2, n - 6)
    else:
        f_stat, f_pval = 0, 1.0
    
    return {
        "task1": task1_name,
        "task2": task2_name,
        "outcome": outcome,
        "f_stat": f_stat,
        "p_value": f_pval,
    }

shape_comparisons = []
for outcome in outcomes:
    for t1, t2, df1, df2 in [
        ("5-CSRT", "TAV", df_5csrt_full, df_tav_full),
        ("5-CSRT", "DMS", df_5csrt_full, df_dms_full),
        ("TAV", "DMS", df_tav_full, df_dms_full)
    ]:
        result = compare_trajectory_shapes(df1, df2, outcome, t1, t2)
        if result:
            shape_comparisons.append(result)
            sig = "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"{outcome:12} | {t1:6} vs {t2:6}: F={result['f_stat']:5.2f}, p={result['p_value']:.4f} {sig}")

if shape_comparisons:
    pd.DataFrame(shape_comparisons).to_csv(os.path.join(output_dir, "trajectory_shape_comparisons.csv"), index=False)

# ============================================================================
# PART 2: MATCHED EXPOSURE COMPARISONS (First N attempts)
# ============================================================================

print("\n" + "="*60)
print("PART 2: MATCHED EXPOSURE COMPARISONS (FIRST N ATTEMPTS)")
print("="*60)

def build_matched_exposure_comparison(dict_5csrt, dict_control, control_name, elo_dict, demographics_df):
    """
    For each monkey: Take first N attempts from EACH task (in chronological order),
    where N = minimum attempts available across both tasks.
    Calculates age at midpoint of ELO window (not attempt dates).
    """
    results = []
    
    for species in dict_5csrt.keys():
        for name in dict_5csrt[species].keys():
            # Must be in both tasks
            if species not in dict_control or name not in dict_control[species]:
                continue
            
            if name not in elo_dict or elo_dict[name].empty:
                continue
            
            # Get attempts from both tasks
            att_5csrt = dict_5csrt[species][name]["attempts"].copy()
            att_control = dict_control[species][name]["attempts"].copy()
            
            att_5csrt["instant_begin"] = pd.to_datetime(att_5csrt["instant_begin"], unit="ms", errors="coerce")
            att_5csrt["instant_end"] = pd.to_datetime(att_5csrt["instant_end"], unit="ms", errors="coerce")
            att_control["instant_begin"] = pd.to_datetime(att_control["instant_begin"], unit="ms", errors="coerce")
            att_control["instant_end"] = pd.to_datetime(att_control["instant_end"], unit="ms", errors="coerce")
            
            # Restrict to ELO window
            elo_start = elo_dict[name]["Date"].min()
            elo_end = elo_dict[name]["Date"].max()
            
            att_5csrt = att_5csrt[
                (att_5csrt["instant_begin"] >= elo_start) &
                (att_5csrt["instant_end"] <= elo_end)
            ]
            
            att_control = att_control[
                (att_control["instant_begin"] >= elo_start) &
                (att_control["instant_end"] <= elo_end)
            ]
            
            if att_5csrt.empty or att_control.empty:
                continue
            
            # Find minimum attempts across both tasks
            n_min = min(len(att_5csrt), len(att_control))
            
            if n_min < 100:  # Minimum threshold
                continue
            
            # Take first n_min attempts from each task (chronologically)
            att_5csrt_matched = att_5csrt.sort_values("instant_begin").head(n_min)
            att_control_matched = att_control.sort_values("instant_begin").head(n_min)
            
            # Get demographics
            demo = demographics_df[(demographics_df["name"] == name) & (demographics_df["species"] == species)]
            if demo.empty:
                continue
            
            # CRITICAL FIX: Calculate age at midpoint of ELO WINDOW (same for both tasks)
            # This matches your prep.py approach
            task_midpoint = elo_start + (elo_end - elo_start) / 2
            
            birthdate = pd.to_datetime(demo["birthdate"].values[0])
            age_at_task = (task_midpoint - birthdate).days / 365.25
            
            gender = demo["gender"].values[0]
            
            # Calculate proportions for both tasks
            out_5csrt = att_5csrt_matched["result"].value_counts(normalize=True)
            out_control = att_control_matched["result"].value_counts(normalize=True)
            
            results.append({
                "name": name,
                "species": species,
                "gender": gender,
                "age": age_at_task,
                "control_task": control_name,
                "n_matched": n_min,
                "p_success_5csrt": out_5csrt.get("success", 0.0),
                "p_error_5csrt": out_5csrt.get("error", 0.0),
                "p_omission_5csrt": out_5csrt.get("stepomission", 0.0),
                "p_premature_5csrt": out_5csrt.get("prematured", 0.0),
                "p_success_control": out_control.get("success", 0.0),
                "p_error_control": out_control.get("error", 0.0),
                "p_omission_control": out_control.get("stepomission", 0.0),
                "p_premature_control": out_control.get("prematured", 0.0),
            })
    
    return pd.DataFrame(results)

df_5csrt_vs_tav_matched = build_matched_exposure_comparison(data, tav_dict, "TAV", elo, demographics)
df_5csrt_vs_dms_matched = build_matched_exposure_comparison(data, dms_dict, "DMS", elo, demographics)

print(f"\n5-CSRT vs TAV (matched exposure): {len(df_5csrt_vs_tav_matched)} individuals")
print(f"5-CSRT vs DMS (matched exposure): {len(df_5csrt_vs_dms_matched)} individuals")

if not df_5csrt_vs_tav_matched.empty:
    print(f"  Mean matched attempts: {df_5csrt_vs_tav_matched['n_matched'].mean():.0f} (range: {df_5csrt_vs_tav_matched['n_matched'].min():.0f}-{df_5csrt_vs_tav_matched['n_matched'].max():.0f})")

if not df_5csrt_vs_dms_matched.empty:
    print(f"  Mean matched attempts: {df_5csrt_vs_dms_matched['n_matched'].mean():.0f} (range: {df_5csrt_vs_dms_matched['n_matched'].min():.0f}-{df_5csrt_vs_dms_matched['n_matched'].max():.0f})")

# Save matched data
df_5csrt_vs_tav_matched.to_csv(os.path.join(output_dir, "5csrt_vs_tav_matched.csv"), index=False)
df_5csrt_vs_dms_matched.to_csv(os.path.join(output_dir, "5csrt_vs_dms_matched.csv"), index=False)

# ============================================================================
# CORRELATIONS WITH MATCHED EXPOSURE
# ============================================================================

print("\n" + "="*60)
print("MATCHED EXPOSURE CORRELATIONS")
print("="*60)

def correlate_matched_tasks(df):
    """Correlate tasks with matched exposure"""
    results = []
    
    for outcome in ["p_success", "p_error", "p_premature", "p_omission"]:
        col_5csrt = f"{outcome}_5csrt"
        col_control = f"{outcome}_control"
        
        data = df[[col_5csrt, col_control, "age"]].dropna()
        if len(data) < 5:
            continue
        
        # Raw correlation
        r_raw, p_raw = stats.pearsonr(data[col_5csrt], data[col_control])
        
        # Partial correlation controlling for age
        X_age = data[["age"]].values
        y_5csrt = data[col_5csrt].values
        y_control = data[col_control].values
        
        model_5csrt = LinearRegression().fit(X_age, y_5csrt)
        model_control = LinearRegression().fit(X_age, y_control)
        
        resid_5csrt = y_5csrt - model_5csrt.predict(X_age)
        resid_control = y_control - model_control.predict(X_age)
        
        r_partial, p_partial = stats.pearsonr(resid_5csrt, resid_control)
        
        results.append({
            "outcome": outcome,
            "n": len(data),
            "r_raw": r_raw,
            "p_raw": p_raw,
            "r_partial": r_partial,
            "p_partial": p_partial,
        })
    
    return pd.DataFrame(results)

if not df_5csrt_vs_tav_matched.empty:
    print(f"\n5-CSRT vs TAV (n={len(df_5csrt_vs_tav_matched)}):")
    corr_tav = correlate_matched_tasks(df_5csrt_vs_tav_matched)
    print(corr_tav.to_string(index=False))
    corr_tav.to_csv(os.path.join(output_dir, "correlations_5csrt_tav_matched.csv"), index=False)

if not df_5csrt_vs_dms_matched.empty:
    print(f"\n5-CSRT vs DMS (n={len(df_5csrt_vs_dms_matched)}):")
    corr_dms = correlate_matched_tasks(df_5csrt_vs_dms_matched)
    print(corr_dms.to_string(index=False))
    corr_dms.to_csv(os.path.join(output_dir, "correlations_5csrt_dms_matched.csv"), index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nResults saved to: {output_dir}/")
print(f"\nKey files:")
print(f"  TRAJECTORIES (all data):")
print(f"    - age_trajectories.csv")
print(f"    - trajectory_shape_comparisons.csv")
print(f"  CORRELATIONS (matched exposure):")
print(f"    - 5csrt_vs_tav_matched.csv")
print(f"    - 5csrt_vs_dms_matched.csv")
print(f"    - correlations_5csrt_tav_matched.csv")
print(f"    - correlations_5csrt_dms_matched.csv")