#!/usr/bin/env python3
"""
GEE Analysis for Social Status Effects in Macaque Data

Focus: How does social status affect cognitive performance in macaques?
"""

import os
import pickle
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod import families
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
from scipy.special import expit

# Make text editable in Adobe Illustrator
plt.rcParams['svg.fonttype'] = 'none'

# Configuration
BASE_DIR = "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt"
DATA_DIR = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
MATCHED_DATA_DIR = os.path.join(BASE_DIR, "dual-matched-analysis/derivatives")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/GEE")

def load_matched_data():
    """
    Load the dataset that eliminates confounds
    """
    
    # Load individual-level data
    rhesus_matched = pd.read_csv(os.path.join(MATCHED_DATA_DIR, "rhesus_unrestricted.csv"))
    tonkean_matched = pd.read_csv(os.path.join(MATCHED_DATA_DIR, "tonkean_unrestricted.csv"))
    
    # Combine matched samples
    matched_individuals = pd.concat([rhesus_matched, tonkean_matched], ignore_index=True)
    
    print(f"Loaded sample: {len(matched_individuals)} individuals")
    print(f"- Rhesus: {len(rhesus_matched)}")
    print(f"- Tonkean: {len(tonkean_matched)}")
    
    return matched_individuals

def load_trial_data_for_matched_individuals(matched_individuals):
    """
    Load trial-level data only for the individuals
    """
    
    # Load main data
    pickle_rick = os.path.join(DATA_DIR, "data.pickle")
    with open(pickle_rick, "rb") as handle:
        data = pickle.load(handle)
    
    # Load demographics
    mkid_rhesus = os.path.join(DATA_DIR, "mkid_rhesus.csv")
    mkid_tonkean = os.path.join(DATA_DIR, "mkid_tonkean.csv")
    
    demographics = {
        "mkid_rhesus": pd.read_csv(mkid_rhesus),
        "mkid_tonkean": pd.read_csv(mkid_tonkean),
    }
    
    # Standardize demographics
    for key in demographics:
        demographics[key]["name"] = demographics[key]["name"].str.lower()
        if "Age" in demographics[key].columns:
            demographics[key].rename(columns={"Age": "age"}, inplace=True)
    
    # Load ELO data
    tonkean_hierarchy_file = os.path.join(DATA_DIR, "hierarchy/tonkean/elo_matrix.xlsx")
    rhesus_hierarchy_file = os.path.join(DATA_DIR, "hierarchy/rhesus/elo_matrix.xlsx")
    
    tonkean_hierarchy = pd.read_excel(tonkean_hierarchy_file)
    rhesus_hierarchy = pd.read_excel(rhesus_hierarchy_file)
    
    # Name mappings (same as original script)
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
        "imo": "imoen", "ind": "indigo", "iro": "iron", "isi": "isis"
    }
    
    rhesus_abrv_2_fn = {
        "the": "theoden", "any": "anyanka", "djo": "djocko", "vla": "vladimir",
        "yel": "yelena", "bor": "boromir", "far": "faramir", "yva": "yvan",
        "baa": "baal", "spl": "spliff", "ol": "ol", "nat": "natasha",
        "arw": "arwen", "eow": "eowyn", "mar": "mar"
    }
    
    # Rename hierarchy columns
    tonkean_hierarchy = tonkean_hierarchy.rename(columns=tonkean_abrv_2_fn)
    rhesus_hierarchy = rhesus_hierarchy.rename(columns=rhesus_abrv_2_fn)
    
    # Remove monkeys with RFID issues
    monkeys_to_remove = {"joy", "jipsy"}
    for species in data:
        for name in list(data[species].keys()):
            if name in monkeys_to_remove:
                del data[species][name]
    
    # Build ELO dataset
    elo_list = []
    hierarchies = {"tonkean": tonkean_hierarchy, "rhesus": rhesus_hierarchy}
    
    for species, hierarchy in hierarchies.items():
        hierarchy["Date"] = pd.to_datetime(hierarchy["Date"])
        for col in hierarchy.columns[1:]:  # skip Date column
            temp = hierarchy[['Date', col]].dropna()
            if not temp.empty:
                temp['name'] = col
                temp['species'] = species
                temp = temp.rename(columns={col: 'elo_score'})
                elo_list.append(temp)
    
    all_elo = pd.concat(elo_list, ignore_index=True)
    
    # Get list of matched individual names
    matched_names = set(matched_individuals['name'])
    
    # Build trial-level dataset - ONLY for matched individuals
    attempts_list = []
    for species in ["tonkean", "rhesus"]:
        for name, monkey_data in data[species].items():
            
            # ONLY include individuals from the matched sample
            if name not in matched_names:
                continue
                
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

    if not attempts_list:
        raise ValueError("No trial data found for matched individuals")
        
    final_data = pd.concat(attempts_list, ignore_index=True)

    # Merge ELO scores based on nearest timestamp
    final_data = pd.merge_asof(
        final_data.sort_values('instant_begin'),
        all_elo.sort_values('Date')[['name', 'Date', 'elo_score']],
        left_on='instant_begin',
        right_on='Date',
        by='name',
        direction='nearest'
    )

    # Merge demographics INCLUDING BIRTHDATE
    demographics_combined = pd.concat([
        demographics["mkid_tonkean"].assign(species="tonkean"),
        demographics["mkid_rhesus"].assign(species="rhesus")
    ])
    final_data = final_data.merge(
        demographics_combined[['name', 'birthdate', 'gender']],  # NEW
        on='name',
        how='left'
    )
    
    # Convert birthdate to datetime
    final_data['birthdate'] = pd.to_datetime(final_data['birthdate'])
    
    # Calculate age at each trial (time-varying)
    final_data['age'] = (
        (final_data['instant_begin'] - final_data['birthdate']).dt.days / 365.25
    )

    # Create binary outcome variables
    result_types = ['success', 'error', 'premature', 'stepomission']
    for res in result_types:
        final_data[res] = (final_data['result'] == res).astype(int)

    # Standardize elo_score within species (or overall - your choice)
    final_data['elo_score_z'] = (final_data['elo_score'] - final_data['elo_score'].mean()) / final_data['elo_score'].std()

    # Prepare categorical variables
    final_data['species'] = pd.Categorical(final_data['species'])
    final_data['gender'] = pd.Categorical(final_data['gender'])

    # Create individual-day grouping variable for clustering
    final_data['date_only'] = final_data['instant_begin'].dt.date
    final_data['ind_day'] = final_data['name'].astype(str) + '_' + final_data['date_only'].astype(str)

    print(f"Trial-level data for matched individuals: {final_data.shape[0]:,} observations")
    print(f"From {final_data['name'].nunique()} individuals")
    
    return final_data, result_types

def run_focused_gee_analysis(final_data, result_types):
    """
    Run focused GEE analysis emphasizing social status effects
    """
    
    # Suppress warnings
    SET_USE_BIC_LLF(True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # models focused on the research question
    models_to_fit = {
        "social_status_main": "elo_score_z + age + gender + species",
        "social_status_age_interaction": "elo_score_z + age + elo_score_z:age + gender + species"
    }
    
    # Note: Species is deliberately excluded as main factor (can be added as control if needed)
    # Alternative: include species as control: "elo_score_z + age + gender + species"
    
    results = {}
    
    print(f"{'=' * 100}")
    print("FOCUSED GEE ANALYSIS: SOCIAL STATUS EFFECTS IN MATCHED DATA")
    print(f"{'=' * 100}")
    print(f"Dataset: {final_data.shape[0]:,} observations from {final_data['name'].nunique()} matched individuals")
    print(f"Age range: {final_data['age'].min():.1f} - {final_data['age'].max():.1f} years")
    print(f"ELO range: {final_data['elo_score'].min():.0f} - {final_data['elo_score'].max():.0f}")
    
    for outcome in result_types:
        print(f"\n{'=' * 80}")
        print(f"OUTCOME: {outcome.upper()}")
        print(f"{'=' * 80}")

        results[outcome] = {}

        for model_name, formula_rhs in models_to_fit.items():
            print(f"\n{'-' * 50}")
            print(f"Model: {model_name.replace('_', ' ').title()}")
            print(f"Formula: {outcome} ~ {formula_rhs}")
            print(f"{'-' * 50}")

            try:
                full_formula = f"{outcome} ~ {formula_rhs}"

                model = GEE.from_formula(
                    full_formula,
                    groups=final_data["ind_day"],
                    data=final_data,
                    family=families.Binomial(),
                    cov_struct=Exchangeable(),
                ).fit()

                results[outcome][model_name] = model

                # Print clean coefficient table
                coef_table = model.summary().tables[1]
                print(coef_table)

                # Print fit statistics
                print(f"\nModel Fit Statistics:")
                print(f"  AIC: {model.aic:.2f}")
                print(f"  BIC: {model.bic_llf:.2f}")
                print(f"  Log-Likelihood: {model.llf:.2f}")
                print(f"  N observations: {model.nobs:,}")
                print(f"  N clusters: {final_data['ind_day'].nunique():,}")

            except Exception as e:
                print(f"ERROR: Failed to fit model - {str(e)}")
                results[outcome][model_name] = None

    return results

def create_social_status_plots(results, final_data):
    """
    Create focused plots showing social status effects
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine best models for each outcome
    best_models = {}
    for outcome in results:
        if outcome in results:
            # Choose model with best fit (lowest AIC)
            valid_models = {k: v for k, v in results[outcome].items() if v is not None}
            if valid_models:
                best_model_name = min(valid_models, key=lambda x: valid_models[x].aic)
                best_models[outcome] = best_model_name
    
    for outcome, best_model_name in best_models.items():
        model = results[outcome][best_model_name]
        
        # Create outcome-specific plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Social Status Effects on {outcome.title()}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Main social status effect
        plot_main_social_status_effect(axes[0], model, final_data, outcome)
        
        # Plot 2: Social status by age (if interaction exists)
        if 'elo_score_z:age' in model.params.index:
            plot_social_status_age_interaction(axes[1], model, final_data, outcome)
        else:
            plot_age_effect(axes[1], model, final_data, outcome)
        
        # Plot 3: Social status by gender (if interaction exists)
        if 'elo_score_z:gender[T.2]' in model.params.index:
            plot_social_status_gender_interaction(axes[2], model, final_data, outcome)
        else:
            plot_gender_effect(axes[2], model, final_data, outcome)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(OUTPUT_DIR, f"{outcome}_social_status_effects.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        plt.show()

def create_comprehensive_figure(results, final_data):
    """
    Create comprehensive 4x3 figure showing all outcomes
    Left: Social Status main effect, Middle: Age main effect, Right: Age x Social Status interaction
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define outcome order and labels
    outcome_order = ['success', 'error', 'premature', 'stepomission']
    outcome_labels = {
        'success': 'Hits',
        'error': 'Errors', 
        'premature': 'Premature',
        'stepomission': 'Omissions'
    }
    
    # Determine best models for each outcome
    best_models = {}
    for outcome in outcome_order:
        if outcome in results:
            valid_models = {k: v for k, v in results[outcome].items() if v is not None}
            if valid_models:
                best_model_name = min(valid_models, key=lambda x: valid_models[x].aic)
                best_models[outcome] = (best_model_name, results[outcome][best_model_name])
    
    # Create figure - 4 rows x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(21, 28))
    
    for row_idx, outcome in enumerate(outcome_order):
        if outcome not in results:
            continue
            
        # Get both models
        main_model = results[outcome].get('social_status_main')
        int_model = results[outcome].get('social_status_age_interaction')
        
        # Column 0: Social Status main effect (from main model)
        ax = axes[row_idx, 0]
        if main_model is not None:
            plot_main_social_status_effect_comprehensive(
                ax, main_model, final_data, outcome, outcome_labels[outcome],
                show_ylabel=True, model_obj=main_model
            )
        
        # Column 1: Age main effect (from main model)
        ax = axes[row_idx, 1]
        if main_model is not None:
            plot_age_effect_comprehensive(
                ax, main_model, final_data, outcome, outcome_labels[outcome],
                show_ylabel=False, model_obj=main_model
            )
        
        # Column 2: Age x Social Status interaction - ALWAYS plot if model exists
        ax = axes[row_idx, 2]
        if int_model is not None and 'elo_score_z:age' in int_model.params.index:
            plot_social_status_age_interaction_comprehensive(
                ax, int_model, final_data, outcome, outcome_labels[outcome],
                show_ylabel=False, model_obj=int_model
            )
        else:
            # Show empty panel if no interaction model exists
            ax.text(0.5, 0.5, 'Model not available', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes, color='gray')
            ax.set_xlabel("Social Status (z)", fontsize=15)
            ax.set_title("Social Status × Age", fontsize=19, fontweight='bold', pad=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Adjust spacing to reduce clutter
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, "comprehensive_social_status_effects.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {save_path}")
    
    plt.show()

def plot_main_social_status_effect_comprehensive(ax, model, final_data, outcome, outcome_label, 
                                                 show_ylabel=True, model_obj=None):
    """Plot main social status effect for comprehensive figure with significance stars or n.s."""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_coef = coefs["elo_score_z"]
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    mean_age = final_data["age"].mean()
    age_coef = coefs.get("age", 0)
    
    logit_vals = intercept + elo_coef * elo_range + age_coef * mean_age
    prob_vals = expit(logit_vals)
    
    ax.plot(elo_range, prob_vals, color='#8B4A9C', linewidth=3, alpha=0.8)
    ax.fill_between(elo_range, prob_vals, alpha=0.2, color='#8B4A9C')
    
    ax.set_xlabel("Social Status (z)", fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f"P({outcome_label})", fontsize=15)
    
    # Add significance stars or n.s.
    title = "Social Status"
    if model_obj is not None:
        p_val = model_obj.pvalues.get("elo_score_z", 1.0)
        if p_val < 0.001:
            title += " ***"
        elif p_val < 0.01:
            title += " **"
        elif p_val < 0.05:
            title += " *"
        else:
            title += " n.s."
    
    ax.set_title(title, fontsize=19, fontweight='bold', pad=12)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.locator_params(axis='both', nbins=5)

def plot_social_status_age_interaction_comprehensive(ax, model, final_data, outcome, outcome_label,
                                                     show_ylabel=True, model_obj=None):
    """Plot social status × age interaction for comprehensive figure with significance stars or n.s."""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_main = coefs["elo_score_z"]
    age_main = coefs["age"]
    elo_age_int = coefs["elo_score_z:age"]
    
    age_quartiles = np.percentile(final_data["age"], [25, 50, 75])
    colors = ["#8B4A9C", "#B8860B", "#CD853F"]
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    for i, age in enumerate(age_quartiles):
        logit_vals = (intercept + age_main * age + 
                     (elo_main + elo_age_int * age) * elo_range)
        prob_vals = expit(logit_vals)
        
        ax.plot(elo_range, prob_vals, color=colors[i], linewidth=3,
               label=f"{age:.1f}y", alpha=0.8)
        ax.fill_between(elo_range, prob_vals, alpha=0.15, color=colors[i])
    
    ax.set_xlabel("Social Status (z)", fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f"P({outcome_label})", fontsize=15)
    
    # Add significance stars or n.s.
    title = "Social Status × Age"
    if model_obj is not None:
        p_val = model_obj.pvalues.get("elo_score_z:age", 1.0)
        if p_val < 0.001:
            title += " ***"
        elif p_val < 0.01:
            title += " **"
        elif p_val < 0.05:
            title += " *"
        else:
            title += " n.s."
    
    ax.set_title(title, fontsize=19, fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='best', framealpha=0.9, title='Age', title_fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.locator_params(axis='both', nbins=5)

def plot_age_effect_comprehensive(ax, model, final_data, outcome, outcome_label,
                                 show_ylabel=True, model_obj=None):
    """Plot age effect for comprehensive figure with significance stars or n.s."""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    age_coef = coefs["age"]
    elo_coef = coefs.get("elo_score_z", 0)
    
    age_range = np.linspace(final_data["age"].min(), 
                           final_data["age"].max(), 50)
    
    mean_elo = final_data["elo_score_z"].mean()
    
    logit_vals = intercept + elo_coef * mean_elo + age_coef * age_range
    prob_vals = expit(logit_vals)
    
    ax.plot(age_range, prob_vals, color='#2E8B57', linewidth=3, alpha=0.8)
    ax.fill_between(age_range, prob_vals, alpha=0.2, color='#2E8B57')
    
    ax.set_xlabel("Age", fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f"P({outcome_label})", fontsize=15)
    
    # Add significance stars or n.s.
    title = "Age"
    if model_obj is not None:
        p_val = model_obj.pvalues.get("age", 1.0)
        if p_val < 0.001:
            title += " ***"
        elif p_val < 0.01:
            title += " **"
        elif p_val < 0.05:
            title += " *"
        else:
            title += " n.s."
    
    ax.set_title(title, fontsize=19, fontweight='bold', pad=12)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)

def plot_social_status_gender_interaction_comprehensive(ax, model, final_data, outcome, outcome_label,
                                                        show_ylabel=True, show_legend=False):
    """Plot social status × gender interaction for comprehensive figure"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_main = coefs["elo_score_z"]
    gender_main = coefs.get("gender[T.2]", 0)
    elo_gender_int = coefs["elo_score_z:gender[T.2]"]
    age_coef = coefs.get("age", 0)
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    mean_age = final_data["age"].mean()
    
    logit_male = intercept + elo_main * elo_range + age_coef * mean_age
    prob_male = expit(logit_male)
    
    logit_female = (intercept + gender_main + 
                   (elo_main + elo_gender_int) * elo_range + 
                   age_coef * mean_age)
    prob_female = expit(logit_female)
    
    ax.plot(elo_range, prob_male, color='#4A90E2', linewidth=3, label='Male', alpha=0.8)
    ax.plot(elo_range, prob_female, color='#8E44AD', linewidth=3, label='Female', alpha=0.8)
    ax.fill_between(elo_range, prob_male, alpha=0.2, color='#4A90E2')
    ax.fill_between(elo_range, prob_female, alpha=0.2, color='#8E44AD')
    
    ax.set_xlabel("Social Status (z)", fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f"P({outcome_label})", fontsize=15)
    
    # Bigger title
    ax.set_title("Social Status × Gender", fontsize=19, fontweight='bold', pad=12)
    
    if show_legend:
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.locator_params(axis='both', nbins=5)

def plot_gender_effect_comprehensive(ax, model, final_data, outcome, outcome_label,
                                    show_ylabel=True, show_legend=False):
    """Plot gender effect for comprehensive figure"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    gender_coef = coefs.get("gender[T.2]", 0)
    elo_coef = coefs["elo_score_z"]
    age_coef = coefs.get("age", 0)
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    mean_age = final_data["age"].mean()
    
    logit_male = intercept + elo_coef * elo_range + age_coef * mean_age
    prob_male = expit(logit_male)
    
    logit_female = intercept + gender_coef + elo_coef * elo_range + age_coef * mean_age
    prob_female = expit(logit_female)
    
    ax.plot(elo_range, prob_male, color='#4A90E2', linewidth=3, label='Male', alpha=0.8)
    ax.plot(elo_range, prob_female, color='#8E44AD', linewidth=3, label='Female', alpha=0.8)
    ax.fill_between(elo_range, prob_male, alpha=0.2, color='#4A90E2')
    ax.fill_between(elo_range, prob_female, alpha=0.2, color='#8E44AD')
    
    ax.set_xlabel("Social Status (z)", fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f"P({outcome_label})", fontsize=15)
    
    # Bigger title
    ax.set_title("Gender", fontsize=19, fontweight='bold', pad=12)
    
    if show_legend:
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.locator_params(axis='both', nbins=5)
def plot_main_social_status_effect(ax, model, final_data, outcome):
    """Plot main social status effect"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_coef = coefs["elo_score_z"]
    
    # Use actual ELO range
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    # Calculate at mean age and reference gender
    mean_age = final_data["age"].mean()
    age_coef = coefs.get("age", 0)
    
    logit_vals = intercept + elo_coef * elo_range + age_coef * mean_age
    prob_vals = expit(logit_vals)
    
    ax.plot(elo_range, prob_vals, color='#8B4A9C', linewidth=3, alpha=0.8)
    ax.fill_between(elo_range, prob_vals, alpha=0.2, color='#8B4A9C')
    
    ax.set_xlabel("Social Status (standardized)")
    ax.set_ylabel(f"Probability of {outcome.title()}")
    ax.set_title("Main Social Status Effect")
    ax.grid(True, alpha=0.3)

def plot_social_status_age_interaction(ax, model, final_data, outcome):
    """Plot social status × age interaction"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_main = coefs["elo_score_z"]
    age_main = coefs["age"]
    elo_age_int = coefs["elo_score_z:age"]
    
    # Use age quartiles
    age_quartiles = np.percentile(final_data["age"], [25, 50, 75])
    colors = ["#8B4A9C", "#B8860B", "#CD853F"]
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    for i, age in enumerate(age_quartiles):
        logit_vals = (intercept + age_main * age + 
                     (elo_main + elo_age_int * age) * elo_range)
        prob_vals = expit(logit_vals)
        
        ax.plot(elo_range, prob_vals, color=colors[i], linewidth=3,
               label=f"Age {age:.1f}y", alpha=0.8)
        ax.fill_between(elo_range, prob_vals, alpha=0.15, color=colors[i])
    
    ax.set_xlabel("Social Status (standardized)")
    ax.set_ylabel(f"Probability of {outcome.title()}")
    ax.set_title("Social Status × Age")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_age_effect(ax, model, final_data, outcome):
    """Plot age effect when no interaction"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    age_coef = coefs["age"]
    elo_coef = coefs["elo_score_z"]
    
    age_range = np.linspace(final_data["age"].min(), 
                           final_data["age"].max(), 50)
    
    # At mean social status
    mean_elo = final_data["elo_score_z"].mean()
    
    logit_vals = intercept + elo_coef * mean_elo + age_coef * age_range
    prob_vals = expit(logit_vals)
    
    ax.plot(age_range, prob_vals, color='#2E8B57', linewidth=3, alpha=0.8)
    ax.fill_between(age_range, prob_vals, alpha=0.2, color='#2E8B57')
    
    ax.set_xlabel("Age")
    ax.set_ylabel(f"Probability of {outcome.title()}")
    ax.set_title("Age Effect")
    ax.grid(True, alpha=0.3)

def plot_social_status_gender_interaction(ax, model, final_data, outcome):
    """Plot social status × gender interaction"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    elo_main = coefs["elo_score_z"]
    gender_main = coefs.get("gender[T.2]", 0)
    elo_gender_int = coefs["elo_score_z:gender[T.2]"]
    age_coef = coefs.get("age", 0)
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    mean_age = final_data["age"].mean()
    
    # Male (reference)
    logit_male = intercept + elo_main * elo_range + age_coef * mean_age
    prob_male = expit(logit_male)
    
    # Female 
    logit_female = (intercept + gender_main + 
                   (elo_main + elo_gender_int) * elo_range + 
                   age_coef * mean_age)
    prob_female = expit(logit_female)
    
    ax.plot(elo_range, prob_male, color='#4A90E2', linewidth=3, label='Male', alpha=0.8)
    ax.plot(elo_range, prob_female, color='#8E44AD', linewidth=3, label='Female', alpha=0.8)
    ax.fill_between(elo_range, prob_male, alpha=0.2, color='#4A90E2')
    ax.fill_between(elo_range, prob_female, alpha=0.2, color='#8E44AD')
    
    ax.set_xlabel("Social Status (standardized)")
    ax.set_ylabel(f"Probability of {outcome.title()}")
    ax.set_title("Social Status × Gender")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_gender_effect(ax, model, final_data, outcome):
    """Plot gender effect when no interaction"""
    
    coefs = model.params
    intercept = coefs["Intercept"]
    gender_coef = coefs.get("gender[T.2]", 0)
    elo_coef = coefs["elo_score_z"]
    age_coef = coefs.get("age", 0)
    
    elo_range = np.linspace(final_data["elo_score_z"].min(), 
                           final_data["elo_score_z"].max(), 50)
    
    mean_age = final_data["age"].mean()
    
    # Male (reference)
    logit_male = intercept + elo_coef * elo_range + age_coef * mean_age
    prob_male = expit(logit_male)
    
    # Female
    logit_female = intercept + gender_coef + elo_coef * elo_range + age_coef * mean_age
    prob_female = expit(logit_female)
    
    ax.plot(elo_range, prob_male, color='#4A90E2', linewidth=3, label='Male', alpha=0.8)
    ax.plot(elo_range, prob_female, color='#8E44AD', linewidth=3, label='Female', alpha=0.8)
    ax.fill_between(elo_range, prob_male, alpha=0.2, color='#4A90E2')
    ax.fill_between(elo_range, prob_female, alpha=0.2, color='#8E44AD')
    
    ax.set_xlabel("Social Status (standardized)")
    ax.set_ylabel(f"Probability of {outcome.title()}")
    ax.set_title("Gender Effect")
    ax.legend()
    ax.grid(True, alpha=0.3)

from statsmodels.stats.multitest import multipletests

def select_best_models_and_report(results, alpha=0.05):
    """
    Select best model per outcome using AIC, then report with optional FDR correction
    """
    best_models = {}
    model_comparison = []
    
    print("\n" + "="*100)
    print("MODEL SELECTION AND INFERENCE")
    print("="*100)
    # Step 1: Model selection via AIC (pairwise comparison)
    for outcome in results:
        print(f"\n{outcome.upper()}:")
        valid_models = {k: v for k, v in results[outcome].items() if v is not None}
        
        if valid_models:
            # Get the two models (assuming only main and interaction models)
            main_model = valid_models.get('social_status_main')
            int_model = valid_models.get('social_status_age_interaction')
            
            if main_model and int_model:
                # Calculate ΔAIC as: Main AIC - Interaction AIC
                main_aic = main_model.aic
                int_aic = int_model.aic
                delta_aic = main_aic - int_aic
                
                # Determine best model
                if delta_aic > 2:
                    best_model_name = 'social_status_age_interaction'
                    best_model = int_model
                    status = "Interaction model better"
                elif delta_aic < -2:
                    best_model_name = 'social_status_main'
                    best_model = main_model
                    status = "Main model better"
                else:
                    # When similar, choose simpler model (main)
                    best_model_name = 'social_status_main'
                    best_model = main_model
                    status = "Models similar"
                
                best_models[outcome] = (best_model_name, best_model)
                
                print(f"  Main model AIC: {main_aic:.2f}")
                print(f"  Interaction model AIC: {int_aic:.2f}")
                print(f"  ΔAIC (Main - Interaction): {delta_aic:.2f}")
                print(f"  Selected: {best_model_name} ({status})")
            
            elif main_model:
                best_models[outcome] = ('social_status_main', main_model)
                print(f"  Only main model available (AIC = {main_model.aic:.2f})")
            elif int_model:
                best_models[outcome] = ('social_status_age_interaction', int_model)
                print(f"  Only interaction model available (AIC = {int_model.aic:.2f})")
    # Step 2: Collect p-values from best models only
    all_pvalues = []
    param_info = []
    
    for outcome, (model_name, model) in best_models.items():
        for param, pval in model.pvalues.items():
            if param != 'Intercept':  # Typically exclude intercept from correction
                all_pvalues.append(pval)
                param_info.append({
                    'outcome': outcome,
                    'parameter': param,
                    'coefficient': model.params[param],
                    'p_value': pval
                })
    
    # Step 3: Apply FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        all_pvalues, 
        alpha=alpha, 
        method='fdr_bh'
    )
    
    # Step 4: Create results table
    for i, info in enumerate(param_info):
        info['p_adjusted'] = pvals_corrected[i]
        info['significant_uncorrected'] = info['p_value'] < alpha
        info['significant_FDR'] = reject[i]
    
    results_df = pd.DataFrame(param_info)
    
    # Step 5: Summary reporting
    print("\n" + "="*100)
    print("INFERENCE: SELECTED MODELS WITH FDR CORRECTION")
    print("="*100)
    print(f"Total tests: {len(all_pvalues)}")
    print(f"Significant (uncorrected α = {alpha}): {sum(results_df['significant_uncorrected'])}")
    print(f"Significant (FDR-corrected α = {alpha}): {sum(results_df['significant_FDR'])}")
    
    print("\n" + "-"*100)
    print("SIGNIFICANT EFFECTS (FDR-corrected):")
    print("-"*100)
    
    sig_results = results_df[results_df['significant_FDR']].sort_values('p_adjusted')
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            print(f"{row['outcome']:12s} | {row['parameter']:25s} | "
                  f"β = {row['coefficient']:7.4f} | "
                  f"p = {row['p_value']:.4f} | p_adj = {row['p_adjusted']:.4f}")
    else:
        print("No effects survived FDR correction")
    
    # Also show effects that were significant before correction but not after
    print("\n" + "-"*100)
    print("EFFECTS SIGNIFICANT BEFORE BUT NOT AFTER FDR CORRECTION:")
    print("-"*100)
    
    lost_sig = results_df[
        results_df['significant_uncorrected'] & ~results_df['significant_FDR']
    ].sort_values('p_value')
    
    if len(lost_sig) > 0:
        for _, row in lost_sig.iterrows():
            print(f"{row['outcome']:12s} | {row['parameter']:25s} | "
                  f"β = {row['coefficient']:7.4f} | "
                  f"p = {row['p_value']:.4f} | p_adj = {row['p_adjusted']:.4f}")
    else:
        print("All significant effects survived FDR correction")
    
    return best_models, results_df

# Add this function right before the main() function (around line 882)

def create_social_status_individual_plots(results, final_data, best_models):
    """
    Create 4 separate plots showing social status effects from the SELECTED model for each outcome.
    - SUCCESS: from main model (no interaction)
    - ERROR, PREMATURE, OMISSIONS: from interaction model (showing age stratification)
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all fitted models
    final_data : DataFrame
        The trial-level data
    best_models : dict
        Dictionary mapping outcome -> (model_name, model_object)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define outcome order and labels
    outcome_order = ['success', 'error', 'premature', 'stepomission']
    outcome_labels = {
        'success': 'Hits',
        'error': 'Errors', 
        'premature': 'Premature',
        'stepomission': 'Omissions'
    }
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Define color scheme (smooth purple to blue gradient)
    colors = ["#8B5FBF", "#6A8FD8", "#4DB8E8"]  # Purple, Mid Blue-Purple, Light Blue
    age_labels = ["Young", "Middle", "Old"]
    
    # Track if we've added the legend
    legend_added = False
    
    for idx, outcome in enumerate(outcome_order):
        ax = axes[idx]
        
        if outcome not in best_models:
            ax.text(0.5, 0.5, 'No model available', ha='center', va='center', 
                   fontsize=18, transform=ax.transAxes, color='gray')
            ax.set_title(outcome_labels[outcome], fontsize=22, fontweight='bold')
            continue
        
        model_name, model = best_models[outcome]
        
        # Get model coefficients
        coefs = model.params
        intercept = coefs["Intercept"]
        elo_coef = coefs["elo_score_z"]
        age_coef = coefs.get("age", 0)
        
        elo_range = np.linspace(final_data["elo_score_z"].min(), 
                               final_data["elo_score_z"].max(), 100)
        
        # Check if this is the interaction model
        if model_name == 'social_status_age_interaction' and 'elo_score_z:age' in coefs.index:
            # Plot with age stratification
            elo_age_int = coefs["elo_score_z:age"]
            
            # Use age quartiles for stratification
            age_quartiles = np.percentile(final_data["age"], [25, 50, 75])
            
            for i, age in enumerate(age_quartiles):
                logit_vals = (intercept + age_coef * age + 
                             (elo_coef + elo_age_int * age) * elo_range)
                prob_vals = expit(logit_vals)
                
                # Only add label for the first subplot with interaction
                label = f"{age_labels[i]} ({age:.1f}y)" if not legend_added else None
                
                ax.plot(elo_range, prob_vals, color=colors[i], linewidth=3,
                       label=label, alpha=0.8)
                ax.fill_between(elo_range, prob_vals, alpha=0.15, color=colors[i])
            
            # Add legend only once (to the first subplot with interaction model)
            if not legend_added:
                ax.legend(fontsize=16, loc='best', framealpha=0.9)
                legend_added = True
            
        else:
            # Plot simple main effect (SUCCESS case)
            mean_age = final_data["age"].mean()
            logit_vals = intercept + elo_coef * elo_range + age_coef * mean_age
            prob_vals = expit(logit_vals)
            
            ax.plot(elo_range, prob_vals, color='#8B5FBF', linewidth=3, alpha=0.8)
            ax.fill_between(elo_range, prob_vals, alpha=0.2, color='#8B5FBF')
        
        # Add labels and formatting
        # Only show x-axis label for bottom row (idx >= 2)
        if idx >= 2:
            ax.set_xlabel("Social Status", fontsize=18)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)
        
        # Only show y-axis label for left column (idx 0 or 2)
        if idx % 2 == 0:
            ax.set_ylabel("Probability", fontsize=18)
        else:
            ax.set_ylabel("")
        
        # Add significance stars to title
        # For interaction models, check the interaction term; otherwise check main effect
        if model_name == 'social_status_age_interaction' and 'elo_score_z:age' in coefs.index:
            p_val = model.pvalues.get("elo_score_z:age", 1.0)
        else:
            p_val = model.pvalues.get("elo_score_z", 1.0)
        
        title = outcome_labels[outcome]
        if p_val < 0.001:
            title += " ***"
        elif p_val < 0.01:
            title += " **"
        elif p_val < 0.05:
            title += " *"
        else:
            title += " n.s."
        
        ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Make axis tick labels more sparse
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save the plot as SVG
    save_path = os.path.join(OUTPUT_DIR, "social_status_effects_by_outcome.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Social status individual plots saved to: {save_path}")
    
    plt.show()


# Then update your main() function to call this new function:

def main():
    """
    Main function for GEE analysis
    """
    
    print("Starting GEE Analysis on Data...")
    print("=" * 60)
    
    # Load matched individuals
    matched_individuals = load_matched_data()
    
    # Load trial-level data for matched individuals only
    final_data, result_types = load_trial_data_for_matched_individuals(matched_individuals)
    
    # Run focused GEE analysis
    results = run_focused_gee_analysis(final_data, result_types)
    
    # NEW: Model selection and FDR correction
    best_models, fdr_results = select_best_models_and_report(results, alpha=0.05)
    
    # Save FDR results
    fdr_path = os.path.join(OUTPUT_DIR, "model_selection_and_fdr_results.csv")
    fdr_results.to_csv(fdr_path, index=False)
    print(f"\nFDR-corrected results saved to: {fdr_path}")
    
    # Create comprehensive figure (existing 4x3 grid)
    create_comprehensive_figure(results, final_data)
    
    # NEW: Create individual social status plots from selected models (2x2 grid)
    create_social_status_individual_plots(results, final_data, best_models)
    
    # Summary
    print(f"\n{'=' * 100}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 100}")
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    
    # Reset warnings
    warnings.resetwarnings()
    

if __name__ == "__main__":
    main()