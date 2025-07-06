import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from scipy.stats import mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def check_normality(data):
    stat, p = shapiro(data)
    return p > 0.05


def run_comparison(group1, group2, label1, label2, metric_name):
    normal1 = check_normality(group1)
    normal2 = check_normality(group2)

    mean1 = group1.mean()
    std1 = group1.std()
    median1 = group1.median()
    min1 = group1.min()
    max1 = group1.max()

    mean2 = group2.mean()
    std2 = group2.std()
    median2 = group2.median()
    min2 = group2.min()
    max2 = group2.max()

    direction = f"{label1} > {label2}" if mean1 > mean2 else f"{label2} > {label1}"

    if normal1 and normal2:
        stat, p = ttest_ind(group1, group2, equal_var=False)
        df = len(group1) + len(group2) - 2
        result_str = f"t({df}) = {stat:.2f}, p = {p:.4f}"
    else:
        stat, p = mannwhitneyu(group1, group2)
        result_str = f"U = {stat:.0f}, p = {p:.4f}"

    print(f"\n{metric_name}: {label1} vs {label2}")
    print(f"  {result_str}")
    print(
        f"  {label1} → Mean: {mean1:.3f}, SD: {std1:.3f}, Median: {median1:.3f}, Min: {min1:.3f}, Max: {max1:.3f}"
    )
    print(
        f"  {label2} → Mean: {mean2:.3f}, SD: {std2:.3f}, Median: {median2:.3f}, Min: {min2:.3f}, Max: {max2:.3f}"
    )
    print(f"  Direction: {direction}")

    if p < 0.05:
        corrected_p = smm.multipletests([p], method="bonferroni")[1][0]
        print(f"  Bonferroni-corrected p = {corrected_p:.4f}")


def run_correlation_tests(x, y, label):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    print(f"\n{label} Correlation:")
    print(f"  Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"  Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}")


def run_quadratic_model(df, dependent_var):
    formula = f"{dependent_var} ~ species * gender + np.power(age, 2)"
    model = smf.ols(formula, data=df).fit()
    print(f"\nModel for {dependent_var}")
    print(model.summary())


if __name__ == "__main__":
    # Load data
    rhesus_table = pd.read_excel(
        "C:/Users/vaiteku3/Desktop/simian-behavior/tables/rhesus_table.xlsx"
    )
    tonkean_table = pd.read_excel(
        "C:/Users/vaiteku3/Desktop/simian-behavior/tables/tonkean_table.xlsx"
    )

    combined_df = pd.concat(
        [
            rhesus_table.assign(species="Rhesus"),
            tonkean_table.assign(species="Tonkean"),
        ],
        ignore_index=True,
    )

    combined_df = combined_df.rename(
        columns={"mean_att#_/sess": "mean_att_per_sess", "#_attempts": "n_attempts"}
    )
    combined_df["species"] = combined_df["species"].astype("category")
    combined_df["gender"] = combined_df["gender"].astype("category")

    metrics = [
        "mean_att_per_sess",
        "n_attempts",
        "p_success",
        "p_error",
        "p_omission",
        "p_premature",
        "age",
        "rt_success",
        "rt_error",
    ]

    # Focused additional tests for mean_att_per_sess
    print("\n--- mean_att_per_sess: Species Comparison ---")
    run_comparison(
        combined_df[combined_df["species"] == "Rhesus"]["mean_att_per_sess"],
        combined_df[combined_df["species"] == "Tonkean"]["mean_att_per_sess"],
        "Rhesus",
        "Tonkean",
        "mean_att_per_sess",
    )

    print("\n--- mean_att_per_sess: Gender Comparison ---")
    run_comparison(
        combined_df[combined_df["gender"] == 1]["mean_att_per_sess"],
        combined_df[combined_df["gender"] == 2]["mean_att_per_sess"],
        "Males",
        "Females",
        "mean_att_per_sess",
    )

    print("\n--- mean_att_per_sess: Correlation with Age ---")
    run_correlation_tests(
        combined_df["mean_att_per_sess"],
        combined_df["age"],
        "mean_att_per_sess vs age",
    )

    tonkean_table_elo = pd.read_excel(
        "C:/Users/vaiteku3/Desktop/simian-behavior/data/py/hierarchy/tonkean_table_elo.xlsx"
    )
    if "mean_att#_/sess" in tonkean_table_elo.columns:
        tonkean_table_elo = tonkean_table_elo.rename(
            columns={"mean_att#_/sess": "mean_att_per_sess"}
        )

    print("\n--- mean_att_per_sess: Tonkean Elo Correlation ---")
    run_correlation_tests(
        tonkean_table_elo["mean_att_per_sess"],
        tonkean_table_elo["elo_mean"],
        "mean_att_per_sess vs elo_mean",
    )

    # Full analyses for all other metrics
    print("\n--- Species Comparisons ---")
    for metric in metrics:
        if metric != "mean_att_per_sess":
            run_comparison(
                combined_df[combined_df["species"] == "Rhesus"][metric],
                combined_df[combined_df["species"] == "Tonkean"][metric],
                "Rhesus",
                "Tonkean",
                metric,
            )

    print("\n--- Gender Comparisons ---")
    for metric in metrics:
        if metric != "mean_att_per_sess":
            run_comparison(
                combined_df[combined_df["gender"] == 1][metric],
                combined_df[combined_df["gender"] == 2][metric],
                "Males",
                "Females",
                metric,
            )

    print("\n--- Within-Species Gender Comparisons ---")
    for species in ["Rhesus", "Tonkean"]:
        species_df = combined_df[combined_df["species"] == species]
        males = species_df[species_df["gender"] == 1]
        females = species_df[species_df["gender"] == 2]
        for metric in metrics:
            if metric != "mean_att_per_sess":
                run_comparison(
                    males[metric],
                    females[metric],
                    f"{species} Males",
                    f"{species} Females",
                    f"{metric} ({species})",
                )

    print("\n--- Quadratic Models (species, gender, age²) ---")
    for metric in metrics:
        if metric != "mean_att_per_sess":
            run_quadratic_model(combined_df, metric)
