
# ======================
# ENHANCED GLM ANALYSIS 
# ======================

print("\n=== GLM ANALYSIS ON TRIMMED DATA ===")

# ---------------------------------------------------
# 1. FIX GENDER CONVERSION (CRITICAL FIX)
# ---------------------------------------------------
print("\nFixing gender encoding...")

# Debug original values
print("Original gender values:", combined_df_trimmed['gender'].unique())

def convert_gender(g):
    """Robust gender conversion handling multiple formats"""
    if pd.isna(g):
        return np.nan
    try:
        g = str(g).lower().strip()
        if g in ['1', 'male', 'm']:
            return 1
        elif g in ['2', 'female', 'f']:
            return 2
        else:
            print(f"Unexpected gender value: '{g}'")
            return np.nan
    except Exception as e:
        print(f"Error converting gender value {g}: {str(e)}")
        return np.nan

# Apply conversion
combined_df_trimmed['gender_numeric'] = combined_df_trimmed['gender'].apply(convert_gender)

# Verify conversion
print("\nGender conversion results:")
print("Numeric values:", combined_df_trimmed['gender_numeric'].unique())
print("Missing values:", combined_df_trimmed['gender_numeric'].isna().sum())
print("Value counts:")
print(combined_df_trimmed['gender_numeric'].value_counts(dropna=False))

# ---------------------------------------------------
# 2. PREPARE ANALYSIS DATASETS
# ---------------------------------------------------
print("\nPreparing analysis datasets...")

# Create complete-case datasets
glm_elo_df = combined_df_trimmed.dropna(subset=['elo_mean', 'age', 'gender_numeric']).copy()
glm_no_elo_df = combined_df_trimmed.dropna(subset=['age', 'gender_numeric']).copy()

# Convert to proper categorical types
for df in [glm_elo_df, glm_no_elo_df]:
    df['gender_factor'] = pd.Categorical(
        df['gender_numeric'].map({1: 'male', 2: 'female'}),
        categories=['male', 'female']
    )
    df['species_factor'] = pd.Categorical(df['species'])

# Diagnostic output
print(f"\nELO analysis sample size: {len(glm_elo_df)}")
print("Variables:")
print("- ELO range:", (glm_elo_df['elo_mean'].min(), glm_elo_df['elo_mean'].max()))
print("- Age range:", (glm_elo_df['age'].min(), glm_elo_df['age'].max()))
print("- Gender distribution:")
print(glm_elo_df['gender_factor'].value_counts(dropna=False))
print("- Species distribution:")
print(glm_elo_df['species_factor'].value_counts(dropna=False))

print(f"\nNon-ELO analysis sample size: {len(glm_no_elo_df)}")
print("Variables:")
print("- Age range:", (glm_no_elo_df['age'].min(), glm_no_elo_df['age'].max()))
print("- Gender distribution:")
print(glm_no_elo_df['gender_factor'].value_counts(dropna=False))
print("- Species distribution:")
print(glm_no_elo_df['species_factor'].value_counts(dropna=False))

# ---------------------------------------------------
# 3. RUN GLM ANALYSES
# ---------------------------------------------------

def run_glm(df, model_type):
    """Safe GLM execution with diagnostics"""
    print(f"\nRunning {model_type} model (n={len(df)})...")
    
    try:
        # Dynamic formula construction
        predictors = {
            'with_elo': "p_success ~ elo_mean + age + C(gender_factor) + C(species_factor)",
            'without_elo': "p_success ~ age + C(gender_factor) + C(species_factor)"
        }[model_type]
        
        model = ols(predictors, data=df).fit()
        
        # Check for separation
        if any(np.isinf(model.params)):
            print("Warning: Complete separation detected")
        
        print(model.summary())
        return model
        
    except Exception as e:
        print(f"GLM failed: {str(e)}")
        print("Debug info:")
        print(df[['p_success', 'age', 'gender_factor', 'species_factor'] + 
               (['elo_mean'] if model_type == 'with_elo' else [])].describe())
        return None

# Run analyses if sufficient data
MIN_SAMPLE_SIZE = 10

if len(glm_elo_df) >= MIN_SAMPLE_SIZE:
    elo_model = run_glm(glm_elo_df, 'with_elo')
else:
    print(f"\nInsufficient data for ELO analysis (n={len(glm_elo_df)} < {MIN_SAMPLE_SIZE})")

if len(glm_no_elo_df) >= MIN_SAMPLE_SIZE:
    no_elo_model = run_glm(glm_no_elo_df, 'without_elo')
else:
    print(f"\nInsufficient data for non-ELO analysis (n={len(glm_no_elo_df)} < {MIN_SAMPLE_SIZE})")

# ---------------------------------------------------
# 4. FINAL DIAGNOSTICS
# ---------------------------------------------------
print("\n=== FINAL DATA DIAGNOSTICS ===")
print("Missing values in complete dataset:")
print(combined_df_trimmed.isna().sum())

print("\nSpecies distribution in ELO analysis sample:")
if len(glm_elo_df) > 0:
    print(glm_elo_df['species_factor'].value_counts())
else:
    print("No data")

print("\nSpecies distribution in non-ELO analysis sample:")
if len(glm_no_elo_df) > 0:
    print(glm_no_elo_df['species_factor'].value_counts())
else:
    print("No data")


# =============================================
# ENHANCED GLM ANALYSIS WITH ADVANCED FEATURES
# =============================================

print("\n=== ENHANCED GLM ANALYSIS ===")

# ---------------------------------------------------
# 1. MULTICOLLINEARITY CHECK (VIF)
# ---------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df, predictors):
    """Calculate Variance Inflation Factors"""
    X = df[predictors].copy()
    X['intercept'] = 1  # Add intercept term
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(len(X.columns))]
    return vif_data.drop(vif_data[vif_data['Variable'] == 'intercept'].index)

# Check VIF for ELO model predictors
print("\nVIF for ELO model predictors:")
vif_results = check_vif(glm_elo_df, ['elo_mean', 'age'])
print(vif_results)

# Rule of thumb: VIF > 5-10 indicates problematic multicollinearity
if any(vif_results['VIF'] > 5):
    print("\nWarning: High multicollinearity detected in ELO model predictors")

# ---------------------------------------------------
# 2. NON-LINEAR AGE EFFECTS
# ---------------------------------------------------
# Add quadratic age term
glm_elo_df['age_squared'] = glm_elo_df['age']**2
glm_no_elo_df['age_squared'] = glm_no_elo_df['age']**2

# ---------------------------------------------------
# 3. MIXED-EFFECTS MODEL (using statsmodels)
# ---------------------------------------------------
import statsmodels.formula.api as smf

def run_mixed_model(df, formula, groups='name'):
    """Run mixed-effects model with random intercepts by monkey"""
    try:
        model = smf.mixedlm(formula, data=df, groups=df[groups]).fit()
        print(model.summary())
        return model
    except Exception as e:
        print(f"Mixed model failed: {str(e)}")
        return None

# ---------------------------------------------------
# 4. RUN ENHANCED MODELS
# ---------------------------------------------------

# Model formulas
elo_formula = ("p_success ~ elo_mean + age + age_squared + "
               "C(gender_factor) + C(species_factor)")
no_elo_formula = "p_success ~ age + age_squared + C(gender_factor) + C(species_factor)"

# Run standard OLS with quadratic age
print("\nRunning OLS with quadratic age term (ELO model):")
quad_elo_model = ols(elo_formula, data=glm_elo_df).fit()
print(quad_elo_model.summary())

print("\nRunning OLS with quadratic age term (non-ELO model):")
quad_no_elo_model = ols(no_elo_formula, data=glm_no_elo_df).fit()
print(quad_no_elo_model.summary())

# Run mixed-effects models
print("\nRunning mixed-effects model (ELO):")
mixed_elo_model = run_mixed_model(
    glm_elo_df,
    "p_success ~ elo_mean + age + age_squared + C(gender_factor) + C(species_factor)"
)

print("\nRunning mixed-effects model (non-ELO):")
mixed_no_elo_model = run_mixed_model(
    glm_no_elo_df,
    "p_success ~ age + age_squared + C(gender_factor) + C(species_factor)"
)

# ---------------------------------------------------
# 5. MODEL COMPARISON
# ---------------------------------------------------
from statsmodels.regression.linear_model import OLSResults

def compare_models(base_model, enhanced_model):
    """Compare model fit using likelihood ratio test"""
    if isinstance(enhanced_model, OLSResults):  # For OLS models
        print(f"\nR-squared improvement: {enhanced_model.rsquared - base_model.rsquared:.3f}")
    else:  # For mixed models
        print(f"\nAIC improvement: {base_model.aic - enhanced_model.aic:.1f}")
        if enhanced_model.aic < base_model.aic:
            print("Enhanced model provides better fit")

print("\n=== MODEL COMPARISONS ===")
print("ELO models:")
compare_models(elo_model, quad_elo_model)
compare_models(quad_elo_model, mixed_elo_model)

print("\nNon-ELO models:")
compare_models(no_elo_model, quad_no_elo_model)
compare_models(quad_no_elo_model, mixed_no_elo_model)

# ---------------------------------------------------
# 6. VISUALIZE NON-LINEAR EFFECTS
# ---------------------------------------------------
if 'age_squared' in quad_elo_model.params:
    print("\nVisualizing non-linear age effects...")
    plt.figure(figsize=(10,6))
    
    # Create prediction data
    age_range = np.linspace(glm_elo_df['age'].min(), glm_elo_df['age'].max(), 100)
    pred_data = pd.DataFrame({
        'age': age_range,
        'age_squared': age_range**2,
        'elo_mean': glm_elo_df['elo_mean'].median(),  # Hold constant at median
        'gender_factor': 'male',  # Reference category
        'species_factor': 'rhesus'  # Reference category
    })
    
    # Get predictions
    preds = quad_elo_model.get_prediction(pred_data).summary_frame()
    
    # Plot
    plt.plot(age_range, preds['mean'], color='blue', label='Predicted success')
    plt.fill_between(age_range, preds['mean_ci_lower'], preds['mean_ci_upper'], 
                    color='blue', alpha=0.1)
    plt.scatter(glm_elo_df['age'], glm_elo_df['p_success'], alpha=0.5, color='red')
    
    plt.xlabel('Age')
    plt.ylabel('Success Rate')
    plt.title('Non-linear Age Effects on Success Rate')
    plt.legend()
    plt.show()