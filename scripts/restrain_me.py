import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings  # they annoy me
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
plt.rcParams['svg.fonttype'] = 'none'  

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py' # string
pickle_rick = os.path.join(directory, 'data.pickle')
with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

rhesus = []
tonkean = []

for species, species_data in data.items():
    count = len(species_data)

    for name, monkey_data in species_data.items():
        attempts = monkey_data['attempts']
        
        attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
        attempts['instant_end'] = pd.to_datetime(attempts['instant_end'], unit='ms', errors='coerce')

        first_attempt = attempts['instant_begin'].min().date()  # overall onset of task egagement 
        last_attempt = attempts['instant_end'].max().date()  # overall offset
 
        attempt_count = attempts['id'].nunique()  # count should increase once i remove '> 10' condition from pickle rick
        session_count = attempts['session'].nunique()  # likewise
        
        outcome = attempts['result'].value_counts(normalize=True)
        
        rt_success = attempts[attempts['result'] == 'success']['reaction_time'].mean()  # rts populated in pickle rick, correspond to 'answer' step 
        rt_error = attempts[attempts['result'] == 'error']['reaction_time'].mean()  # likewise
                
        if species == 'rhesus':
            rhesus.append({'name': name, '#_attempts': attempt_count, '#_sessions': session_count, 'start_date': first_attempt, 'end_date': last_attempt,
                           'p_success': outcome.get('success', 0.0),
                           'p_error': outcome.get('error', 0.0),
                           'p_omission': outcome.get('stepomission', 0.0),
                           'p_premature': outcome.get('prematured', 0.0),
                           'rt_success': rt_success,
                           'rt_error': rt_error,
                           'mean_sess_duration': attempts['session_duration'].mean(),  # tried to plot but not particularly informative
                           'mean_att#_/sess': attempts.groupby('session')['id'].count().mean()})  # mean attempt count per session
        elif species == 'tonkean':
            tonkean.append({'name': name, '#_attempts': attempt_count, '#_sessions': session_count, 'start_date': first_attempt, 'end_date': last_attempt,
                            'p_success': outcome.get('success', 0.0),
                            'p_error': outcome.get('error', 0.0),
                            'p_omission': outcome.get('stepomission', 0.0),
                            'p_premature': outcome.get('prematured', 0.0),
                            'rt_success': rt_success,
                            'rt_error': rt_error,
                            'mean_sess_duration': attempts['session_duration'].mean(),
                            'mean_att#_/sess': attempts.groupby('session')['id'].count().mean()})

rhesus_table = pd.DataFrame(rhesus).round(2)  # remove .round(2) for final plots
tonkean_table = pd.DataFrame(tonkean).round(2)

# load tonkean_table_elo that's been aligned with elo-rating period (see social_status_bby.py) 
tonkean_table_elo = pd.read_excel(os.path.join(directory, 'hierarchy', 'tonkean_table_elo.xlsx'))

print()
print("macaca mulatta (n = 16)")
print(rhesus_table) 
print()
print("macaca tonkeana (n = 35)")
print(tonkean_table) 
print()
print("macaca tonkeana (n = 33) incl. elo")
print(tonkean_table_elo)

# For Rhesus macaques
rhesus_sessions_range = {
    'min_sessions': rhesus_table['#_sessions'].min(),
    'max_sessions': rhesus_table['#_sessions'].max(),
    'mean_sessions': rhesus_table['#_sessions'].mean()
}

rhesus_attempts_range = {
    'min_attempts': rhesus_table['#_attempts'].min(),
    'max_attempts': rhesus_table['#_attempts'].max(),
    'mean_attempts': rhesus_table['#_attempts'].mean()
}

# For Tonkean macaques
tonkean_sessions_range = {
    'min_sessions': tonkean_table['#_sessions'].min(),
    'max_sessions': tonkean_table['#_sessions'].max(),
    'mean_sessions': tonkean_table['#_sessions'].mean()
}

tonkean_attempts_range = {
    'min_attempts': tonkean_table['#_attempts'].min(),
    'max_attempts': tonkean_table['#_attempts'].max(),
    'mean_attempts': tonkean_table['#_attempts'].mean()
}

# Output the results
print("Rhesus Sessions Range:")
print(f"Min Sessions: {rhesus_sessions_range['min_sessions']}")
print(f"Max Sessions: {rhesus_sessions_range['max_sessions']}")
print(f"Mean Sessions: {rhesus_sessions_range['mean_sessions']:.2f}")
print(f"Rhesus Mean Number of Attempts: {rhesus_attempts_range['mean_attempts']:.2f}")
print(f"Min Attempts: {rhesus_attempts_range['min_attempts']}")
print(f"Max Attempts: {rhesus_attempts_range['max_attempts']}")

print("\nTonkean Sessions Range:")
print(f"Min Sessions: {tonkean_sessions_range['min_sessions']}")
print(f"Max Sessions: {tonkean_sessions_range['max_sessions']}")
print(f"Mean Sessions: {tonkean_sessions_range['mean_sessions']:.2f}")
print(f"Tonkean Mean Number of Attempts: {tonkean_attempts_range['mean_attempts']:.2f}")
print(f"Min Attempts: {tonkean_attempts_range['min_attempts']}")
print(f"Max Attempts: {tonkean_attempts_range['max_attempts']}")


# let's append demographics 
# 1 = male, 2 = female 

demographics = {}

mkid_rhesus = directory + '/mkid_rhesus.csv'
mkid_tonkean = directory + '/mkid_tonkean.csv'

demographics = {
    'mkid_rhesus': pd.read_csv(mkid_rhesus),
    'mkid_tonkean': pd.read_csv(mkid_tonkean)
}

# switch names 2 lowercase 
demographics['mkid_tonkean']['name'] = demographics['mkid_tonkean']['name'].str.lower()  
demographics['mkid_rhesus']['name'] = demographics['mkid_rhesus']['name'].str.lower()

# im too OCD 
demographics['mkid_tonkean'].rename(columns={'Age': 'age'}, inplace=True)
demographics['mkid_rhesus'].rename(columns={'Age': 'age'}, inplace=True)

print("rhesus demographics (n = 16)")
print(demographics['mkid_rhesus'])
print()
print("tonkean demographics (n = 35)")
print(demographics['mkid_tonkean'])

if 'mkid_tonkean' in demographics:
    tonkean_table = tonkean_table.join(demographics['mkid_tonkean'].set_index('name')[['id', 'gender', 'birthdate', 'age']],
                                     on='name', how='left')
    tonkean_table_elo = tonkean_table_elo.join(demographics['mkid_tonkean'].set_index('name')[['id', 'gender', 'birthdate', 'age']],
                                     on='name', how='left')
else:
    print("whoops, 'mkid_tonkean' not found ~.")

if 'mkid_rhesus' in demographics:
    rhesus_table = rhesus_table.join(demographics['mkid_rhesus'].set_index('name')[['id', 'gender', 'birthdate', 'age']],
                                     on='name', how='left')
else:
    print("whoops, 'mkid_rhesus' not found ~.")

save_directory = '/Users/similovesyou/Downloads/'
rhesus_table.to_excel(os.path.join(save_directory, 'rhesus_table.xlsx'), index=False)
tonkean_table.to_excel(os.path.join(save_directory, 'tonkean_table.xlsx'), index=False)

# tables now summarize device engagement, task performance & demographics
print()
print("tonkean task engagement")
print(tonkean_table)  
print()
print("rhesus task engagement")
print(rhesus_table)
print()
print("tonkean task engagement (incl. elo)")
print(tonkean_table_elo)  

plt.rcParams['font.family'] = 'Arial' # set global font to be used across all plots ^^ **change later**

# plotting proportions across the lifespan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def p_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel):
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='steelblue', s=80, linewidth=1)
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 2], x='age', y=metric, color=colors['rhesus'], 
                    s=80, alpha=0.6, edgecolor='steelblue', linewidth=1)

    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='purple', s=80, linewidth=1)
    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 2], x='age', y=metric, color=colors['tonkean'], 
                    s=80, alpha=0.6, edgecolor='purple', linewidth=1)

    sns.regplot(ax=ax, data=rhesus_table, x='age', y=metric, color=colors['rhesus'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=tonkean_table, x='age', y=metric, color=colors['tonkean'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=pd.concat([rhesus_table, tonkean_table]), x='age', y=metric, color='black', order=2, 
                scatter=False, ci=None)
    
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0', '', '', '', '', '1'])
    ax.grid(True, which='both', linestyle='--')
    ax.grid(True, which='major', axis='x')

    # Remove bold x and y axes but keep thick tick marks
    ax.spines['bottom'].set_linewidth(1)  # Default thickness
    ax.spines['left'].set_linewidth(1)    # Default thickness
    ax.tick_params(axis='both', which='major', width=2.5, length=7)  # Thicker and longer ticks

    if show_xlabel:
        ax.set_xlabel('Age', fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    if show_ylabel:
        ax.set_ylabel('Proportion', fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')

sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "arial"}) 

fig, axes = plt.subplots(1, 4, figsize=(20, 6))  # Keep in a row

# Adjusted colors
colors = {'tonkean': 'purple', 'rhesus': 'steelblue'}

metrics = ['p_success', 'p_error', 'p_premature', 'p_omission']
titles = ['Hits', 'Errors', 'Premature', 'Omissions']

for ax, metric, title, show_xlabel, show_ylabel in zip(axes, metrics, titles, [True] * 4, [True] + [False] * 3):
    p_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

handles = [
    plt.Line2D([], [], color=colors['rhesus'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='steelblue', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (m)'),
    plt.Line2D([], [], color=colors['tonkean'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='purple', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (m)')
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

fig.suptitle('Simian Success: How well do they play?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')

plt.tight_layout(rect=[0, 0.02, 1, 0.93])

# Save as SVG
save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/plots/simian_success.svg"
plt.savefig(save_path, format='svg')

plt.show()


def rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel):
    # Adjust very low values to sit slightly above the x-axis
    rhesus_table[metric] = np.where(rhesus_table[metric] < 0.02, 0.02, rhesus_table[metric])
    tonkean_table[metric] = np.where(tonkean_table[metric] < 0.02, 0.02, tonkean_table[metric])

    # Scatter plot for rhesus monkeys
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='steelblue', s=80, linewidth=1)
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 2], x='age', y=metric, color=colors['rhesus'], 
                    s=80, alpha=0.6, edgecolor='steelblue', linewidth=1)

    # Scatter plot for tonkean monkeys
    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='purple', s=80, linewidth=1)
    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 2], x='age', y=metric, color=colors['tonkean'], 
                    s=80, alpha=0.6, edgecolor='purple', linewidth=1)
    
    # Regression lines
    sns.regplot(ax=ax, data=rhesus_table, x='age', y=metric, color=colors['rhesus'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=tonkean_table, x='age', y=metric, color=colors['tonkean'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=pd.concat([rhesus_table, tonkean_table]), x='age', y=metric, color='black', order=2, 
                scatter=False, ci=None)
    
    # Titles and labels
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.grid(True, which='both', linestyle='--')
    ax.grid(True, which='major', axis='x')
    
    if show_xlabel:
        ax.set_xlabel('Age', fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    if show_ylabel:
        ax.set_ylabel('RTs (ms)', fontsize=14)
    else:
        ax.set_ylabel('')

# Set plot style
sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "arial"})

# Create subplots with the fixed height (16x7)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# **Adjusted Colors**
colors = {'tonkean': 'purple', 'rhesus': 'steelblue'}

metrics = ['rt_success', 'rt_error']
titles = ['Hits', 'Error']

for ax, metric, title, show_xlabel, show_ylabel in zip(axes.flatten(), metrics, titles, [True, True], [True, False]):
    rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

# Legend (Reusing handles)
handles = [
    plt.Line2D([], [], color=colors['rhesus'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='steelblue', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (m)'),
    plt.Line2D([], [], color=colors['tonkean'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='purple', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (m)')
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# Main title
fig.suptitle('Swift Simian: How fast do they react?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as SVG
save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/plots/swift_simian.svg"
plt.savefig(save_path, format='svg')

plt.show()


# plotting session frequencies (mean attempt # per session)
combined_df = pd.concat([  # combine both species into one df
    rhesus_table.assign(species='Rhesus'),
    tonkean_table.assign(species='Tonkean')
]).drop_duplicates(subset='name', ignore_index=True)

# ffs these warnings r gonna end me 

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    
    plt.figure(figsize=(10, 8))    
    jitter = 0.1
    colors = {'Rhesus': 'steelblue', 'Tonkean': 'purple'}  # Adjusted colors
    
    for species in ['Rhesus', 'Tonkean']:
        males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]  # males hollow
        females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]  # females filled

        sns.stripplot(x='species', y='age', data=females, jitter=jitter, marker='o', 
                      color=colors[species], alpha=0.6, size=10)
        sns.stripplot(x='species', y='age', data=males, jitter=jitter, marker='o', 
                      facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5)
    
    for species in ['Rhesus', 'Tonkean']:  # Loop to plot error bars for each species (irrespective of gender)
        subset = combined_df[combined_df['species'] == species] 
        mean_age = subset['age'].mean() 
        std_age = subset['age'].std()
        plt.errorbar(x=[species], y=[mean_age], yerr=[std_age], fmt='none', 
                     ecolor=colors[species], capsize=5, elinewidth=2)
    
    plt.ylabel('Age', fontsize=14)  # Increased y-label size
    plt.title('Demographics', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')  # Updated title

    plt.xticks(ticks=[0, 1], 
               labels=[f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
                       f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})'], 
               fontsize=14)  # Increased x-tick label size
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Remove decimals from age
    plt.ylim(0, 25)
    plt.xlim(-0.4, 1.4)  # Adjust x-lims to move the plots together
    
    plt.xlabel('')  # Removed x-axis label

    # Save as SVG
    save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/plots/demographics.svg"
    plt.savefig(save_path, format='svg')

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Load your PRIME-DE data
prime_de = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/data/prime-de.csv")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    
    # Create figure with original dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- LEFT PLOT (EXACT original reproduction) ---
    jitter = 0.1
    colors = {'Rhesus': 'steelblue', 'Tonkean': 'purple', 'mulatta': 'steelblue'}
    
    for species in ['Rhesus', 'Tonkean']:
        males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]
        females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]

        sns.stripplot(x='species', y='age', data=females, jitter=jitter, marker='o', 
                     color=colors[species], alpha=0.6, size=10, ax=ax1)
        sns.stripplot(x='species', y='age', data=males, jitter=jitter, marker='o', 
                     facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5, ax=ax1)
    
    for species in ['Rhesus', 'Tonkean']:
        subset = combined_df[combined_df['species'] == species] 
        mean_age = subset['age'].mean() 
        std_age = subset['age'].std()
        ax1.errorbar(x=[species], y=[mean_age], yerr=[std_age], fmt='none', 
                    ecolor=colors[species], capsize=5, elinewidth=2, zorder=10)
    
    ax1.set_ylabel('Age', fontsize=14)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels([f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
                        f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})'], 
                       fontsize=14)
    
    # Add SILABE subtitle
    ax1.text(0.5, 1.05, 'SILABE', transform=ax1.transAxes,
            fontsize=18, fontweight='bold', ha='center', va='bottom')
    
    # --- RIGHT PLOT (PRIME-DE) ---
    prime_de['gender'] = prime_de['sex'].map({'M': 1, 'F': 2})
    prime_de['dummy_x'] = 0
    
    # Plot females (filled)
    sns.stripplot(x='dummy_x', y='age_years', data=prime_de[prime_de['gender'] == 2], 
                 jitter=jitter, marker='o', color=colors['mulatta'], 
                 alpha=0.6, size=10, ax=ax2)
    
    # Plot males (hollow)
    sns.stripplot(x='dummy_x', y='age_years', data=prime_de[prime_de['gender'] == 1], 
                 jitter=jitter, marker='o', facecolors='none', 
                 edgecolor=colors['mulatta'], size=10, linewidth=1.5, ax=ax2)
    
    # Add mean/std
    mean_age = prime_de['age_years'].mean()
    std_age = prime_de['age_years'].std()
    ax2.errorbar(x=[0], y=[mean_age], yerr=[std_age], fmt='none', 
                ecolor=colors['mulatta'], capsize=5, elinewidth=2, zorder=10)
    
    # Format x-axis identically to left plot
    ax2.set_xticks([0])
    ax2.set_xticklabels([f'$\\it{{Macaca\\ mulatta}}$\n(n={len(prime_de)})'], fontsize=14)
    
    # Add PRIME-DE subtitle
    ax2.text(0.5, 1.05, 'PRIME-DE', transform=ax2.transAxes,
            fontsize=18, fontweight='bold', ha='center', va='bottom')
    
    # --- CRITICAL: ORIGINAL SPINE STYLING ---
    # Left plot - exactly as in your original (default matplotlib spines)
    ax1.spines['top'].set_visible(False)  # Only top spine removed
    
    # Right plot - match original styling
    ax2.spines['top'].set_visible(False)   # Only top spine removed
    ax2.set_ylabel('')
    ax2.set_yticklabels([])  # Remove y-axis numbers but keep spine
    
    # --- Shared formatting ---
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_ylim(0, 25)
        ax.set_xlabel('')
    
    # Adjust x-limits
    ax1.set_xlim(-0.5, 1.5)
    ax2.set_xlim(-0.5, 0.5)
    
    # Main title (Times New Roman only for title)
    fig.suptitle('Demographics', fontsize=25, fontweight='bold', 
                fontname='Times New Roman', color='purple', y=1.02)
    
    plt.tight_layout()
    
    # Save as SVG
    save_path = "/Users/similovesyou/Desktop/qts/simian-behavior/plots/demographics_comparison.svg"
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    
    plt.show()
gender_counts = combined_df.groupby(['species', 'gender']).size().unstack()
gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
gender_percentages.rename(columns={1: 'Male (%)', 2: 'Female (%)'}, inplace=True)
gender_percentages

def tossNaN(df, columns):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace inf with NaN 
    return df.dropna(subset=columns)  # drop NaNs

columns = ['mean_att#_/sess', 'gender']

# apply the function to toss NaNs 
rhesus_table = tossNaN(rhesus_table, columns)
tonkean_table = tossNaN(tonkean_table, columns)

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define save path for SVG
save_path = Path("/Users/similovesyou/Desktop/qts/simian-behavior/plots/Simian_Activity_Playful_Patterns.svg")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    jitter = 0.1
    colors = {'Rhesus': 'steelblue', 'Tonkean': 'purple'}  
    titles = ['Species', 'Gender', 'Age', 'Social Status (Elo-Rating)']  

    for i, ax in enumerate(axes):
        if i == 0:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]
                
                sns.stripplot(x='species', y='mean_att#_/sess', data=females, jitter=jitter, marker='o', 
                              color=colors[species], alpha=0.6, size=10, ax=ax)
                sns.stripplot(x='species', y='mean_att#_/sess', data=males, jitter=jitter, marker='o', 
                              facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5, ax=ax)

            for species in ['Rhesus', 'Tonkean']:
                subset = combined_df[combined_df['species'] == species]
                mean = subset['mean_att#_/sess'].mean()
                std = subset['mean_att#_/sess'].std()
                ax.errorbar(x=[species], y=[mean], yerr=[std], fmt='none', ecolor=colors[species], capsize=5, elinewidth=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'$\\it{{Macaca\\ mulatta}}$\n(n = {combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
                                f'$\\it{{Macaca\\ tonkeana}}$\n(n = {combined_df[combined_df["species"] == "Tonkean"].shape[0]})'],
                               fontsize=14)
            ax.set_xlim(-0.4, 1.4)

        elif i == 1:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 1)]
                
                sns.stripplot(x='gender', y='mean_att#_/sess', data=females, jitter=jitter, marker='o',
                              color=colors[species], alpha=0.6, size=10, ax=ax)
                sns.stripplot(x='gender', y='mean_att#_/sess', data=males, jitter=jitter, marker='o',
                              facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5, ax=ax)

            for gender in [1, 2]:
                subset = combined_df[combined_df['gender'] == gender]
                mean = subset['mean_att#_/sess'].mean()
                std = subset['mean_att#_/sess'].std()
                ax.errorbar(x=[gender-1], y=[mean], yerr=[std], fmt='none', ecolor='black', capsize=5, elinewidth=2)
                
            female_count = combined_df[combined_df["gender"] == 2].shape[0]
            male_count = combined_df[combined_df["gender"] == 1].shape[0]
           
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'Females\n(n = {female_count})', f'Males\n(n = {male_count})'], fontsize=14)
            ax.set_xlim(-0.4, 1.4)

        elif i == 2:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]
                
                ax.scatter(females['age'], females['mean_att#_/sess'], color=colors[species], alpha=0.6, s=100)
                ax.scatter(males['age'], males['mean_att#_/sess'], edgecolor=colors[species], facecolor='none', s=100, linewidth=1.5)

            sns.regplot(x='age', y='mean_att#_/sess', data=combined_df, order=2, scatter=False, ax=ax, color='black')
            ax.set_xticks(range(0, 22, 5))

        elif i == 3:
            females = tonkean_table_elo[tonkean_table_elo['gender'] == 2]
            males = tonkean_table_elo[tonkean_table_elo['gender'] == 1]

            ax.scatter(females['elo_mean'], females['mean_att#_/sess'], color=colors['Tonkean'], alpha=0.6, s=100)
            ax.scatter(males['elo_mean'], males['mean_att#_/sess'], edgecolor=colors['Tonkean'], facecolor='none', s=100, linewidth=1.5)

            sns.regplot(x='elo_mean', y='mean_att#_/sess', data=tonkean_table_elo, scatter=False, ax=ax, color=colors['Tonkean'])

        ax.set_title(titles[i], fontsize=18, fontweight='bold', fontname='Times New Roman')  
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_ylim(0, 33)
        ax.set_yticks(range(0, 33, 10))
        
        ax.tick_params(axis='x', labelsize=14)  
        ax.tick_params(axis='y', labelsize=14)  
        
        ax.set_xlabel('')
        if i in [0, 2]:
            ax.set_ylabel('Mean Count', fontsize=15)  
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Simian Activity: Playful Patterns', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')

    # Save as SVG
    plt.savefig(save_path, format='svg')

    plt.show()
    
    import scipy.stats as stats

    # Perform species comparison (as done previously)
    rhesus_data = combined_df[combined_df['species'] == 'Rhesus']['mean_att#_/sess']
    tonkean_data = combined_df[combined_df['species'] == 'Tonkean']['mean_att#_/sess']

    t_stat, p_value = stats.ttest_ind(rhesus_data, tonkean_data, equal_var=False)

    # Descriptive statistics for Rhesus
    rhesus_mean = rhesus_data.mean()
    rhesus_std = rhesus_data.std()
    rhesus_n = rhesus_data.count()
    rhesus_ci = stats.t.interval(0.95, rhesus_n-1, loc=rhesus_mean, scale=rhesus_std/np.sqrt(rhesus_n))

    # Descriptive statistics for Tonkean
    tonkean_mean = tonkean_data.mean()
    tonkean_std = tonkean_data.std()
    tonkean_n = tonkean_data.count()
    tonkean_ci = stats.t.interval(0.95, tonkean_n-1, loc=tonkean_mean, scale=tonkean_std/np.sqrt(tonkean_n))

    print(f"Rhesus: Mean attempts = {rhesus_mean:.2f}, SD = {rhesus_std:.2f}, n = {rhesus_n}, 95% CI = ({rhesus_ci[0]:.2f}, {rhesus_ci[1]:.2f})")
    print(f"Tonkean: Mean attempts = {tonkean_mean:.2f}, SD = {tonkean_std:.2f}, n = {tonkean_n}, 95% CI = ({tonkean_ci[0]:.2f}, {tonkean_ci[1]:.2f})")

    print(f"\nT-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("There is a significant difference between the species.")
    else:
        print("There is no significant difference between the species.")

    # Gender testing across species
    print("\nGender comparison across species:")

    rhesus_females = combined_df[(combined_df['species'] == 'Rhesus') & (combined_df['gender'] == 2)]['mean_att#_/sess']
    rhesus_males = combined_df[(combined_df['species'] == 'Rhesus') & (combined_df['gender'] == 1)]['mean_att#_/sess']
    tonkean_females = combined_df[(combined_df['species'] == 'Tonkean') & (combined_df['gender'] == 2)]['mean_att#_/sess']
    tonkean_males = combined_df[(combined_df['species'] == 'Tonkean') & (combined_df['gender'] == 1)]['mean_att#_/sess']

    # Perform t-tests for gender within species
    print("\nWithin species gender comparison:")
    
    print("Rhesus: ")
    t_stat_rhesus, p_value_rhesus = stats.ttest_ind(rhesus_females, rhesus_males, equal_var=False)
    print(f"T-statistic: {t_stat_rhesus:.4f}, P-value: {p_value_rhesus:.4f}")

    print("Tonkean: ")
    t_stat_tonkean, p_value_tonkean = stats.ttest_ind(tonkean_females, tonkean_males, equal_var=False)
    print(f"T-statistic: {t_stat_tonkean:.4f}, P-value: {p_value_tonkean:.4f}")

    # Gender comparison across species (combined)
    print("\nAcross species gender comparison:")
    all_females = combined_df[combined_df['gender'] == 2]['mean_att#_/sess']
    all_males = combined_df[combined_df['gender'] == 1]['mean_att#_/sess']
    t_stat_gender, p_value_gender = stats.ttest_ind(all_females, all_males, equal_var=False)

    print(f"\nT-statistic: {t_stat_gender:.4f}, P-value: {p_value_gender:.4f}")
    if p_value_gender < 0.05:
        print("There is a significant gender difference across species.")
    else:
        print("There is no significant gender difference across species.")
    
    # Pearson correlation across species
    pearson_corr_all, pearson_p_all = stats.pearsonr(combined_df['age'], combined_df['mean_att#_/sess'])
    print(f"Overall Pearson correlation between age and mean attempts: r = {pearson_corr_all:.3f}, p = {pearson_p_all:.3f}")

    # Spearman correlation across species (non-parametric)
    spearman_corr_all, spearman_p_all = stats.spearmanr(combined_df['age'], combined_df['mean_att#_/sess'])
    print(f"Overall Spearman correlation between age and mean attempts: r = {spearman_corr_all:.3f}, p = {spearman_p_all:.3f}")

    # Within species - Rhesus
    rhesus_data = combined_df[combined_df['species'] == 'Rhesus']
    pearson_corr_rhesus, pearson_p_rhesus = stats.pearsonr(rhesus_data['age'], rhesus_data['mean_att#_/sess'])
    spearman_corr_rhesus, spearman_p_rhesus = stats.spearmanr(rhesus_data['age'], rhesus_data['mean_att#_/sess'])
    print(f"Rhesus - Pearson: r = {pearson_corr_rhesus:.3f}, p = {pearson_p_rhesus:.3f}")
    print(f"Rhesus - Spearman: r = {spearman_corr_rhesus:.3f}, p = {spearman_p_rhesus:.3f}")

    # Within species - Tonkean
    tonkean_data = combined_df[combined_df['species'] == 'Tonkean']
    pearson_corr_tonkean, pearson_p_tonkean = stats.pearsonr(tonkean_data['age'], tonkean_data['mean_att#_/sess'])
    spearman_corr_tonkean, spearman_p_tonkean = stats.spearmanr(tonkean_data['age'], tonkean_data['mean_att#_/sess'])
    print(f"Tonkean - Pearson: r = {pearson_corr_tonkean:.3f}, p = {pearson_p_tonkean:.3f}")
    print(f"Tonkean - Spearman: r = {spearman_corr_tonkean:.3f}, p = {spearman_p_tonkean:.3f}")


    # Pearson correlation (if normally distributed)
    elo_pearson_corr, elo_pearson_p = stats.pearsonr(tonkean_table_elo['elo_mean'], tonkean_table_elo['mean_att#_/sess'])
    print(f"Pearson correlation between Elo-rating and mean attempts: r = {elo_pearson_corr:.3f}, p = {elo_pearson_p:.3f}")

    # Spearman correlation (non-parametric)
    elo_spearman_corr, elo_spearman_p = stats.spearmanr(tonkean_table_elo['elo_mean'], tonkean_table_elo['mean_att#_/sess'])
    print(f"Spearman correlation between Elo-rating and mean attempts: r = {elo_spearman_corr:.3f}, p = {elo_spearman_p:.3f}")

    # Linear regression: Predicting attempts based on Elo-rating
    X_elo = sm.add_constant(tonkean_table_elo['elo_mean'])
    y_elo = tonkean_table_elo['mean_att#_/sess']

    elo_model = sm.OLS(y_elo, X_elo).fit()
    print(elo_model.summary())

    
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    jitter = 0.1
    colors = {'Rhesus': 'steelblue', 'Tonkean': 'purple'}  # Colors
    titles = ['Species', 'Gender', 'Age', 'Social Status (Elo-Rating)']  # Updated last title

    for i, ax in enumerate(axes):
        if i == 0:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]

                sns.stripplot(x='species', y='#_sessions', data=females, jitter=jitter, marker='o',
                              color=colors[species], alpha=0.6, size=10, ax=ax)
                sns.stripplot(x='species', y='#_sessions', data=males, jitter=jitter, marker='o',
                              facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5, ax=ax)

            for species in ['Rhesus', 'Tonkean']:
                subset = combined_df[combined_df['species'] == species]
                mean = subset['#_sessions'].mean()
                std = subset['#_sessions'].std()
                ax.errorbar(x=[species], y=[mean], yerr=[std], fmt='none', ecolor=colors[species], capsize=5, elinewidth=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'$\\it{{Macaca\\ mulatta}}$\n(n = {combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
                                f'$\\it{{Macaca\\ tonkeana}}$\n(n = {combined_df[combined_df["species"] == "Tonkean"].shape[0]})'],
                               fontsize=14)  # Increased size by 2
            ax.set_xlim(-0.4, 1.4)
        
        elif i == 1:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 1)]

                sns.stripplot(x='gender', y='#_sessions', data=females, jitter=jitter, marker='o',
                              color=colors[species], alpha=0.6, size=10, ax=ax)
                sns.stripplot(x='gender', y='#_sessions', data=males, jitter=jitter, marker='o',
                              facecolors='none', edgecolor=colors[species], size=10, linewidth=1.5, ax=ax)

            for gender in [1, 2]:
                subset = combined_df[combined_df['gender'] == gender]
                mean = subset['#_sessions'].mean()
                std = subset['#_sessions'].std()
                ax.errorbar(x=[gender - 1], y=[mean], yerr=[std], fmt='none', ecolor='black', capsize=5, elinewidth=2)

            female_count = combined_df[combined_df["gender"] == 2].shape[0]
            male_count = combined_df[combined_df["gender"] == 1].shape[0]

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'Females\n(n = {female_count})', f'Males\n(n = {male_count})'], fontsize=14)  # Increased size by 2
            ax.set_xlim(-0.4, 1.4)

        elif i == 2:
            for species in ['Rhesus', 'Tonkean']:
                females = combined_df[(combined_df['species'] == species) & (combined_df['gender'] == 2)]
                males = combined_df[(combined_df['species'] == species) & (combined_df['gender'] != 2)]

                ax.scatter(females['age'], females['#_sessions'], color=colors[species], alpha=0.6, s=100)
                ax.scatter(males['age'], males['#_sessions'], edgecolor=colors[species], facecolor='none', s=100, linewidth=1.5)

            sns.regplot(x='age', y='#_sessions', data=combined_df, order=2, scatter=False, ax=ax, color='black')
            ax.set_xticks(range(0, 22, 5))

        elif i == 3:
            females = tonkean_table_elo[tonkean_table_elo['gender'] == 2]
            males = tonkean_table_elo[tonkean_table_elo['gender'] == 1]

            ax.scatter(females['elo_mean'], females['#_sessions'], color=colors['Tonkean'], alpha=0.6, s=100)
            ax.scatter(males['elo_mean'], males['#_sessions'], edgecolor=colors['Tonkean'], facecolor='none', s=100, linewidth=1.5)

            sns.regplot(x='elo_mean', y='#_sessions', data=tonkean_table_elo, scatter=False, ax=ax, color=colors['Tonkean'])
        
        ax.set_title(titles[i], fontsize=18, fontweight='bold', fontname='Times New Roman')  # Increased subtitle size
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_ylim(0, combined_df['#_sessions'].max() + 100)
        ax.set_yticks(range(0, int(combined_df['#_sessions'].max() + 1000), 2000))
        ax.grid(True, linestyle='--', alpha=0.7)

        # Increase size of x and y labels
        ax.tick_params(axis='x', labelsize=14)  # Increased by 2 points
        ax.tick_params(axis='y', labelsize=14)  # Increased by 2 points

        ax.set_xlabel('')  # No specific x-axis label
        if i in [0, 2]:
            ax.set_ylabel('Session Count', fontsize=15)  # Increased by 2 points
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Simian Activity: How much do they play?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')  # Updated main title
    
    # plt.savefig('/Users/similovesyou/Downloads/Simian_Activity_Sessions.png', format='png')
    plt.show()

    
# Statistical Tests
def run_stat_tests(group1, group2, group1_name, group2_name, metric_name):
    """
    Runs appropriate statistical test (t-test or Mann-Whitney U) for two groups.
    Prints a clear summary of the results.
    """
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
    'Mean Attempts Per Session': ('mean_att#_/sess', rhesus_table, tonkean_table),
    'Total Number of Attempts': ('#_attempts', rhesus_table, tonkean_table),
    'Proportion of Successful Attempts': ('p_success', rhesus_table, tonkean_table),
    'Proportion of Erroneous Attempts': ('p_error', rhesus_table, tonkean_table),
    'Proportion of Omitted Attempts': ('p_omission', rhesus_table, tonkean_table),
    'Proportion of Premature Attempts': ('p_premature', rhesus_table, tonkean_table),
    'Age of Subjects': ('age', rhesus_table, tonkean_table),
    'Mean Reaction Time (Successful Attempts)': ('rt_success', rhesus_table, tonkean_table),
    'Mean Reaction Time (Erroneous Attempts)': ('rt_error', rhesus_table, tonkean_table),
}

# Species-wise comparisons
for metric_name, (metric, rhesus_data, tonkean_data) in metrics.items():
    run_stat_tests(rhesus_data[metric], tonkean_data[metric], "Rhesus", "Tonkean", metric_name)

# Gender-wise comparisons
for metric_name, (metric, _, _) in metrics.items():
    male_data = combined_df[combined_df['gender'] == 1][metric]
    female_data = combined_df[combined_df['gender'] == 2][metric]
    run_stat_tests(male_data, female_data, "Male", "Female", metric_name)

species_list = ['Rhesus', 'Tonkean']

for species in species_list:
    for metric_name, (metric, _, _) in metrics.items():
        species_data = combined_df[combined_df['species'] == species]
        
        male_data_species = species_data[species_data['gender'] == 1][metric]
        female_data_species = species_data[species_data['gender'] == 2][metric]
        
        run_stat_tests(male_data_species, female_data_species, f"{species} Male", f"{species} Female", f"{metric_name} ({species})")



proportions = {}

for species, monkeys in data.items():
    proportions[species] = {}
    
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        
        if attempts is not None:
            # Calculate proportions of each result type
            result_counts = attempts['result'].value_counts(normalize=True)
            
            proportions[species][name] = {
                'p_success': result_counts.get('success', 0.0),
                'p_error': result_counts.get('error', 0.0),
                'p_omission': result_counts.get('stepomission', 0.0),
                'p_premature': result_counts.get('premature', 0.0)
            }

# Convert to a DataFrame for better visualization
proportions_df = []
for species, species_data in proportions.items():
    for name, stats in species_data.items():
        stats['species'] = species
        stats['name'] = name
        proportions_df.append(stats)

proportions_df = pd.DataFrame(proportions_df)

for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        print(f"{name}: {monkey_data['attempts']['result'].value_counts(normalize=True)}")


import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import statsmodels.stats.multitest as smm

# Function to run statistical tests and post hoc tests
def run_stat_tests(group1, group2, group1_name, group2_name, metric_name):
    """
    Runs appropriate statistical test (t-test or Mann-Whitney U) for two groups.
    If significant, applies post hoc Bonferroni correction for multiple testing.
    Prints a clear summary of the results.
    """
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
        print("  Significant difference found! Applying Bonferroni correction for post hoc tests.")
        pvals = [test_result.pvalue]
        corrected_pvals = smm.multipletests(pvals, method='bonferroni')[1]
        print(f"  Bonferroni corrected p-value: {corrected_pvals[0]:.3f}")

# Metrics to be tested
metrics = {
    'Mean Attempts Per Session': ('mean_att#_/sess', rhesus_table, tonkean_table),
    'Total Number of Attempts': ('#_attempts', rhesus_table, tonkean_table),
    'Proportion of Successful Attempts': ('p_success', rhesus_table, tonkean_table),
    'Proportion of Erroneous Attempts': ('p_error', rhesus_table, tonkean_table),
    'Proportion of Omitted Attempts': ('p_omission', rhesus_table, tonkean_table),
    'Proportion of Premature Attempts': ('p_premature', rhesus_table, tonkean_table),
    'Age of Subjects': ('age', rhesus_table, tonkean_table),
    'Mean Reaction Time (Successful Attempts)': ('rt_success', rhesus_table, tonkean_table),
    'Mean Reaction Time (Erroneous Attempts)': ('rt_error', rhesus_table, tonkean_table),
}

# Species-wise comparisons
for metric_name, (metric, rhesus_data, tonkean_data) in metrics.items():
    run_stat_tests(rhesus_data[metric], tonkean_data[metric], "Rhesus", "Tonkean", metric_name)

# Gender-wise comparisons
for metric_name, (metric, _, _) in metrics.items():
    male_data = combined_df[combined_df['gender'] == 1][metric]
    female_data = combined_df[combined_df['gender'] == 2][metric]
    run_stat_tests(male_data, female_data, "Male", "Female", metric_name)

species_list = ['Rhesus', 'Tonkean']

for species in species_list:
    for metric_name, (metric, _, _) in metrics.items():
        species_data = combined_df[combined_df['species'] == species]
        
        male_data_species = species_data[species_data['gender'] == 1][metric]
        female_data_species = species_data[species_data['gender'] == 2][metric]
        
        run_stat_tests(male_data_species, female_data_species, f"{species} Male", f"{species} Female", f"{metric_name} ({species})")

### 

# Calculate proportions
proportions = {}

for species, monkeys in data.items():
    proportions[species] = {}
    
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        
        if attempts is not None:
            # Calculate proportions of each result type
            result_counts = attempts['result'].value_counts(normalize=True)
            
            proportions[species][name] = {
                'p_success': result_counts.get('success', 0.0),
                'p_error': result_counts.get('error', 0.0),
                'p_omission': result_counts.get('stepomission', 0.0),
                'p_premature': result_counts.get('premature', 0.0)
            }

# Convert to a DataFrame for better visualization
proportions_df = []
for species, species_data in proportions.items():
    for name, stats in species_data.items():
        stats['species'] = species
        stats['name'] = name
        proportions_df.append(stats)

proportions_df = pd.DataFrame(proportions_df)

for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        print(f"{name}: {monkey_data['attempts']['result'].value_counts(normalize=True)}")


###

import statsmodels.api as sm

# Preparing the data for regression analysis
# Ensure you replace 'tonkean_table_elo' with your actual DataFrame variable
hierarchy_attempts_data = tonkean_table_elo[['elo_mean', 'mean_att#_/sess']].dropna()

X = hierarchy_attempts_data['elo_mean']  # Independent variable: mean Elo-rating (hierarchy)
y = hierarchy_attempts_data['mean_att#_/sess']  # Dependent variable: mean attempts per session

# Adding a constant term for the regression model (intercept)
X = sm.add_constant(X)

# Fitting the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Extracting the regression coefficient and its significance
print("Regression Coefficient for Elo-rating:", model.params['elo_mean'])
print("P-Value for Elo-rating Coefficient:", model.pvalues['elo_mean'])

# Printing a full regression summary
print(model.summary())

# Combine rhesus_table and tonkean_table with an extra column specifying the species
combined_table = pd.concat([
    rhesus_table.assign(species='Rhesus'),
    tonkean_table.assign(species='Tonkean')
], ignore_index=True)

# Display the combined table
print(combined_table)

# Save the combined table to a file, if needed
save_directory = '/Users/similovesyou/Desktop/qts/simian-behavior'
combined_table.to_excel(os.path.join(save_directory, 'demos-performance-combined.xlsx'), index=False)

