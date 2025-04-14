import pickle
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns
import warnings  # they annoy me
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py' # string
dmt_rick = os.path.join(directory, 'dms.pickle')
with open(dmt_rick, 'rb') as handle:
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
                           'mean_sess_duration': attempts['session_duration'].mean(),
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

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Adjusted colors
colors = {'tonkean': 'purple', 'rhesus': 'steelblue'}

metrics = ['p_success', 'p_error', 'p_premature', 'p_omission']
titles = ['Hits', 'Errors', 'Premature', 'Omissions']

for ax, metric, title, show_xlabel, show_ylabel in zip(axes.flatten(), metrics, titles, [False, False, True, True], [True, False, True, False]):
    p_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

handles = [
    plt.Line2D([], [], color=colors['rhesus'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='steelblue', linestyle='None', markersize=8, label=r'$\it{Macaca\ mulatta}$ (m)'),
    plt.Line2D([], [], color=colors['tonkean'], marker='o', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='purple', linestyle='None', markersize=8, label=r'$\it{Macaca\ tonkeana}$ (m)')
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

fig.suptitle('Simian Success: How well do they play?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')

# Adjust tight_layout to reduce vertical squeezing
plt.tight_layout(rect=[0, 0.02, 1, 0.93])  # Reduced top margin
plt.savefig('/Users/similovesyou/Downloads/accuracy.png', format='png')
plt.show()

def rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel):
    
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='steelblue', s=80, linewidth=1)  # Increased size
    sns.scatterplot(ax=ax, data=rhesus_table[rhesus_table['gender'] == 2], x='age', y=metric, color=colors['rhesus'], 
                    s=80, alpha=0.6, edgecolor='steelblue', linewidth=1)  # Increased size

    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 1], x='age', y=metric, facecolors='none', 
                    edgecolor='purple', s=80, linewidth=1)  # Increased size
    sns.scatterplot(ax=ax, data=tonkean_table[tonkean_table['gender'] == 2], x='age', y=metric, color=colors['tonkean'], 
                    s=80, alpha=0.6, edgecolor='purple', linewidth=1)  # Increased size
    
    sns.regplot(ax=ax, data=rhesus_table, x='age', y=metric, color=colors['rhesus'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=tonkean_table, x='age', y=metric, color=colors['tonkean'], order=2, scatter=False, ci=None, 
                line_kws={'linestyle':'--'})
    sns.regplot(ax=ax, data=pd.concat([rhesus_table, tonkean_table]), x='age', y=metric, color='black', order=2, 
                scatter=False, ci=None)
    
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman')  # Bigger subtitle size
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.grid(True, which='both', linestyle='--')
    ax.grid(True, which='major', axis='x')
    if show_xlabel:
        ax.set_xlabel('Age', fontsize=14)  # Increased label size
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    if show_ylabel:
        ax.set_ylabel('RTs (ms)', fontsize=14)  # Updated and increased label size
    else:
        ax.set_ylabel('')

sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "arial"})

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# **Adjusted Colors**
colors = {'tonkean': 'purple', 'rhesus': 'steelblue'}

metrics = ['rt_success', 'rt_error']
titles = ['Hits', 'Error']  # Updated subtitles

for ax, metric, title, show_xlabel, show_ylabel in zip(axes.flatten(), metrics, titles, [True, True], [True, False]):
    rt_lifespan(ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel)

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)  # Reusing handles

fig.suptitle('Swift Simian: How fast do they react?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')  # Updated main title

# Adjust placements
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/similovesyou/Downloads/RTs.png', format='png')

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
    plt.savefig('/Users/similovesyou/Downloads/Demographics.png', format='png')

    plt.show()

def tossNaN(df, columns):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace inf with NaN 
    return df.dropna(subset=columns)  # drop NaNs

columns = ['mean_att#_/sess', 'gender']

# apply the function to toss NaNs 
rhesus_table = tossNaN(rhesus_table, columns)
tonkean_table = tossNaN(tonkean_table, columns)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    jitter = 0.1
    colors = {'Rhesus': 'steelblue', 'Tonkean': 'purple'}  # Colors
    titles = ['Species', 'Gender', 'Age', 'Social Status (Elo-Rating)']  # Updated last title

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

        ax.set_title(titles[i], fontsize=18, fontweight='bold', fontname='Times New Roman')  # Subtitles
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_ylim(0, 33)
        ax.set_yticks(range(0, 33, 10))
        
        ax.tick_params(axis='x', labelsize=14)  # Increased x-axis label size
        ax.tick_params(axis='y', labelsize=14)  # Increased y-axis label size
        
        ax.set_xlabel('')
        if i in [0, 2]:
            ax.set_ylabel('Mean Count', fontsize=15)  # Increased by 2 points
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Simian Activity: How much do they play?', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')
    plt.savefig('/Users/similovesyou/Downloads/session_freq.png', format='png')
    plt.show()
