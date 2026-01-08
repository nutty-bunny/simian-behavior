import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings  # they annoy me

# re-run script after cleaning pickle rick 
directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py' # string
dmt_rick = os.path.join(directory, 'dms.pickle')
with open(dmt_rick, 'rb') as handle:
    data = pickle.load(handle)

# load tonkean_table_elo that's been aligned with elo-rating period (see social_status_bby.py) 
tonkean_table_elo = pd.read_excel(os.path.join(directory, 'hierarchy', 'tonkean_table_elo.xlsx'))

# note 2 simi -- proportions differ bcuz the task period has been shortened 
print()
print("macaca tonkeana (n = 33) incl. elo")
print(tonkean_table_elo)

# let's append demographics 
# 1 = male, 2 = female 

mkid_tonkean = directory + '/mkid_tonkean.csv'
demographics = pd.read_csv(mkid_tonkean)
demographics['name'] = demographics['name'].str.lower()  
demographics.rename(columns={'Age': 'age'}, inplace=True) # switch names 2 lowercase cuz im 2 ocd 

print()
print("tonkean demographics (n = 35)")
print(demographics)

# merge demographics
tonkean_table_elo = tonkean_table_elo.join(demographics.set_index('name')[['id', 'gender', 'birthdate', 'age']],
                                 on='name', how='left')
# summarize device engagement, task performance & demographics
print()
print("tonkean task engagement (incl. elo)")
print(tonkean_table_elo)  

plt.rcParams['font.family'] = 'Arial' # set global font to be used across all plots ^^ **change later**
sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "arial"})  # ** modify font selection here! **

def p_lifespan(ax, metric, title, data, show_xlabel, show_ylabel, show_colorbar):
    # Scale elo_mean to use as point sizes
    min_size = 50  # Increased dot sizes
    max_size = 300
    sizes = ((data['elo_mean'] - data['elo_mean'].min()) / (data['elo_mean'].max() - data['elo_mean'].min())) * (max_size - min_size) + min_size
    
    # Get color mapping from elo_mean
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(data['elo_mean'].min(), data['elo_mean'].max())
    
    # Plot data with sizes based on elo_mean
    male_data = data[data['gender'] == 1]
    female_data = data[data['gender'] == 2]

    scatter_male = ax.scatter(male_data['age'], male_data[metric], 
                              c='none',  # Hollow centers
                              edgecolors=cmap(norm(male_data['elo_mean'])),
                              s=sizes[data['gender'] == 1], linewidth=1.5)
    scatter_female = ax.scatter(female_data['age'], female_data[metric], 
                                c=female_data['elo_mean'], cmap='plasma', s=sizes[data['gender'] == 2], 
                                linewidth=1.5, edgecolors='face')
    
    # Add regression line
    sns.regplot(ax=ax, data=data, x='age', y=metric, color='purple', order=2, scatter=False, ci=None, line_kws={'linestyle': '--'})

    # Set title and limits
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0', '', '', '', '', '1'])  # Only retain ticks 0 and 1
    ax.grid(True, which='both', linestyle='--')
    ax.grid(True, which='major', axis='x')
    
    # Show or hide x and y labels
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
   
    # Only show colorbar on the right subplots
    if show_colorbar:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
        cbar.set_label('Elo Mean', fontsize=14)

# Create subplots and plot data
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['p_success', 'p_error', 'p_premature', 'p_omission']
titles = ['Hits', 'Errors', 'Premature', 'Omissions']

for ax, metric, title, show_xlabel, show_ylabel, show_colorbar in zip(
        axes.flatten(), metrics, titles, 
        [False, False, True, True], [True, False, True, False], [False, True, False, True]):
    p_lifespan(ax, metric, title, tonkean_table_elo, show_xlabel, show_ylabel, show_colorbar)

# Legend
handles = [
    plt.Line2D([], [], color='purple', marker='o', linestyle='None', markersize=10, label=r'$\it{Macaca\ tonkeana}$ (f)'),
    plt.Line2D([], [], color='none', marker='o', markeredgecolor='purple', linestyle='None', markersize=10, label=r'$\it{Macaca\ tonkeana}$ (m)')
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

fig.suptitle('Simian Success: How well do they play? (Incl. Elo-Rating)', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')

# Adjust layout and save as png
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
plt.savefig('/Users/similovesyou/Downloads/accuracy_elo.png', format='png')
plt.show()

def rt_hierarchy(ax, metric, title, data, show_xlabel, show_ylabel, show_colorbar):
    # Scale elo_mean to use as point sizes
    min_size = 50  # Dot size range
    max_size = 300
    sizes = ((data['elo_mean'] - data['elo_mean'].min()) / (data['elo_mean'].max() - data['elo_mean'].min())) * (max_size - min_size) + min_size
    
    # Color mapping for Elo
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(data['elo_mean'].min(), data['elo_mean'].max())
    
    # Plot male and female with hierarchy mapping
    male_data = data[data['gender'] == 1]
    female_data = data[data['gender'] == 2]

    ax.scatter(male_data['age'], male_data[metric], 
               c='none', edgecolors=cmap(norm(male_data['elo_mean'])),
               s=sizes[data['gender'] == 1], linewidth=1.5, label='Male')
    ax.scatter(female_data['age'], female_data[metric], 
               c=female_data['elo_mean'], cmap='plasma',
               s=sizes[data['gender'] == 2], edgecolors='face', label='Female')
    
    # Regression line
    sns.regplot(ax=ax, data=data, x='age', y=metric, color='purple', order=2, scatter=False, ci=None, line_kws={'linestyle': '--'})
    
    # Titles and formatting
    ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.set_xlim(0, 25)
    ax.grid(True, linestyle='--')
    
    if show_xlabel:
        ax.set_xlabel('Age', fontsize=14)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel('RTs', fontsize=14)
    else:
        ax.set_ylabel('')
    
    # Add colorbar for Elo rating only on the right plot
    if show_colorbar:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
        cbar.set_label('Elo Rating', fontsize=14)

# Apply to both `rt_success` and `rt_error`
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
metrics = ['rt_success', 'rt_error']
titles = ['Hits', 
          'Errors']

for ax, metric, title, show_xlabel, show_ylabel, show_colorbar in zip(
        axes.flatten(), metrics, titles, 
        [True, True], [True, True], [False, True]):  # Only last plot gets the colorbar
    rt_hierarchy(ax, metric, title, tonkean_table_elo, show_xlabel, show_ylabel, show_colorbar)

fig.suptitle('How Fast Are They? (Incl. Elo-Score)', fontsize=25, fontweight='bold', fontname='Times New Roman', color='purple')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/similovesyou/Downloads/Tonkean_RT_Hierarchy.png', format='png')
# plt.savefig('/Users/similovesyou/Downloads/Tonkean_RT_Hierarchy.svg', format='svg')

plt.show()

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

# sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "arial"})

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

