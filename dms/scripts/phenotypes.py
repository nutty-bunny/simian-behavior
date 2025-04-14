import pickle
import os
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
# ** re-run this sscript once pickle rick is finalised! **
directory = Path('/Users/similovesyou/Desktop/qts/simian-behavior/data/py')
dmt_rick = os.path.join(directory, 'dms.pickle')
with open(dmt_rick, 'rb') as handle:
    data = pickle.load(handle)

hierarchy_file = os.path.join(directory, 'hierarchy/tonkean_elo.xlsx')
hierarchy = pd.read_excel(hierarchy_file)

# elo-ratings 
abrv_2_fn = {
    'abr': 'abricot',
    'ala': 'alaryc',
    'alv': 'alvin',
    'anu': 'anubis',
    'bar': 'barnabe',
    'ber': 'berenice',
    'ces': 'cesar',
    'dor': 'dory',
    'eri': 'eric',
    'jea': 'jeanne',
    'lad': 'lady',
    'las': 'lassa',
    'nem': 'nema',
    'nen': 'nenno',
    'ner': 'nereis',
    'ola': 'olaf',
    'olg': 'olga',
    'oli': 'olli',
    'pac': 'patchouli',
    'pat': 'patsy',
    'wal': 'wallace',
    'wat': 'walt',
    'wot': 'wotan',
    'yan': 'yang',
    'yak': 'yannick',
    'yin': 'yin',
    'yoh': 'yoh',
    'ult': 'ulysse',
    'fic': 'ficelle',
    'gan': 'gandhi',
    'hav': 'havanna',
    'con': 'controle',
    'gai': 'gaia',
    'her': 'hercules',
    'han': 'hanouk',
    'hor': 'horus',
    'imo': 'imoen',
    'ind': 'indigo',
    'iro': 'iron',
    'isi': 'isis',
    'joy': 'joy',
    'jip': 'jipsy'
}

hierarchy = hierarchy.rename(columns=abrv_2_fn)
print(hierarchy.columns)
hierarchy['Date'] = pd.to_datetime(hierarchy['Date']).dt.date

elo_values = hierarchy.iloc[:, 1:].values.flatten() # flatten 2 find min and max
elo_values = elo_values[~pd.isnull(elo_values)] # remove NaNs

print(hierarchy)

tonkean = []
elo = {}  # stores filtered hierarchy data

for species, species_data in data.items():
    count = len(species_data)

    for name, monkey_data in species_data.items():
        attempts = monkey_data['attempts']
        
        attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
        attempts['instant_end'] = pd.to_datetime(attempts['instant_end'], unit='ms', errors='coerce')

        first_attempt = attempts['instant_begin'].min().date()  # overall onset of task engagement 
        last_attempt = attempts['instant_end'].max().date()  # overall offset
        
        if species == 'tonkean':
            if name in hierarchy.columns:
                filtered_hierarchy = hierarchy[(hierarchy['Date'] >= first_attempt) & (hierarchy['Date'] <= last_attempt)]  # filter hierarchy scores based on task period
                elo[name] = filtered_hierarchy[['Date', name]].dropna() # drop NaNs where there is no hierarchy assessment
            
            if name not in elo or elo[name].empty: # skip monkeys with no hierarchy data (there's two of 'em)
                continue
            
            first_elo = elo[name]['Date'].min()
            last_elo = elo[name]['Date'].max()
            
            new_start_date = max(first_attempt, first_elo) # overall onset that aligns with available elo 
            new_end_date = min(last_attempt, last_elo) # overall offset that aligns too 
            
            # only include attempts that fall within the elo timeframe (using the newly defined new_start_date and new_end_date)
            filtered_attempts = attempts[(attempts['instant_begin'].dt.date >= new_start_date) & (attempts['instant_end'].dt.date <= new_end_date)]
            
            # everything that follows is performed on the restricted df
            attempt_count = filtered_attempts['id'].nunique()
            session_count = filtered_attempts['session'].nunique()
            outcome = filtered_attempts['result'].value_counts(normalize=True)
            rt_success = filtered_attempts[filtered_attempts['result'] == 'success']['reaction_time'].mean()
            rt_error = filtered_attempts[filtered_attempts['result'] == 'error']['reaction_time'].mean()
            
            tonkean.append({'name': name, 
                            '#_attempts': attempt_count, 
                            '#_sessions': session_count, 
                            'start_date': new_start_date, 
                            'end_date': new_end_date,
                            'p_success': outcome.get('success', 0.0),
                            'p_error': outcome.get('error', 0.0),
                            'p_omission': outcome.get('stepomission', 0.0),
                            'p_premature': outcome.get('prematured', 0.0),
                            'rt_success': rt_success,
                            'rt_error': rt_error,
                            'mean_sess_duration': filtered_attempts['session_duration'].mean(),
                            'mean_att#_/sess': filtered_attempts.groupby('session')['id'].count().mean()})

# elo['Date'] = pd.to_datetime(elo['Date'])
df = hierarchy.set_index('Date')

# Plot for 'nema' with rolling average and adjusted y-axis
plt.figure(figsize=(8, 4))  # Shorter plot width for compact view
rolling_avg = df['olga'].rolling(window=40).mean()

# Plotting with customizations
plt.plot(df.index, rolling_avg, color='#FFD700', linewidth=2)  # Dark yellow color
plt.ylim(df['olga'].min() - 20, df['olga'].max() + 20)  # Extend y-axis to fit all values
plt.yticks([850, 1000, 1150])  # Keep ticks at specified points

plt.title('Olga')
plt.xlabel('Date')
plt.ylabel('Hierarchy Score')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

tonkean_table_elo = pd.DataFrame(tonkean)
tonkean_table_elo.to_excel(Path.home() / 'Downloads/tonkean_table_elo_aligned_periods.xlsx', index=False)

# sanity check 2 c if periods indeed align 
for item in tonkean:
    name = item['name']
    print()
    print(f"monkey: {name}")  
    print(f"task frame: {item['start_date']} to {item['end_date']}")
    if name in elo and not elo[name].empty:
        print(f"elo frame: {elo[name]['Date'].min()} to {elo[name]['Date'].max()}")
    else:
        print("no hierarchy data available for this period")
# periods do align! 

elo_mean = {}
for name, data in elo.items():
    if not data.empty:
        mean = round(data[name].mean()) # i think rounding here is appropriate since elo scores are integers anyways
        elo_mean[name] = mean
    else:
        elo_mean[name] = None

# append elo mean scores 2 table
for index, row in tonkean_table_elo.iterrows():
    name = row['name']
    if name in elo_mean:
        tonkean_table_elo.loc[index, 'elo_mean'] = elo_mean[name]
    else:
        tonkean_table_elo.loc[index, 'elo_mean'] = None

print("Updated tonkean_table_elo:")
print(tonkean_table_elo)

tonkean_table_elo_file = directory / 'hierarchy/tonkean_table_elo.xlsx'
tonkean_table_elo.to_excel(tonkean_table_elo_file, index=False)

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
dmt_rick = os.path.join(directory, 'data.pickle')

with open(dmt_rick, 'rb') as handle:
    data = pickle.load(handle)

data = {key: data[key] for key in ['rhesus', 'tonkean']}

print("Loaded data:")
print(data.keys())

rhesus = []
tonkean = []

durations = []  # durations irrespective of species 
frequencies = []  # attempt counts irrespective of species

# loop over data 
for species, monkeys in data.items():
    durations_species = []  # durations wrt species 
    frequencies_species = []  # frequencies wrt species 
    
    for name, data_dict in monkeys.items():
        attempts = data_dict['attempts']
        
        session_durations = []  # storing durations 
        session_frequencies = []  # storing frequencies 
        sessions_grouped = attempts.groupby('session')  # group by session 
        
        for session, session_data in sessions_grouped:
            start = session_data['instant_begin'].min()
            end = session_data['instant_end'].max()
            session_duration = (end - start) / 1000  # convert ms to s
            # session_duration = end - start  # in case you do not want 2 conver to s
            session_durations.append(session_duration) 
            session_frequencies.append(len(session_data))  # len() of selected session = attempt # per session 
        
        if species == 'rhesus':
            rhesus.append({
                'monkey': name,
                'min duration': min(session_durations) if session_durations else 0,
                'max duration': max(session_durations) if session_durations else 0,
                'mean duration': sum(session_durations) / len(session_durations) if session_durations else 0,
                'min count': min(session_frequencies) if session_frequencies else 0,
                'max count': max(session_frequencies) if session_frequencies else 0,
                'mean count': sum(session_frequencies) / len(session_frequencies) if session_frequencies else 0,
                'mode count': max(set(session_frequencies), key=session_frequencies.count) if session_frequencies else 0
            })
        elif species == 'tonkean':
            tonkean.append({
                'monkey': name,
                'min duration': min(session_durations) if session_durations else 0,
                'max duration': max(session_durations) if session_durations else 0,
                'mean duration': sum(session_durations) / len(session_durations) if session_durations else 0,
                'min count': min(session_frequencies) if session_frequencies else 0,
                'max count': max(session_frequencies) if session_frequencies else 0,
                'mean count': sum(session_frequencies) / len(session_frequencies) if session_frequencies else 0,
                'mode count': max(set(session_frequencies), key=session_frequencies.count) if session_frequencies else 0
            })
        
        durations_species.extend(session_durations)  # accumulate session durations respective of species
        frequencies_species.extend(session_frequencies)  # accumulate frequencies respective of species
        
        durations.extend(session_durations)  # accumulate overall session durations
        frequencies.extend(session_frequencies)  # accumulate overall session frequencies

    print(f"\n{species.capitalize()} Session Duration Statistics")
    if species == 'rhesus':
        rhesus_df = pd.DataFrame(rhesus)
        rhesus_df = rhesus_df.round(0)
        print(rhesus_df.to_string(index=False))
    elif species == 'tonkean':
        tonkean_df = pd.DataFrame(tonkean)
        tonkean_df = tonkean_df.round(0)
        print(tonkean_df.to_string(index=False))

tonkean_proportions = []

for name, data_dict in data['tonkean'].items():
    attempts = data_dict['attempts']
    
    # Ensure instant_begin and instant_end are in datetime format
    attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms', errors='coerce')
    attempts['instant_end'] = pd.to_datetime(attempts['instant_end'], unit='ms', errors='coerce')
    
    # Filter hierarchy data and determine task period for this monkey
    if name in elo and not elo[name].empty:
        first_attempt = attempts['instant_begin'].min().date()
        last_attempt = attempts['instant_end'].max().date()
        
        first_elo = elo[name]['Date'].min()
        last_elo = elo[name]['Date'].max()
        
        new_start_date = max(first_attempt, first_elo)
        new_end_date = min(last_attempt, last_elo)
        
        # Filter attempts within task period
        filtered_attempts = attempts[
            (attempts['instant_begin'].dt.date >= new_start_date) &
            (attempts['instant_end'].dt.date <= new_end_date)
        ]
        
        # Group by session and calculate proportions as before
        sessions_grouped = filtered_attempts.groupby('session')
        
        for session, session_data in sessions_grouped:
            total_attempts = len(session_data)
            
            outcome_counts = session_data['result'].value_counts().to_dict()
            
            proportions = {
                'monkey': name,
                'session': session,
                'p_success': outcome_counts.get('success', 0.0) / total_attempts,
                'p_error': outcome_counts.get('error', 0.0) / total_attempts,
                'p_omission': outcome_counts.get('stepomission', 0.0) / total_attempts,
                'p_premature': outcome_counts.get('prematured', 0.0) / total_attempts
            }
            
            tonkean_proportions.append(proportions)

# Convert to DataFrame for further analysis
tonkean_proportions_df = pd.DataFrame(tonkean_proportions)
print(tonkean_proportions_df)

import matplotlib.pyplot as plt
import pandas as pd

# Filter for Monkey Olga
monkey_olga_df = tonkean_proportions_df[tonkean_proportions_df['monkey'] == 'olga']

# Sort by session and reset index
monkey_olga_df = monkey_olga_df.sort_values('session').reset_index(drop=True)
# Apply rolling average (window size = 20)
rolling_window = 40
monkey_olga_df['p_success_smoothed'] = monkey_olga_df['p_success'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_error_smoothed'] = monkey_olga_df['p_error'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_omission_smoothed'] = monkey_olga_df['p_omission'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_premature_smoothed'] = monkey_olga_df['p_premature'].rolling(rolling_window, center=True).mean()

# Combine smoothed values into a DataFrame
stacked_df = monkey_olga_df[['p_success_smoothed', 'p_error_smoothed', 'p_premature_smoothed', 'p_omission_smoothed']]

# Remove rows with NaNs
stacked_df = stacked_df.dropna()

# Renormalize to ensure proportions sum to 1
stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

# Check if all session proportions add up to 1 after renormalization
# stacked_df['proportions_sum'] = stacked_df.sum(axis=1)
# assert all(abs(stacked_df['proportions_sum'] - 1) < 1e-5), "Proportions still do not add up to 1!"

# Plot the stacked proportions
# Plot the stacked proportions
plt.figure(figsize=(8, 8))  # Square plot
plt.stackplot(
    range(1, len(stacked_df) + 1),  # Sequential session count as x-axis
    stacked_df['p_success_smoothed'],
    stacked_df['p_error_smoothed'],
    stacked_df['p_premature_smoothed'],
    stacked_df['p_omission_smoothed'],
    colors=['#5386b6', '#d57459', '#e8bc60', '#8d5993'],  # blue, orange, yellow, purple
    labels=['Success', 'Error', 'Premature', 'Stepomission']
)

# Customize the plot
plt.ylabel("Stacked Proportion", fontsize=15, fontname='Times New Roman')
plt.xlabel("Sessions", fontsize=15, fontname='Times New Roman')
plt.yticks([0, 1], fontsize=13, fontname='Times New Roman')
plt.xticks(fontsize=13, fontname='Times New Roman')

# Set the x-axis and y-axis limits
plt.xlim(1, len(stacked_df))  # Ensure plot starts at session 1 and ends at the last session
plt.ylim(0, 1)  # Proportions range between 0 and 1

plt.legend(loc='upper left', fontsize=12, prop={'family': 'Times New Roman'})
plt.title("Impulsive Olga", fontsize=18, fontweight='bold', fontname='Times New Roman')

plt.tight_layout()  # Adjust layout to remove extra whitespace
plt.show()

monkey_olga_df = tonkean_proportions_df[tonkean_proportions_df['monkey'] == 'olga']
monkey_olga_df = monkey_olga_df.sort_values('session').reset_index(drop=True)

# Apply rolling average (window size = 40)
rolling_window = 40
monkey_olga_df['p_success_smoothed'] = monkey_olga_df['p_success'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_error_smoothed'] = monkey_olga_df['p_error'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_omission_smoothed'] = monkey_olga_df['p_omission'].rolling(rolling_window, center=True).mean()
monkey_olga_df['p_premature_smoothed'] = monkey_olga_df['p_premature'].rolling(rolling_window, center=True).mean()

# Combine smoothed values into a DataFrame
stacked_df = monkey_olga_df[['p_success_smoothed', 'p_error_smoothed', 'p_premature_smoothed', 'p_omission_smoothed']].dropna()
stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

# Hierarchy plot
rolling_avg = df['olga'].rolling(window=1, min_periods=1).mean()  # Ensures values are computed at edges

# Combined plot
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Stacked proportions 3x height, hierarchy 1x

# Stacked proportions plot
ax1 = fig.add_subplot(gs[0])
ax1.stackplot(
    range(1, len(stacked_df) + 1),
    stacked_df['p_success_smoothed'],
    stacked_df['p_error_smoothed'],
    stacked_df['p_premature_smoothed'],
    stacked_df['p_omission_smoothed'],
    colors=['#5386b6', '#d57459', '#e8bc60', '#8d5993'],
    labels=['Success', 'Error', 'Premature', 'Stepomission']
)
ax1.set_ylabel("Stacked Proportion", fontsize=14, fontname='Times New Roman')
ax1.set_xlim(1, len(stacked_df))
ax1.set_ylim(0, 1)
ax1.set_yticks([0, 1])
ax1.legend(loc='upper left', fontsize=12, prop={'family': 'Times New Roman'})
ax1.set_title("Olga", fontsize=18, fontweight='bold', fontname='Times New Roman')
ax1.set_xlabel('Session Count', fontsize=14, fontname='Times New Roman')

# Hierarchy plot
ax2 = fig.add_subplot(gs[1])
ax2.plot(df.index, rolling_avg, color='#FFD700', linewidth=2)

# Adjust y-axis and x-axis to fit entire data range
hierarchy_min = rolling_avg.min()
hierarchy_max = rolling_avg.max()
padding = 20
centered_ticks = [850, 1000, 1150]

ax2.set_ylim(min(hierarchy_min - padding, 850), max(hierarchy_max + padding, 1150))
ax2.set_yticks(centered_ticks)
ax2.set_xlim(df.index.min(), df.index.max())

ax2.set_ylabel('Elo', fontsize=14, fontname='Times New Roman')
ax2.set_xlabel('Date', fontsize=14, fontname='Times New Roman')

plt.tight_layout()
plt.show()

unique_monkeys = tonkean_proportions_df['monkey'].unique()

for monkey in unique_monkeys:
    # Filter data for the current monkey
    monkey_df = tonkean_proportions_df[tonkean_proportions_df['monkey'] == monkey]
    
    # Sort by session and reset index
    monkey_df = monkey_df.sort_values('session').reset_index(drop=True)
    
    # Determine number of sessions and set rolling window size
    num_sessions = len(monkey_df)
    if num_sessions > 5000:
        rolling_window = 50
    elif num_sessions < 500:
        rolling_window = 30
    else:
        rolling_window = 40
    
    # Apply rolling average for stacked proportions
    monkey_df['p_success_smoothed'] = monkey_df['p_success'].rolling(rolling_window, center=True).mean()
    monkey_df['p_error_smoothed'] = monkey_df['p_error'].rolling(rolling_window, center=True).mean()
    monkey_df['p_omission_smoothed'] = monkey_df['p_omission'].rolling(rolling_window, center=True).mean()
    monkey_df['p_premature_smoothed'] = monkey_df['p_premature'].rolling(rolling_window, center=True).mean()
    
    # Combine smoothed values into a DataFrame
    stacked_df = monkey_df[['p_success_smoothed', 'p_error_smoothed', 'p_premature_smoothed', 'p_omission_smoothed']].dropna()
    stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)
    
    # Renormalize to ensure proportions sum to 1
    # stacked_df['proportions_sum'] = stacked_df.sum(axis=1)
    # assert all(abs(stacked_df['proportions_sum'] - 1) < 1e-5), "Proportions do not sum to 1!"
    
    # Filter hierarchy data for the current monkey
    hierarchy_monkey_df = elo[monkey]
    
    # Apply rolling average to hierarchy data
    rolling_avg = hierarchy_monkey_df[monkey].rolling(window=rolling_window, min_periods=1).mean()
    
    # Combined plot
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Stacked proportions plot
    ax1 = fig.add_subplot(gs[0])
    ax1.stackplot(
        range(1, len(stacked_df) + 1),
        stacked_df['p_success_smoothed'],
        stacked_df['p_error_smoothed'],
        stacked_df['p_premature_smoothed'],
        stacked_df['p_omission_smoothed'],
        colors=['#5386b6', '#d57459', '#e8bc60', '#8d5993'],
        labels=['Success', 'Error', 'Premature', 'Stepomission']
    )
    
    # Add the dashed black line for p_success
    overall_p_success = tonkean_table_elo.loc[tonkean_table_elo['name'] == monkey, 'p_success'].values[0]
    ax1.axhline(y=overall_p_success, color='black', linestyle=(0, (5, 10)), linewidth=1.5)  # Dashed line with longer gaps
    
    # Add a label next to the dashed line
    ax1.text(len(stacked_df) + 7, overall_p_success, f'{overall_p_success:.2f}', 
             color='black', fontsize=14, fontname='Times New Roman',  # Match font and size to x-axis
             verticalalignment='center')
    
    ax1.set_ylabel("Stacked Proportion", fontsize=14, fontname='Times New Roman')
    ax1.set_xlim(1, len(stacked_df))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 1])
    ax1.legend(loc='upper left', fontsize=12, prop={'family': 'Times New Roman'})
    ax1.set_title(f"{monkey.capitalize()}", fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax1.set_xlabel('Session Count', fontsize=14, fontname='Times New Roman')
    
    # Hierarchy plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(hierarchy_monkey_df['Date'], rolling_avg, color='#FFD700', linewidth=2)
    ax2.set_title("Hierarchy", fontsize=16, fontweight='bold', fontname='Times New Roman')

    # Set dynamic y-axis for hierarchy plot
    hierarchy_min = rolling_avg.min()
    hierarchy_max = rolling_avg.max()
    
    # Extend y-axis to center 1000 with 850 and 1150 as bounds
    y_mid = 1000
    y_range = max(abs(hierarchy_max - y_mid), abs(hierarchy_min - y_mid)) + 20  # Extend 20 units padding
    ax2.set_ylim(y_mid - y_range, y_mid + y_range)
    ax2.set_yticks([700, 1000, 1300])
    
    # Set dynamic x-axis limits and date format
    ax2.set_xlim(hierarchy_monkey_df['Date'].min(), hierarchy_monkey_df['Date'].max())
    
    
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))  # Adjust tick density dynamically
    
    ax2.set_ylabel('Elo', fontsize=14, fontname='Times New Roman')
    ax2.set_xlabel('Date', fontsize=14, fontname='Times New Roman')
    
        
    ax2.plot(range(1, len(hierarchy_monkey_df) + 1), rolling_avg, color='#FFD700', linewidth=2)
    ax2.set_xlabel('Session Count', fontsize=14, fontname='Times New Roman')
    plt.tight_layout()
    save_path = f'/Users/similovesyou/Downloads/{monkey}_stacked_proportions_hierarchy_Rw40_no10.png'
    # plt.savefig(save_path, format='png')
    
    plt.show()

    for index, row in tonkean_table_elo.iterrows():
        name = row['name']
        elo_mean = row['elo_mean']
        print(f"Monkey: {name}, Elo Mean: {elo_mean}")
