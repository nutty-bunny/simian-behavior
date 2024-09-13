import pickle
import os
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

# ** re-run this sscript once pickle rick is finalised! **
directory = Path('/Users/similovesyou/Desktop/qts/simian-behavior/data/py')
pickle_rick = os.path.join(directory, 'data.pickle')
with open(pickle_rick, 'rb') as handle:
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
    'ces': 'ceasar',
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
    'hav': 'havana',
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
plt.plot(df.index, df['nema'])

tonkean_table_elo = pd.DataFrame(tonkean)

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
