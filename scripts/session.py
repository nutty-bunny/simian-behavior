import pickle
import os
import pandas as pd 

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')

with open(pickle_rick, 'rb') as handle:
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

    min_duration_species = min(durations_species) if durations_species else 0
    max_duration_species = max(durations_species) if durations_species else 0
    mean_duration_species = sum(durations_species) / len(durations_species) if durations_species else 0
    min_count_species = min(frequencies_species) if frequencies_species else 0
    max_count_species = max(frequencies_species) if frequencies_species else 0
    mean_count_species = sum(frequencies_species) / len(frequencies_species) if frequencies_species else 0
    mode_count_species = max(set(frequencies_species), key=frequencies_species.count) if frequencies_species else 0
    
    print(f"\n{species.capitalize()} Session Duration Statistics")
    if species == 'rhesus':
        rhesus_df = pd.DataFrame(rhesus)
        rhesus_df = rhesus_df.round(0)
        print(rhesus_df.to_string(index=False))
    elif species == 'tonkean':
        tonkean_df = pd.DataFrame(tonkean)
        tonkean_df = tonkean_df.round(0)
        print(tonkean_df.to_string(index=False))
    
    print(f"\n{species.capitalize()} Species-Level Descriptives")
    print(f"min duration: {min_duration_species:.0f} s")
    print(f"max duration: {max_duration_species:.0f} s")
    print(f"mean duration: {mean_duration_species:.0f} s")
    print(f"min count: {min_count_species}")
    print(f"max count: {max_count_species}")
    print(f"mean count: {mean_count_species:.0f}")
    print(f"mode count: {mode_count_species}")

count_species = min(durations) if durations else 0
max_duration = max(durations) if durations else 0
mean_duration = sum(durations) / len(durations) if durations else 0
min_count = min(frequencies) if frequencies else 0
max_count = max(frequencies) if frequencies else 0
mean_count = sum(frequencies) / len(frequencies) if frequencies else 0
mode_count = max(set(frequencies), key=frequencies.count) if frequencies else 0

print("\nOverall Statistics Across Both Species")
print(f"min duration: {count_species:.0f} s")
print(f"max duration: {max_duration:.0f} s")
print(f"mean duration: {mean_duration:.0f} s")
print(f"min count: {min_count}")
print(f"max count: {max_count}")
print(f"mean count: {mean_count:.0f}")
print(f"mode count: {mode_count}")

