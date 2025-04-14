# attempt based descriptives 
import pickle
import os
import pandas as pd 

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')

with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

rhesus = []
tonkean = []

durations = []  # durations irrespective of species

# Loop over data
for species, monkeys in data.items():
    durations_species = []  # durations wrt species
    
    for name, data_dict in monkeys.items():
        attempts = data_dict['attempts']
        
        task_durations = attempts['reaction_time'].dropna().tolist()  # get task durations
        
        if species == 'rhesus':
            rhesus.append({
                'monkey': name,
                'min duration': min(task_durations) if task_durations else 0,
                'max duration': max(task_durations) if task_durations else 0,
                'mean duration': sum(task_durations) / len(task_durations) if task_durations else 0,
            })
        elif species == 'tonkean':
            tonkean.append({
                'monkey': name,
                'min duration': min(task_durations) if task_durations else 0,
                'max duration': max(task_durations) if task_durations else 0,
                'mean duration': sum(task_durations) / len(task_durations) if task_durations else 0,
            })
        
        durations_species.extend(task_durations)  # accumulate task durations respective of species
        durations.extend(task_durations)  # accumulate overall task durations

    min_duration_species = min(durations_species) if durations_species else 0
    max_duration_species = max(durations_species) if durations_species else 0
    mean_duration_species = sum(durations_species) / len(durations_species) if durations_species else 0
    
    print(f"\n{species.capitalize()} Task Duration Statistics")
    if species == 'rhesus':
        rhesus_df = pd.DataFrame(rhesus)
        print(rhesus_df.to_string(index=False))
    elif species == 'tonkean':
        tonkean_df = pd.DataFrame(tonkean)
        print(tonkean_df.to_string(index=False))
    
    print(f"\n{species.capitalize()} Species-Level Descriptives")
    print(f"min duration: {min_duration_species} ms")
    print(f"max duration: {max_duration_species} ms")
    print(f"mean duration: {mean_duration_species} ms")

min_duration = min(durations) if durations else 0
max_duration = max(durations) if durations else 0
mean_duration = sum(durations) / len(durations) if durations else 0

print("\nOverall Statistics Across Both Species")
print(f"min duration: {min_duration} s")
print(f"max duration: {max_duration} s")
print(f"mean duration: {mean_duration} s")

