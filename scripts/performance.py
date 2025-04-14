# performance across sessions miauw (per monkey)
import pickle
import os
import pandas as pd 

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')

with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

print("Loaded data:")
print(data)

proportions = {} 
for species, monkeys in data.items():
    for monkey, data_dict in monkeys.items():
        attempts_data = data_dict.get('attempts')
        
        if attempts_data is not None:
            result_counts = attempts_data['result'].value_counts(normalize=True)
            
            proportions[f'{species} - {monkey}'] = result_counts

# proportions across monkeys 
for monkey, result_proportions in proportions.items():
    print(f"Proportions for {monkey}:")
    print(result_proportions)
    print()  

split_proportions = {}

for species, monkeys in data.items():
    for monkey, data_dict in monkeys.items():
        attempts_data = data_dict.get('attempts')
        
        if attempts_data is not None:
            session_proportions = attempts_data.groupby('session')['result'].value_counts(normalize=True).unstack(fill_value=0)
            
            proportions[f'{species} - {monkey}'] = session_proportions

# proportions across sessions per monkey 
for monkey, session_proportions in proportions.items():
    print(f"Proportions for {monkey}:")
    print(session_proportions)
    print()  
    