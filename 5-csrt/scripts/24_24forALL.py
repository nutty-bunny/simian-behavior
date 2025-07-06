# sessions across the 24 hours of the day 
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')
with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

print("Loaded data:")
print(data.keys())

# created a function based on olga exploratory analyses 
def create_plots(species, name, steps, attempts):
    print(f"\nCreating plots for '{species}' '{name}'...")

    steps['instant_begin'] = pd.to_datetime(steps['instant_begin'], unit='ms')
    steps['instant_end'] = pd.to_datetime(steps['instant_end'], unit='ms')

    attempts['instant_begin'] = pd.to_datetime(attempts['instant_begin'], unit='ms')
    attempts['instant_end'] = pd.to_datetime(attempts['instant_end'], unit='ms')

    steps['hour'] = steps['instant_begin'].dt.hour
    unique_attempts = steps.drop_duplicates(subset='attempt')
    attempt_frequency_by_hour = unique_attempts.groupby('hour').size().reset_index(name='attempt_count')

    # plt.figure(figsize=(12, 6))
    # plt.plot(attempt_frequency_by_hour['hour'], attempt_frequency_by_hour['attempt_count'], marker='o')
    # plt.xlabel('Hour of the Day')
    # plt.ylabel('Number of Unique Attempts')
    # plt.title(f'Frequency of Unique Attempts Across a 24-Hour Interval - {species} {name}')
    # plt.grid(True)
    # plt.xticks(range(0, 24))
    # plt.show()

    attempts['hour'] = attempts['instant_begin'].dt.hour
    unique_sessions = attempts.drop_duplicates(subset='session')
    session_frequency_by_hour = unique_sessions.groupby('hour').size().reset_index(name='session_count')

    plt.figure(figsize=(12, 6))
    plt.plot(session_frequency_by_hour['hour'], session_frequency_by_hour['session_count'], marker='o')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Unique Sessions')
    plt.title(f'Frequency of Unique Sessions Across a 24-Hour Interval - {species} {name}')
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.show()

for species, monkeys in data.items():
    for name, info in monkeys.items():
        steps = info.get('steps')
        attempts = info.get('attempts')

        if steps is not None and attempts is not None:
            create_plots(species, name, steps, attempts)
        else:
            print(f"Data for '{species}' '{name}' is not available.")

# to do
    # turn into kernel densities 