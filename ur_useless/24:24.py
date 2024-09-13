import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')

with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

print("Loaded data:")
print(data)

# 'id' in attempts = 'attempt' in steps
# check 

# let's explore olga 
olga = data.get('tonkean', {}).get('olga', {}) # isolate olga

olga_steps = olga.get('steps')
olga_attempts = olga.get('attempts')

if olga_steps is not None and olga_attempts is not None:
    print()
    print("olga's steps:")
    print(olga_steps.head())  

    print()
    print("olga's attempts:")
    print(olga_attempts.head()) 
else:
    print("Data for 'rhesus' species and 'olga' monkey is not available.")

# convert timestamps

olga_steps['instant_begin'] = pd.to_datetime(olga_steps['instant_begin'], unit='ms')
olga_steps['instant_end'] = pd.to_datetime(olga_steps['instant_end'], unit='ms')

print(olga_steps.head())

olga_steps['hour'] = olga_steps['instant_begin'].dt.hour
unique_attempts = olga_steps.drop_duplicates(subset='attempt') 
attempt_frequency_by_hour = unique_attempts.groupby('hour').size().reset_index(name='attempt_count')

# plot freq of attempts across a 24-hour interval 
# plot also replicated using attempt data (sanity check)

plt.figure(figsize=(12, 6))
plt.plot(attempt_frequency_by_hour['hour'], attempt_frequency_by_hour['attempt_count'], marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Unique Attempts')
plt.title('Frequency of Unique Attempts Across a 24-Hour Interval')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()

# convert timestamps 

olga_attempts['instant_begin'] = pd.to_datetime(olga_attempts['instant_begin'], unit='ms')
olga_attempts['instant_end'] = pd.to_datetime(olga_attempts['instant_end'], unit='ms')

print(olga_attempts.head())

olga_attempts['hour'] = olga_attempts['instant_begin'].dt.hour
unique_sessions = olga_attempts.drop_duplicates(subset='session')
session_frequency_by_hour = unique_sessions.groupby('hour').size().reset_index(name='session_count')

# plot freq of sessions across a 24-hour interval 

plt.figure(figsize=(12, 6))
plt.plot(attempt_frequency_by_hour['hour'], session_frequency_by_hour['session_count'], marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Unique Sessions')
plt.title('Frequency of Unique Sessions Across a 24-Hour Interval')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()

# accuracy across sessions

success = olga_attempts.groupby('session')['result'].apply(
    lambda x: (x == 'success').sum() / x.count()).reset_index(name='p_success')
print(success)

error = olga_attempts.groupby('session')['result'].apply(
    lambda x: (x == 'error').sum() / x.count()).reset_index(name='p_error')
print(error)

omission = olga_attempts.groupby('session')['result'].apply(
    lambda x: (x == 'stepomission').sum() / x.count()).reset_index(name='p_omission')
print(omission)

premature = olga_attempts.groupby('session')['result'].apply(
    lambda x: (x == 'prematured').sum() / x.count()).reset_index(name='p_premature')
print(premature)

# pp across sessions heh

pp_session = success.merge(error, on='session').merge(omission, on='session').merge(premature, on='session')
print(pp_session)

# number of sessions (y) across time (x) by date

olga_attempts['date'] = olga_attempts['instant_begin'].dt.date
unique_sessions = olga_attempts.drop_duplicates(subset='session')
session_frequency_by_date = unique_sessions.groupby('date').size().reset_index(name='session_count')

olga_first = session_frequency_by_date['date'].min() 
olga_last = session_frequency_by_date['date'].max() 

print(f"first session date: {olga_first}")
print(f"fast session date: {olga_last}")

# number of sessions (y) across time (x) by month

olga_attempts['month'] = olga_attempts['instant_begin'].dt.to_period('M')
unique_sessions = olga_attempts.drop_duplicates(subset='session')
session_frequency_by_month = unique_sessions.groupby('month').size().reset_index(name='session_count')

plt.figure(figsize=(12, 6))
plt.bar(session_frequency_by_month['month'].astype(str), session_frequency_by_month['session_count'], color='blue', alpha=0.7)
plt.xlabel('Month')
plt.ylabel('Number of Sessions')
plt.title('Histogram of Session Counts Over Time')
plt.xticks(ticks=range(0, len(session_frequency_by_month), 3), rotation=45)  # Show every third month
plt.grid(True)
plt.tight_layout()
plt.show()

# number of sessions (y) across time (y) by season 

olga_attempts['month'] = olga_attempts['instant_begin'].dt.month
unique_sessions = olga_attempts.drop_duplicates(subset='session')
session_frequency_by_month = unique_sessions.groupby('month').size().reset_index(name='session_count')

plt.figure(figsize=(12, 6))
plt.plot(session_frequency_by_month['month'], session_frequency_by_month['session_count'], marker='o')
plt.xlabel('Month')
plt.ylabel('Number of Unique Sessions')
plt.title('Frequency of Unique Sessions Across Months')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.tight_layout()
plt.show()
