# exploratory -- consider to include in GLM 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import statsmodels.api as sm

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
pickle_rick = os.path.join(directory, 'data.pickle')
with open(pickle_rick, 'rb') as handle:
    data = pickle.load(handle)

all_attempts = []
for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        attempts = monkey_data['attempts']
        
        attempts['species'] = species
        attempts['name'] = name
        all_attempts.append(attempts)

attempts_df = pd.concat(all_attempts)

attempts_df['variable_delay'] = attempts_df['variable_delay'] / 1000  # convert to seconds (if you want)
attempts_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # toss NaNs

# remove outliers
q1 = attempts_df['variable_delay'].quantile(0.01) 
q99 = attempts_df['variable_delay'].quantile(0.99)
attempts_df = attempts_df[(attempts_df['variable_delay'] >= q1) & (attempts_df['variable_delay'] <= q99)]  # update the df with removed outliers

count = attempts_df.groupby(['variable_delay', 'result']).size().unstack(fill_value=0)  # absolute result count 
proportion = count.div(count.sum(axis=1), axis=0).reset_index()  # relative proportion 

melt_me = proportion.melt(id_vars=['variable_delay'], var_name='result', value_name='proportion')  # melt df to plot 

# loess smoothing function 
def loess(x, y, frac=0.3):
    loess = sm.nonparametric.lowess
    smoothed = loess(y, x, frac=frac)
    return smoothed[:, 0], smoothed[:, 1]

# plotting proportions wrt variable delay
results = ['success', 'error', 'stepomission'] 
colors = {'success': 'green', 'error': 'red', 'stepomission': 'blue'}  # change colors pls
fig, axes = plt.subplots(3, 1, figsize=(14, 24), sharex=True)

for ax, result in zip(axes, results):
    subset = melt_me[melt_me['result'] == result]
    sns.scatterplot(x='variable_delay', y='proportion', data=subset, label=result.capitalize(), color=colors[result], ax=ax, legend=False)
    x_smooth, y_smooth = loess(subset['variable_delay'], subset['proportion'])
    ax.plot(x_smooth, y_smooth, label=f'{result.capitalize()} (LOESS)', color=colors[result])
    ax.set_title(f'Proportion of {result.capitalize()} by Variable Delay')
    ax.set_xlabel('Variable Delay (Seconds)')
    ax.set_ylabel('Proportion')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# calculating mean rts wrt to variable delay 
mean_rts = attempts_df.groupby(['variable_delay', 'result'])['reaction_time'].mean().reset_index()  # mean RT for each variable delay value
mean_rt_success = mean_rts[mean_rts['result'] == 'success']  # filter for success
mean_rt_error = mean_rts[mean_rts['result'] == 'error']  # filter for error

# plotting mean rts wrt to variable delay
results = ['success', 'error'] 
colors = {'success': 'green', 'error': 'red'} 
rts = [mean_rt_success, mean_rt_error]
fig, axes = plt.subplots(2, 1, figsize=(14, 16), sharex=True)

for ax, data, result in zip(axes, rts, results):
    sns.scatterplot(x='variable_delay', y='reaction_time', data=data, label=result, color=colors[result], ax=ax, legend=False)
    x_smooth, y_smooth = loess(data['variable_delay'], data['reaction_time'])
    ax.plot(x_smooth, y_smooth, label=f'{result.capitalize()} (LOESS)', color=colors[result])
    ax.set_title(f'Mean Reaction Times for {result.capitalize()} by Variable Delay')
    ax.set_xlabel('Variable Delay (seconds)')
    ax.set_ylabel('Mean Reaction Time (ms)')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()