import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# here's a function to extract descriptives ^^
def descriptives(durations, title="descriptive statistics"):
    descriptive_statistics = {
        'Statistic': ['Min', 'Max', 'Mean', 'Standard Deviation', 'Median', '25th Percentile', '75th Percentile', '10th Percentile', '90th Percentile', 'Range', 'IQR', 'Outlier Count', 'Skewness', 'Kurtosis'],
        'Value': [
            np.min(durations),
            np.max(durations),
            np.mean(durations),
            np.std(durations),
            np.median(durations),
            np.percentile(durations, 25),
            np.percentile(durations, 75),
            np.percentile(durations, 10),
            np.percentile(durations, 90),
            np.max(durations) - np.min(durations),
            np.percentile(durations, 75) - np.percentile(durations, 25),
            len(durations[(durations < (np.percentile(durations, 25) - 1.5 * (np.percentile(durations, 75) - np.percentile(durations, 25)))) | (durations > (np.percentile(durations, 75) + 1.5 * (np.percentile(durations, 75) - np.percentile(durations, 25))))]),
            stats.skew(durations),
            stats.kurtosis(durations)
        ]
    }
    
    # round to 1 decimal
    descriptive_statistics['Value'] = [round(value, 1) for value in descriptive_statistics['Value']]

    descriptive_statistics_df = pd.DataFrame(descriptive_statistics)

    print(f"\n{title}:")
    print(descriptive_statistics_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, edgecolor='black')
    plt.title(f'Distribution of {title}')
    plt.xlabel(f'Duration (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # kernel density plot (potentially more useful in the future)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(durations, fill=True)
    plt.title(f'Density Plot of {title}')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
