import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from matplotlib.lines import Line2D

# Suppress FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Load and combine data
    data_paths = {
        'TAV': {
            'Rhesus': '/Users/similovesyou/Desktop/qts/simian-behavior/TAVo-tevas/derivatives/rhesus-elo-min-attempts.csv',
            'Tonkean': '/Users/similovesyou/Desktop/qts/simian-behavior/TAVo-tevas/derivatives/tonkean-elo-min-attempts.csv'
        },
        'DMS': {
            "Rhesus": '/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/rhesus-elo-min-attempts.csv',
            'Tonkean': '/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/tonkean-elo-min-attempts.csv'
        },
        '5-CSRT': {
            'Rhesus': '/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo-min-attempts.csv',
            'Tonkean': '/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo-min-attempts.csv'
        }
    }

    dfs = []
    for task, species_files in data_paths.items():
        for species, path in species_files.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['task'] = task
                df['species'] = species
                dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Style parameters
    colors = {"Rhesus": "steelblue", "Tonkean": "purple"}
    titles = ["Success Rate", "Error Rate", "Omission Rate", "Premature Rate"]
    metrics = ['p_success', 'p_error', 'p_omission', 'p_premature']
    task_order = ['TAV', 'DMS', '5-CSRT']  # Updated task order with new labels

    # Create custom legend handles
    handles = [
        Line2D([0], [0], color='steelblue', lw=2, label='Rhesus'),
        Line2D([0], [0], color='purple', lw=2, label='Tonkean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Individual'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Group Mean')
    ]

    for i, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        # Plot individual connected lines
        for name in combined_df['name'].unique():
            subject_data = combined_df[combined_df['name'] == name].sort_values('task', key=lambda x: x.map({v:k for k,v in enumerate(task_order)}))
            color = colors[subject_data['species'].iloc[0]]
            ax.plot(subject_data['task'], 
                   subject_data[metric], 
                   color=color, 
                   alpha=0.2, 
                   linewidth=0.5)
            ax.scatter(subject_data['task'], 
                      subject_data[metric], 
                      color='gray', 
                      alpha=0.4, 
                      s=30)

        # Plot group mean lines
        for species in ["Rhesus", "Tonkean"]:
            species_data = combined_df[combined_df['species'] == species]
            mean_values = species_data.groupby('task')[metric].mean().reindex(task_order)
            ax.plot(mean_values.index, 
                   mean_values.values, 
                   color=colors[species], 
                   linestyle='--', 
                   linewidth=2, 
                   marker='o', 
                   markersize=8)

        # Styling
        ax.set_title(title, fontsize=18, fontweight="bold", fontname="DejaVu Sans")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        
        ax.set_xlabel("")
        if i in [0, 2]:
            ax.set_ylabel("Proportion", fontsize=15)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        
        # Set x-axis labels with sample sizes
        task_labels = []
        for task in task_order:
            n = combined_df[combined_df['task'] == task].shape[0]
            task_labels.append(f"{task}\n(n={n})")
        ax.set_xticks(range(len(task_labels)))
        ax.set_xticklabels(task_labels, fontsize=14)

    # Add legend and title
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(
        "Performance Metrics Across Cognitive Tasks",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
    )
    
    # Save the plots
    output_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}/performance-metrics-across-tasks.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/performance-metrics-across-tasks.svg", bbox_inches='tight')
    
    plt.show()