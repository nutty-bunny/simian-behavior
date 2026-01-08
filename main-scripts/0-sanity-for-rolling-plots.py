import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"

# Directory setup
directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"

# Load all three task datasets
csrt_pickle = os.path.join(directory, "data.pickle")
tav_pickle = os.path.join(directory, "tav.pickle")
dms_pickle = os.path.join(directory, "dms.pickle")

print("Loading data...")
with open(csrt_pickle, 'rb') as handle:
    csrt_data = pickle.load(handle)

with open(tav_pickle, 'rb') as handle:
    tav_data = pickle.load(handle)

with open(dms_pickle, 'rb') as handle:
    dms_data = pickle.load(handle)

# Remove monkeys with RFID issues
monkeys_to_remove = {"joy", "jipsy"}

def extract_session_data(task_data, monkey_name):
    """Extract date-aligned session data from a task for a specific monkey."""
    session_data = []
    
    for species, species_data in task_data.items():
        if monkey_name not in species_data:
            continue
        
        monkey_data = species_data[monkey_name]
        attempts = monkey_data.get("attempts")
        
        if attempts is None or attempts.empty:
            continue
        
        # Convert timestamps
        attempts["instant_begin"] = pd.to_datetime(attempts["instant_begin"], unit="ms", errors="coerce")
        attempts["instant_end"] = pd.to_datetime(attempts["instant_end"], unit="ms", errors="coerce")
        
        # Group by session
        for session_id in attempts["session"].unique():
            session_attempts = attempts[attempts["session"] == session_id]
            
            # Get session date (normalize to day)
            session_date = session_attempts["instant_begin"].min().normalize()
            
            # Calculate proportions
            total = len(session_attempts)
            outcomes = session_attempts["result"].value_counts()
            
            p_success = outcomes.get("success", 0) / total
            p_error = outcomes.get("error", 0) / total
            p_premature = outcomes.get("prematured", 0) / total
            p_omission = outcomes.get("stepomission", 0) / total
            
            session_data.append({
                'date': session_date,
                'session': session_id,
                'p_success': p_success,
                'p_error': p_error,
                'p_premature': p_premature,
                'p_omission': p_omission,
                'n_trials': total
            })
    
    if session_data:
        df = pd.DataFrame(session_data)
        df = df.sort_values('date').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

# Collect all monkeys from 5-CSRT (main task)
csrt_monkeys = set()
for species, species_data in csrt_data.items():
    csrt_monkeys.update(species_data.keys())

# Remove problematic monkeys
csrt_monkeys = csrt_monkeys - monkeys_to_remove

print(f"Found {len(csrt_monkeys)} monkeys in 5-CSRT")
print(f"Monkeys: {sorted(csrt_monkeys)}")

# Rolling window parameter
rolling_window = 10

# Stacked plot colors (matching your original scripts)
colors = ["#5386b6", "#d57459", "#e8bc60", "#8d5993"]

print("\n" + "="*60)
print("Creating date-aligned stacked proportion plots for all monkeys")
print("="*60)

for monkey in sorted(csrt_monkeys):
    print(f"\nProcessing {monkey}...")
    
    # Extract data from all three tasks
    csrt_df = extract_session_data(csrt_data, monkey)
    tav_df = extract_session_data(tav_data, monkey)
    dms_df = extract_session_data(dms_data, monkey)
    
    # Collect available tasks
    available_tasks = []
    if not csrt_df.empty:
        available_tasks.append(('5-CSRT', csrt_df))
    if not tav_df.empty:
        available_tasks.append(('TAV', tav_df))
    if not dms_df.empty:
        available_tasks.append(('DMS', dms_df))
    
    if len(available_tasks) == 0:
        print(f"  Skipping {monkey} - no data")
        continue
    
    n_tasks = len(available_tasks)
    print(f"  Tasks available: {[t[0] for t in available_tasks]}")
    
    # Find the overall date range across all tasks for this monkey
    all_dates = []
    for task_name, df in available_tasks:
        if not df.empty:
            all_dates.extend(df['date'].tolist())
    
    if not all_dates:
        print(f"  Skipping {monkey} - no valid dates")
        continue
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    # Create figure with subplots for each task
    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 5*n_tasks), sharex=True)
    
    # Handle case where there's only one task (axes won't be an array)
    if n_tasks == 1:
        axes = [axes]
    
    for idx, (task_name, df) in enumerate(available_tasks):
        ax = axes[idx]
        
        # Normalize proportions (ensure they sum to 1)
        stacked_df = df[["p_success", "p_error", "p_premature", "p_omission"]].copy()
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)
        
        # Apply rolling window
        stacked_df_roll = stacked_df.rolling(window=rolling_window, min_periods=1, center=True).mean()
        
        # Add dates back
        stacked_df_roll['date'] = df['date'].values
        
        # Create stacked area plot
        ax.stackplot(
            stacked_df_roll['date'],
            stacked_df_roll["p_success"],
            stacked_df_roll["p_error"],
            stacked_df_roll["p_premature"],
            stacked_df_roll["p_omission"],
            edgecolor="none",
            linewidth=0,
            baseline="zero",
            colors=colors,
            labels=["Success", "Error", "Premature", "Omission"],
            alpha=0.9
        )
        
        # Add horizontal line for overall success rate
        overall_p_success = df["p_success"].mean()
        ax.axhline(
            y=overall_p_success,
            color="black",
            linestyle=(0, (8, 4)),
            linewidth=1.5,
            alpha=0.8
        )
        
        # Formatting
        ax.set_ylabel("Stacked Proportion", fontsize=16, fontname="Times New Roman")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(f"{task_name}", fontsize=20, fontweight='bold', 
                    fontname="Times New Roman", pad=15)
        
        # Set same x-axis limits for all subplots
        ax.set_xlim(min_date, max_date)
        
        # Legend
        if idx == 0:  # Only on first subplot
            ax.legend(loc='upper left', fontsize=14, 
                     framealpha=0.9, prop={"family": "Times New Roman"})
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', labelsize=14)
        
        # Only show x-label on bottom plot
        if idx == len(available_tasks) - 1:
            ax.set_xlabel("Date", fontsize=16, fontname="Times New Roman")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall title
    fig.suptitle(
        f'{monkey.capitalize()}',
        fontsize=24,
        fontweight='bold',
        fontname="Times New Roman",
        y=0.995
    )
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("\n" + "="*60)
print("Analysis complete!")
print(f"Processed {len(csrt_monkeys)} monkeys")
print("="*60)

print("\n" + "="*60)
print("Creating wide 5-CSRT-only plots with detailed date labels")
print("="*60)

# Now create wide plots focused only on 5-CSRT
for monkey in sorted(csrt_monkeys):
    print(f"\nProcessing {monkey} (5-CSRT only)...")
    
    # Extract only 5-CSRT data
    csrt_df = extract_session_data(csrt_data, monkey)
    
    if csrt_df.empty:
        print(f"  Skipping {monkey} - no 5-CSRT data")
        continue
    
    print(f"  Sessions: {len(csrt_df)}, Date range: {csrt_df['date'].min().date()} to {csrt_df['date'].max().date()}")
    
    # Create wide figure
    fig, ax = plt.subplots(1, 1, figsize=(24, 6))
    
    # Normalize proportions (ensure they sum to 1)
    stacked_df = csrt_df[["p_success", "p_error", "p_premature", "p_omission"]].copy()
    stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)
    
    # Apply rolling window
    stacked_df_roll = stacked_df.rolling(window=rolling_window, min_periods=1, center=True).mean()
    
    # Add dates back
    stacked_df_roll['date'] = csrt_df['date'].values
    
    # Create stacked area plot
    ax.stackplot(
        stacked_df_roll['date'],
        stacked_df_roll["p_success"],
        stacked_df_roll["p_error"],
        stacked_df_roll["p_premature"],
        stacked_df_roll["p_omission"],
        edgecolor="none",
        linewidth=0,
        baseline="zero",
        colors=colors,
        labels=["Success", "Error", "Premature", "Omission"],
        alpha=0.9
    )
    
    # Add horizontal line for overall success rate
    overall_p_success = csrt_df["p_success"].mean()
    ax.axhline(
        y=overall_p_success,
        color="black",
        linestyle=(0, (8, 4)),
        linewidth=1.8,
        alpha=0.8
    )
    
    # Formatting
    ax.set_ylabel("Stacked Proportion", fontsize=18, fontname="Times New Roman", fontweight='bold')
    ax.set_xlabel("Date", fontsize=18, fontname="Times New Roman", fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.5, axis='both')
    
    # Title
    ax.set_title(
        f'{monkey.capitalize()} - 5-CSRT Performance (Rolling avg = {rolling_window} sessions)',
        fontsize=22,
        fontweight='bold',
        fontname="Times New Roman",
        pad=20
    )
    
    # Legend
    ax.legend(loc='upper left', fontsize=16, 
             framealpha=0.95, prop={"family": "Times New Roman"})
    
    # Detailed date formatting on x-axis
    # Calculate appropriate interval based on date range
    date_range_days = (csrt_df['date'].max() - csrt_df['date'].min()).days
    
    if date_range_days < 90:  # Less than 3 months
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    elif date_range_days < 365:  # Less than 1 year
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    elif date_range_days < 730:  # Less than 2 years
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:  # 2+ years
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate x-axis labels for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add minor gridlines for months if showing weeks
    if date_range_days < 90:
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("\n" + "="*60)
print("Wide 5-CSRT plots complete!")
print("="*60)

print("\n" + "="*60)
print("Creating time-aligned stacked proportion plots for all monkeys")
print("="*60)

# Collect all 5-CSRT data for all monkeys
all_monkey_data = {}
overall_min_date = None
overall_max_date = None

for monkey in sorted(csrt_monkeys):
    csrt_df = extract_session_data(csrt_data, monkey)
    
    if csrt_df.empty:
        continue
    
    all_monkey_data[monkey] = csrt_df
    
    # Track overall date range
    if overall_min_date is None or csrt_df['date'].min() < overall_min_date:
        overall_min_date = csrt_df['date'].min()
    if overall_max_date is None or csrt_df['date'].max() > overall_max_date:
        overall_max_date = csrt_df['date'].max()

if not all_monkey_data:
    print("No data to plot!")
else:
    n_monkeys = len(all_monkey_data)
    print(f"Plotting {n_monkeys} monkeys")
    print(f"Overall date range: {overall_min_date.date()} to {overall_max_date.date()}")
    
    # Create figure with one row per monkey, all time-aligned
    fig, axes = plt.subplots(n_monkeys, 1, figsize=(24, 4*n_monkeys), sharex=True)
    
    # Handle case where there's only one monkey
    if n_monkeys == 1:
        axes = [axes]
    
    for idx, (monkey, df) in enumerate(sorted(all_monkey_data.items())):
        ax = axes[idx]
        
        print(f"  Plotting {monkey}...")
        
        # Normalize proportions (ensure they sum to 1)
        stacked_df = df[["p_success", "p_error", "p_premature", "p_omission"]].copy()
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)
        
        # Apply rolling window
        stacked_df_roll = stacked_df.rolling(window=rolling_window, min_periods=1, center=True).mean()
        
        # Add dates back
        stacked_df_roll['date'] = df['date'].values
        
        # Create stacked area plot
        if idx == 0:
            ax.stackplot(
                stacked_df_roll['date'],
                stacked_df_roll["p_success"],
                stacked_df_roll["p_error"],
                stacked_df_roll["p_premature"],
                stacked_df_roll["p_omission"],
                edgecolor="none",
                linewidth=0,
                baseline="zero",
                colors=colors,
                labels=["Success", "Error", "Premature", "Omission"],
                alpha=0.9
            )
        else:
            ax.stackplot(
                stacked_df_roll['date'],
                stacked_df_roll["p_success"],
                stacked_df_roll["p_error"],
                stacked_df_roll["p_premature"],
                stacked_df_roll["p_omission"],
                edgecolor="none",
                linewidth=0,
                baseline="zero",
                colors=colors,
                alpha=0.9
            )
        
        # Add horizontal line for overall success rate
        overall_p_success = df["p_success"].mean()
        ax.axhline(
            y=overall_p_success,
            color="black",
            linestyle=(0, (8, 4)),
            linewidth=1.5,
            alpha=0.8
        )
        
        # Set same x-axis limits for all subplots (time-aligned)
        ax.set_xlim(overall_min_date, overall_max_date)
        
        # Formatting
        ax.set_ylabel("Stacked\nProportion", fontsize=12, fontname="Times New Roman", rotation=0, ha='right', va='center')
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Title with monkey name (on left side)
        ax.set_title(f"{monkey.capitalize()}", fontsize=14, fontweight='bold', 
                    fontname="Times New Roman", loc='left', pad=10)
        
        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=12, 
                     framealpha=0.95, prop={"family": "Times New Roman"}, ncol=4)
    
    # Format x-axis (only bottom plot shows labels)
    date_range_days = (overall_max_date - overall_min_date).days
    if date_range_days < 365:
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    elif date_range_days < 730:
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    axes[-1].set_xlabel("Date", fontsize=16, fontname="Times New Roman", fontweight='bold')
    axes[-1].tick_params(axis='x', labelsize=13)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall title
    fig.suptitle(
        f'All Monkeys - 5-CSRT Performance Time-Aligned\n(Rolling avg = {rolling_window} sessions, n={n_monkeys} monkeys)',
        fontsize=20,
        fontweight='bold',
        fontname="Times New Roman",
        y=0.998
    )
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.995])
    plt.show()
    plt.close(fig)

print("\n" + "="*60)
print("Time-aligned stacked plots complete!")
print("="*60)