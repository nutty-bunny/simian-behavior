import os
import pickle
import pandas as pd
from datetime import datetime
from tabulate import tabulate

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except Exception as e:
            print(f"Failed to read {path}: {str(e)}")
            return None

def save_reports_to_csv(df, species, report_type):
    """Save reports to CSV files in a reports folder"""
    os.makedirs('monkey_reports_csv', exist_ok=True)
    filename = f"{species}_{report_type}.csv"
    df.to_csv(os.path.join('monkey_reports_csv', filename), index=False)
    print(f"Saved {report_type} report for {species} to monkey_reports_csv/{filename}")

def analyze_monkey_data(directory):
    data = {}
    found_monkeys = []
    missing_monkeys = []
    
    print("\nScanning directory structure...")
    
    for species in ['rhesus', 'tonkean']:
        species_path = os.path.join(directory, species)
        if not os.path.exists(species_path):
            print(f"Species folder not found: {species_path}")
            continue
            
        data[species] = {}
        for name in os.listdir(species_path):
            name_path = os.path.join(species_path, name)
            if not os.path.isdir(name_path):
                continue
                
            step_file, attempt_file = None, None
            for f in os.listdir(name_path):
                f_lower = f.lower()
                if "step" in f_lower and f_lower.endswith(".csv"):
                    step_file = os.path.join(name_path, f)
                elif ("id" in f_lower and f_lower.endswith(".csv") 
                      and "step" not in f_lower):
                    attempt_file = os.path.join(name_path, f)
            
            if step_file and attempt_file:
                try:
                    data[species][name] = {
                        "steps": safe_read_csv(step_file),
                        "attempts": safe_read_csv(attempt_file)
                    }
                    found_monkeys.append(f"{species}/{name}")
                except Exception as e:
                    print(f"Error reading files for {species}/{name}: {e}")
                    missing_monkeys.append(f"{species}/{name} (error)")
            else:
                missing_monkeys.append(f"{species}/{name} (missing files)")
    
    print("\nData collection summary:")
    print(f"Found valid data for {len(found_monkeys)} monkeys")
    print(f"Missing/invalid data for {len(missing_monkeys)} monkeys")
    
    if not found_monkeys:
        print("\nNo valid monkey data found - cannot proceed with analysis")
        return
    
    print("\nMonkeys with valid data:")
    for monkey in found_monkeys:
        print(f"- {monkey}")
    
    return data

def generate_level_duration_reports(data):
    for species in ['rhesus', 'tonkean']:
        if species not in data:
            continue
            
        level_stats = []
        
        for monkey_name, monkey_data in data[species].items():
            attempts = monkey_data.get('attempts')
            if attempts is None or 'task' not in attempts.columns:
                continue
                
            # Filter for tasks 88-98
            task_filter = attempts[attempts['task'].between(88, 98)].copy()
            if task_filter.empty:
                continue
                
            # Convert timestamps
            task_filter['datetime'] = pd.to_datetime(task_filter['instant_begin'], unit='ms', errors='coerce')
            task_filter = task_filter.dropna(subset=['datetime'])
            
            # Calculate time spent per level
            for task_num, group in task_filter.groupby('task'):
                if len(group) > 1:
                    time_span = (group['datetime'].max() - group['datetime'].min()).total_seconds() / 86400
                    level_stats.append({
                        'level': task_num,
                        'duration_days': time_span,
                        'monkey': monkey_name
                    })
        
        if not level_stats:
            print(f"\nNo level duration data found for {species}")
            continue
            
        # Create DataFrame and calculate stats
        level_df = pd.DataFrame(level_stats)
        stats_df = level_df.groupby('level')['duration_days'].agg(
            ['min', 'max', 'mean', 'median', 'std', 'count']
        ).reset_index()
        
        # Rename columns for clarity
        stats_df = stats_df.rename(columns={
            'min': 'min_days',
            'max': 'max_days',
            'mean': 'mean_days',
            'median': 'median_days',
            'std': 'std_days',
            'count': 'n_monkeys'
        })
        
        # Save CSV report
        save_reports_to_csv(stats_df.round(2), species, "level_duration_stats")
        
        # Print summary
        print(f"\n{species.capitalize()} Level Duration Statistics:")
        print(tabulate(stats_df.round(2), headers='keys', tablefmt='grid', showindex=False))

def generate_individual_reports(data):
    for species in ['rhesus', 'tonkean']:
        if species not in data:
            continue
            
        report_data = []
        
        for monkey_name, monkey_data in data[species].items():
            attempts = monkey_data.get('attempts')
            if attempts is None or 'task' not in attempts.columns:
                continue
                
            # Filter for tasks 88-98
            task_filter = attempts[attempts['task'].between(88, 98)].copy()
            if task_filter.empty:
                continue
                
            # Convert timestamps
            task_filter['datetime'] = pd.to_datetime(task_filter['instant_begin'], unit='ms', errors='coerce')
            task_filter = task_filter.dropna(subset=['datetime'])
            
            monkey_data = {'Monkey': monkey_name}
            for task_num, group in task_filter.groupby('task'):
                if len(group) > 1:
                    monkey_data[f'Task {task_num} Start'] = group['datetime'].min().strftime('%Y-%m-%d')
                    monkey_data[f'Task {task_num} End'] = group['datetime'].max().strftime('%Y-%m-%d')
                    monkey_data[f'Task {task_num} Attempts'] = len(group)
            
            report_data.append(monkey_data)
        
        if not report_data:
            print(f"\nNo individual report data found for {species}")
            continue
            
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Reorder columns to group by task
        cols = ['Monkey']
        tasks = sorted(set([col.split()[1] for col in report_df.columns if col.startswith('Task')]))
        for task in tasks:
            cols.extend([f'Task {task} Start', f'Task {task} End', f'Task {task} Attempts'])
        
        report_df = report_df[cols]
        
        # Save CSV report
        save_reports_to_csv(report_df, species, "individual_task_dates")
        
        # Print summary
        print(f"\n{species.capitalize()} Individual Monkey Task Dates:")
        print(tabulate(report_df, headers='keys', tablefmt='grid', showindex=False))

# Main execution
directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
data = analyze_monkey_data(directory)

if data:
    # Generate all reports
    generate_level_duration_reports(data)
    generate_individual_reports(data)
    
def generate_malt_usage_by_period(data):
    os.makedirs('monkey_reports_csv', exist_ok=True)

    for label, task_range in {
        "learning_period": range(88, 97),
        "learned_period": range(97, 99)
    }.items():
        summary_records = []

        for species in ['rhesus', 'tonkean']:
            if species not in data:
                continue

            monkey_stats = []

            for monkey_name, monkey_data in data[species].items():
                attempts = monkey_data.get('attempts')
                if attempts is None or 'task' not in attempts.columns or 'instant_begin' not in attempts.columns:
                    continue

                # Filter tasks in given range
                filtered = attempts[attempts['task'].isin(task_range)].copy()
                if filtered.empty:
                    continue

                filtered['datetime'] = pd.to_datetime(filtered['instant_begin'], unit='ms', errors='coerce')
                filtered = filtered.dropna(subset=['datetime'])

                # Unique days and months
                unique_days = filtered['datetime'].dt.date.unique()
                day_count = len(unique_days)

                month_series = pd.Series(unique_days)
                unique_months = month_series.map(lambda d: d.replace(day=1)).unique()
                month_count = len(unique_months)

                # Attempts
                total_attempts = len(filtered)
                attempts_per_day = total_attempts / day_count if day_count > 0 else 0
                attempts_per_month = total_attempts / month_count if month_count > 0 else 0

                monkey_stats.append({
                    'days': day_count,
                    'months': month_count,
                    'attempts_per_day': attempts_per_day,
                    'attempts_per_month': attempts_per_month
                })

            if not monkey_stats:
                continue

            # Extract per-stat lists
            days_used = [x['days'] for x in monkey_stats]
            months_used = [x['months'] for x in monkey_stats]
            attempts_per_day = [x['attempts_per_day'] for x in monkey_stats]
            attempts_per_month = [x['attempts_per_month'] for x in monkey_stats]

            summary_records.append({
                'Species': species.capitalize(),
                'N Monkeys': len(monkey_stats),
                'Mean Days Used': round(pd.Series(days_used).mean(), 2),
                'Min Days Used': min(days_used),
                'Max Days Used': max(days_used),
                'Mean Months Used': round(pd.Series(months_used).mean(), 2),
                'Min Months Used': min(months_used),
                'Max Months Used': max(months_used),
                'Mean Attempts per Day': round(pd.Series(attempts_per_day).mean(), 2),
                'Min Attempts per Day': round(min(attempts_per_day), 2),
                'Max Attempts per Day': round(max(attempts_per_day), 2),
                'Mean Attempts per Month': round(pd.Series(attempts_per_month).mean(), 2),
                'Min Attempts per Month': round(min(attempts_per_month), 2),
                'Max Attempts per Month': round(max(attempts_per_month), 2)
            })

        # Save to CSV with semicolon for Excel friendliness
        df = pd.DataFrame(summary_records)
        filename = f"malt_usage_summary_{label}.csv"
        df.to_csv(os.path.join('monkey_reports_csv', filename), index=False, sep=';')
        print(f"\nMALT usage summary ({label}):")
        print(df.to_string(index=False))
        print(f"\nSaved MALT usage summary ({label}) to monkey_reports_csv/{filename}\n")

if data:
    generate_malt_usage_by_period(data)
