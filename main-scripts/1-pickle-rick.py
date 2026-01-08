import os
import pickle

import numpy as np
import pandas as pd
from functions import (
    descriptives,
)  # functions.py stores my functions - use name of function 2 call it ^^

# CLEAN THIS GODDAMN SCRIPT

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        print(f"UTF-8 failed for {path}, trying Latin-1...")
        return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"

data = {}  # dictionary to store data for each monkey

# load filter & pre-process data
for species in os.listdir(directory):
    species_path = os.path.join(
        directory, species
    )  # iterate through each species folder

    if os.path.isdir(species_path):
        data[species] = {}  # initialize dictionary

        for name in os.listdir(species_path):
            name_path = os.path.join(species_path, name)

            if os.path.isdir(name_path):
                data[species][name] = {}

                step_file = None
                attempt_file = None

                for file_name in os.listdir(name_path):
                    file_name_lower = file_name.lower()
                    if "step" in file_name_lower and file_name_lower.endswith(".csv"):
                        step_file = os.path.join(name_path, file_name)
                    elif (
                        "id" in file_name_lower
                        and file_name_lower.endswith(".csv")
                        and "step" not in file_name_lower
                    ):
                        attempt_file = os.path.join(name_path, file_name)

                if step_file and attempt_file:
                    try:
                        steps = safe_read_csv(step_file)
                        attempts = safe_read_csv(attempt_file)
                        if "task" in attempts.columns and "level" in attempts.columns:
                            task_filter = attempts[
                                attempts["task"].isin([97, 98])
                            ]  # filter tasks 97 and 98
                            palier_filter = task_filter[
                                task_filter["level"] > 25
                            ]  # filter palier > 25 (exclude learning)

                            if not palier_filter.empty:
                                session_counts = palier_filter[
                                    "session"
                                ].value_counts()  # session counts
                                # valid_sessions = session_counts[session_counts >= 10].index # filter sessions with less than 10 attempts

                                # exclude animals with less than 10 sessions (basically iron)
                                if (
                                    len(session_counts) < 10
                                ):  # less than 10 unique sessions; sub with valid_sessions if needed
                                    print(
                                        f"skipping '{species}' '{name}' - less than 10 unique sessions"
                                    )
                                    del data[species][name]
                                    continue

                                # valid_attempts = palier_filter[palier_filter['session'].isin(valid_sessions)]['id'].unique()  # exclusion of sessions with less than 10 attempts (see valid_sessions)
                                valid_attempts = palier_filter[
                                    "id"
                                ].unique()  # iterate filtered attempts ('id' column)

                                # NEW: exclude individuals with less than 1000 attempts
                                if len(valid_attempts) < 1000:
                                    print(
                                        f"skipping '{species}' '{name}' - less than 1000 attempts ({len(valid_attempts)} attempts)"
                                    )
                                    del data[species][name]
                                    continue

                                step_filter = steps[
                                    steps["attempt"].isin(valid_attempts)
                                ]  # filter step data (based on matching 'attempt')

                                if not step_filter.empty:
                                    # store data in the dict
                                    data[species][name]["steps"] = step_filter
                                    data[species][name]["attempts"] = palier_filter[
                                        palier_filter["id"].isin(valid_attempts)
                                    ]
                                    # print(f"filtered and stored data for '{species}' '{name}'")

                                else:
                                    print(
                                        f"skipping '{species}' '{name}' - no valid step data after filtering"
                                    )
                                    del data[species][
                                        name
                                    ]  # remove monkey from data dictionary

                            else:
                                print(
                                    f"skipping '{species}' '{name}' - no data for tasks 97 or 98 and level > 25"
                                )
                                del data[species][
                                    name
                                ]  # remove monkey from data dictionary

                        else:
                            print(
                                f"skipping '{species}' '{name}' - missing 'task' or 'level' columns"
                            )
                            del data[species][name]

                    except Exception as e:
                        print(
                            f"error reading files for '{species}' '{name}'. Error: {e}"
                        )
                        del data[species][name]

                else:
                    print(f"missing CSV files for '{species}' '{name}'")
                    del data[species][name]
# data[species][monkey_name]['steps'] or data[species][monkey_name]['attempts']

# note-2-simi:
# 'task' filter to [97, 98]
# 'level' filter to > 25
# include steps post-filter
# exclude monkeys with less than 10 sessions
# exclude monkeys with less than 1000 attempts

# calculating attempt duration (instant_end - instant_begin)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")
        if attempts is not None:
            attempts["attempt_duration"] = (
                attempts["instant_end"] - attempts["instant_begin"]
            )  # appended in attempts
# print('calculated attempt durations')

# calculating step duration (instant_end - instant_begin)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        steps = monkey_data.get("steps")
        if steps is not None:
            steps["step_duration"] = (
                steps["instant_end"] - steps["instant_begin"]
            )  # appended in steps
# print('calculated step durations')

# note-2-simi:
# appended attempt durations 2 attempts
# appended step durations 2 steps

# sanity checks confirm:
# instant_begin of attempts matches (with some exceptions, likely due to precision-based mismatches) instant_begin of start_step (steps)
# e.g., step instant_begin: 1516021840268, attempt instant_begin: 1516021840267 (difference = 1 ms; possibly a delay until square appears)
# in prior analyses, i linked steps & attempts by instant_begin timestamp, which likely caused data loss
# at present, steps and attempts were linked by 'id' (attempts) and 'attempt' (steps)
# instant_end of 'flash' = instant_begin of 'answer'
# instant_end of 'start_step' != instant_begin of 'flash'
# the residual represents variable_delay
# range: 2 - 4 seconds ^^
# all sorted attempts have a match in steps

for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        if "steps" in monkey_data:
            steps = monkey_data["steps"]
            attempts = monkey_data["attempts"]
            attempt_count_s = steps[
                "attempt"
            ].nunique()  # number of attempts in step data
            attempt_count_a = attempts.shape[
                0
            ]  # number of attempts in attempt data (using .shape[0] instead of nunique() to not overlook duplicates)

            step_count = steps[
                "attempt"
            ].value_counts()  # extracting step counts for attempts

            one = sum(count == 1 for count in step_count.values)
            two = sum(count == 2 for count in step_count.values)
            three = sum(count == 3 for count in step_count.values)
            four = sum(count == 4 for count in step_count.values)

            # print(f"{name}")
            # print(f"attempt count with 1 step: {one}")
            # print(f"attempt count with 2 steps: {two}") # olli had 10 attempts with 2 steps
            # print(f"attempt count with 3 steps: {three}")
            # print(f"attempt count with 4 steps: {four}") # olli had 260 attempts with 4 steps
            # print()

            # if one + two + three + four != attempt_count_a:
            #    print("oh no!")  # wallace & nema

            # excluding duplicates (i.e., attempts with > 4 steps)
            fuckit = step_count[step_count > 4].index.tolist()
            if fuckit:
                # print("deleting attempts with more than 4 steps")
                attempts = attempts[~attempts["id"].isin(fuckit)]
                steps = steps[~steps["attempt"].isin(fuckit)]

            # (only) olli has attempts with old step encoding (i.e., w start_delay) - removing for cohesiveness
            for odd_step_count in [2, 4]:
                start_delay = step_count[step_count == odd_step_count].index.tolist()
                for attempt_id in start_delay:
                    i = steps[steps["attempt"] == attempt_id]
                    if "start_delay" in i["name"].values:
                        steps = steps[
                            ~(
                                (steps["attempt"] == attempt_id)
                                & (steps["name"] == "start_delay")
                            )
                        ]

            # update counts after deletion
            attempt_count_s = steps["attempt"].nunique()
            attempt_count_a = attempts.shape[0]

            step_count = steps[
                "attempt"
            ].value_counts()  # extracting step counts for attempts

            # sanity check 2 c if # of attempts is the same in steps & attempts
            if attempt_count_s != attempt_count_a:
                print("oh no! attempt count in steps & attempts does not match (yeesh)")

# note 2 simi:
# exclusion of attempts with more than 4 steps
# sanity check at the end checks out ^^

# attempt duration compiled from step data (i.e., attempt duration excl. black screen & reward; task duration)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")
        steps = monkey_data.get("steps")

        if attempts is not None and steps is not None:
            attempts["task_duration"] = float("nan")

            for attempt_id in attempts["id"]:
                attempt_steps = steps[steps["attempt"] == attempt_id]

                # extract 'start' and 'answer' steps
                start = attempt_steps[attempt_steps["name"] == "start_step"]
                answer = attempt_steps[attempt_steps["name"] == "answer"]

                if not start.empty:
                    start_onset = start["instant_begin"].values[0]
                    start_offset = start["instant_end"].values[0]

                    if not answer.empty:
                        answer_offset = answer["instant_end"].values[0]
                        task_duration = answer_offset - start_offset
                    else:
                        task_duration = (
                            start_offset - start_onset
                        )  # relevant for premature attempts

                    attempts.loc[attempts["id"] == attempt_id, "task_duration"] = (
                        task_duration
                    )
            data[species][name]["attempts"] = attempts

# note-2-simi:
# i've noticed an abnormality in task_durations - a handful of premature task_durations are longer than omissions (not possible)
# revisit after %-tile exclusion

# decided to not implement attempt duration restriction (1%-tile) -- ig it makes sense if we're only interested in reaction times (and super short rts will be excluded later anyways)
""" 
# store attempt durations (4 descriptive processing)
attempt_durations = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            attempt_durations.extend(attempts['attempt_duration'].tolist())

attempt_durations = np.array(attempt_durations)
# descriptives(attempt_durations, "Attempt Duration Descriptives")

# removing attempts where (attempt) duration falls within 1% on each tail 
q1 = np.percentile(attempt_durations, 1) 
print(f"q1: {q1}")

q99 = np.percentile(attempt_durations, 99) 
print(f"q99: {q99}")

for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        steps = monkey_data.get('steps')

        if attempts is not None and steps is not None:
            attempt_filter = attempts[(attempts['attempt_duration'] >= q1) & (attempts['attempt_duration'] <= q99)]
            step_filter = steps[steps['attempt'].isin(attempt_filter['id'])]

            data[species][name]['attempts'] = attempt_filter
            data[species][name]['steps'] = step_filter

        else:
            print(f"no attempt and/or step data found for '{species}' '{name}'")

# collect attempt_durations *after* 1%-tile removal & print descriptives 
attempt_durations = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            attempt_durations.extend(attempts['attempt_duration'].tolist())

attempt_durations = np.array(attempt_durations)
# descriptives(attempt_durations, "Attempt Duration Descriptives After 1%-tile Removal") # descriptive function
# justification? device malfunction, double attempts, etc.
"""

# calculating session duration (instant_end.max - instant_begin.min; see session.py for session info)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")
        if attempts is not None:
            session_durations = attempts.groupby("session", group_keys=False).apply(
                lambda x: x["instant_end"].max() - x["instant_begin"].min()
            )
            attempts["session_duration"] = attempts["session"].map(session_durations)

# appending reaction times (= duration of question_step) to attempts df
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")
        steps = monkey_data.get("steps")

        if attempts is not None and steps is not None:
            answer_steps = steps[steps["name"] == "answer"]  # corresponds to RTs
            link = dict(
                zip(answer_steps["attempt"], answer_steps["step_duration"])
            )  # mapping dictionary
            attempts.loc[:, "reaction_time"] = attempts["id"].map(
                link
            )  # append RTs (which correspond to answer steps)
            data[species][name]["attempts"] = attempts.copy()

        else:
            print(f"No attempts and/or steps data found for '{species}' '{name}'")

# calculating variable_delay & appending it to attempts df ~.
for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        if "steps" in monkey_data and "attempts" in monkey_data:
            steps = monkey_data["steps"]
            attempts = monkey_data["attempts"]
            steps_grouped = steps.groupby("attempt")

            for attempt, group in steps_grouped:
                flash = group[group["name"] == "flash"]  # filter for flash steps
                start_step = group[
                    group["name"] == "start_step"
                ]  # filter for start steps

                if not flash.empty and not start_step.empty:
                    flash_onset = flash["instant_begin"].values[0]
                    start_offset = start_step["instant_end"].values[0]
                    flash_start_difference = flash_onset - start_offset
                    attempts.loc[attempts["id"] == attempt, "variable_delay"] = (
                        flash_start_difference
                    )

                else:
                    attempts.loc[attempts["id"] == attempt, "variable_delay"] = (
                        np.nan
                    )  # relevant for premature attempts
            data[species][name]["attempts"] = attempts.copy()

# note 2 simi:
# session_duration appended to attempts df
# reaction_time corresponds to RTs for successful, erroneous, and omitted responses
# calculating the reaction_time for premature responses necessitates an additional calculation
# update: calculating the reaction_time for preamtures is not possible (previous calculation were not correct)
# attempt_duration - (start_)step_duration != premature RT
# the residual is confounded by (1) variable delay, (2) fixed delay, and (3) reward period
# calculation can be estimated but lack of precision pushed me to neglect it -- so BYE!
# see script attempts.py
# i extracted min/max/mean response ('answer' step) durations
# some times were as low as 1 ms
# decided to exclude 1%-tile (left)
# max was ~ 5000 ms, which corresponds to omissions (no problem)


# rt quartile restriction
"""
# split rts by response type 
rts_by_result = {
    'success': [],
    'error': [],
    'stepomission': []
}

# collect rts for each result type
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            for result in rts_by_result.keys():
                result_filtered_attempts = attempts[attempts['result'] == result]
                rts_by_result[result].extend(result_filtered_attempts['reaction_time'].dropna().tolist())  # drop NaNs

# percentiles per result type 
percentiles = {}
for result, reaction_times in rts_by_result.items():
    reaction_times = np.array(reaction_times)
    descriptives(reaction_times, f"{result.capitalize()} Reaction Time Descriptives")  # Generate descriptives
    q1 = np.percentile(reaction_times, 1)
    q99 = np.percentile(reaction_times, 99)
    percentiles[result] = (q1, q99)

# apply percentile filtering
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        steps = monkey_data.get('steps')

        if attempts is not None and steps is not None:
            filtered_attempts = pd.DataFrame()  
            for result, (q1, q99) in percentiles.items():
                result_filtered_attempts = attempts[attempts['result'] == result]
                attempt_filter = result_filtered_attempts[
                    (result_filtered_attempts['reaction_time'] >= q1) &
                    (result_filtered_attempts['reaction_time'] <= q99) |
                    (result_filtered_attempts['reaction_time'].isna())
                ]
                filtered_attempts = pd.concat([filtered_attempts, attempt_filter])

            # filter steps based on the filtered attempts
            step_filter = steps[steps['attempt'].isin(filtered_attempts['id'])]

            data[species][name]['attempts'] = filtered_attempts
            data[species][name]['steps'] = step_filter

# re-compile reaction times after filtering for each result type and print descriptives
for result in rts_by_result.keys():
    rts_by_result[result] = []
    for species, monkeys in data.items():
        for name, monkey_data in monkeys.items():
            attempts = monkey_data.get('attempts')
            if attempts is not None:
                result_filtered_attempts = attempts[attempts['result'] == result]
                rts_by_result[result].extend(result_filtered_attempts['reaction_time'].dropna().tolist())

    reaction_times = np.array(rts_by_result[result])
    descriptives(reaction_times, f"{result.capitalize()} Reaction Time Descriptives After 1%-tile Removal")
"""

# classify rts for successes and errors as premature if they fall under 100 ms (or other threshold)
premature_threshold = 100  # ms (or other threshold)

# iterate and reclassify premature attempts
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")

        if attempts is not None:
            # identify attempts where reaction time is shorter than the premature threshold
            premature_attempts = attempts[
                attempts["reaction_time"] < premature_threshold
            ]

            # reclassify the 'result' of these attempts as 'premature'
            if not premature_attempts.empty:
                attempts.loc[premature_attempts.index, "result"] = "premature"
                print(
                    f"reclassified {len(premature_attempts)} attempts as 'premature' for '{species}' '{name}'"
                )

            data[species][name]["attempts"] = attempts
        else:
            print(f"No attempt data found for '{species}' '{name}'")
# proportion of premature attempts should have increased (since whatever was classified as success and error were re-classified as premature due to < 150 ms duration)

# flash quartile restrictions
"""
# extracting flash_durations
flash_durations = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        steps = monkey_data.get('steps')
        if steps is not None:
            flash = steps[steps['name'] == 'flash']
            flash_durations.extend(flash['step_duration'].tolist())

flash_durations = np.array(flash_durations)
# descriptives(flash_durations, "Flash Duration Descriptives")

q1 = np.percentile(flash_durations, 1) # visual inspection of q1 deems unnecessary to exclude 
print(f"q1: {q1}")

q99 = np.percentile(flash_durations, 99) # some flashes are really long -- decided to exlude 1%-tile (right) 
print(f"q99: {q99}")

# removing 
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        steps = monkey_data.get('steps')
        attempts = monkey_data.get('attempts')

        if steps is not None:
            flash = steps[steps['name'] == 'flash']
            step_filter = flash[flash['step_duration'] <= q99]  # filter applied

            id = step_filter['attempt'].tolist()  # find match in attempts based on step_filter 
            attempt_filter = attempts[~attempts['id'].isin(id)]

            data[species][name]['steps'] = step_filter
            data[species][name]['attempts'] = attempt_filter
        else:
            print(f"no step data found for '{species}' '{name}'")

flash_durations = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        steps = monkey_data.get('steps')
        if steps is not None:
            flash = steps[steps['name'] == 'flash']
            flash_durations.extend(flash['step_duration'].tolist())

flash_durations = np.array(flash_durations)
# descriptives(flash_durations, "Flash Duration Descriptives After 1%-tile Removal")
"""

# for species, monkeys in data.items():
#     for monkey_name, details in monkeys.items():
#         if 'steps' in details and 'attempts' in details:
#             steps_df = details['steps']
#             attempts_df = details['attempts']
#             print(f"Monkey '{species}' '{monkey_name}' has both steps and attempts data.")
#             print(f"Number of columns in 'steps' dataframe: {len(steps_df.columns)}")
#             print(f"Number of columns in 'attempts' dataframe: {len(attempts_df.columns)}")
#         else:
#             print(f"Monkey '{species}' '{monkey_name}' is missing steps or attempts data.")


# save the cleaned dataframe (only note Olli's bs coding that will later have to be accounted for)
pickle_rick = os.path.join(directory, "data.pickle")

with open(pickle_rick, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"data saved to {pickle_rick}")


import pandas as pd

# Combine all attempts into one DataFrame
all_attempts = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data['attempts'].copy()
        attempts['species'] = species
        attempts['monkey'] = name
        all_attempts.append(attempts)

df = pd.concat(all_attempts)

# Convert timestamps to datetime (adjust column if needed)
df['date'] = pd.to_datetime(df['instant_begin'], unit='ms')

# --- Calculate Metrics ---
metrics = df.groupby(['species', 'monkey']).agg(
    # Session stats
    mean_attempts_per_session=('session', lambda x: x.value_counts().mean()),
    min_attempts_per_session=('session', lambda x: x.value_counts().min()),
    max_attempts_per_session=('session', lambda x: x.value_counts().max()),
    
    # Weekly stats (assuming 7 sessions/week)
    mean_attempts_per_week=('session', lambda x: x.value_counts().mean() * 7),
    min_attempts_per_week=('session', lambda x: x.value_counts().min() * 7),
    max_attempts_per_week=('session', lambda x: x.value_counts().max() * 7),
    
    # Monthly stats (4 weeks/month)
    mean_attempts_per_month=('session', lambda x: x.value_counts().mean() * 7 * 4),
    
    # Date range
    first_play_date=('date', 'min'),
    last_play_date=('date', 'max'),
    days_active=('date', lambda x: (x.max() - x.min()).days),
).reset_index()

# Round numeric values
metrics = metrics.round(2)

# --- Save/Display Table ---
print(metrics)
metrics.to_csv('monkey_engagement_metrics.csv', index=False)


import pandas as pd
import os

# --- Data Preparation ---
# Combine all attempts into one DataFrame
all_attempts = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data['attempts'].copy()
        attempts['species'] = species
        attempts['monkey'] = name
        all_attempts.append(attempts)

df = pd.concat(all_attempts)

# Convert timestamps to datetime
df['date'] = pd.to_datetime(df['instant_begin'], unit='ms')
df['week'] = df['date'].dt.isocalendar().year.astype(str) + '-W' + df['date'].dt.isocalendar().week.astype(str)
df['month'] = df['date'].dt.to_period('M').astype(str)

# --- Session-level metrics ---
session_stats = df.groupby(['species', 'monkey', 'session']).agg(
    attempts=('result', 'count'),
    successes=('result', lambda x: (x == 'success').sum())
).reset_index()

session_summary = session_stats.groupby(['species', 'monkey']).agg(
    mean_attempts_session=('attempts', 'mean'),
    min_attempts_session=('attempts', 'min'),
    max_attempts_session=('attempts', 'max'),
    mean_success_session=('successes', 'mean'),
    min_success_session=('successes', 'min'),
    max_success_session=('successes', 'max')
)

# --- Weekly-level metrics ---
weekly = df.groupby(['species', 'monkey', 'week']).agg(
    attempts=('result', 'count'),
    successes=('result', lambda x: (x == 'success').sum())
).reset_index()

weekly_summary = weekly.groupby(['species', 'monkey']).agg(
    mean_attempts_week=('attempts', 'mean'),
    min_attempts_week=('attempts', 'min'),
    max_attempts_week=('attempts', 'max'),
    mean_success_week=('successes', 'mean'),
    min_success_week=('successes', 'min'),
    max_success_week=('successes', 'max')
)

# --- Monthly-level metrics ---
monthly = df.groupby(['species', 'monkey', 'month']).agg(
    attempts=('result', 'count'),
    successes=('result', lambda x: (x == 'success').sum())
).reset_index()

monthly_summary = monthly.groupby(['species', 'monkey']).agg(
    mean_attempts_month=('attempts', 'mean'),
    min_attempts_month=('attempts', 'min'),
    max_attempts_month=('attempts', 'max'),
    mean_success_month=('successes', 'mean'),
    min_success_month=('successes', 'min'),
    max_success_month=('successes', 'max')
)

# --- Days active ---
active_days = df.groupby(['species', 'monkey']).agg(
    first_date=('date', 'min'),
    last_date=('date', 'max')
)
active_days['days_active'] = (active_days['last_date'] - active_days['first_date']).dt.days

# --- Merge all per-monkey stats ---
monkey_stats = session_summary \
    .join(weekly_summary) \
    .join(monthly_summary) \
    .join(active_days[['days_active']]) \
    .reset_index()

# --- Group-Level Summary ---
group_metrics = monkey_stats.groupby('species').agg(
    n_monkeys=('monkey', 'nunique'),

    # Session
    mean_attempts_session=('mean_attempts_session', 'mean'),
    min_attempts_session=('min_attempts_session', 'min'),
    max_attempts_session=('max_attempts_session', 'max'),
    mean_success_session=('mean_success_session', 'mean'),
    min_success_session=('min_success_session', 'min'),
    max_success_session=('max_success_session', 'max'),

    # Weekly
    mean_attempts_week=('mean_attempts_week', 'mean'),
    min_attempts_week=('min_attempts_week', 'min'),
    max_attempts_week=('max_attempts_week', 'max'),
    mean_success_week=('mean_success_week', 'mean'),
    min_success_week=('min_success_week', 'min'),
    max_success_week=('max_success_week', 'max'),

    # Monthly
    mean_attempts_month=('mean_attempts_month', 'mean'),
    min_attempts_month=('min_attempts_month', 'min'),
    max_attempts_month=('max_attempts_month', 'max'),
    mean_success_month=('mean_success_month', 'mean'),
    min_success_month=('min_success_month', 'min'),
    max_success_month=('max_success_month', 'max'),

    # Activity
    median_days_active=('days_active', 'median')
).round(2)

# --- Save to Downloads ---
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'monkey_play_metrics_counts.csv')
group_metrics.to_csv(downloads_path)

print(f"Results saved to: {downloads_path}")

"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Extract task duration data for tasks 88-98 with timestamps
task_time_data = []
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            # Filter for tasks 88-98 and convert timestamps
            task_filter = attempts[attempts['task'].between(88, 98)].copy()
            if not task_filter.empty:
                task_filter['datetime'] = pd.to_datetime(task_filter['instant_begin'], unit='ms')
                for task_num, group in task_filter.groupby('task'):
                    if len(group) > 1:  # Need at least 2 attempts to calculate time span
                        time_span = (group['datetime'].max() - group['datetime'].min()).total_seconds() / 86400  # Convert to days
                        task_time_data.append({
                            'species': species,
                            'monkey': name,
                            'task': task_num,
                            'time_span_days': time_span,
                            'n_attempts': len(group)
                        })

# Create DataFrame
task_time_df = pd.DataFrame(task_time_data)

# Calculate statistics by species and task
time_stats = task_time_df.groupby(['species', 'task']).agg({
    'time_span_days': ['mean', 'min', 'max', 'median', 'std'],
    'n_attempts': 'sum',
    'monkey': 'nunique'  # Count of unique monkeys per task
}).reset_index()

# Flatten multi-index columns
time_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in time_stats.columns.values]

# Rename columns for clarity
time_stats = time_stats.rename(columns={
    'monkey_nunique': 'n_monkeys',
    'n_attempts_sum': 'total_attempts'
})

# Pivot the table for better readability
pivot_time_stats = time_stats.pivot_table(
    index='task',
    columns='species',
    values=['time_span_days_mean', 'time_span_days_min', 'time_span_days_max', 
            'time_span_days_median', 'n_monkeys', 'total_attempts'],
    aggfunc='first'
)

# Flatten multi-index columns
pivot_time_stats.columns = ['_'.join(col).strip() for col in pivot_time_stats.columns.values]
pivot_time_stats.reset_index(inplace=True)

# Reorder columns for better presentation
column_order = ['task'] + sorted([col for col in pivot_time_stats.columns if col != 'task'])
pivot_time_stats = pivot_time_stats[column_order]

# Save to CSV
output_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'task_time_spans_88-98.csv')
pivot_time_stats.to_csv(output_path, index=False)

print(f"Task time span statistics saved to: {output_path}")
print("\nMean Time Spent on Tasks 88-98 (days):")
print(pivot_time_stats.round(2).to_string(index=False))"""