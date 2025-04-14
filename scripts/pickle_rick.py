import os
import pandas as pd 
import pickle 
import numpy as np
from functions import descriptives  # functions.py stores my functions - use name of function 2 call it ^^ 

directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'

data = {} # dictionary to store data for each monkey 

# load filter & pre-process data
for species in os.listdir(directory): 
    species_path = os.path.join(directory, species) # iterate through each species folder
    
    if os.path.isdir(species_path):
        data[species] = {} # initialize dictionary
        
        for name in os.listdir(species_path):
            name_path = os.path.join(species_path, name)
            
            if os.path.isdir(name_path):
                data[species][name] = {}
                
                step_file = None 
                attempt_file = None 
                
                for file_name in os.listdir(name_path):
                    if "step" in file_name:
                        step_file = os.path.join(name_path, file_name)
                    elif "ID" in file_name:
                        attempt_file = os.path.join(name_path, file_name)
                
                if step_file and attempt_file:
                    try: 
                        steps = pd.read_csv(step_file)
                        attempts = pd.read_csv(attempt_file)
   
                        if 'task' in attempts.columns and 'level' in attempts.columns:
                            task_filter = attempts[attempts['task'].isin([97, 98])] # filter tasks 97 and 98
                            palier_filter = task_filter[task_filter['level'] > 25] # filter palier > 25 (exclude learning)
        
                            if not palier_filter.empty:
                               
                                session_counts = palier_filter['session'].value_counts() # session counts  
                                # valid_sessions = session_counts[session_counts >= 10].index # filter sessions with less than 10 attempts 
                                
                                # exclude animals with less than 10 sessions (basically iron)
                                if len(session_counts) < 10:  # less than 10 unique sessions; sub with valid_sessions if needed
                                    print(f"skipping '{species}' '{name}' - less than 10 unique sessions")
                                    del data[species][name]
                                    continue
                                
                                # valid_attempts = palier_filter[palier_filter['session'].isin(valid_sessions)]['id'].unique()  # exclusion of sessions with less than 10 attempts (see valid_sessions)
                                valid_attempts = palier_filter['id'].unique() # iterate filtered attempts ('id' column) 
                                step_filter = steps[steps['attempt'].isin(valid_attempts)] # filter step data (based on matching 'attempt')
                                
                                if not step_filter.empty:
                                    # store data in the dict 
                                    data[species][name]['steps'] = step_filter
                                    data[species][name]['attempts'] = palier_filter[palier_filter['id'].isin(valid_attempts)]
                                    # print(f"filtered and stored data for '{species}' '{name}'")
  
                                else: 
                                    print(f"skipping '{species}' '{name}' - no valid step data after filtering")
                                    del data[species][name]  # remove monkey from data dictionary

                            else:
                                print(f"skipping '{species}' '{name}' - no data for tasks 97 or 98 and level > 25")
                                del data[species][name]  # remove monkey from data dictionary

                        else:
                            print(f"skipping '{species}' '{name}' - missing 'task' or 'level' columns")
                            del data[species][name]
                            
                    except Exception as e:
                        print(f"error reading files for '{species}' '{name}'. Error: {e}")
                        del data[species][name]
                        
                else:
                    print(f"missing CSV files for '{species}' '{name}'")                        
                    del data[species][name]
                    
# data[species][monkey_name]['steps'] or data[species][monkey_name]['attempts']

# note-2-simi: 
    # 'task' filter to [97, 98]
    # 'level' filter to > 25
    # (un)exclude sessions with < 10 attempts (re-implement if needed)
    # include steps post-filter 

# calculating attempt duration (instant_end - instant_begin)  
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            attempts['attempt_duration'] = attempts['instant_end'] - attempts['instant_begin'] # appended in attempts 
# print('calculated attempt durations')

# calculating step duration (instant_end - instant_begin)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        steps = monkey_data.get('steps')
        if steps is not None:
            steps['step_duration'] = steps['instant_end'] - steps['instant_begin'] # appended in steps
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
        if 'steps' in monkey_data:
            steps = monkey_data['steps']
            attempts = monkey_data['attempts']
            attempt_count_s = steps['attempt'].nunique() # number of attempts in step data 
            attempt_count_a = attempts.shape[0] # number of attempts in attempt data (using .shape[0] instead of nunique() to not overlook duplicates)

            step_count = steps['attempt'].value_counts() # extracting step counts for attempts 
           
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
                attempts = attempts[~attempts['id'].isin(fuckit)]
                steps = steps[~steps['attempt'].isin(fuckit)]
            
            # (only) olli has attempts with old step encoding (i.e., w start_delay) - removing for cohesiveness         
            for odd_step_count in [2, 4]:
                start_delay = step_count[step_count == odd_step_count].index.tolist()
                for attempt_id in start_delay:
                    i = steps[steps['attempt'] == attempt_id]
                    if 'start_delay' in i['name'].values:
                        steps = steps[~((steps['attempt'] == attempt_id) & (steps['name'] == 'start_delay'))]

            # update counts after deletion 
            attempt_count_s = steps['attempt'].nunique()
            attempt_count_a = attempts.shape[0]
            
            step_count = steps['attempt'].value_counts() # extracting step counts for attempts 

            # sanity check 2 c if # of attempts is the same in steps & attempts   
            if attempt_count_s != attempt_count_a:
               print("oh no! attempt count in steps & attempts does not match (yeesh)")

# note 2 simi:  
    # exclusion of attempts with more than 4 steps
    # sanity check at the end checks out ^^ 

# attempt duration compiled from step data (i.e., attempt duration excl. black screen & reward; task duration)
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        steps = monkey_data.get('steps')
        
        if attempts is not None and steps is not None:
            attempts['task_duration'] = float('nan')
            
            for attempt_id in attempts['id']:
                attempt_steps = steps[steps['attempt'] == attempt_id]
                
                # extract 'start' and 'answer' steps
                start = attempt_steps[attempt_steps['name'] == 'start_step']  
                answer = attempt_steps[attempt_steps['name'] == 'answer']  
                
                if not start.empty:
                    start_onset = start['instant_begin'].values[0]
                    start_offset = start['instant_end'].values[0]
                    
                    if not answer.empty:
                        answer_offset = answer['instant_end'].values[0]
                        task_duration = answer_offset - start_offset
                    else:
                        task_duration = start_offset - start_onset  # relevant for premature attempts
                    
                    attempts.loc[attempts['id'] == attempt_id, 'task_duration'] = task_duration
            data[species][name]['attempts'] = attempts

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
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            session_durations = attempts.groupby('session').apply(
                lambda x: x['instant_end'].max() - x['instant_begin'].min())
            attempts['session_duration']= attempts['session'].map(session_durations)

# appending reaction times (= duration of question_step) to attempts df 
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        steps = monkey_data.get('steps')
        
        if attempts is not None and steps is not None:
            answer_steps = steps[steps['name'] == 'answer'] # corresponds to RTs
            link = dict(zip(answer_steps['attempt'], answer_steps['step_duration'])) # mapping dictionary
            attempts.loc[:, 'reaction_time'] = attempts['id'].map(link) # append RTs (which correspond to answer steps)   
            data[species][name]['attempts'] = attempts.copy()
  
        else:
            print(f"No attempts and/or steps data found for '{species}' '{name}'")
            
# calculating variable_delay & appending it to attempts df ~.
for species, species_data in data.items():
    for name, monkey_data in species_data.items():
        if 'steps' in monkey_data and 'attempts' in monkey_data:
            steps = monkey_data['steps']
            attempts = monkey_data['attempts']
            steps_grouped = steps.groupby('attempt')
            
            for attempt, group in steps_grouped:
                flash = group[group['name'] == 'flash']  # filter for flash steps 
                start_step = group[group['name'] == 'start_step']  # filter for start steps
                
                if not flash.empty and not start_step.empty:
                    flash_onset = flash['instant_begin'].values[0]  
                    start_offset = start_step['instant_end'].values[0]  
                    flash_start_difference = flash_onset - start_offset   
                    attempts.loc[attempts['id'] == attempt, 'variable_delay'] = flash_start_difference
               
                else:
                    attempts.loc[attempts['id'] == attempt, 'variable_delay'] = np.nan  # relevant for premature attempts
            data[species][name]['attempts'] = attempts.copy()

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

# classify rts for successes and errors as premature if they fall under 150 ms (or other threshold)
premature_threshold = 50 # ms (or other threshold)

# iterate and reclassify premature attempts
for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        
        if attempts is not None:
            # identify attempts where reaction time is shorter than the premature threshold
            premature_attempts = attempts[attempts['reaction_time'] < premature_threshold]

            # reclassify the 'result' of these attempts as 'premature'
            if not premature_attempts.empty:
                attempts.loc[premature_attempts.index, 'result'] = 'premature'
                print(f"reclassified {len(premature_attempts)} attempts as 'premature' for '{species}' '{name}'")

            data[species][name]['attempts'] = attempts
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
pickle_rick = os.path.join(directory, 'data.pickle')

with open(pickle_rick, 'wb') as handle: 
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"data saved to {pickle_rick}")
     
