import os
import pandas as pd
import pickle

# Define directory and initialize data dictionary
directory = '/Users/similovesyou/Desktop/qts/simian-behavior/data/py'
data = {}

# Load, filter, and preprocess data
for species in os.listdir(directory):
    species_path = os.path.join(directory, species)

    if os.path.isdir(species_path):
        data[species] = {}

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
                            task_filter = attempts[attempts['task'].isin([100])]
                            # palier_filter = task_filter[task_filter['level'] > 25]

                            if not task_filter.empty:
                                session_counts = task_filter['session'].value_counts()

                                if len(session_counts) < 10:
                                    print(f"Skipping '{species}' '{name}' - less than 10 unique sessions") # aribitrary, re-consider
                                    del data[species][name]
                                    continue

                                valid_attempts = task_filter['id'].unique()
                                step_filter = steps[steps['attempt'].isin(valid_attempts)]

                                if not step_filter.empty:
                                    data[species][name]['steps'] = step_filter
                                    data[species][name]['attempts'] = task_filter[task_filter['id'].isin(valid_attempts)]
                                else:
                                    print(f"Skipping '{species}' '{name}' - no valid step data after filtering")
                                    del data[species][name]
                            else:
                                print(f"Skipping '{species}' '{name}' - no data for task id 100")
                                del data[species][name]
                        else:
                            print(f"Skipping '{species}' '{name}' - missing 'task' or 'level' columns")
                            del data[species][name]
                    except Exception as e:
                        print(f"Error reading files for '{species}' '{name}'. Error: {e}")
                        del data[species][name]
                else:
                    print(f"Missing CSV files for '{species}' '{name}'")
                    del data[species][name]
       
data[species]['olga']['attempts'] 

# note-2-simi: 
    # 'task' filter to [100] - DMS_T03
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

# calculating session duration (instant_end.max - instant_begin.min; see session.py for session info)
for species, monkeys in data.items(): 
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get('attempts')
        if attempts is not None:
            session_durations = attempts.groupby('session').apply(
                lambda x: x['instant_end'].max() - x['instant_begin'].min())
            attempts['session_duration']= attempts['session'].map(session_durations)

# note-2-simi: 
    # appended attempt durations 2 attempts
    # appended step durations 2 steps

# unique 'name' and 'result' in steps and attempts, resp
    # start_step, show and answer 
    # error, success, stepomission, premature

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

# reclassify reaction times under threshold as premature
premature_threshold = 100  # ms, or adjust as needed

for species, monkeys in data.items():
    for name, monkey_data in monkeys.items():
        attempts = monkey_data.get("attempts")

        if attempts is not None and 'reaction_time' in attempts.columns:
            premature_attempts = attempts[attempts["reaction_time"] < premature_threshold]

            if not premature_attempts.empty:
                attempts.loc[premature_attempts.index, "result"] = "premature"
                print(f"reclassified {len(premature_attempts)} attempts as 'premature' for '{species}' '{name}'")

            data[species][name]["attempts"] = attempts
        else:
            print(f"No attempt data or reaction times found for '{species}' '{name}'")

# save the cleaned dataframe (only note Olli's bs coding that will later have to be accounted for)
dmt_rick = os.path.join(directory, 'dms.pickle')

with open(dmt_rick, 'wb') as handle: 
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"data saved to {dmt_rick}")
     
