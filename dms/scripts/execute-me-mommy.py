import subprocess
import os

scripts_dir = '/Users/similovesyou/Desktop/qts/simian-behavior/dms/scripts'

scripts_2_run = [
    'dmt_rick.py',
    'feed_me_dmt.py',
    'phenotypes.py',
    'hierarchy.py'
]

for script in scripts_2_run:
    script_path = os.path.join(scripts_dir, script)
    
    if os.path.exists(script_path):
        print(f"Running {script}...")
        
        try:
            result = subprocess.run(['python3', script_path], check=True, capture_output=True, text=True)
            print(f"Output of {script}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}:\n{e.stderr}")
            break
    else:
        print(f"Script {script} not found. Skipping...")
