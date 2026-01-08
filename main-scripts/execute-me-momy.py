import os
import subprocess

def main():
    script_dir = "/Users/similovesyou/Desktop/qts/simian-behavior/scripts"

    script_order = [
        "pickle_rick.py",
        "restrain_me.py",
        "social_status_bb.py",
        "(h)el(l)o.py.py"
    ]

    for script in script_order:
        script_path = os.path.join(script_dir, script)

        if os.path.isfile(script_path):
            try:
                print(f"Executing: {script_path}")
                result = subprocess.run(["python3", script_path], check=True)
                print(f"Finished: {script_path} with return code {result.returncode}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error while executing {script_path}: {e}\n")
                break
        else:
            print(f"Script not found: {script_path}\n")

if __name__ == "__main__":
    main()
