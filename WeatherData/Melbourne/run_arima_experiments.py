import os
import subprocess
subdirs = ["./ARIMA/NoRetraining", "./ARIMA/ScheduledRetraining", "./ARIMA/DynamicRetraining"]

for subdir in subdirs:
    try:
        subprocess.run(['python', 'run_experiments.py'], cwd=subdir, check=True)
        print(f"{subdir} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing {subdir}: {e}")
        break  

