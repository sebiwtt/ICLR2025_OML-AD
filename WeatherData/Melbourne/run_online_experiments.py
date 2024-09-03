import os
import subprocess
try:
    subprocess.run(['python', 'run_experiments.py'], cwd="./OnlineLearning/", check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running scripts: {e}")


