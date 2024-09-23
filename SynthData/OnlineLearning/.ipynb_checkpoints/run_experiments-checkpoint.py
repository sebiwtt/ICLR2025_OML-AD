import subprocess
scripts = ["time.py", "resources.py", "performance.py", "plots.py"]

for script in scripts:
    try:
        subprocess.run(["python", script], check=True)
        print(f"{script} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing {script}: {e}")
        break  
