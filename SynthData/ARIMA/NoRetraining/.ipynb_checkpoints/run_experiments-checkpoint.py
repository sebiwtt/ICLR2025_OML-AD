import subprocess
scripts = ["time.py", "resources.py", "performance.py", "plots.py"]

for script in scripts:
    try:
        print(f"started with {script} execution.")
        subprocess.run(["python", script], check=True)
        print(f"{script} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing {script}: {e}")
        break  
