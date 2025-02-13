import subprocess

# List of scripts to execute sequentially
scripts = [
    "get-team-stats-simplified.py",
    "combine-individual-csvs.py",
    "current-best-model.py"
]

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)  # Runs each script and checks for errors
        print(f"Finished {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break  # Stops execution if any script fails

print("All scripts executed successfully.")
