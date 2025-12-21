import subprocess
from pathlib import Path
import sys

# Always get the correct path independently of where the folder is
SCRIPT_PATH = Path(__file__).parent / "main_simulation.py"


processes = []

# start 10 runs in parallel
for i in range(10):
    print(f"Starting run {i+1}/10")
    p = subprocess.Popen([sys.executable, SCRIPT_PATH])
    processes.append(p)

# (optional) wait for all to finish
for i, p in enumerate(processes):
    ret = p.wait()
    print(f"Run {i+1} finished with return code {ret}")
