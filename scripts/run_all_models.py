import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

env = os.environ.copy()
env["PYTHONPATH"] = SRC_DIR

scripts = [
    os.path.join(SRC_DIR, "data_processing", "dataset_initializer.py"),
    os.path.join(SRC_DIR, "data_processing", "security_preprocessing.py"),
    os.path.join(SRC_DIR, "data_processing", "maintenance_preprocessing.py"),
    os.path.join(SRC_DIR, "security", "baseline_models.py"),
    os.path.join(SRC_DIR, "maintenance", "baseline_models.py"),
    os.path.join(SRC_DIR, "adaptive", "retrain.py")
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}:")
        print(result.stderr)
    else:
        print(result.stdout)
        print(f"Successfully ran {script}.")
