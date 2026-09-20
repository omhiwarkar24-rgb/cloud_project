import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

notebooks = [
    "01_data_loading.ipynb",
    "02_security_eda.ipynb",
    "03_maintenance_eda.ipynb",
    "04_security_preprocessing.ipynb",
    "05_security_baseline_models.ipynb",
    "06_security_prediction_test.ipynb",
    "07_maintenance_preprocessing.ipynb",
    "08_maintenance_models.ipynb",
    "09_federated_learning.ipynb",
    "10_adaptive_learning.ipynb",
    "11_risk_engine_test.ipynb",
    "12_prevention_test.ipynb"
]

def run_notebooks():
    print(f"Starting execution of {len(notebooks)} notebooks...")
    for nb in notebooks:
        nb_path = os.path.join(NOTEBOOKS_DIR, nb)
        print(f"\n--- Running {nb} ---")
        try:
            subprocess.run([
                "python", "-m", "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute", 
                "--inplace", 
                nb_path
            ], check=True)
            print(f"Successfully ran {nb}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {nb}: {e}")
            break

if __name__ == "__main__":
    run_notebooks()
