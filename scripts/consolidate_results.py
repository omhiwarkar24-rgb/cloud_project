import os
import pandas as pd
import glob
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FINAL_DIR = os.path.join(RESULTS_DIR, "final_summary")
PLOTS_DIR = os.path.join(FINAL_DIR, "plots")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_markdown_table(df, title):
    md = f"### {title}\n\n"
    md += "|" + "|".join(df.columns) + "|\n"
    md += "|" + "|".join(["---"] * len(df.columns)) + "|\n"
    for _, row in df.iterrows():
        md += "|" + "|".join([str(x) for x in row.values]) + "|\n"
    return md + "\n"

def consolidate():
    md_content = "# Final Consolidated Results\n\n"
    
    # 1. Security Models
    sec_csv = os.path.join(RESULTS_DIR, "security", "model_comparison.csv")
    if os.path.exists(sec_csv):
        df_sec = pd.read_csv(sec_csv)
        df_sec.to_csv(os.path.join(FINAL_DIR, "security_comparison.csv"), index=False)
        md_content += generate_markdown_table(df_sec, "Security Model Comparison (Binary & Multiclass)")

    # 2. Maintenance Models
    maint_csv = os.path.join(RESULTS_DIR, "maintenance", "model_comparison.csv")
    if os.path.exists(maint_csv):
        df_maint = pd.read_csv(maint_csv)
        df_maint.to_csv(os.path.join(FINAL_DIR, "maintenance_comparison.csv"), index=False)
        md_content += generate_markdown_table(df_maint, "Predictive Maintenance Model Comparison")

    # 3. Adaptive Learning
    adapt_csv = os.path.join(RESULTS_DIR, "adaptive", "adaptive_retraining_comparison.csv")
    if os.path.exists(adapt_csv):
        df_adapt = pd.read_csv(adapt_csv)
        df_adapt.to_csv(os.path.join(FINAL_DIR, "adaptive_learning_metrics.csv"), index=False)
        md_content += generate_markdown_table(df_adapt, "Adaptive Learning Adaptation Metrics")
        
    with open(os.path.join(FINAL_DIR, "summary_tables.md"), "w") as f:
        f.write(md_content)
        
    # Gather all plots
    print("Gathering plots...")
    plot_sources = [
        os.path.join(RESULTS_DIR, "security", "plots", "*.png"),
        os.path.join(RESULTS_DIR, "maintenance", "plots", "*.png"),
        os.path.join(RESULTS_DIR, "adaptive", "plots", "*.png")
    ]
    for pattern in plot_sources:
        for filepath in glob.glob(pattern):
            shutil.copy(filepath, os.path.join(PLOTS_DIR, os.path.basename(filepath)))
            
    print("Consolidation complete.")

if __name__ == "__main__":
    consolidate()
