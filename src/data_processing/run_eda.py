"""
run_eda.py - Clean execution of Part 2 EDA logic with saved PNG plots.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEC_DATA = os.path.join(BASE_DIR, "data", "security", "UNSW_NB15_training-set.csv")
MAINT_DATA = os.path.join(BASE_DIR, "data", "maintenance", "ai4i2020.csv")

SEC_SAVE = os.path.join(BASE_DIR, "results", "security", "eda")
MAINT_SAVE = os.path.join(BASE_DIR, "results", "maintenance", "eda")

os.makedirs(SEC_SAVE, exist_ok=True)
os.makedirs(MAINT_SAVE, exist_ok=True)


def run_security_eda():
    print("Executing UNSW-NB15 Security EDA...")
    df = pd.read_csv(SEC_DATA)

    # 1. Binary target
    fig, ax = plt.subplots(figsize=(7, 5))
    target_counts = df['label'].value_counts()
    percentages = df['label'].value_counts(normalize=True) * 100
    bars = ax.bar(['Normal (0)', 'Attack (1)'], target_counts, color=['#2ecc71', '#e74c3c'], width=0.5)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{height:,} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax.set_title("UNSW-NB15: Binary Target Distribution", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count")
    plt.savefig(os.path.join(SEC_SAVE, 'binary_target_distribution.png'), dpi=300)
    plt.close()

    # 2. Attack category
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_counts = df['attack_cat'].value_counts()
    cat_pcts = df['attack_cat'].value_counts(normalize=True) * 100
    palette = sns.color_palette("viridis", len(cat_counts))
    bars = ax.barh(cat_counts.index, cat_counts.values, color=palette)
    for bar, pct in zip(bars, cat_pcts):
        width = bar.get_width()
        ax.annotate(f'{width:,} ({pct:.1f}%)', xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
    ax.set_title("UNSW-NB15: Multi-Class Attack Category Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Count")
    ax.set_ylabel("Attack Category")
    ax.invert_yaxis()
    plt.savefig(os.path.join(SEC_SAVE, 'attack_category_distribution.png'), dpi=300)
    plt.close()

    # 3. Numeric feature distributions
    num_cols = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        data = np.log1p(df[col]) if df[col].min() >= 0 else df[col]
        sns.histplot(data, ax=axes[idx], kde=True, color='#3498db', bins=30)
        axes[idx].set_title(f"Distribution of {col} (log1p scale)", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(f"{col} (log1p)")
    plt.suptitle("UNSW-NB15: Key Numeric Feature Distributions", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(SEC_SAVE, 'numeric_feature_distributions.png'), dpi=300)
    plt.close()

    # 4. Correlation heatmap
    selected_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'smean', 'dmean', 'label']
    corr_matrix = df[selected_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("UNSW-NB15: Numeric Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(SEC_SAVE, 'numeric_correlation_heatmap.png'), dpi=300)
    plt.close()

    # 5. Categorical distributions
    cat_cols = ['proto', 'service', 'state']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(cat_cols):
        top_cats = df[col].value_counts().nlargest(6)
        sns.barplot(x=top_cats.values, y=top_cats.index, ax=axes[idx], hue=top_cats.index, palette="Blues_r", legend=False)
        axes[idx].set_title(f"Top Categories: {col}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Count")
    plt.suptitle("UNSW-NB15: Categorical Feature Distributions", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(SEC_SAVE, 'categorical_feature_distributions.png'), dpi=300)
    plt.close()

    print("Security EDA Complete. All plots saved to results/security/eda/")


def run_maintenance_eda():
    print("Executing AI4I 2020 Maintenance EDA...")
    df = pd.read_csv(MAINT_DATA)

    # 1. Machine failure distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['Machine failure'].value_counts()
    pcts = df['Machine failure'].value_counts(normalize=True) * 100
    bars = ax.bar(['No Failure (0)', 'Machine Failure (1)'], counts, color=['#2ecc71', '#e74c3c'], width=0.5)
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.annotate(f'{height:,} ({pct:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax.set_title("AI4I 2020: Machine Failure Class Balance", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count")
    plt.savefig(os.path.join(MAINT_SAVE, 'machine_failure_distribution.png'), dpi=300)
    plt.close()

    # 2. Individual sensor distributions
    sensors = [
        ('Air temperature [K]', 'air_temperature_distribution.png', '#e67e22'),
        ('Process temperature [K]', 'process_temperature_distribution.png', '#e74c3c'),
        ('Rotational speed [rpm]', 'rotational_speed_distribution.png', '#3498db'),
        ('Torque [Nm]', 'torque_distribution.png', '#9b59b6'),
        ('Tool wear [min]', 'tool_wear_distribution.png', '#1abc9c')
    ]
    for col, filename, color in sensors:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=30)
        ax.set_title(f"AI4I 2020: {col} Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        plt.savefig(os.path.join(MAINT_SAVE, filename), dpi=300)
        plt.close()

    # 3. Correlation heatmap
    num_cols_maint = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
    corr_maint = df[num_cols_maint].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_maint, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("AI4I 2020: Maintenance Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(MAINT_SAVE, 'maintenance_correlation_heatmap.png'), dpi=300)
    plt.close()

    # 4. Boxplots split by failure
    sensor_names = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, col in enumerate(sensor_names):
        sns.boxplot(data=df, x='Machine failure', y=col, ax=axes[idx], hue='Machine failure', palette=['#2ecc71', '#e74c3c'], legend=False)
        axes[idx].set_title(f"{col} vs. Failure", fontsize=11, fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Normal (0)', 'Failure (1)'])
    fig.delaxes(axes[5])
    plt.suptitle("AI4I 2020: Sensor Distributions Split by Machine Failure", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(MAINT_SAVE, 'sensor_boxplots_by_failure.png'), dpi=300)
    plt.close()

    print("Maintenance EDA Complete. All plots saved to results/maintenance/eda/")


if __name__ == "__main__":
    run_security_eda()
    run_maintenance_eda()
