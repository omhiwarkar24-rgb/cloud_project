"""
drift_detection.py - Explainable Concept & Covariate Drift Detection engine using the Kolmogorov-Smirnov (KS) Test.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_feature_drift_ks(df_env_a, df_env_b, feature_cols, p_value_threshold=0.05, ks_stat_threshold=0.15):
    """
    Computes Kolmogorov-Smirnov (KS) 2-sample test for each feature between Environment A and Environment B.
    
    KS-Test Explanation:
    - Null Hypothesis (H0): Feature distributions in Env A and Env B are identical.
    - If p-value < p_value_threshold (e.g. 0.05) AND KS statistic > ks_stat_threshold, H0 is rejected -> Feature Drift Flagged.
    """
    drift_results = []
    drifted_features_count = 0

    for col in feature_cols:
        val_a = df_env_a[col].values
        val_b = df_env_b[col].values
        
        # Calculate KS 2-sample statistic and p-value
        ks_stat, p_val = ks_2samp(val_a, val_b)
        
        is_drift = bool(p_val < p_value_threshold and ks_stat > ks_stat_threshold)
        if is_drift:
            drifted_features_count += 1

        drift_results.append({
            "feature": col,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "is_drifted": is_drift,
            "mean_env_a": float(np.mean(val_a)),
            "mean_env_b": float(np.mean(val_b)),
            "std_env_a": float(np.std(val_a)),
            "std_env_b": float(np.std(val_b))
        })

    df_drift = pd.DataFrame(drift_results)
    
    pct_drifted = (drifted_features_count / len(feature_cols)) * 100
    # Overall System Drift Flagged if >= 15% of features exhibit statistical drift
    system_drift_detected = bool(pct_drifted >= 15.0)

    summary = {
        "total_features": len(feature_cols),
        "drifted_features_count": drifted_features_count,
        "pct_drifted": round(pct_drifted, 2),
        "system_drift_detected": system_drift_detected,
        "p_value_threshold": p_value_threshold,
        "ks_stat_threshold": ks_stat_threshold
    }

    return df_drift, summary
