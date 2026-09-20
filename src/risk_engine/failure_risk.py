"""
failure_risk.py - Configurable Predictive Maintenance Failure Risk Assessment Module.
Converts failure probability into qualitative risk levels.
"""

import numpy as np

# Configurable probability thresholds
DEFAULT_FAILURE_THRESHOLDS = {
    'low_max': 0.30,
    'medium_max': 0.70
}


def evaluate_failure_risk(failure_probability, thresholds=None):
    """
    Evaluates maintenance failure risk level based on model failure probability.
    
    Note: The output is a statistical model prediction, not a real-world physical guarantee.
    
    Parameters:
    - failure_probability (float): Model output probability of machine failure [0.0, 1.0]
    - thresholds (dict, optional): Custom dictionary override for low_max and medium_max thresholds.
    
    Returns:
    - dict: {"failure_probability": float, "failure_risk_level": str, "failure_risk_score": float, "disclaimer": str}
    """
    t = thresholds if thresholds is not None else DEFAULT_FAILURE_THRESHOLDS
    prob = float(np.clip(failure_probability, 0.0, 1.0))

    if prob < t['low_max']:
        level = 'LOW'
    elif prob < t['medium_max']:
        level = 'MEDIUM'
    else:
        level = 'HIGH'

    return {
        "failure_probability": round(prob, 4),
        "failure_risk_level": level,
        "failure_risk_score": round(prob, 4),
        "disclaimer": "Model-estimated statistical probability (not an absolute physical guarantee)"
    }


if __name__ == "__main__":
    import numpy as np
    print(evaluate_failure_risk(0.15))
    print(evaluate_failure_risk(0.85))
