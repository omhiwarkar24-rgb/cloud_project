"""
integrated_risk.py - Configurable Integrated Risk Fusion Module.
Combines Security Risk Score and Predictive Maintenance Failure Risk Score into a unified system risk score.
"""

from security_risk import evaluate_security_risk
from failure_risk import evaluate_failure_risk

DEFAULT_INTEGRATED_THRESHOLDS = {
    'low_max': 0.30,
    'medium_max': 0.60,
    'high_max': 0.85
}


def evaluate_integrated_risk(attack_category,
                            failure_probability,
                            w_sec=0.5,
                            w_fail=0.5,
                            sec_mapping=None,
                            fail_thresholds=None,
                            integrated_thresholds=None):
    """
    Computes integrated overall system risk by fusing Security Risk and Failure Risk.
    
    Formula:
    overall_risk_score = w_sec * security_risk_score + w_fail * failure_risk_score
    
    Parameters:
    - attack_category (str): Predicted attack string (e.g. 'Normal', 'DoS', 'Exploits')
    - failure_probability (float): Model output failure probability [0.0, 1.0]
    - w_sec (float): Weight for security risk score (default 0.5)
    - w_fail (float): Weight for failure risk score (default 0.5)
    
    Returns:
    - dict: Comprehensive risk assessment payload.
    """
    # 1. Compute component risks
    sec_res = evaluate_security_risk(attack_category, risk_mapping=sec_mapping)
    fail_res = evaluate_failure_risk(failure_probability, thresholds=fail_thresholds)

    sec_score = sec_res['security_risk_score']
    fail_score = fail_res['failure_risk_score']

    # Normalize weights so they sum to 1.0
    total_w = w_sec + w_fail
    norm_w_sec = w_sec / total_w if total_w > 0 else 0.5
    norm_w_fail = w_fail / total_w if total_w > 0 else 0.5

    # 2. Fuse risk scores
    overall_score = norm_w_sec * sec_score + norm_w_fail * fail_score
    overall_score = round(overall_score, 4)

    # 3. Map overall score to level
    t = integrated_thresholds if integrated_thresholds is not None else DEFAULT_INTEGRATED_THRESHOLDS
    if overall_score < t['low_max']:
        overall_level = 'LOW'
    elif overall_score < t['medium_max']:
        overall_level = 'MEDIUM'
    elif overall_score < t['high_max']:
        overall_level = 'HIGH'
    else:
        overall_level = 'CRITICAL'

    return {
        "attack_category": attack_category,
        "security_risk_level": sec_res['security_risk_level'],
        "security_risk_score": sec_score,
        "failure_probability": fail_res['failure_probability'],
        "failure_risk_level": fail_res['failure_risk_level'],
        "failure_risk_score": fail_score,
        "overall_risk_score": overall_score,
        "overall_risk_level": overall_level,
        "weights_used": {"w_sec": norm_w_sec, "w_fail": norm_w_fail}
    }


if __name__ == "__main__":
    print(evaluate_integrated_risk("Normal", 0.10))
    print(evaluate_integrated_risk("DoS", 0.15))
    print(evaluate_integrated_risk("Exploits", 0.85))
