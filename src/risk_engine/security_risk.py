"""
security_risk.py - Configurable Security Risk Assessment Module.
Converts predicted attack categories into risk levels and numerical risk scores.
"""

# Application-level rule mapping dictionary (easily configurable/editable)
# Note: This is an application-level rule, not a claim about the dataset's inherent risk semantics.
DEFAULT_SECURITY_RISK_MAPPING = {
    'Normal': {'level': 'LOW', 'score': 0.10, 'reason': 'Standard benign network communication'},
    'Analysis': {'level': 'LOW', 'score': 0.25, 'reason': 'Port scanning or info gathering'},
    'Reconnaissance': {'level': 'MEDIUM', 'score': 0.50, 'reason': 'Active network mapping'},
    'Fuzzers': {'level': 'MEDIUM', 'score': 0.60, 'reason': 'Automated vulnerability discovery'},
    'Generic': {'level': 'MEDIUM', 'score': 0.65, 'reason': 'Unspecified attack signature'},
    'DoS': {'level': 'HIGH', 'score': 0.80, 'reason': 'Denial of Service availability threat'},
    'Exploits': {'level': 'CRITICAL', 'score': 0.90, 'reason': 'System vulnerability exploitation'},
    'Backdoor': {'level': 'CRITICAL', 'score': 0.95, 'reason': 'Unauthorized persistent access'},
    'Backdoors': {'level': 'CRITICAL', 'score': 0.95, 'reason': 'Unauthorized persistent access'},
    'Shellcode': {'level': 'CRITICAL', 'score': 0.95, 'reason': 'Arbitrary code execution payload'},
    'Worms': {'level': 'CRITICAL', 'score': 1.00, 'reason': 'Self-propagating malware infection'}
}


def evaluate_security_risk(attack_category, risk_mapping=None):
    """
    Evaluates security risk level and score for a predicted attack category.
    
    Parameters:
    - attack_category (str): Predicted attack string (e.g. 'Normal', 'DoS', 'Exploits')
    - risk_mapping (dict, optional): Custom dictionary override mapping categories to risk levels/scores.
    
    Returns:
    - dict: {"attack_category": str, "risk_level": str, "risk_score": float, "reason": str}
    """
    mapping = risk_mapping if risk_mapping is not None else DEFAULT_SECURITY_RISK_MAPPING
    
    category_key = str(attack_category).strip()
    
    if category_key in mapping:
        info = mapping[category_key]
        level = info['level']
        score = info['score']
        reason = info['reason']
    else:
        # Fallback for unexpected categories
        level = 'MEDIUM'
        score = 0.50
        reason = f'Unrecognized attack category: {category_key}'

    return {
        "attack_category": category_key,
        "security_risk_level": level,
        "security_risk_score": round(score, 4),
        "reason": reason
    }


if __name__ == "__main__":
    print(evaluate_security_risk("DoS"))
    print(evaluate_security_risk("Exploits"))
