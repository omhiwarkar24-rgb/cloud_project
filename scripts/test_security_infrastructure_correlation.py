"""Validate time alignment and correlation reporting with current data."""

import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from analytics.security_infrastructure_correlation import (
    analyze_all_nodes,
    _correlation_result,
)


def run_tests():
    synthetic = pd.DataFrame({
        "security_risk": [0.1, 0.2, 0.3, 0.4, 0.5],
        "cpu_utilization": [10, 20, 30, 40, 50],
    })
    correlation = _correlation_result(synthetic, "cpu_utilization", "CPU utilization")
    assert correlation["statistically_meaningful"] is True
    assert correlation["correlation"] > 0.99

    results = analyze_all_nodes()
    assert results
    for result in results:
        assert {"node_id", "aligned_data", "correlations", "event_timeline", "explanation"}.issubset(result)
        assert "causes" in result["explanation"]
        assert isinstance(result["correlations"], list)

    print(f"[SUCCESS] CORRELATION TESTS PASSED for {len(results)} nodes")


if __name__ == "__main__":
    run_tests()
