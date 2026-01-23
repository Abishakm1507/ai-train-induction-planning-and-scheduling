def detect_surge(predicted_demand: int, threshold: int = 7000):
    """
    Detect abnormal passenger surges
    """
    if predicted_demand >= threshold:
        return {
            "surge": True,
            "severity": "High"
        }
    elif predicted_demand >= threshold * 0.8:
        return {
            "surge": True,
            "severity": "Medium"
        }
    else:
        return {
            "surge": False,
            "severity": "Low"
        }
