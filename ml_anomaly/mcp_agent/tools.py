def get_ml_context():
    # later connect real CSV
    return {
        "fusion_score": 0.78,
        "severity": "STRONG",
        "byte_asymmetry": 0.92,
        "packet_asymmetry": 0.88
    }


def get_ip_info(ip="unknown"):
    # dummy tool (later API)
    return {
        "ip": ip,
        "country": "Unknown",
        "reputation": "Suspicious"
    }