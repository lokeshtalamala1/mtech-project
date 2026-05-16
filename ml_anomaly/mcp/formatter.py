def format_for_llm(row):
    return f"""
You are a cybersecurity analyst.

Analyze this anomaly:

- Total Bytes: {row.get('total_bytes')}
- Total Packets: {row.get('total_packets')}
- Byte Asymmetry: {row.get('byte_asymmetry')}
- Packet Asymmetry: {row.get('packet_asymmetry')}
- Severity: {row.get('severity')}

Return STRICT JSON:

{{
  "attack_type": "...",
  "reason": "...",
  "risk_level": "...",
  "recommendation": "..."
}}
"""