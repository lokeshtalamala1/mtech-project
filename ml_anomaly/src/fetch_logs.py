import json
import requests, sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INDEXER = "https://10.21.232.220:9201"
INDEX = "wazuh-alerts-*"
OUTFILE = "/home/user/Downloads/MTP/mtp/ml_anomaly/data/logs.jsonl"

AUTH = ("admin", "WazuhDashboard")

TIME_FROM = "now-2w"
TIME_TO = "now"

QUERY = {
    "size": 1000,
    "sort": [{"@timestamp": "asc"}],
    "query": {
        "bool": {
            "must": [
               {"match_all":{}}, 
                {"range": {"@timestamp": {"gte": TIME_FROM, "lte": TIME_TO}}}
            ]
        }
    },
}

# QUERY = {
#     "size": 1000,
#     "sort": [{"@timestamp": "asc"}],
#     "_source": [
#         "agent.ip",
#         "agent.name",
#         "data.src_ip",
#         "data.dest_ip",
#         "data.src_port",
#         "data.dest_port",
#         "data.proto",
#         "data.app_proto",
#         "data.flow.bytes_toserver",
#         "data.flow.bytes_toclient",
#         "data.flow.pkts_toserver",
#         "data.flow.pkts_toclient",
#         # "data.timestamp",
#         # "data.flow.start",
#         # "data.direction",
#         # "data.flow_id",
#         "rule.level",
#         "rule.id",
#         # "rule.firedtimes",
#         # "data.alert.signature_id",
#         # "data.alert.category",
#         "@timestamp",
#         # "timestamp",
#     ],
#     "query": {
#         "range": {
#             "@timestamp": {"gte": TIME_FROM, "lte": TIME_TO}
#         }
#     },
# }



def fetch_alerts() -> int:
# 1) initial search with scroll
    session = requests.Session()
    session.auth = AUTH
    try : 
        r = session.post(
        f"{INDEXER}/{INDEX}/_search?scroll=2m",
        # f"{INDEXER}/{INDEX}/_search",
        json=QUERY,
        verify=False,
        timeout=120,
        )   
        r.raise_for_status()
        resp = r.json()

    except requests.RequestException as exc :
        print(f"Error  : {exc} ")
        return 0

    scroll_id = resp.get("_scroll_id")
    hits      = resp["hits"]["hits"]
    total_est = resp["hits"].get("total", {})
    total_est = total_est.get("value", "?") if isinstance(total_est, dict) else total_est
    print(f"   Indexer reports ~{total_est} matching documents")

    count = 0
    with open(OUTFILE, "w", encoding="utf-8") as fout:
        while hits:
            for h in hits:
                fout.write(json.dumps(h["_source"], ensure_ascii=False) + "\n")
                count += 1

            if count % 5000 == 0:
                print(f"   … fetched {count:,} documents so far")

            # ── 2. Next batch ─────────────────────────────────────────────────
            try:
                r = session.post(
                    f"{INDEXER}/_search/scroll",
                    json={"scroll": "2m", "scroll_id": scroll_id},
                    verify=False,
                    timeout=120,
                )
                r.raise_for_status()
            except requests.RequestException as exc:
                print(f"❌  Scroll request failed at doc {count}: {exc}", file=sys.stderr)
                break

            resp      = r.json()
            scroll_id = resp.get("_scroll_id", scroll_id)
            hits      = resp["hits"]["hits"]

    # ── 3. Clear scroll (cleanup) ─────────────────────────────────────────────
    if scroll_id:
        try:
            session.delete(
                f"{INDEXER}/_search/scroll",
                json={"scroll_id": [scroll_id]},
                verify=False,
                timeout=60,
            )
        except Exception:
            pass  # non-critical

    print(f"\n✅  Exported {count:,} documents → {OUTFILE}")
    return count


if __name__ == "__main__":
    fetch_alerts()