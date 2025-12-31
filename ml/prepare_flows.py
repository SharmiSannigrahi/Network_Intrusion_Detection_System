# ml/prepare_flows.py
import glob
import pandas as pd

rows = []

for f in glob.glob("data/chunks/logs_chunk_*/conn.log"):
    print(f"Processing {f} ...")
    with open(f) as fh:
        header_map = None
        for line in fh:
            if line.startswith("#fields"):
                # Build a map of field name -> index
                fields = line.strip().split("\t")[1:]  # skip '#fields'
                header_map = {name: idx for idx, name in enumerate(fields)}
                continue
            if line.startswith("#") or not line.strip():
                continue
            if not header_map:
                print(f"⚠️ No header found in {f}")
                break

            parts = line.strip().split("\t")
            try:
                orig_h = parts[header_map["id.orig_h"]]
                resp_h = parts[header_map["id.resp_h"]]
                duration = float(parts[header_map["duration"]]) if parts[header_map["duration"]] != "-" else 0.0
                orig_bytes = float(parts[header_map["orig_bytes"]]) if parts[header_map["orig_bytes"]] != "-" else 0.0
                resp_bytes = float(parts[header_map["resp_bytes"]]) if parts[header_map["resp_bytes"]] != "-" else 0.0
                orig_pkts = float(parts[header_map["orig_pkts"]]) if "orig_pkts" in header_map and parts[header_map["orig_pkts"]] != "-" else 0.0
                resp_pkts = float(parts[header_map["resp_pkts"]]) if "resp_pkts" in header_map and parts[header_map["resp_pkts"]] != "-" else 0.0

                pkt_count = orig_pkts + resp_pkts
                byte_count = orig_bytes + resp_bytes
                avg_pkt_size = byte_count / max(1.0, pkt_count)
                pps = pkt_count / max(1e-6, duration)

                rows.append({
                    "src_ip": orig_h,
                    "dst_ip": resp_h,
                    "pkt_count": pkt_count,
                    "byte_count": byte_count,
                    "duration": duration,
                    "avg_pkt_size": avg_pkt_size,
                    "pps": pps
                })
            except Exception as e:
                # Debugging: show why a row failed
                print(f"Skipping line in {f} due to error: {e}")
                continue

df = pd.DataFrame(rows)
out_file = "data/flows_features.csv"
df.to_csv(out_file, index=False)
print(f"Wrote {len(df)} flows to {out_file}")
