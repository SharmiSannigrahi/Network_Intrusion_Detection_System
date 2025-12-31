import os
import json
import gzip
import pandas as pd

# Root directory containing your chunked logs
BASE_DIR = os.path.join("data", "chunks")
OUTPUT_PATH = os.path.join("data", "flows.jsonl")

def zeek_log_to_jsonl(conn_log_path):
    """Convert a single conn.log file to JSON lines format."""
    json_records = []
    with open(conn_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            try:
                # Split based on tab (default Zeek log separator)
                parts = line.strip().split("\t")

                # Minimal example fields (you can adjust per your Zeek version)
                if len(parts) < 10:
                    continue  # skip invalid lines

                record = {
                    "ts": float(parts[0]) if parts[0].replace(".", "", 1).isdigit() else None,
                    "uid": parts[1],
                    "src_ip": parts[2],
                    "src_port": parts[3],
                    "dst_ip": parts[4],
                    "dst_port": parts[5],
                    "proto": parts[6],
                    "service": parts[7],
                    "duration": float(parts[8]) if parts[8] not in ("-", "") else 0.0,
                    "orig_bytes": int(parts[9]) if parts[9].isdigit() else 0,
                    "resp_bytes": int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 0,
                }
                json_records.append(record)
            except Exception as e:
                print(f"⚠️ Skipping line in {conn_log_path}: {e}")
                continue
    return json_records

def process_all_chunks():
    """Loop through all chunk log directories and merge conn.log -> JSONL"""
    all_records = []

    for folder in sorted(os.listdir(BASE_DIR)):
        folder_path = os.path.join(BASE_DIR, folder)
        conn_log = os.path.join(folder_path, "conn.log")

        if os.path.isfile(conn_log):
            print(f"📂 Processing {conn_log}")
            recs = zeek_log_to_jsonl(conn_log)
            print(f"✅ Extracted {len(recs)} connections from {folder}")
            all_records.extend(recs)
        else:
            print(f"⚠️ No conn.log found in {folder}")

    if not all_records:
        print("❌ No valid Zeek conn.log data found.")
        return

    # Save as JSON Lines for later training
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for rec in all_records:
            json.dump(rec, f_out)
            f_out.write("\n")

    print(f"\n✅ Done. Saved {len(all_records)} total records to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_chunks()
