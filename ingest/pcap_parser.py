# ingest/pcap_parser.py
"""
Simple PCAP -> Flow extractor using Scapy.
Writes per-flow JSON lines to output file or indexes to Elasticsearch.
"""
from scapy.all import rdpcap, IP, IPv6, TCP, UDP
import json, math, time
from collections import defaultdict
from elasticsearch import Elasticsearch
import argparse

FLOW_TIMEOUT = 60.0  # seconds (simple approach: we don't split flows by timeout in this MVP)

def packet_key(pkt):
    # returns a 5-tuple key
    if IP in pkt:
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
    elif IPv6 in pkt:
        src = pkt[IPv6].src
        dst = pkt[IPv6].dst
        proto = pkt[IPv6].nh
    else:
        return None
    sport = pkt.sport if (TCP in pkt or UDP in pkt) else 0
    dport = pkt.dport if (TCP in pkt or UDP in pkt) else 0
    # keep direction as-is for MVP
    return (src, dst, int(sport), int(dport), int(proto))

def extract_flows_from_pcap(pcap_path):
    pkts = rdpcap(pcap_path)
    flows = {}
    for pkt in pkts:
        key = packet_key(pkt)
        if key is None:
            continue
        ts = float(pkt.time)
        length = len(pkt)  # total bytes
        if key not in flows:
            flows[key] = {
                "src": key[0],
                "dst": key[1],
                "src_port": key[2],
                "dst_port": key[3],
                "protocol": key[4],
                "start_ts": ts,
                "end_ts": ts,
                "pkt_count": 0,
                "byte_count": 0,
                "tcp_flags": set()
            }
        f = flows[key]
        f["pkt_count"] += 1
        f["byte_count"] += length
        f["end_ts"] = max(f["end_ts"], ts)
        # TCP flags
        if TCP in pkt:
            try:
                f["tcp_flags"].add(str(pkt[TCP].flags))
            except Exception:
                pass
    # finalize
    out = []
    for f in flows.values():
        duration = max(0.0, f["end_ts"] - f["start_ts"])
        avg_pkt_size = f["byte_count"] / f["pkt_count"] if f["pkt_count"] else 0
        pps = f["pkt_count"] / duration if duration > 0 else f["pkt_count"]
        tcp_flags = list(f["tcp_flags"])
        rec = {
            "timestamp": f["start_ts"],
            "src_ip": f["src"],
            "dst_ip": f["dst"],
            "src_port": f["src_port"],
            "dst_port": f["dst_port"],
            "protocol": f["protocol"],
            "pkt_count": f["pkt_count"],
            "byte_count": f["byte_count"],
            "duration": duration,
            "avg_pkt_size": avg_pkt_size,
            "pps": pps,
            "tcp_flags": tcp_flags
        }
        out.append(rec)
    return out

def index_flows_to_es(flows, es_host="http://localhost:9200", index_name="flows"):
    es = Elasticsearch(es_host)
    for doc in flows:
        es.index(index=index_name, body=doc)

def write_jsonl(flows, out_path):
    with open(out_path, "w") as fh:
        for d in flows:
            fh.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="path to pcap file")
    parser.add_argument("--out", default=None, help="output JSONL path (optional)")
    parser.add_argument("--es", default=None, help="Elasticsearch URL (optional, will index flows if provided)")
    parser.add_argument("--index", default="flows", help="ES index name when using --es")
    args = parser.parse_args()

    flows = extract_flows_from_pcap(args.pcap)
    print(f"Extracted {len(flows)} flows from {args.pcap}")

    if args.out:
        write_jsonl(flows, args.out)
        print("Wrote JSONL to", args.out)
    if args.es:
        index_flows_to_es(flows, es_host=args.es, index_name=args.index)
        print("Indexed to ES", args.es, "index", args.index)
