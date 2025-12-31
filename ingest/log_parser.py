# ingest/log_parser.py
import re, json
from elasticsearch import Elasticsearch
import argparse
from datetime import datetime

# Example Apache access log regex (common log format)
APACHE_RE = re.compile(r'(?P<ip>\S+) \S+ \S+ \[(?P<time>.+?)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+-?)')

def parse_apache(line):
    m = APACHE_RE.match(line)
    if not m:
        return None
    g = m.groupdict()
    # parse time
    dt = datetime.strptime(g["time"].split()[0], "%d/%b/%Y:%H:%M:%S")
    return {
        "timestamp": dt.isoformat(),
        "src_ip": g["ip"],
        "method": g["method"],
        "path": g["path"],
        "status": int(g["status"]),
        "size": int(g["size"]) if g["size"].isdigit() else 0,
        "raw": line.strip()
    }

def index_line(es, index, doc):
    es.index(index=index, body=doc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--es", default="http://localhost:9200")
    parser.add_argument("--index", default="raw-logs")
    args = parser.parse_args()
    es = Elasticsearch(args.es)
    with open(args.file) as fh:
        for line in fh:
            doc = parse_apache(line)
            if doc:
                index_line(es, args.index, doc)
