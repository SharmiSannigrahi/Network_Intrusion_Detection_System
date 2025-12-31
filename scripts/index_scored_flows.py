# scripts/index_scored_flows.py
import csv, os, json
from elasticsearch import Elasticsearch, helpers

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = "enriched-flows"
IN_CSV = os.path.join("data", "scored_flows.csv")   # or data/eval_results.csv

def generator(csvfile):
    with open(csvfile, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Optionally convert numeric fields to floats/ints
            doc = {}
            for k, v in row.items():
                if v is None or v == "":
                    doc[k] = None
                    continue
                # try numeric
                try:
                    if "." in v:
                        doc[k] = float(v)
                    else:
                        doc[k] = int(v)
                except Exception:
                    doc[k] = v
            yield {"_index": INDEX, "_source": doc}

def main():
    es = Elasticsearch(ES_URL)
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(IN_CSV)
    print("Indexing", IN_CSV, "->", ES_URL, "index:", INDEX)
    success, failed = 0, 0
    for ok, item in helpers.streaming_bulk(es, generator(IN_CSV), chunk_size=500):
        if ok:
            success += 1
        else:
            failed += 1
    print("Done. Success:", success, "Failed:", failed)

if __name__ == "__main__":
    main()
