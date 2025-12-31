# ingest/generate_sample_pcap.py
from scapy.all import IP, ICMP, TCP, Raw, wrpcap
import os

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "sample.pcap")

pkts = []
# synthetic ICMP flows from a few sources
for i in range(40):
    src = f"192.168.1.{2 + (i % 8)}"
    pkts.append(IP(src=src, dst="8.8.8.8")/ICMP())

# a few simple HTTP-like TCP SYN + payloads (not real HTTP parsing but OK for flow features)
for i in range(30):
    src = f"192.168.1.{2 + (i % 8)}"
    sport = 2000 + i
    pkts.append(IP(src=src, dst="93.184.216.34")/TCP(sport=sport, dport=80, flags="S")/Raw(b"GET / HTTP/1.1\r\nHost: example\r\n\r\n"))

wrpcap(out_file, pkts)
print("Wrote", out_file)
