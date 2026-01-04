import os
import re

# Directory containing the files
folder_path = r"D:\log-pcap-siem\data\chunks"

# Iterate through each file in the directory
for filename in os.listdir(folder_path):
    # Match files named chunk_00XXX_YYYYMMDDHHMMSS.pcap (where XXX = 002 to 034)
    match = re.match(r"(chunk_0{2,3}\d{2})_\d{14}\.pcap", filename)
    if match:
        # Construct new name: chunk_00XXX.pcap
        new_filename = f"{match.group(1)}.pcap"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_filename}")
