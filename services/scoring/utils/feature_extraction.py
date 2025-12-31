import subprocess
import pandas as pd

def extract_features_from_pcap(file_path: str):
    try:
        # Run tshark command to extract basic info from packets
        command = [
            "tshark",
            "-r", file_path,
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "frame.len",
            "-e", "_ws.col.Protocol"
        ]

        # Execute and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split("\n")
        if not lines or lines == ['']:
            print("❌ No packets found in file.")
            return None

        # Parse tshark output
        packets = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    timestamp = float(parts[0])
                    length = int(parts[1])
                    protocol = parts[2] if parts[2] else "UNKNOWN"
                    packets.append({
                        "timestamp": timestamp,
                        "length": length,
                        "protocol": protocol
                    })
                except:
                    continue

        if not packets:
            print("❌ No valid packets parsed.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(packets)

        # Compute simple statistical features
        features = {
            "total_packets": len(df),
            "avg_packet_size": df["length"].mean(),
            "max_packet_size": df["length"].max(),
            "min_packet_size": df["length"].min(),
            "std_packet_size": df["length"].std(),
            "duration": df["timestamp"].max() - df["timestamp"].min()
        }

        return pd.DataFrame([features])

    except subprocess.CalledProcessError as e:
        print(f"❌ Tshark error: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None
