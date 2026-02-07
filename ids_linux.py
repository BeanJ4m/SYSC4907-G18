import os
import subprocess
import time
import signal

# ---------------- CONFIG FOR LINUX ----------------
BASE_DIR = "/opt/iot_ids"              # change if you want
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

EXTRACTOR = "extract_v2.py"            # same extractor
TSHARK = "tshark"                      # must be in PATH

INTERFACE = "eth0"                     # use: tshark -D or ip link
CAPTURE_DURATION = 100                 # seconds

POLL_INTERVAL = 30                     # seconds
MIN_FILE_AGE = 60                      # seconds
SIZE_STABLE_INTERVAL = 10              # seconds
# ---------------------------------------

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

tshark_proc = None


# ---------- tshark control (Linux-safe) ----------
def start_tshark():
    global tshark_proc
    print("[*] Starting tshark capture")

    tshark_proc = subprocess.Popen(
        [
            TSHARK,
            "-i", INTERFACE,
            "-b", f"duration:{CAPTURE_DURATION}",
            "-w", os.path.join(
                CAPTURE_DIR,
                "iot_capture_%Y%m%d_%H%M%S.pcapng"
            )
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid   # allow clean group termination
    )


def is_tshark_running():
    return tshark_proc is not None and tshark_proc.poll() is None


def stop_tshark():
    global tshark_proc
    if tshark_proc:
        print("[*] Stopping tshark")
        os.killpg(os.getpgid(tshark_proc.pid), signal.SIGTERM)
        tshark_proc.wait(timeout=5)
        tshark_proc = None


# ---------- file stability ----------
def is_file_stable(path):
    """
    File is complete if:
    - size does not change
    - age > MIN_FILE_AGE
    """
    try:
        size1 = os.path.getsize(path)
        time.sleep(SIZE_STABLE_INTERVAL)
        size2 = os.path.getsize(path)

        age = time.time() - os.path.getmtime(path)
        return size1 == size2 and age > MIN_FILE_AGE
    except FileNotFoundError:
        return False


# ---------- main loop ----------
print("[*] Linux IoT IDS supervisor started")

try:
    start_tshark()

    while True:
        # Restart tshark if it died
        if not is_tshark_running():
            print("[!] tshark stopped — restarting")
            start_tshark()

        # Process completed PCAPs
        for pcap in sorted(os.listdir(CAPTURE_DIR)):
            if not pcap.endswith(".pcapng"):
                continue

            pcap_path = os.path.join(CAPTURE_DIR, pcap)
            csv_name = pcap.replace(".pcapng", ".csv")
            out_csv = os.path.join(DATASET_DIR, csv_name)

            if os.path.exists(out_csv):
                continue

            if not is_file_stable(pcap_path):
                continue

            print(f"[+] Extracting {pcap}")

            subprocess.run(
                ["python3", EXTRACTOR, pcap_path, out_csv],
                check=True
            )

            print(f"[+] Wrote {out_csv}")

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print("\n[*] Shutting down IDS supervisor")
    stop_tshark()
