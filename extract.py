import pyshark
import pandas as pd

PCAP_FILE = "test.pcapng"
OUTPUT_CSV = "features.csv"

# --------------------------------------------------
# LOAD ORIGINAL SCHEMA (SOURCE OF TRUTH)
# --------------------------------------------------
FEATURES = list(pd.read_csv("data.csv", nrows=0).columns)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def init_row():
    """
    Initialize row with:
    - all values = 0.0
    - BUT any default-state columns (*-0 or *-0.0) set to 1.0
    """
    row = {}
    for c in FEATURES:
        if c.endswith("-0") or c.endswith("-0.0"):
            row[c] = 1.0
        else:
            row[c] = 0.0
    return row


def safe_float(v):
    try:
        if v is None:
            return 0.0
        if isinstance(v, str) and v.startswith("0x"):
            return float(int(v, 16))
        return float(v)
    except:
        return 0.0


# --------------------------------------------------
# Capture
# --------------------------------------------------
cap = pyshark.FileCapture(
    PCAP_FILE,
    keep_packets=False,
    use_json=True,
    include_raw=False
)

rows = []

for pkt in cap:
    row = init_row()

    # ---------------- ARP ----------------
    if hasattr(pkt, "arp"):
        if "arp.opcode" in row:
            row["arp.opcode"] = safe_float(pkt.arp.get_field("opcode"))
        if "arp.hw.size" in row:
            row["arp.hw.size"] = safe_float(pkt.arp.get_field("hw.size"))

    # ---------------- ICMP ----------------
    if hasattr(pkt, "icmp"):
        i = pkt.icmp
        if "icmp.checksum" in row:
            row["icmp.checksum"] = safe_float(getattr(i, "checksum", None))
        if "icmp.seq_le" in row:
            row["icmp.seq_le"] = safe_float(getattr(i, "seq", None))
        if "icmp.unused" in row:
            row["icmp.unused"] = safe_float(getattr(i, "unused", None))

    # ---------------- TCP ----------------
    if hasattr(pkt, "tcp"):
        t = pkt.tcp
        flags = int(t.flags, 16)

        if "tcp.flags" in row:
            row["tcp.flags"] = float(flags)
        if "tcp.flags.ack" in row:
            row["tcp.flags.ack"] = 1.0 if flags & 0x10 else 0.0
        if "tcp.connection.syn" in row:
            row["tcp.connection.syn"] = 1.0 if flags & 0x02 else 0.0
        if "tcp.connection.fin" in row:
            row["tcp.connection.fin"] = 1.0 if flags & 0x01 else 0.0
        if "tcp.connection.rst" in row:
            row["tcp.connection.rst"] = 1.0 if flags & 0x04 else 0.0
        if "tcp.connection.synack" in row:
            row["tcp.connection.synack"] = 1.0 if flags & 0x12 == 0x12 else 0.0

        if "tcp.seq" in row:
            row["tcp.seq"] = safe_float(t.seq)
        if "tcp.len" in row:
            row["tcp.len"] = safe_float(t.len)
        if "tcp.ack" in row:
            row["tcp.ack"] = safe_float(getattr(t, "ack", None))
        if "tcp.ack_raw" in row:
            row["tcp.ack_raw"] = safe_float(getattr(t, "ack_raw", None))
        if "tcp.checksum" in row:
            row["tcp.checksum"] = safe_float(t.checksum)

    # ---------------- UDP ----------------
    if hasattr(pkt, "udp"):
        if "udp.stream" in row:
            row["udp.stream"] = safe_float(pkt.udp.stream)
        if "udp.time_delta" in row and hasattr(pkt, "frame_info"):
            row["udp.time_delta"] = safe_float(pkt.frame_info.time_delta)

    # ---------------- DNS ----------------
    if hasattr(pkt, "dns") and hasattr(pkt.dns, "qry_name"):
        name = pkt.dns.qry_name

        if "dns.qry.name" in row:
            row["dns.qry.name"] = 1.0

        ln = f"dns.qry.name.len-{len(name)}"
        if ln in row:
            row[ln] = 1.0

        kn = f"dns.qry.name.len-{name}"
        if kn in row:
            row[kn] = 1.0

    # ---------------- HTTP ----------------
    if hasattr(pkt, "http"):
        h = pkt.http

        # flip defaults
        if "http.request.method-0.0" in row:
            row["http.request.method-0.0"] = 0.0
        if "http.request.version-0.0" in row:
            row["http.request.version-0.0"] = 0.0
        if "http.referer-0.0" in row:
            row["http.referer-0.0"] = 0.0

        if hasattr(h, "request_method"):
            col = f"http.request.method-{h.request_method}"
            if col in row:
                row[col] = 1.0

        if hasattr(h, "request_version"):
            col = f"http.request.version-{h.request_version}"
            if col in row:
                row[col] = 1.0

        if hasattr(h, "request_uri") and hasattr(h, "request_version"):
            comp = f"{h.request_uri} {h.request_version}"
            col = f"http.request.version-{comp}"
            if col in row:
                row[col] = 1.0

        if hasattr(h, "referer"):
            col = f"http.referer-{h.referer}"
            if col in row:
                row[col] = 1.0

    # ---------------- MQTT ----------------
    if hasattr(pkt, "mqtt"):
        m = pkt.mqtt

        # flip defaults
        for c in [
            "mqtt.conack.flags-0",
            "mqtt.protoname-0",
            "mqtt.topic-0"
        ]:
            if c in row:
                row[c] = 0.0

        if hasattr(m, "msgtype") and "mqtt.msgtype" in row:
            row["mqtt.msgtype"] = safe_float(m.msgtype)
        if hasattr(m, "len") and "mqtt.len" in row:
            row["mqtt.len"] = safe_float(m.len)
        if hasattr(m, "ver") and "mqtt.ver" in row:
            row["mqtt.ver"] = safe_float(m.ver)

        if hasattr(m, "proto_name"):
            col = f"mqtt.protoname-{m.proto_name}"
            if col in row:
                row[col] = 1.0

        if hasattr(m, "topic"):
            col = f"mqtt.topic-{m.topic}"
            if col in row:
                row[col] = 1.0

        if hasattr(m, "conack_flags"):
            col = f"mqtt.conack.flags-{m.conack_flags}"
            if col in row:
                row[col] = 1.0

    # ---------------- LABEL ----------------
    row["Attack_type"] = 0.0
    rows.append(row)

cap.close()

# --------------------------------------------------
# Write output (exact match)
# --------------------------------------------------
df = pd.DataFrame(rows, columns=FEATURES)
df.to_csv(OUTPUT_CSV, index=False)

print(f"[+] Exact schema + value match written to {OUTPUT_CSV}")

