import os
import json
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import platform
import time

# ------------------------------------------------
# LLM CONFIG
# ------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

DEFAULT_MODEL_NAME = "gemma3:4b"

AVAILABLE_MODELS = {
    "gemma3:4b": "gemma3:4b",
    "qwen:7b": "qwen:7b",
    "qwen2:7b": "qwen2:7b", # added qwen2:7b
    "llama3.1": "llama3.1",
    "llama3.1:8b": "llama3.1:8b",
}

def is_ollama_installed():
    """Check if ollama binary is installed."""
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        return result.returncode == 0       
    except Exception:
        return False

def _get_torch():
    import torch
    return torch

def install_ollama():
    print(" Installing Ollama...")
    system = platform.system()  
    try:
        if system == "Linux":
            print("   Downloading and installing Ollama for Linux...")
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            result = subprocess.run(
                install_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(" Ollama installed successfully")
                return True
            else:
                print(f" Installation failed: {result.stderr}")
                return False    
        elif system == "Darwin":  # macOS
            print("   Please install Ollama manually:")
            print("   $ brew install ollama")
            print("   Or download from: https://ollama.com/download")
            return False   
        else:
            print(f" Unsupported platform: {system}")
            print("   Please install Ollama manually from: https://ollama.com/download")
            return False          
    except Exception as e:
        print(f" Installation error: {e}")
        return False


def ensure_ollama_running():
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=2)
        if r.ok:
            print(" Ollama server already running.")
            return True
    except Exception:
        pass
    print(" Starting Ollama server...")
    log_file = "ollama_server.log"
    try:
        with open(log_file, "ab", buffering=0) as log:
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=log, 
                stderr=log, 
                start_new_session=True
            )
        for i in range(60):
            try:
                r = requests.get(OLLAMA_TAGS_URL, timeout=2)
                if r.ok:
                    print(" Ollama API is ready.")
                    return True
            except Exception:
                time.sleep(2)
        raise RuntimeError(" Ollama did not start in time. Check ollama_server.log")  
    except FileNotFoundError:
        print(" Ollama binary not found in PATH")
        return False
    except Exception as e:
        print(f" Failed to start Ollama: {e}")
        return False

def check_model_exists(model_name):
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if r.ok:
            models = r.json().get("models", [])
            for model in models:
                if model_name == model.get("name").split(":")[0] + ":" + model.get("name").split(":")[1]: # exact match for "model:tag"
                    return True
        print(f"  Model '{model_name}' not found.")
        return False
    except Exception as e:
        print(f"  Could not check models: {e}")
        return False

def pull_model(model_name):
    print(f"  Pulling model '{model_name}'...")
    try:
        pull_cmd = f"ollama pull {model_name}"
        result = subprocess.run(
            pull_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600 # longer timeout for model pulls
        )
        if result.returncode == 0:
            print(f"  Model '{model_name}' pulled successfully.")
            return True
        else:
            print(f"  Failed to pull model '{model_name}': {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error pulling model '{model_name}': {e}")
        return False


def ask_llm(prompt, model_name=DEFAULT_MODEL_NAME):

    # Ensure Ollama server is running
    if not ensure_ollama_running():
        raise RuntimeError("Ollama server is not running.")
    
    # We no longer check/pull here, assuming models are pre-pulled at app startup
    # if not check_model_exists(model_name):
    #     if not pull_model(model_name):
    #         raise RuntimeError(f"Failed to ensure model '{model_name}' is available.")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(
        OLLAMA_URL,
        json=payload
    )

    r.raise_for_status()

    return r.json()["response"]

ATTACK_FEATURE_FILTER = {

    "Normal": [],

    "DDoS_UDP": ["udp", "icmp", "tcp.len", "tcp.flags"],
    "DDoS_ICMP": ["icmp"],
    "DDoS_TCP": ["tcp"],
    "DDoS_HTTP": ["http"],

    "Password": ["http", "tcp", "dns"],

    "Vulnerability_scanner": ["tcp", "http", "dns"],

    "SQL_injection": ["http"],

    "Uploading": ["tcp", "http", "mqtt"],

    "Backdoor": ["tcp", "mqtt"],

    "Port_Scanning": ["tcp", "icmp"],

    "XSS": ["http"],

    "Ransomware": ["tcp", "dns"],

    "MITM": ["arp", "tcp"],

    "OS_Fingerprinting": ["tcp", "icmp"]
}

def print_event_debug(row, df):

    attack_type = row["Predicted_Label_Name"]

    print("\nEVENT DEBUG")
    print("Timestamp:", row["Timestamp"])
    print("Predicted:", attack_type)
    print("Actual:", row["Actual_Label_Name"])
    print("Correct:", bool(row["Correct"]))

    ignore_cols = [
        "Row_Index",
        "Timestamp",
        "Actual_Label_ID",
        "Predicted_Label_ID",
        "Actual_Label_Name",
        "Predicted_Label_Name",
        "Correct",
        "Attack_type"
    ]

    relevant_prefixes = ATTACK_FEATURE_FILTER.get(attack_type, [])

    active = {}

    for col in df.columns:

        if col in ignore_cols:
            continue

        value = row[col]

        if value == 0 or value == 0.0:
            continue

        # filter by protocol relevance
        if relevant_prefixes:
            if not any(col.startswith(p) for p in relevant_prefixes):
                continue

        # decode one-hot encoded fields
        if "-" in col:
            base, category = col.split("-", 1)
            active[base] = category
        else:
            active[col] = value

    print("\nRelevant packet features:")

    for k, v in list(active.items())[:10]:
        print(f"  {k}: {v}")

    print()

def generate_compact_statistics(df):

    stats = {}

    total = len(df)

    stats["num_samples"] = int(total)
    counts = df["Predicted_Label_Name"].value_counts().to_dict()

    stats["predicted_counts"] = {
        k: int(v) for k, v in counts.items()
    }

    stats["predicted_percent"] = {
        k: round(v / total, 6)
        for k, v in counts.items()
    }

    all_labels = [
        "Normal",
        "DDoS_UDP",
        "DDoS_ICMP",
        "DDoS_TCP",
        "DDoS_HTTP",
        "Password",
        "Vulnerability_scanner",
        "SQL_injection",
        "Uploading",
        "Backdoor",
        "Port_Scanning",
        "XSS",
        "Ransomware",
        "MITM",
        "OS_Fingerprinting"
    ]

    stats["inactive_labels"] = [
        l for l in all_labels if l not in counts
    ]
    top = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    stats["top_labels"] = [
        {
            "label": k,
            "count": int(v),
            "percent": round(v / total, 4)
        }
        for k, v in top
    ]
    pairs = {}

    preds = df["Predicted_Label_Name"].tolist()

    for i in range(len(preds) - 1):

        a = preds[i]
        b = preds[i + 1]

        key = (a, b)

        pairs[key] = pairs.get(key, 0) + 1

    cooccur = []

    for (a, b), count in pairs.items():

        rate = count / counts.get(a, 1)

        if rate >= 0.1:

            cooccur.append({
                "from_predicted": a,
                "to_predicted": b,
                "count": int(count),
                "rate": round(rate, 4)
            })

        cooccur = sorted(
        cooccur,
        key=lambda x: x["count"],
        reverse=True
    )

    stats["cooccur_pairs"] = cooccur[:20]

    return stats
# ------------------------------------------------
# TIMESTAMP GENERATION
# ------------------------------------------------

def add_timestamps(df, start_time, hours:float=24):

    rows = len(df)
    total_seconds = hours * 3600
    interval = total_seconds / rows

    timestamps = [
        start_time + timedelta(seconds=i * interval)
        for i in range(rows)
    ]

    df.insert(1, "Timestamp", timestamps)

    new_start = timestamps[-1] + timedelta(seconds=interval)

    return df, new_start

def locate_packet(datasets, timestamp):

    ts = pd.to_datetime(timestamp)

    for d in datasets:

        if d["start"] <= ts <= d["end"]:

            df = d["df"]

            idx = (df["Timestamp"] - ts).abs().idxmin()

            return df, idx

    closest_df = None
    closest_idx = None
    best_delta = None

    for d in datasets:

        df = d["df"]

        idx = (df["Timestamp"] - ts).abs().idxmin()

        delta = abs(df.iloc[idx]["Timestamp"] - ts)
        

        if best_delta is None or delta < best_delta:

            best_delta = delta
            closest_df = df
            closest_idx = idx

    return closest_df, closest_idx
# ------------------------------------------------
# DATASET LOADING
# ------------------------------------------------

def load_all_datasets(folder):

    datasets = []

    import glob
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    # total monitoring window
    TOTAL_HOURS = 24

    # distribute across files
    hours_per_file = TOTAL_HOURS / len(csv_files)

    # start timeline at midnight for cleaner queries
    current_time = datetime.now().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    for path in csv_files:

        name = os.path.basename(path)

        df = pd.read_csv(path)

        stats = generate_compact_statistics(df)

        json_path = path.replace(".csv", "_stats.json")

        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        # distribute hours across datasets
        df, current_time = add_timestamps(
            df,
            current_time,
            hours=hours_per_file
        )

        start = df["Timestamp"].iloc[0]
        end = df["Timestamp"].iloc[-1]

        datasets.append({
            "name": name,
            "df": df,
            "start": start,
            "end": end
        })
        
        print(name, "timeline:", start, "→", end)

    return datasets


# ------------------------------------------------
# FEATURE UTILITIES
# ------------------------------------------------

def convert_numpy(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def get_active_features(features):

    return {k: v for k, v in features.items() if v != 0 and v != 0.0}

# BUILD EVENT PAYLOAD

def build_payload(row):

    metadata = [
        "Row_Index",
        "Timestamp",
        "Actual_Label_ID",
        "Predicted_Label_ID",
        "Actual_Label_Name",
        "Predicted_Label_Name",
        "Correct"
    ]

    features = {
        col: convert_numpy(row[col])
        for col in row.index
        if col not in metadata
    }

    active = get_active_features(features)

    payload = {
        "timestamp": str(row["Timestamp"]),
        "prediction": {
            "actual": row["Actual_Label_Name"],
            "predicted": row["Predicted_Label_Name"],
            "correct": bool(row["Correct"])
        },
        "active_features": active
    }

    return payload


# ------------------------------------------------
# PACKET WINDOW
# ------------------------------------------------

import random

def get_packet_window(df, index, min_window=5, max_window=20):

    window = random.randint(min_window, max_window)

    start = max(index - window, 0)
    end = min(index + window + 1, len(df))

    return df.iloc[start:end]


def build_window_payload(window_df):

    predicted_events = {}
    actual_events = {}

    for _, row in window_df.iterrows():
        predicted_label = row["Predicted_Label_Name"]
        actual_label = row["Actual_Label_Name"]
        timestamp = str(row["Timestamp"])

        if predicted_label not in predicted_events:
            predicted_events[predicted_label] = []
        predicted_events[predicted_label].append(timestamp)

        if actual_label not in actual_events:
            actual_events[actual_label] = []
        actual_events[actual_label].append(timestamp)

    return {
        "packet_count": len(window_df),
        "predicted_events": predicted_events, # Renamed from predicted_counts
        "actual_events": actual_events       # Renamed from actual_counts
    }


# ------------------------------------------------
# QUERY PARSER
# ------------------------------------------------
def parse_query(question, global_start, global_end, model_name=DEFAULT_MODEL_NAME):

    prompt = f"""
You convert user questions into structured timeline queries.

Monitoring timeline:
Start: {global_start}
End: {global_end}

The goal of the system is to investigate what the intrusion detection model
predicted at a particular time and compare it with the actual classification.

Interpret the user's question and map it to ONE of the following intents.

Allowed intents:

exact:
    the user asks about a specific time.

around:
    the user asks about activity near a specific time.

range:
    the user asks about activity between two times.

earlier_today:
    the user asks about earlier activity today.

unsupported:
    the question does not ask about a specific time.

Examples:

Question: what happened at 6:05 pm
Output:
{{"type":"exact","time":"2026-03-08T18:05:00"}}

Question: what occurred around 3 pm
Output:
{{"type":"around","time":"2026-03-08T15:00:00"}}

Question: what happened between 7:50 pm and 8:30 pm
Output:
{{"type":"range","start":"2026-03-08T19:50:00","end":"2026-03-08T20:30:00"}}

Question: from 8 pm to 10 pm what happened on my network
Output:
{{"type":"range","start":"2026-03-08T20:00:00","end":"2026-03-08T22:00:00"}}

Question: what happened earlier today
Output:
{{"type":"earlier_today"}}
VERY IMPORTANT ONLY REPLY WITH JSON
User question:
{question}
"""

    
    response = ask_llm(prompt, model_name=model_name)

    # remove ```json blocks if model adds them
    response = response.replace("```json", "").replace("```", "").strip()

    data = json.loads(response)

    # convert timestamps to datetime
    if "time" in data:
        data["time"] = pd.to_datetime(data["time"])

    if "start" in data:
        data["start"] = pd.to_datetime(data["start"])

    if "end" in data:
        data["end"] = pd.to_datetime(data["end"])

    return data


def explain(event_payload, window_payload, question, model_name=DEFAULT_MODEL_NAME):

    label_definitions = {
        "Normal": "benign network activity",
        "DDoS_UDP": "a UDP flooding attack intended to overwhelm a service",
        "DDoS_ICMP": "an ICMP flood used to consume network bandwidth",
        "DDoS_TCP": "a TCP connection flood targeting a server",
        "DDoS_HTTP": "a large volume of HTTP requests targeting a web server",
        "Password": "a password brute force attempt",
        "Vulnerability_scanner": "automated scanning for software vulnerabilities",
        "SQL_injection": "malicious SQL queries targeting a database",
        "Uploading": "data being uploaded from a client to a server",
        "Backdoor": "traffic consistent with remote control access",
        "Port_Scanning": "network port probing",
        "XSS": "cross-site scripting attempts against a web application",
        "Ransomware": "ransomware-related activity",
        "MITM": "man-in-the-middle interception",
        "OS_Fingerprinting": "traffic attempting to identify a system's OS"
    }

    prompt = f"""
You are a cybersecurity analyst explaining results from an intrusion detection system .

The system internally uses labels for attack categories, but the user should NOT see these labels.

Instead, convert them into plain language explanations.

Important rules:

1. You may reference the attack types using the descriptions provided.
2. Translate labels into human explanations.
3. Only use the evidence provided.
4. Do NOT invent attacker behavior.
5. Normal traffic means benign activity.
6. Do NOT infer protocols, packet contents, or network behavior.
7. Do NOT mention HTTP, DNS, TCP flags, ports, or packet payloads.
8. Only use the classification labels provided.
9. Only describe the predicted vs actual labels and the surrounding pattern

User question:
{question}

Event packet:
{json.dumps(event_payload, indent=2)}

Packet sequence around the event:
{json.dumps(window_payload, indent=2)}

Attack type descriptions:
{json.dumps(label_definitions, indent=2)}

Your task:

1. Explain what the detection system believed the traffic represented.
2. Explain what the actual traffic pattern represents.
3. If the prediction differs from the actual classification, explain why the system may have confused them.
4. Describe the observable pattern in the surrounding packet window.

Write a short explanation using only the provided classifications in plain English without mentioning internal labels.
"""

    return ask_llm(prompt, model_name=model_name)

def recommend_security_actions(window_payload, model_name=DEFAULT_MODEL_NAME):
    """
    Generates security recommendations based on predicted events in the packet window.
    """
    predicted_events_summary = ""
    if window_payload and window_payload.get("predicted_events"):
        for label, timestamps in window_payload["predicted_events"].items():
            predicted_events_summary += f"- {label} ({len(timestamps)} events): {', '.join(timestamps)}\n"
    
    if not predicted_events_summary:
        return "no specific security recommendations based on current predicted events."

    prompt = f"""
You are a cybersecurity expert providing actionable security recommendations for IOT systems.

Based on the following predicted network events and their timestamps,
provide concise and practical security recommendations.
Focus on immediate actions and best practices relevant to the detected threat types.
If the events include "Normal" traffic, prioritize recommendations for the non-normal events.

Predicted Events:
{predicted_events_summary}

Provide specific and actionable advice.
"""
    return ask_llm(prompt, model_name=model_name)


# ------------------------------------------------
# CLI
# ------------------------------------------------

def start_cli(datasets):

    global_start = datasets[0]["start"]
    global_end = datasets[-1]["end"]

    print("\n==============================")
    print("Network Timeline Investigator")
    print("==============================")

    print("\nAsk questions like:")
    print("what happened at 8:03 pm")
    print("what happened around 3 pm")
    print("what happened between 7:50 pm and 8:30 pm")

    forbidden = [
        "summarize",
        "summary",
        "most common",
        "recent activity",
        "attacks today",
        "how strong",
        "who attacked",
        "vulnerabilities",
        "last hour"
    ]

    while True:

        question = input("\n> ")

        if question.lower() == "exit":
            break

        q = question.lower()

        if any(x in q for x in forbidden):
            print("\nUnsupported question.")
            continue
        
        # In CLI mode, we'll just use the default model or can add a prompt for it later
        query = parse_query(question, global_start, global_end, model_name=DEFAULT_MODEL_NAME)
        print("PARSED QUERY:", query)

        if query["type"] == "unsupported":

            print("\nUnsupported question format.")
            continue


        # ------------------------------------------------
        # EXACT / AROUND
        # ------------------------------------------------

        if query["type"] in ["exact", "around"]:

            df, idx = locate_packet(datasets, query["time"])

            if df is None:
                print("No event found.")
                continue

            window_df = get_packet_window(df, idx)

            payload = build_payload(df.iloc[idx])

            window_payload = build_window_payload(window_df)
            row = df.iloc[idx]

            print_event_debug(row, df)

            answer = explain(payload, window_payload, question, model_name=DEFAULT_MODEL_NAME)
            recommendations = recommend_security_actions(window_payload, model_name=DEFAULT_MODEL_NAME)


            print("\n", answer)
            print("\nSecurity Recommendations:\n", recommendations)


        # ------------------------------------------------
        # RANGE
        # ------------------------------------------------

        elif query["type"] == "range":

            df_start, idx_start = locate_packet(datasets, query["start"])
            df_end, idx_end = locate_packet(datasets, query["end"])

            if df_start is None or df_end is None:
                print("No events found.")
                continue

            start_idx = max(idx_start - 10, 0)
            end_idx = min(idx_end + 10, len(df_start))

            window_df = df_start.iloc[start_idx:end_idx]

            row = df_start.iloc[idx_start]

            payload = build_payload(row)
            window_payload = build_window_payload(window_df)

            print_event_debug(row, df_start)

            answer = explain(payload, window_payload, question, model_name=DEFAULT_MODEL_NAME)
            recommendations = recommend_security_actions(window_payload, model_name=DEFAULT_MODEL_NAME)


            print("\n", answer)
            print("\nSecurity Recommendations:\n", recommendations)

# DATASET_FOLDER = "/teamspace/studios/this_studio/confusion"
# datasets = load_all_datasets(DATASET_FOLDER) #
# start_cli(datasets)