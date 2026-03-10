import os
import json
import re
import requests
import streamlit as st
import subprocess

# ----------------------------
# LOAD CONFIG
# ----------------------------
with open("config.json", "r") as f:
    CFG = json.load(f)

RUN_PATH = CFG["PATH_TEMPLATE"].format(**CFG)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2"

# ----------------------------
# FILE PATTERNS
# ----------------------------
ROUND_RE = re.compile(r"Round_(\d+)_Summary\.json$", re.IGNORECASE)
DETECT_RE = re.compile(r"Detect_LIVE\.json$", re.IGNORECASE)


import pandas as pd

def load_detection_log(run_path):
    log_path = os.path.join(run_path, "detections", "Detect_LOG.jsonl")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    rows = []
    with open(log_path) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")
# ----------------------------
# FILE LOADERS
# ----------------------------
def list_round_files(run_path: str):
    if not os.path.isdir(run_path):
        return []
    items = []
    for fn in os.listdir(run_path):
        m = ROUND_RE.search(fn)
        if m:
            items.append((int(m.group(1)), fn))
    return sorted(items, key=lambda x: x[0])

def load_round_json(run_path: str, filename: str):
    full = os.path.join(run_path, filename)
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)

def list_detection_files(run_path: str):
    detect_path = os.path.join(run_path, "detections")
    if not os.path.isdir(detect_path):
        return []
    return [f for f in os.listdir(detect_path) if DETECT_RE.search(f)]

def load_detection_json(run_path: str, filename: str):
    full = os.path.join(run_path, "detections", filename)
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)

LABEL_MAP = {
    0: "Normal",
    1: "DDoS_UDP",
    2: "DDoS_ICMP",
    3: "DDoS_TCP",
    4: "DDoS_HTTP",
    5: "Password",
    6: "Vulnerability_scanner",
    7: "SQL_injection",
    8: "Uploading",
    9: "Backdoor",
    10: "Port_Scanning",
    11: "XSS",
    12: "Ransomware",
    13: "MITM",
    14: "OS_Fingerprinting"
}

def humanize_evidence(evidence_json):

    if "label_counts" not in evidence_json:
        return evidence_json

    new_counts = {}

    for k, v in evidence_json["label_counts"].items():
        try:
            label = LABEL_MAP[int(k)]
        except:
            label = f"Unknown_{k}"

        new_counts[label] = v

    evidence_json = evidence_json.copy()
    evidence_json["label_counts"] = new_counts

    return evidence_json


ATTACK_LABELS = [
    "Normal", "DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP", "Password",
    "Vulnerability_scanner", "SQL_injection", "Uploading", "Backdoor",
    "Port_Scanning", "XSS", "Ransomware", "MITM", "OS_Fingerprinting"
]

def build_firewall_reasoning_prompt(posture_json: str, question: str):

    return f"""
You are a firewall security analyst.

You are reasoning about how an EXISTING firewall behaves.

DO NOT recommend improvements.
DO NOT suggest new rules.

You must explain:

- What traffic is allowed
- What traffic is blocked
- Whether the host is exposed
- Whether protections are only for containers
- How the firewall would respond to the user's scenario

Firewall posture:

{posture_json}

User question:
{question}

Explain clearly:

- whether the firewall meaningfully protects against the scenario
- where protection exists
- where exposure remains

If host is exposed, state it explicitly.

Respond in plain English.
"""

def build_firewall_summary_prompt(raw_rules: str):

    return f"""
You are a firewall posture analyzer.

Your job is to determine the REAL host protection level.

You MUST extract:

- INPUT policy
- OUTPUT policy
- FORWARD policy

From nftables / iptables rules.

CRITICAL LOGIC:

If INPUT policy = ACCEPT
AND no explicit drop rules for inbound traffic exist,
then:

host_filtered = false
exposed_host = true

If FORWARD policy = DROP
AND docker isolation chains exist,
then:

docker_only_isolation = true

If docker_only_isolation = true
AND host_filtered = false
then:

Protections only apply to containers.
Host remains exposed.

You MUST NOT say "unknown" if policies are visible.

Output STRICT JSON:

{{
"input_policy": "",
"forward_policy": "",
"output_policy": "",
"host_filtered": true/false,
"docker_only_isolation": true/false,
"exposed_host": true/false,
"blocked_ports": [],
"segmentation": [],
"protections_present": [],
"protections_missing": []
}}


Ruleset:

{raw_rules}
"""
def build_firewall_recommendation_prompt(posture_json: str, question: str):

    return f"""
You are a firewall hardening advisor.

You are given the CURRENT firewall posture.

Your job is to suggest SPECIFIC defensive firewall rules.

You MUST:

- Suggest concrete protections
- Focus on host exposure
- Address real attack surfaces
- Consider IoT environments

Do NOT describe the current posture.
Do NOT explain behavior.

ONLY recommend:

- inbound protections
- rate limiting
- segmentation
- service exposure control
- IP filtering

Firewall posture:

{posture_json}

User request:
{question}

Respond with actionable firewall rules in plain English.
"""

def build_firewall_explainer_prompt(posture_json: str, question: str):
    return f"""
You are a firewall security analyst.

You are explaining an EXISTING firewall posture.

DO NOT recommend new rules.
DO NOT suggest improvements.

Only explain:

- what protections currently exist
- what protections are missing
- how the firewall responds to the user’s question

Firewall posture:

{posture_json}

User question:
{question}

Respond clearly in plain English.
"""
# Build the prompt for Gemma
def build_prompt(round_json: dict, user_question: str) -> str:
    return f"""
You are a cybersecurity incident-response assistant focused on practical countermeasures that reduce REAL device and network risk in an IoT environment.

The intrusion detection system (IDS) output is READ-ONLY evidence.
Attack labels are NOT things you can block, restrict, or disable directly.
Treat each label as: "observed signals consistent with <label>".
You MUST translate labels into actions on assets (hosts, services, accounts, network segments) and observable signals (logs, flows, ports, authentication events).
The IDS detects ONLY the following behaviors:

Important:
`Normal` represents benign traffic and is NOT an attack.

Attack behaviors are:

{[x for x in ATTACK_LABELS if x != "Normal"]}


========================
CRITICAL CONSTRAINTS (MANDATORY)
========================
- Do NOT suggest any action that changes the detection system itself.
- Do NOT mention or suggest changes to:
  accuracy, precision, recall, thresholds, confusion matrices, retraining, tuning, calibration, or model behavior.
- Do NOT write actions like "block SQL_injection" or "restrict Password".
- All actions MUST be external to the model and applicable even if the IDS did not exist.
- Use the word "countermeasure" (not mitigation).

========================
PRIORITY SCALE (MANDATORY — MUST USE)
========================
- P0 = Immediate action (hours)
       Active harm, ongoing compromise, or high likelihood of imminent service disruption.
- P1 = Urgent action (24–48 hours)
       Clear attack attempts or preparation activity; exposure reduction needed quickly.
- P2 = Planned action (1–2 weeks)
       Hardening, monitoring, or control improvements that reduce risk over time.

Whenever you assign a Priority:
- You MUST include a one-line justification.
- The justification MUST clearly align with the above timeline.

========================
EVIDENCE LANGUAGE RULES (MANDATORY)
========================
- Do NOT use absolute terms like "high" or "low" unless a unlocking baseline is provided.
- Describe frequency RELATIVE to other behaviors in the same round.
- If uncertainty exists, state it explicitly.

========================
COUNT NORMALIZATION RULE (MANDATORY)
========================
- Whenever you mention a label count, you MUST write it as:
  `Label`: count / num_samples (percent)
- Use num_samples from the JSON evidence.
- If num_samples is unavailable, write "(out of unknown total)".

========================
LABEL STYLING RULE (MANDATORY)
========================
- EVERY label name MUST be wrapped in backticks, e.g. `SQL_injection`, `Normal`, `DDoS_HTTP`.
- NEVER write a label name without backticks.

========================
FORMATTING RULES (STRICT)
========================
- NEVER place Priority, Targets, Evidence, Action, Verify, or Fallback on the same line.
- EACH field MUST start on a new line.
- EACH field MUST start with a dash and a bold label.
- Countermeasure titles MUST be bold and on their own line.
- If any field is inline or compressed, REWRITE the entire countermeasure.

========================
OUTPUT RULE (VERY IMPORTANT)
========================
- Use the structure below, but DO NOT include any instructional text in the final answer.
- Do NOT include guidance notes, parenthetical instructions, or formatting hints.
- Print ONLY clean section headers and content.

========================
FINAL ANSWER FORMAT (EXACT)
========================

SITUATION SUMMARY
- <bullet>
- <bullet>
- <additional bullets as needed>

PRIORITY LEGEND
- P0: immediate (hours)
- P1: urgent (24–48 hours)
- P2: planned (1–2 weeks)

COUNTERMEASURES
1) **<Countermeasure title>**
- **Priority:** <P0 / P1 / P2> — <one-line justification tied to the priority scale>
- **Targets:** <assets / services / accounts / segments>
- **Evidence:** list labels using backticks and normalized counts
- **Action:** <single concrete countermeasure>
- **Verify:** <how to confirm it worked>
- **Fallback:** <safe alternative if evidence is uncertain>

2) **<Countermeasure title>**
- **Priority:** ...
- **Targets:** ...
- **Evidence:** ...
- **Action:** ...
- **Verify:** ...
- **Fallback:** ...

NEXT VERIFICATIONS
- <check> → <what it confirms> → <what to do if positive>
- <check> → <what it confirms> → <what to do if positive>
- <additional checks as needed>

========================
EVIDENCE (READ-ONLY)
========================
{json.dumps(round_json, indent=2)}

========================
USER QUESTION
========================
{user_question}

Before answering, internally verify:
- Are all label names wrapped in backticks?
- Are counts normalized using num_samples?
- Are P0/P1/P2 used consistently with the defined timeline?
- Is every field on its own line with a bold label?
If any answer is no, rewrite.
""".strip()

def get_nftables_rules():
    try:
        output = subprocess.check_output(
            ["sudo", "/usr/local/bin/readfw"],
            stderr=subprocess.STDOUT
        ).decode()
        return output
    except:
        return None


def summarize_firewall(raw_rules):
    fw_prompt = build_firewall_summary_prompt(raw_rules)
    try:
        summary = ask_ollama(fw_prompt)
        return summary
    except:
        return None

def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

CHAT_PATH = os.path.join(RUN_PATH, "chats")
os.makedirs(CHAT_PATH, exist_ok=True)

from datetime import datetime

def new_chat_id():
    return datetime.utcnow().strftime("chat_%Y-%m-%d_%H-%M-%S.json")

def save_chat(chat_id, history):
    with open(os.path.join(CHAT_PATH, chat_id), "w") as f:
        json.dump(history, f, indent=2)

def load_chat(chat_id):
    with open(os.path.join(CHAT_PATH, chat_id)) as f:
        return json.load(f)

def list_chats():
    return sorted(os.listdir(CHAT_PATH), reverse=True)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="IDS Intelligence Analyst", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}
</style>
""", unsafe_allow_html=True)

st.title("IDS Intelligence Analyst")

# ----------------------------
# MODE TOGGLE
# ----------------------------
mode = st.toggle("Live Inference Mode", value=False)
mode_label = "Live Detection" if mode else "Training Intelligence"
st.caption(f"Mode: {mode_label}")

# ----------------------------
# SESSION MEMORY
# ----------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = new_chat_id()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "firewall_context" not in st.session_state:
    st.session_state.firewall_context = None

if "firewall_state" not in st.session_state:
    st.session_state.firewall_state = None  # None / REQUESTED / WAITING / LOADED

if "processing" not in st.session_state:
    st.session_state.processing = False

evidence_json = None

# ----------------------------
# LOAD EVIDENCE
# ----------------------------
if not mode:

    round_files = list_round_files(RUN_PATH)
    if not round_files:
        st.warning("No training rounds available.")
        st.stop()

    round_nums = [r for r, _ in round_files]
    selected_round = st.selectbox("Round", round_nums, index=len(round_nums)-1)

    selected_file = dict(round_files)[selected_round]
    evidence_json = load_round_json(RUN_PATH, selected_file)

else:
    df = load_detection_log(RUN_PATH)
    if df.empty:
        st.warning("No live detection logs yet.")
        st.stop()

    agg_counts = {}

    for row in df["label_counts"]:
        for k, v in row.items():
            agg_counts[k] = agg_counts.get(k, 0) + v

    total_samples = int(df["num_samples"].sum())

    evidence_json = {
        "log_windows": len(df),
        "num_samples": total_samples,
        "label_counts": agg_counts
    }

# ----------------------------
# SUGGESTED QUESTIONS
# ----------------------------
if not st.session_state.chat_history:

    suggestions = (
        [
            "Which attack types are increasing over recent rounds?",
            "What infrastructure weaknesses are most exposed?",
            "What countermeasure would reduce Uploading activity?",
            "Which behavior appears most persistent across rounds?"
        ]
        if not mode else
        [
            "What attack is currently dominant?",
            "What should be addressed immediately?",
            "What infrastructure is most at risk?",
            "What countermeasure should be prioritized now?"
        ]
    )

    cols = st.columns(len(suggestions))

    for col, s in zip(cols, suggestions):
        if col.button(s, use_container_width=True):
            st.session_state.chat_history.append({"question": s, "answer": None})
            st.rerun()

# ----------------------------
# CHAT DISPLAY
# ----------------------------
st.sidebar.markdown("### Conversation History")

past = list_chats()

selected_chat = st.sidebar.selectbox(
    "Open previous session",
    ["Current"] + past
)

if selected_chat != "Current":
    st.session_state.chat_history = load_chat(selected_chat)
    st.session_state.chat_id = selected_chat


for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    if chat["answer"]:
        with st.chat_message("assistant"):
            st.write(chat["answer"])

# ----------------------------
# CHAT INPUT
# ----------------------------
user_question = st.chat_input("Ask a question about infrastructure risk...")

if user_question:

    st.session_state.chat_history.append({
        "question": user_question,
        "answer": None
    })

    if "firewall" in user_question.lower() and st.session_state.firewall_state is None:
        st.session_state.firewall_state = "REQUESTED"

    st.rerun()

# ----------------------------
# PROCESS LAST MESSAGE
# ----------------------------
if st.session_state.chat_history:

    last = st.session_state.chat_history[-1]

    if last["answer"] is None and not st.session_state.processing:

        st.session_state.processing = True

        with st.chat_message("assistant"):

            # ----------------------------
            # FIREWALL ONBOARDING
            # ----------------------------
            if st.session_state.firewall_state == "REQUESTED":

                st.write(
                    "Firewall context can improve recommendations.\n\n"
                    "Reply with:\n"
                    "- 'auto' to analyze nftables\n"
                    "- paste rules directly\n"
                    "- 'skip'"
                )

                st.session_state.firewall_state = "WAITING"
                st.session_state.processing = False
                st.stop()

            # ----------------------------
            # WAITING FOR FIREWALL INPUT
            # ----------------------------
            elif st.session_state.firewall_state == "WAITING":

                response = last["question"].lower()

                placeholder = st.empty()
                placeholder.markdown("*Loading firewall posture…*")

                if response == "auto":

                    raw = get_nftables_rules()

                    if raw:
                        summary = summarize_firewall(raw)

                        st.session_state.firewall_context = summary
                        st.session_state.firewall_state = "LOADED"

                        explain_prompt = build_firewall_explainer_prompt(
                            summary,
                            "Explain my firewall posture"
                        )

                        answer = ask_ollama(explain_prompt)

                        last["answer"] = answer
                        placeholder.empty()
                        st.write(answer)
                        save_chat(
                        st.session_state.chat_id,
                        st.session_state.chat_history
                        )

                    else:
                        last["answer"] = "Unable to access nftables rules."
                        st.session_state.firewall_state = None
                        placeholder.empty()
                        st.write(last["answer"])

                elif response == "skip":
                    st.session_state.firewall_context = None
                    st.session_state.firewall_state = None
                    last["answer"] = "Firewall posture not provided."

                else:
                    summary = summarize_firewall(last["question"])

                    st.session_state.firewall_context = summary
                    st.session_state.firewall_state = "LOADED"

                    explain_prompt = build_firewall_explainer_prompt(
                        summary,
                        "Explain my firewall posture"
                    )

                    answer = ask_ollama(explain_prompt)

                    last["answer"] = answer
                    placeholder.empty()
                    st.write(answer)
                    save_chat(
                    st.session_state.chat_id,
                    st.session_state.chat_history
                    )

                st.session_state.processing = False
                st.stop()

            # ----------------------------
            # FIREWALL ANALYSIS MODE
            # ----------------------------
            elif st.session_state.firewall_context:

                q = last["question"].lower()

                # Recommendation intent
                if any(k in q for k in [
                    "suggest",
                    "recommend",
                    "improve",
                    "secure",
                    "protect",
                    "rules to implement",
                    "what rules"
                ]):
                    placeholder = st.empty()
                    placeholder.markdown("*Designing firewall protections…*")

                    try:
                        rec_prompt = build_firewall_recommendation_prompt(
                            st.session_state.firewall_context,
                            last["question"]
                        )

                        answer = ask_ollama(rec_prompt)

                        last["answer"] = answer
                        placeholder.empty()
                        st.write(answer)
                        save_chat(
                        st.session_state.chat_id,
                        st.session_state.chat_history
                        )

                    except:
                        placeholder.empty()
                        st.write("Firewall recommendation failed.")

                    st.session_state.processing = False
                    st.stop()

                # Reasoning intent
                elif any(k in q for k in [
                    "how",
                    "respond",
                    "handle",
                    "protect against",
                    "exposed",
                    "blocked"
                ]):
                    placeholder = st.empty()
                    placeholder.markdown("*Reviewing firewall behavior…*")

                    try:
                        reasoning_prompt = build_firewall_reasoning_prompt(
                            st.session_state.firewall_context,
                            last["question"]
                        )

                        answer = ask_ollama(reasoning_prompt)

                        last["answer"] = answer
                        placeholder.empty()
                        st.write(answer)
                        save_chat(
                        st.session_state.chat_id,
                        st.session_state.chat_history
                        )

                    except:
                        placeholder.empty()
                        st.write("Firewall analysis failed.")

                    st.session_state.processing = False
                    st.stop()

            # ----------------------------
            # IDS ANALYSIS MODE
            # ----------------------------
            else:

                placeholder = st.empty()
                placeholder.markdown("*Analyzing…*")

                prompt = build_prompt(
                    evidence_json,
                    last["question"],
                    st.session_state.firewall_context
                )

                try:
                    answer = ask_ollama(prompt)
                    last["answer"] = answer
                    placeholder.empty()
                    st.write(answer)
                    save_chat(
                    st.session_state.chat_id,
                    st.session_state.chat_history
                    )
                except:
                    placeholder.empty()
                    st.write("Model request failed.")

        st.session_state.processing = False
