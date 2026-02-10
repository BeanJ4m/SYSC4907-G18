import os
import json
import re
import requests
import streamlit as st
import json

with open("config.json", "r") as f:
    CFG = json.load(f)

RUN_PATH = CFG["PATH_TEMPLATE"].format(**CFG)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

# Filename pattern for round summaries
ROUND_RE = re.compile(r"Round_(\d+)_Summary\.json$", re.IGNORECASE)

# Find all available JSON files for dropdown list
def list_round_files(run_path: str):
    if not os.path.isdir(run_path):
        return []
    items = []
    for fn in os.listdir(run_path):
        m = ROUND_RE.search(fn)
        if m:
            items.append((int(m.group(1)), fn))
    return sorted(items, key=lambda x: x[0])

# load one round's JSON file and convert it to a Python dictionary
def load_round_json(run_path: str, filename: str):
    full = os.path.join(run_path, filename)
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)

# Build the prompt for Gemma
def build_prompt(round_json: dict, user_question: str) -> str:
    return f"""
You are a cybersecurity incident-response assistant focused on practical countermeasures that reduce REAL device and network risk in an IoT environment.

The intrusion detection system (IDS) output is READ-ONLY evidence.
Attack labels are NOT things you can block, restrict, or disable directly.
Treat each label as: "observed signals consistent with <label>".
You MUST translate labels into actions on assets (hosts, services, accounts, network segments) and observable signals (logs, flows, ports, authentication events).

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

# Sets the browser tab title, use a wide layout, and display the main title at the top of the pae
st.set_page_config(page_title="FL Round Analyst", layout="wide")
st.title("Federated Learning Round Analyst (Gemma 3)")

st.write("Select a round summary and ask questions about attacks, confusions, and mitigations.")

# load available rounds
round_files = list_round_files(RUN_PATH)
if not round_files:
    st.error(f"No Round_*_Summary.json files found in: {RUN_PATH}")
    st.stop()

round_nums = [r for r, _ in round_files]
latest_round = round_nums[-1]

# split the web page into two columns (left side for round selecton, and right side for question box)
col1, col2 = st.columns([1, 2])
with col1:
    selected_round = st.selectbox("Round", round_nums, index=len(round_nums) - 1)
    selected_file = dict(round_files)[selected_round]

    st.caption(f"File: {selected_file}")
    if st.button("Load Latest Round"):
        selected_round = latest_round

with col2:
    question = st.text_area(
        "Ask a question",
        value="Given this round's evidence, produce a prioritized mitigation plan to reduce real device and network risk. Focus on practical defensive steps and what to verify next.",
        height=140
    )

# Load JSON for selected round
round_json = load_round_json(RUN_PATH, selected_file)
with st.expander("View loaded JSON", expanded=False):
    st.json(round_json)

# When clicked, it builds the prompt,, sends it to Ollama, and displays teh answer
if st.button("Ask Gemma 3"):
    with st.spinner("Thinking..."):
        prompt = build_prompt(round_json, question)
        try:
            answer = ask_ollama(prompt)
            st.subheader("Answer")
            st.write(answer)
        except requests.RequestException as e:
            st.error(f"Ollama request failed: {e}")
            st.info("Make sure Ollama is running and the model is available: `ollama run gemma3`")
