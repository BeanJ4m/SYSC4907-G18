import os
import json
import re
import requests
import streamlit as st
from config import PATH

RUN_PATH = PATH
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

# Enter python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

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
    # Defensive-only + “based on JSON” to keep answers grounded
    return f"""
        You are a cybersecurity analyst reviewing federated IDS results.
        Answer using ONLY the JSON data provided. Focus on DEFENSIVE mitigations.

        Round summary JSON:
        {json.dumps(round_json, indent=2)}

        User question:
        {user_question}

        When relevant, structure your answer as:
        - Observations (from the JSON)
        - Likely causes (based on misclassifications)
        - Recommended mitigations (technical but human-readable)
        """.strip()

def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
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
        value="Using only this round summary, produce an operational runbook: (1) top 3 risks to production, (2) what to alert on first, (3) specific mitigations per confused pair, (4) validation steps and acceptance criteria, (5) what additional telemetry/features you’d request to reduce those confusions.",
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
