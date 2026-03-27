from flask import Flask, render_template, request, jsonify
import json
import os
import datetime
from datetime import datetime, date
from llm_ui import *

app = Flask(__name__)

DATASET_FOLDER = "/teamspace/studios/this_studio/demo/inference_output"
datasets = load_all_datasets(DATASET_FOLDER)

SESSION_FILE = "sessions.json"

# ensure ollama is installed and running when app starts
if not is_ollama_installed():
    print("Ollama not found, attempting to install...")
    if not install_ollama():
        print("Failed to install Ollama. Please install it manually.")
        # you might want to exit or handle this more gracefully
        exit(1)
else:
    print("Ollama is installed.")

ensure_ollama_running()

# Pull all available models at startup
print("Attempting to pull all models specified in AVAILABLE_MODELS...")
for user_facing_name, ollama_model_name in AVAILABLE_MODELS.items():
    if not check_model_exists(ollama_model_name):
        pull_model(ollama_model_name)
    else:
        print(f"Model '{ollama_model_name}' already exists.")
print("Finished checking/pulling all models.")

# -----------------------
# session utilities
# -----------------------

def load_sessions():
    if not os.path.exists(SESSION_FILE):
        return {}
    
    # Check if the file is empty before attempting to load JSON
    if os.path.getsize(SESSION_FILE) == 0:
        return {}

    with open(SESSION_FILE, "r") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"Warning: {SESSION_FILE} is corrupted or empty. Starting with empty sessions.")
            return {}


def save_sessions(data):

    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def today():

    return date.today().isoformat()


# -----------------------
# homepage
# -----------------------

@app.route("/")
def index():

    sessions = load_sessions()

    t = today()

    current_chat = sessions.get(t, [])

    return render_template(
        "index.html",
        sessions=sessions,
        chat=current_chat,
        today=t,
        available_models=AVAILABLE_MODELS.keys(), # pass available models to the frontend
        default_model=DEFAULT_MODEL_NAME # pass default model
    )


# -----------------------
# load old session
# -----------------------

@app.route("/session/<date>")
def get_session(date):

    sessions = load_sessions()

    return jsonify(sessions.get(date, []))


# -----------------------
# ask endpoint
# -----------------------

@app.route("/ask", methods=["POST"])
def ask():

    question = request.json.get("question")
    selected_model = request.json.get("model", DEFAULT_MODEL_NAME) # get selected model, default to DEFAULT_MODEL_NAME

    # run_ai already returns a dict, so we directly use it
    answer_data = run_ai(question, selected_model)

    sessions = load_sessions()

    t = today()

    if t not in sessions:
        sessions[t] = []

    sessions[t].append({
        "question": question,
        "answer": answer_data, # Save the entire answer_data dictionary
        "model": selected_model # save selected model with the session
    })

    save_sessions(sessions)

    return jsonify({"answer": answer_data})


# -----------------------
# AI pipeline
# -----------------------

def run_ai(question, model_name): # accept model_name

    global_start = datasets[0]["start"]
    global_end = datasets[-1]["end"]

    query = parse_query(question, global_start, global_end, model_name=model_name)

    if query["type"] == "unsupported":
        return {
            "error": True,
            "message": (
                "Unsupported question format.",
                [
                    "what happened at 8:03 pm",
                    "what happened around 3 pm",
                    "what happened between 7:50 pm and 8:30 pm"
                ]
            )
        }

    # ------------------------------------------------
    # EXACT / AROUND
    # ------------------------------------------------

    if query["type"] in ["exact", "around"]:

        df, idx = locate_packet(datasets, query["time"])

        if df is None:
            return {"error": True, "message": "No event found for that time."}

        row = df.iloc[idx]

        window_df = get_packet_window(df, idx)

        payload = build_payload(row)
        window_payload = build_window_payload(window_df)

        explanation = explain(payload, window_payload, question, model_name=model_name)
        security_recommendations = recommend_security_actions(window_payload, model_name=model_name)

        return {
            "error": False,
            "timestamp": str(row["Timestamp"]),
            "predicted": row["Predicted_Label_Name"],
            "actual": row["Actual_Label_Name"],
            "correct": bool(row["Correct"]),
            "packet_features": payload["active_features"],
            "packet_window": window_payload,
            "explanation": explanation,
            "security_recommendations": security_recommendations # add recommendations
        }

    # ------------------------------------------------
    # RANGE
    # ------------------------------------------------

    if query["type"] == "range":

        df_start, idx_start = locate_packet(datasets, query["start"])
        df_end, idx_end = locate_packet(datasets, query["end"])

        if df_start is None or df_end is None:
            return {"error": True, "message": "No events found in that time range."}

        start_idx = max(idx_start - 10, 0)
        end_idx = min(idx_end + 10, len(df_start))

        window_df = df_start.iloc[start_idx:end_idx]

        row = df_start.iloc[idx_start]

        payload = build_payload(row)
        window_payload = build_window_payload(window_df)

        explanation = explain(payload, window_payload, question, model_name=model_name)
        security_recommendations = recommend_security_actions(window_payload, model_name=model_name)

        return {
            "error": False,
            "timestamp": str(row["Timestamp"]),
            "predicted": row["Predicted_Label_Name"],
            "actual": row["Actual_Label_Name"],
            "correct": bool(row["Correct"]),
            "packet_features": payload["active_features"],
            "packet_window": window_payload,
            "explanation": explanation,
            "security_recommendations": security_recommendations # add recommendations
        }

    return {"error": True, "message": "Unable to process the request."}

if __name__ == "__main__":
    app.run(debug=True, port=5000)