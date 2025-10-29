#!/usr/bin/env python3
"""
main_controller.py
------------------
This script uses Ollama to dynamically generate and run a PyTorch
training script (train_dynamic.py) using data and model definitions
from the static modules: prepare_data.py and model_def.py.
"""

import os
import sys
import time
import subprocess
import requests
import re
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError

# ==============================
# Configuration
# ==============================
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_PATH = "trained_model.pth"
TRAIN_SCRIPT = "train_dynamic.py"
MAX_ATTEMPTS = 5

MODEL_CANDIDATES = ("gemma3:latest", "gemma2:latest", "gemma:2b")

REQUIRED_FILES = [
    "prepare_data.py",
    "model_def.py",
    "X_train.npy",
    "y_train.npy",
    "X_test.npy",
    "y_test.npy",
]

# ==============================
# Utilities
# ==============================
def run(cmd, **kwargs):
    """Run a subprocess command."""
    return subprocess.run(cmd, **kwargs)

def ensure_ollama_installed():
    """Check that Ollama is installed."""
    try:
        run(["ollama", "--version"], check=True, capture_output=True)
        print("✅ Ollama is already installed.")
    except Exception:
        print("❌ Ollama not found.\nInstall manually:\n  curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)

def _tail_file(path, n=40):
    """Show the last few lines of a file (for debugging Ollama)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192), os.SEEK_SET)
            lines = f.read().decode(errors="replace").splitlines()[-n:]
            return "\n".join(lines)
    except Exception:
        return "<unavailable>"

def start_ollama():
    """Ensure Ollama server is running."""
    try:
        if requests.get(OLLAMA_TAGS_URL, timeout=2).ok:
            print("Ollama server is already running.")
            return
    except Exception:
        pass
    print("🚀 Starting Ollama server...")
    log_path = "ollama_server.log"
    logf = open(log_path, "ab", buffering=0)
    subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=logf, start_new_session=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if requests.get(OLLAMA_TAGS_URL, timeout=2).ok:
                print("✅ Ollama API is ready.")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not start in time.\n" + _tail_file(log_path))

def ensure_model(candidates=MODEL_CANDIDATES):
    """Ensure a model is available locally."""
    print("🔍 Checking available models...")
    out = run(["ollama", "list"], capture_output=True, text=True)
    have = (out.stdout or "").lower()
    for name in candidates:
        if name.lower() in have:
            print(f"✅ Using model: {name}")
            return name
    for name in candidates:
        print(f"⬇️  Pulling {name} ...")
        if run(["ollama", "pull", name]).returncode == 0:
            return name
    raise RuntimeError("Could not find or pull any model.")

def ask_model(model_name, prompt_text, retries=3):
    """Send a text prompt to Ollama and return its generated response."""
    payload = {"model": model_name, "prompt": prompt_text, "stream": False}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=(5, 420))
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                raise RuntimeError(data["error"])
            return data.get("response", "")
        except (ReadTimeout, ConnectTimeout):
            print(f"⏳ [try {attempt}] Timeout waiting for model response.")
        except ConnectionError as e:
            print(f"⚠️  Connection error: {e}")
        except Exception as e:
            print(f"⚠️  {type(e).__name__}: {e}")
        time.sleep(2 * attempt)
    raise RuntimeError("❌ Failed to get response from Ollama after retries.")

def save_code(code_text, path):
    """Clean and save generated code."""
    lines = []
    for l in code_text.splitlines():
        if l.strip().startswith("```"):
            continue
        if re.match(r"^\s*(Note|Explanation|Below|Output|This code)\b", l):
            break
        lines.append(l)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run_code(path):
    """Execute a generated Python script and capture output."""
    result = run(["python3", path], text=True, capture_output=True)
    return result.stdout.strip(), result.stderr.strip()

def file_exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def verify_dependencies():
    """Ensure required static files and datasets exist."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print("❌ Missing files:\n  " + "\n  ".join(missing))
        sys.exit(1)
    print("✅ All required files found.")

# ==============================
# Prompt for training script
# ==============================
TRAIN_PROMPT = f"""
Output ONLY valid Python 3 code.

Write a PyTorch training script that:
• Imports load_data() from prepare_data.py and Net from model_def.py (do not redefine).
• Calls train_loader, test_loader, num_classes, input_size = load_data().
• Uses device = torch.device("cuda" if torch.cuda.is_available() else "cpu").
• Initializes model = Net(input_size, num_classes).to(device).
• Uses CrossEntropyLoss() and Adam(lr=1e-3).
• Trains for 10 epochs, printing epoch, loss, and accuracy.
• After training, evaluates on the test set and prints final test accuracy.
• Saves final model to 'trained_model.pth'.
No markdown, comments, or explanations.
"""

# ==============================
# Main
# ==============================
def main():
    ensure_ollama_installed()
    verify_dependencies()
    start_ollama()
    model_name = ensure_model()

    print(f"🧠 Using LLM model: {model_name}")

    attempt = 1
    success = False
    last_error = None

    while attempt <= MAX_ATTEMPTS and not success:
        print(f"\n=== Attempt {attempt} ===")

        prompt = TRAIN_PROMPT if attempt == 1 else (
            f"{TRAIN_PROMPT}\n\nPrevious error:\n{last_error or 'No stderr'}\nFix it and regenerate correctly."
        )

        code = ask_model(model_name, prompt)
        save_code(code, TRAIN_SCRIPT)

        print("🏃 Running generated script...")
        stdout, stderr = run_code(TRAIN_SCRIPT)

        if stdout:
            print("Standard output:\n", stdout)
        if stderr:
            print("Standard error:\n", stderr)

        # Check if training succeeded
        if file_exists(MODEL_PATH) and not any(
            kw in stderr for kw in (
                "Traceback", "Error", "Exception", "RuntimeError",
                "ValueError", "NameError", "SyntaxError"
            )
        ):
            print(f"✅ Success: model created → {MODEL_PATH}")
            success = True

            # -------------------------------------------------
            # 📊 Automatically run evaluation after success
            # -------------------------------------------------
            print("\n📈 Running evaluate_model.py...\n")
            eval_path = os.path.join(os.path.dirname(__file__), "evaluate_model.py")
            try:
                result = subprocess.run(
                    ["python3", eval_path],
                    text=True,
                    capture_output=True,
                    check=True
                )
                print(result.stdout)
                if result.stderr.strip():
                    print("Evaluation warnings/errors:\n", result.stderr)
            except FileNotFoundError:
                print(f"⚠️  Evaluation script not found at {eval_path}")
            except subprocess.CalledProcessError as e:
                print("⚠️  Evaluation script failed:")
                print(e.stderr)

        else:
            print("⚠️  Training failed — retrying with error feedback...")
            last_error = stderr or stdout
            attempt += 1
            time.sleep(3)

    if not success:
        print("❌ All attempts failed. Check ollama_server.log or stderr above.")

if __name__ == "__main__":
    main()
