import json
import os

# Load configuration from config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")

with open(config_path, "r") as f:
    _config = json.load(f)

# Dynamically export all configuration variables to namespace
for key, value in _config.items():
    if key != "PATH_TEMPLATE":  # Handle PATH_TEMPLATE separately
        globals()[key] = value

# Build PATH from template
PATH_TEMPLATE = _config.get("PATH_TEMPLATE", "results/{MODE}-FL-{FL}-{NUM_CLIENTS}-clients-{NUM_ATCKS}-atk-{ROUNDS}-rounds-{EPOCHS}-epochs-{LEARNING_RATE}-lr-{DATA_GROUPS}-groups-llm-{LLM}")

# Create PATH using current global variables
PATH = PATH_TEMPLATE.format(**{k: v for k, v in globals().items() if not k.startswith("_")})

# Make PATH absolute if it's relative
if not os.path.isabs(PATH):
    PATH = os.path.join(os.path.dirname(__file__), PATH)

globals()["PATH"] = PATH

# Calculate SIZE_ROUND
SIZE_ROUND = int(BATCH_ROUND * BATCH_SIZE * NUM_CLIENTS)
globals()["SIZE_ROUND"] = SIZE_ROUND

print(f"Configuration loaded from {config_path}")
print(f"Results path: {PATH}")
