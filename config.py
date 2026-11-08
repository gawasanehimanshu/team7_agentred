"""Central settings for AgentRed.

Every tunable value lives here so that the other modules never hard-code
model names, limits or paths. Change a value once and the whole system follows.
"""

# ---------------------------------------------------------------- LLM backend
# Model served by the local Ollama server. Used by the target, the judge and
# the attacker payload generator.
MODEL_NAME = "llama3.2:3b"

# Address of the local Ollama server (default Ollama port).
OLLAMA_BASE_URL = "http://localhost:11434"

# Sampling temperature for attacker payload generation. A higher value gives
# more varied attack payloads.
ATTACKER_TEMPERATURE = 0.8

# Sampling temperature for the judge. Zero keeps verdicts as repeatable as
# possible.
JUDGE_TEMPERATURE = 0.0

# Sampling temperature for the Ollama target application.
TARGET_TEMPERATURE = 0.2

# ---------------------------------------------------------------- workflow
# Guard against uncontrolled execution: the Commander stops after this many
# attacker rounds even if it still has ideas.
MAX_ROUNDS = 8

# How many times the Commander may call the same attacker in one run.
MAX_RUNS_PER_ATTACKER = 3

# Number of payloads an attacker sends for each of its techniques in one round.
PAYLOADS_PER_TECHNIQUE = 1

# How many times an attacker retries when the model fails to write a payload,
# before it falls back to a hand-written seed payload.
GENERATOR_RETRIES = 2

# How many times the judge retries when the model returns an invalid verdict.
JUDGE_RETRIES = 3

# ---------------------------------------------------------------- vocabulary
# Names of the attacker agents. The Commander and the report use these names.
ATTACKER_NAMES = ["prompt_injection", "jailbreak", "role_confusion", "exfiltration"]

# Severity levels from most to least serious.
SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]

# ---------------------------------------------------------------- output
# Folder where the Report Agent writes the JSON and Markdown reports.
REPORT_DIR = "reports"
