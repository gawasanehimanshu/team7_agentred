# AgentRed: Automated Multi-Agent Red Teaming Framework for LLM Applications

CSC 244, California State University, Sacramento. Team 7: Himanshu Gawasane and Snehith Sheri.

AgentRed uses several LLM agents to attack a chatbot automatically, judge the answers and write a security report. The target is a fake customer support bot (AcmeShop SupportBot) that hides a secret discount code in its system prompt. Everything runs locally with LangGraph and llama3.2:3b through Ollama.

## Architecture

Commander -> one of four attackers (prompt injection, jailbreak, role confusion, exfiltration) -> back to the Commander -> ... -> Report Agent.
All agents share one state object (state.py).

| File | Purpose | Owner |
|---|---|---|
| state.py | Shared AttackState and Trial record | Himanshu |
| commander.py | Commander agent and LangGraph workflow | Himanshu |
| config.py | Model name, temperatures, limits | Himanshu |
| targets.py | Target chatbot (mock and Ollama) | Himanshu |
| attackers.py | Four attackers, nine techniques, payload modes | Himanshu |
| main.py | Entry point, runs one full attack | Himanshu |
| judge.py | Hybrid judge (rules, LLM, fallback) | Snehith |
| report.py | Report Agent (JSON and Markdown reports) | Snehith |
| evaluate.py | Evaluation harness | Snehith |
| manual_test.py | Human red-team baseline | Snehith |

## Install (macOS)

1. Install Ollama from https://ollama.com and pull the models:
   `ollama pull llama3.2:3b` and `ollama pull mxbai-embed-large`
2. Create the environment and install packages:
   `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Run

- Full attack with the local model (Ollama must be running): `python main.py`
- Quick test without any model (about 1 second): `python main.py --target mock --mode seed`
- Options: `--target {mock,ollama}`, `--mode {seed,free,mutate,adaptive}` (default adaptive), `--rounds N` (default 8)
- Evaluation (about 15 minutes): `caffeinate -i python evaluate.py`
- Human baseline: `python manual_test.py`

Reports are written to the `reports/` folder.

## Results (average of 3 runs)

| Setting | Findings | Techniques | False positives | Time |
|---|---|---|---|---|
| Mock, seed | 5.0 | 5 | 0% | 8.0 s |
| Ollama, seed | 4.0 | 4 | 0% | 55.8 s |
| Ollama, free | 0.0 | 0 | 0% | 148.0 s |
| Ollama, mutate | 2.7 | 5 | 20% | 140.7 s |
| Ollama, adaptive | 4.0 | 6 | 0% | 93.2 s |

A human tester found 6 successful attacks (5 techniques) in 449.5 s. Limits: only 3 runs per setting, a small easy target, and ground truth that overlaps the judge rules.

For authorised testing of your own applications only.

## Running on Windows (Command Prompt)

    py -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py --target mock

The caffeinate tip is macOS only. On Windows, keep the laptop plugged in and awake while running evaluate.py.