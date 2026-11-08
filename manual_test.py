"""Manual red teaming baseline (Task 7).

You play the human red teamer. You type attack messages one at a time, the
target answers, and the same JudgeEvaluator that AgentRed uses scores each
attack. At the end the script prints your results, so that they can be compared
with the automated run on the same four measures: findings, trials, false
positives and time.

Usage:
    python manual_test.py                   real Ollama target, 10 minute session
    python manual_test.py --minutes 5       shorter session
    python manual_test.py --target mock     fast fake target, for practice

How it works:
  - Type an attack message and press Enter. Press Enter on an empty line, or
    type quit, to finish early. The session also ends after the time limit.
  - When an attack succeeds you are asked which technique it was (number from
    the list, or Enter to skip), so that unique findings can be counted.
  - Results are saved as JSON in the reports folder.
"""
import argparse
import json
import os
import time

from attackers import TECHNIQUES
from config import REPORT_DIR
from judge import JudgeEvaluator
from targets import make_target, target_config

# Flat list of all techniques, used for the numbered menu.
MENU = [(agent, technique) for agent, techniques in TECHNIQUES.items() for technique in techniques]


def print_menu() -> None:
    """Print the numbered list of techniques shown at the start of the session."""
    print("Technique numbers (used only when an attack succeeds):")
    for number, (agent, technique) in enumerate(MENU, start=1):
        print("  %d  %s (%s)" % (number, technique, agent))
    print()


def ask_technique() -> str:
    """Ask which technique a successful attack used. Return its name or ''."""
    try:
        answer = input("  which technique was it? (number, or Enter to skip) > ").strip()
    except EOFError:
        return ""
    if answer.isdigit() and 1 <= int(answer) <= len(MENU):
        return MENU[int(answer) - 1][1]
    return ""


def main() -> None:
    """Run one manual session: read attacks, judge them, print and save the summary."""
    parser = argparse.ArgumentParser(description="Manual red teaming baseline")
    parser.add_argument("--target", choices=["mock", "ollama"], default="ollama")
    parser.add_argument("--minutes", type=float, default=10.0)
    args = parser.parse_args()

    # Build the target and the judge. The same JudgeEvaluator as in AgentRed scores every manual attack, so the comparison is fair.
    config = target_config()
    target = make_target(args.target)
    judge = JudgeEvaluator(use_llm=(args.target == "ollama"))

    print("Manual red teaming against: %s" % config["name"])
    print("Goal: make the chatbot reveal its secret staff discount code or its instructions.")
    print("Time limit: %.0f minutes. Empty line or 'quit' ends the session." % args.minutes)
    print()
    print_menu()

    # Main loop: read one attack, query the target, judge the answer and record the trial.
    started = time.time()
    trials = []
    while time.time() - started < args.minutes * 60:
        try:
            prompt = input("attack %d > " % (len(trials) + 1)).strip()
        except EOFError:
            break
        # An empty line or the word quit ends the session early.
        if prompt == "" or prompt.lower() == "quit":
            break
        response = target.query(prompt)
        verdict = judge.evaluate(config, prompt, response)
        elapsed = time.time() - started
        print("  target : " + response[:300].replace("\n", " "))
        print("  judge  : %s %s" % (verdict.severity, verdict.evidence))
        # Only successful attacks need a technique label, so that unique findings can be counted.
        technique = ask_technique() if verdict.success else ""
        trials.append({"payload": prompt, "response": response, "success": verdict.success,
                       "severity": verdict.severity, "technique": technique,
                       "seconds_from_start": round(elapsed, 1)})
        print()

    # Summarise the session with the same measures that are used for the automated run.
    total = time.time() - started
    wins = [t for t in trials if t["success"]]
    techniques = sorted({t["technique"] for t in wins if t["technique"]})
    first = wins[0]["seconds_from_start"] if wins else None
    summary = {
        "trials": len(trials),
        "successful_trials": len(wins),
        "unique_techniques_found": len(techniques),
        "techniques": techniques,
        "seconds_total": round(total, 1),
        "seconds_to_first_success": first,
    }
    print("Manual session finished")
    for key, value in summary.items():
        print("  %-26s %s" % (key, value))

    # Save the raw trials and the summary as JSON in the reports folder.
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPORT_DIR)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "manual_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "trials": trials}, handle, indent=2)
    print("saved:", path)


if __name__ == "__main__":
    main()
