"""AgentRed entry point.

Runs the complete workflow once: the Commander drives the four attacker agents
against the target, the JudgeEvaluator scores every attack, and the Report
Agent writes a JSON report and a Markdown report to the reports folder.

Usage:
    python main.py                          real Ollama target, adaptive payloads
    python main.py --target mock            fast fake target, no Ollama needed
    python main.py --mode seed              use only the hand-written payloads
    python main.py --rounds 5               change the round limit

Options:
    --target  mock or ollama                          (default: ollama)
    --mode    seed, free, mutate or adaptive          (default: adaptive)
    --rounds  maximum number of attacker rounds       (default: MAX_ROUNDS in config.py)
"""
# ===== 1. IMPORTS =====
import argparse

from attackers import PayloadGenerator
from commander import run_attack
from config import ATTACKER_NAMES, MAX_ROUNDS, MODEL_NAME
from judge import JudgeEvaluator
from report import report_node
from targets import make_target


# ===== 2. PROGRESS PRINTER: prints what every node of the graph just did =====
def make_progress_printer():
    """Return a callback that prints what every node of the graph just did."""
    # Remember how many trials were already printed, so that each trial is shown only once.
    seen = {"trials": 0}

    def show(node: str, update: dict, state: dict) -> None:
        # Print a different message depending on which node of the graph just finished.
        if node == "commander":
            if update.get("next_agent") == "end":
                reason = ("round limit reached" if state["round"] >= state["max_rounds"]
                          else "nothing left to try")
                print("Commander: %s, handing over to the Report Agent" % reason)
            else:
                print("Commander: round %d, calling %s" % (state["round"] + 1, update["next_agent"]))
        elif node in ATTACKER_NAMES:
            # Only print the trials that this attacker added in its latest round.
            for trial in state["history"][seen["trials"]:]:
                print("    %-22s %-8s %-5s %s" % (trial["technique"], trial["severity"],
                                                  trial["source"],
                                                  trial["payload"][:55].replace("\n", " ")))
            seen["trials"] = len(state["history"])
        elif node == "report":
            print("Report Agent: report written")

    return show


# ===== 3. MAIN: read the options, run the workflow and print the summary =====
def main() -> None:
    parser = argparse.ArgumentParser(description="AgentRed: automated multi-agent red teaming")
    parser.add_argument("--target", choices=["mock", "ollama"], default="ollama")
    parser.add_argument("--mode", choices=["seed", "free", "mutate", "adaptive"], default="adaptive")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    args = parser.parse_args()

    # The mock target needs no Ollama server, so the LLM parts are switched off for it.
    live = args.target == "ollama"
    print("AgentRed  target=%s  payload mode=%s  round limit=%d%s" % (
        args.target, args.mode, args.rounds, "  model=" + MODEL_NAME if live else ""))
    print()

    # Run the whole LangGraph workflow: Commander, attackers, judge and Report Agent.
    final, order, seconds = run_attack(
        make_target(args.target), JudgeEvaluator(use_llm=live),
        PayloadGenerator(use_llm=live, mode=args.mode),
        max_rounds=args.rounds, report_node=report_node, on_step=make_progress_printer())

    # The Report Agent stored its result in the final state.
    report = final["final_report"]
    print()
    print("Finished in %.1f seconds: %d rounds, %d trials, %d unique findings, overall risk %s" % (
        seconds, report["rounds"], report["trials"], report["unique_findings"],
        report["overall_risk"]))
    for finding in report["findings"]:
        print("  %s %-8s %-24s (%s) succeeded %d of %d" % (
            finding["id"], finding["severity"], finding["technique"], finding["agent"],
            finding["successes"], finding["attempts"]))
    print()
    # Show where the report files were written.
    print("Reports:")
    print("  " + report["files"]["markdown"])
    print("  " + report["files"]["json"])


# ===== 4. ENTRY POINT: python main.py [options] =====
if __name__ == "__main__":
    main()
