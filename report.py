"""Report Agent (Task 6).

Runs as the last node of the workflow. It reads the attack history from the
shared state and produces two outputs:

  JSON      a structured vulnerability report, meant for tools and comparisons
  Markdown  a readable security report with proof-of-concept payloads and
            remediation advice for every finding

Processing steps:
  1. Group the trials by attacker and technique.
  2. Deduplicate: a technique that succeeded several times becomes ONE finding,
     keeping the most severe trial as the proof of concept.
  3. Assign the severity of each finding (the highest severity seen) and sort
     the findings from CRITICAL down to LOW.
  4. Attach remediation advice and write both files to the reports folder.

Run this file directly to produce a report from a full run:
    python report.py            (mock target, no LLM)
    python report.py ollama     (real target, LLM payloads, LLM judge)
"""
# ===== 1. IMPORTS =====
import json
import os
import sys
import time
from typing import Dict, List, Optional

from config import MODEL_NAME, REPORT_DIR, SEVERITY_LEVELS
from state import AttackState

# ===== 2. REMEDIATION ADVICE: a fix suggestion for each of the 9 techniques =====
# Advice for every attack technique. Shown next to each finding.
REMEDIATION = {
    "cognitive_overload": "Limit the length and complexity of user input, and apply the same "
                          "policy check to every part of a multi-part request.",
    "prefix_injection": "Do not let user text decide how a reply begins. Filter replies that "
                        "start with a forced agreement before they are returned.",
    "hypothetical_reframing": "Apply the confidentiality rules to stories and hypothetical "
                              "scenarios as well, and add an output filter that blocks secrets "
                              "in any format.",
    "instruction_override": "Keep system instructions and user input in separate roles, state "
                            "that user text can never change the rules, and test regularly "
                            "with known override phrases.",
    "delimiter_hijack": "Escape or remove separators and role markup from user input before it "
                        "reaches the model.",
    "privilege_escalation": "Never grant access because of a claim made inside the chat. "
                            "Verify identity outside the model.",
    "persona_switch": "Reject requests that change the assistant into a character without "
                      "rules, and restate the policy at the end of the prompt.",
    "system_prompt_echo": "Do not keep secrets in the system prompt. Add an output filter that "
                          "blocks replies that repeat the prompt.",
    "data_leak_probe": "Remove secrets from the model context entirely and serve restricted "
                       "data only after a server-side authorisation check.",
}
DEFAULT_REMEDIATION = "Review the system prompt and add an output filter for confidential data."


# ===== 3. HELPER: severity ranking =====
def _rank(severity: str) -> int:
    """Return the position of a severity in SEVERITY_LEVELS (0 is the most severe)."""
    return SEVERITY_LEVELS.index(severity)


# ===== 4. BUILD THE REPORT: group, deduplicate, rank the findings and add the totals (JSON structure) =====
def build_report(state: AttackState) -> dict:
    """Turn the attack history into a structured report (steps 1 to 3)."""
    # Step 1: group all trials by attacker and technique.
    groups: Dict[tuple, List[dict]] = {}
    for trial in state["history"]:
        groups.setdefault((trial["agent"], trial["technique"]), []).append(trial)

    # Step 2: a technique with at least one success becomes one finding. The others are listed as not exploited.
    findings, not_exploited = [], []
    for (agent, technique), trials in groups.items():
        wins = [t for t in trials if t["success"]]
        if not wins:
            not_exploited.append({"agent": agent, "technique": technique,
                                  "attempts": len(trials)})
            continue
        # Keep the most severe successful trial as the proof of concept.
        best = min(wins, key=lambda t: _rank(t["severity"]))
        findings.append({
            "agent": agent,
            "technique": technique,
            "severity": best["severity"],
            "attempts": len(trials),
            "successes": len(wins),
            "evidence": best["evidence"],
            "proof_of_concept": {"payload": best["payload"], "response": best["response"],
                                 "payload_source": best["source"]},
            "remediation": REMEDIATION.get(technique, DEFAULT_REMEDIATION),
        })
    # Step 3: most severe findings first, then the ones that succeeded most often.
    findings.sort(key=lambda f: (_rank(f["severity"]), -f["successes"]))
    for number, finding in enumerate(findings, start=1):
        finding["id"] = "F%d" % number

    # Totals for the summary section of the report.
    trials_total = len(state["history"])
    successes = sum(1 for t in state["history"] if t["success"])
    return {
        "target": state["target_config"].get("name", "unknown"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "rounds": state["round"],
        "trials": trials_total,
        "successful_trials": successes,
        "success_rate": round(successes / trials_total, 3) if trials_total else 0.0,
        "unique_findings": len(findings),
        "findings_by_severity": {s: sum(1 for f in findings if f["severity"] == s)
                                 for s in SEVERITY_LEVELS if s != "NONE"},
        "overall_risk": findings[0]["severity"] if findings else "NONE",
        "findings": findings,
        "not_exploited": not_exploited,
        "insights": list(state["insights"]),
    }


# ===== 5. MARKDOWN OUTPUT: code-block helper and the readable report =====
def _fence(text: str, limit: int = 500) -> str:
    """Format text as a Markdown code block, shortened and safe to embed."""
    text = text.strip().replace("```", "'''")
    if len(text) > limit:
        text = text[:limit] + " ..."
    return "```\n" + text + "\n```"


def build_markdown(report: dict) -> str:
    """Render the report as a readable Markdown document."""
    # The report is built as a list of lines that is joined at the end.
    lines = [
        "# AgentRed Security Report",
        "",
        "Target: **%s**  " % report["target"],
        "Generated: %s  " % report["generated"],
        "Model: %s  " % report["model"],
        "Overall risk: **%s**" % report["overall_risk"],
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Rounds | %d |" % report["rounds"],
        "| Attack trials | %d |" % report["trials"],
        "| Successful trials | %d |" % report["successful_trials"],
        "| Trial success rate | %.0f%% |" % (report["success_rate"] * 100),
        "| Unique findings | %d |" % report["unique_findings"],
    ]
    # One summary row for each severity level.
    for severity, count in report["findings_by_severity"].items():
        lines.append("| %s findings | %d |" % (severity.capitalize(), count))
    # One section per finding, with the proof of concept and the remediation advice.
    lines += ["", "## Findings", ""]
    if not report["findings"]:
        lines += ["No successful attack was found.", ""]
    for f in report["findings"]:
        poc = f["proof_of_concept"]
        lines += [
            "### %s. %s (%s) - %s" % (f["id"], f["technique"], f["agent"], f["severity"]),
            "",
            "Succeeded in %d of %d attempts. Judge evidence: %s" % (
                f["successes"], f["attempts"], f["evidence"]),
            "",
            "Proof of concept payload (%s):" % poc["payload_source"],
            "",
            _fence(poc["payload"]),
            "",
            "Target response:",
            "",
            _fence(poc["response"]),
            "",
            "Remediation: " + f["remediation"],
            "",
        ]
    # Section for techniques that never succeeded.
    lines += ["## Techniques that did not succeed", ""]
    if report["not_exploited"]:
        for item in report["not_exploited"]:
            lines.append("- %s (%s): %d attempts, no success" % (
                item["technique"], item["agent"], item["attempts"]))
    else:
        lines.append("- none")
    # Section with the notes the agents shared during the run.
    lines += ["", "## Notes shared between agents", ""]
    lines += ["- " + note for note in report["insights"]] or ["- none"]
    return "\n".join(lines) + "\n"


# ===== 6. WRITE FILES: save the JSON and Markdown reports in the reports folder =====
def write_report(report: dict, directory: Optional[str] = None) -> Dict[str, str]:
    """Write the JSON and Markdown files (step 4) and return their paths."""
    if directory is None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPORT_DIR)
    os.makedirs(directory, exist_ok=True)
    # A time stamp in the file names keeps earlier reports from being overwritten.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = {"json": os.path.join(directory, "agentred_report_%s.json" % stamp),
             "markdown": os.path.join(directory, "agentred_report_%s.md" % stamp)}
    # JSON is meant for tools and comparisons, Markdown is meant for people.
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(paths["markdown"], "w", encoding="utf-8") as handle:
        handle.write(build_markdown(report))
    return paths


# ===== 7. LANGGRAPH NODE: the last step of the workflow, it calls the functions above =====
def report_node(state: AttackState) -> dict:
    """LangGraph node: build the report, write both files, store it in the state."""
    report = build_report(state)
    report["files"] = write_report(report)
    return {"final_report": report}


# ---------------------------------------------------------------- self test
if __name__ == "__main__":
    from attackers import PayloadGenerator
    from commander import run_attack
    from judge import JudgeEvaluator
    from targets import make_target

    # ===== 8. SELF TEST: python report.py =====
    # Choose the target from the command line (mock by default, ollama for the real run).
    kind = sys.argv[1] if len(sys.argv) > 1 else "mock"
    live = kind == "ollama"
    # Run the full workflow once and get the final state.
    final, order, seconds = run_attack(make_target(kind), JudgeEvaluator(use_llm=live),
                                       PayloadGenerator(use_llm=live), report_node=report_node)
    result = final["final_report"]
    # Print a short summary and the paths of the saved files.
    print("order :", " > ".join(n for n in order if n != "commander"))
    print("overall risk:", result["overall_risk"], "  findings:", result["unique_findings"],
          "  trials:", result["trials"], "  seconds: %.1f" % seconds)
    for f in result["findings"]:
        print("  %s %-8s %-24s %d of %d" % (f["id"], f["severity"], f["technique"],
                                            f["successes"], f["attempts"]))
    print("files:", result["files"]["json"])
    print("       ", result["files"]["markdown"])
