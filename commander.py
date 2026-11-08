"""Commander agent and LangGraph workflow (Task 5).

The Commander is the entry point and the conditional router of the graph. After
every attacker it reads the attack history in the shared state and decides who
runs next:

  1. First pass.  Every attacker in plan_queue runs once, in order.
  2. Further runs.  When the queue is empty, the Commander looks for attackers
     that still have untested techniques (techniques that have not succeeded
     yet) and have not used up their runs. It picks by priority:
        a. the attacker with the fewest runs, so no single attacker takes over,
        b. attackers whose last run produced a success come first, because their
           area of the target is already known to be weak,
        c. the attacker with the most severe finding breaks a tie.
     The attacker then retries only its untested techniques, with rewritten
     payloads.
  3. Stop.  The Commander ends the attack phase when no attacker has anything
     left to try, or when the round limit is reached. The round limit guards
     against uncontrolled execution.

Graph shape:
    commander --> attacker --> commander --> ... --> (report) --> END

Run this file directly to test the whole workflow:
    python commander.py            (mock target, no LLM)
    python commander.py ollama     (real target, LLM payloads, LLM judge)
"""
import sys
import time
from typing import Callable, Optional, Tuple

from langgraph.graph import END, StateGraph

from attackers import TECHNIQUES, PayloadGenerator, make_attacker
from config import ATTACKER_NAMES, MAX_ROUNDS, MAX_RUNS_PER_ATTACKER, SEVERITY_LEVELS
from judge import JudgeEvaluator
from state import AttackState, new_state
from targets import TargetAdapter, make_target, target_config

NO_FINDING = len(SEVERITY_LEVELS)   # sort value for an attacker without any success


# ---------------------------------------------------------------- history helpers
def _trials_of(history: list, name: str) -> list:
    return [t for t in history if t["agent"] == name]


def _runs_done(history: list, name: str) -> int:
    """Number of rounds in which the attacker sent at least one payload."""
    return len({t["round"] for t in _trials_of(history, name)})


def _last_run_won(history: list, name: str) -> bool:
    """Did the most recent run of the attacker produce a success?"""
    trials = _trials_of(history, name)
    if not trials:
        return False
    last_round = max(t["round"] for t in trials)
    return any(t["success"] for t in trials if t["round"] == last_round)


def _untested(history: list, name: str) -> list:
    """Techniques of the attacker that have not succeeded yet."""
    succeeded = {t["technique"] for t in _trials_of(history, name) if t["success"]}
    return [technique for technique in TECHNIQUES[name] if technique not in succeeded]


def _best_severity(history: list, name: str) -> int:
    ranks = [SEVERITY_LEVELS.index(t["severity"]) for t in _trials_of(history, name) if t["success"]]
    return min(ranks) if ranks else NO_FINDING


# ---------------------------------------------------------------- commander node
def commander_node(state: AttackState) -> dict:
    """Decide which node runs next and write the decision to the state."""
    if state["round"] >= state["max_rounds"]:
        return {"next_agent": "end"}

    # First pass: take the next attacker from the plan queue.
    queue = list(state["plan_queue"])
    if queue:
        return {"next_agent": queue.pop(0), "plan_queue": queue}

    # Further runs: only attackers with untested techniques and runs left.
    history = state["history"]
    candidates = []
    for name in ATTACKER_NAMES:
        runs = _runs_done(history, name)
        if runs < MAX_RUNS_PER_ATTACKER and _untested(history, name):
            promising = 0 if _last_run_won(history, name) else 1
            candidates.append((runs, promising, _best_severity(history, name), name))
    if not candidates:
        return {"next_agent": "end"}
    candidates.sort()
    return {"next_agent": candidates[0][3]}


def route(state: AttackState) -> str:
    """Conditional edge: the Commander's decision is the next node."""
    return state["next_agent"]


# ---------------------------------------------------------------- graph assembly
def build_graph(target: TargetAdapter, judge: JudgeEvaluator, generator: PayloadGenerator,
                report_node: Optional[Callable] = None):
    """Compile the LangGraph workflow. report_node is the Report Agent (optional)."""
    graph = StateGraph(AttackState)
    graph.add_node("commander", commander_node)
    for name in ATTACKER_NAMES:
        graph.add_node(name, make_attacker(name, target, judge, generator))

    routes = {name: name for name in ATTACKER_NAMES}
    if report_node is not None:
        graph.add_node("report", report_node)
        graph.add_edge("report", END)
        routes["end"] = "report"
    else:
        routes["end"] = END

    graph.set_entry_point("commander")
    graph.add_conditional_edges("commander", route, routes)
    for name in ATTACKER_NAMES:
        graph.add_edge(name, "commander")      # every attacker returns to the Commander
    return graph.compile()


def run_attack(target: TargetAdapter, judge: JudgeEvaluator, generator: PayloadGenerator,
               max_rounds: int = MAX_ROUNDS,
               report_node: Optional[Callable] = None,
               on_step: Optional[Callable] = None) -> Tuple[AttackState, list, float]:
    """Run the whole workflow once.

    on_step, if given, is called after every node as on_step(node, update, state)
    so that a caller can show live progress.

    Returns the final state, the list of nodes in the order they ran, and the
    elapsed time in seconds.
    """
    app = build_graph(target, judge, generator, report_node)
    state = new_state(target_config(), ATTACKER_NAMES, max_rounds)
    order = []
    started = time.time()
    # Each round uses two graph steps (commander and attacker), so allow plenty.
    for chunk in app.stream(dict(state), config={"recursion_limit": 100},
                            stream_mode="updates"):
        for node, update in chunk.items():
            order.append(node)
            state.update(update or {})
            if on_step is not None:
                on_step(node, update or {}, state)
    return state, order, time.time() - started


# ---------------------------------------------------------------- self test
if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "mock"
    live = kind == "ollama"
    final, order, seconds = run_attack(make_target(kind), JudgeEvaluator(use_llm=live),
                                       PayloadGenerator(use_llm=live))
    print("target:", kind, "  rounds:", final["round"], "  trials:", len(final["history"]),
          "  seconds: %.1f" % seconds)
    print("order :", " > ".join(n for n in order if n != "commander"))
    wins = [t for t in final["history"] if t["success"]]
    print("successful attacks:", len(wins))
    for t in wins:
        print("  round %d  %-16s %-24s %-6s %-8s" % (t["round"], t["agent"], t["technique"],
                                                    t["source"], t["severity"]))
