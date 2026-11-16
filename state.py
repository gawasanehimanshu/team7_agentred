"""Shared state for AgentRed (Task 2).

AttackState is the single object that every node in the LangGraph workflow
reads and writes. It is the only channel between agents, so a finding made by
one attacker can change what the Commander and the other attackers do next.
"""
# ===== 1. IMPORTS =====
from typing import Dict, List, Optional, TypedDict

from config import MAX_ROUNDS


# ===== 2. TRIAL: one attack attempt and its judged result =====
class Trial(TypedDict):
    """One attack attempt and its judged result."""

    agent: str        # attacker that sent the payload
    round: int        # round in which the payload was sent (starts at 1)
    technique: str    # short name of the attack technique used
    source: str       # where the payload came from: llm or seed
    payload: str      # text sent to the target
    response: str     # text the target answered with
    success: bool     # judge verdict: did the attack work
    severity: str     # CRITICAL, HIGH, MEDIUM, LOW or NONE
    evidence: str     # judge explanation of the verdict


# ===== 3. ATTACK STATE: the shared object that every node reads and writes =====
class AttackState(TypedDict):
    target_config: Dict[str, str]     # name and system prompt of the target (input)
    plan_queue: List[str]             # attackers still to run, ordered by the Commander
    history: List[Trial]              # every trial so far
    insights: List[str]               # facts discovered by one agent for the others
    final_report: Optional[dict]      # written by the Report Agent at the end
    round: int                        # attacker rounds completed
    max_rounds: int                   # round limit enforced by the Commander
    next_agent: str                   # routing decision made by the Commander


# ===== 4. NEW STATE: build a fresh state for one run =====
def new_state(target_config: Dict[str, str], plan_queue: List[str],
              max_rounds: int = MAX_ROUNDS) -> AttackState:
    """Build a fresh state for one run against one target."""
    return {
        "target_config": target_config,
        "plan_queue": list(plan_queue),
        "history": [],
        "insights": [],
        "final_report": None,
        "round": 0,
        "max_rounds": max_rounds,
        "next_agent": "",
    }
