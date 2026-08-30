"""Continuous Learning & Performance Dashboard.

Reads run history and memory banks to display learning trajectories,
error reduction, pillar discovery, and episodic memory consolidation.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
RUN_HISTORY = JSON_DIR / "run_history.json"
MEMORY_FILE = JSON_DIR / "learning_memory.json"
STATE_FILE = JSON_DIR / "learning_state.json"


def show_dashboard():
    print("\n" + "=" * 80)
    print("      CONTINUOUS LEARNING & ADAPTATION DASHBOARD")
    print("=" * 80)

    # 1. State & Pillars
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        pillars = state.get("pillars", {})
        print(
            f"\n[Active Triangulation Pillars] ({len(pillars)} registered dimensions):"
        )
        for p, query in pillars.items():
            print(f"  - {p:18}: {query[:65]}...")
    else:
        print("\n[Active Triangulation Pillars]: Default 4 pillars")

    # 2. Episodic Memory Consolidation
    if MEMORY_FILE.exists():
        memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        grounded_count = sum(1 for m in memory if m.get("is_grounded", True))
        pitfall_count = sum(1 for m in memory if m.get("feedback_type") == "pitfall")
        print("\n[Episodic Memory Bank]:")
        print(f"  Total Indexed Cases    : {len(memory)}")
        print(f"  Verified Precedents    : {grounded_count}")
        print(f"  Pitfall Anti-Patterns  : {pitfall_count}")
    else:
        print("\n[Episodic Memory Bank]: Empty (0 records)")

    # 3. Cross-Run Learning Progression
    if RUN_HISTORY.exists():
        history = json.loads(RUN_HISTORY.read_text(encoding="utf-8"))
        print(f"\n[Execution & Learning History] ({len(history)} runs recorded):")
        print(
            f"{'Run ID':<10} {'Timestamp':<20} {'Events':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Validity':<10} {'Mem Size':<10}"
        )
        print("-" * 88)
        for r in history:
            print(
                f"{r.get('run_id', 'N/A'):<10} "
                f"{r.get('timestamp', 'N/A'):<20} "
                f"{r.get('events_audited', 0):<8} "
                f"{r.get('f1', 0.0):<8.4f} "
                f"{r.get('precision', 0.0):<10.4f} "
                f"{r.get('recall', 0.0):<8.4f} "
                f"{r.get('citation_validity', 1.0):<10.4f} "
                f"{r.get('memory_precedents_stored', 0):<10}"
            )
    else:
        print("\n[Execution History]: No previous runs recorded.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    show_dashboard()
