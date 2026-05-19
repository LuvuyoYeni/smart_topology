"""
verify_asymmetry.py
-------------------
Diagnostic: confirms that topology structure now matters for task completion.

Each trial uses the same random seed for all three topologies, ensuring a
fair environment-matched comparison.

Expected ordering after the partial-observability refactor:
  empty.task_score   < ring.task_score   <= full.task_score
  empty.message_count = 0  (no edges -> no messages)
"""

from __future__ import annotations

import numpy as np

from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome

NUM_AGENTS = 6
N_REPEATS  = 30
BASE_SEED  = 0

evaluator = FitnessEvaluator(num_trials=1, agent_mode="cooperative")

# ── Build topologies ──────────────────────────────────────────────────────────

empty_mat = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
empty = TopologyGenome(NUM_AGENTS, empty_mat)

ring_mat = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
for i in range(NUM_AGENTS):
    ring_mat[i][(i + 1) % NUM_AGENTS] = True
    ring_mat[(i + 1) % NUM_AGENTS][i] = True
ring = TopologyGenome(NUM_AGENTS, ring_mat)

full_mat = np.ones((NUM_AGENTS, NUM_AGENTS), dtype=bool)
np.fill_diagonal(full_mat, False)
full = TopologyGenome(NUM_AGENTS, full_mat)

topologies = [("Empty", empty), ("Ring", ring), ("Full", full)]

results: dict[str, dict] = {
    name: {"task_scores": [], "msg_counts": []} for name, _ in topologies
}

# ── Each trial: same seed for all three topologies ───────────────────────────
for trial in range(N_REPEATS):
    for name, genome in topologies:
        np.random.seed(BASE_SEED + trial)
        r = evaluator.run_simulation(genome)
        results[name]["task_scores"].append(r.task_score)
        results[name]["msg_counts"].append(r.message_count)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\n=== Asymmetry Verification ({N_REPEATS} matched trials, base_seed={BASE_SEED}) ===\n")
print(f"{'Topology':<10} {'mean TaskScore':>15}  {'mean MsgCount':>14}")
print("─" * 45)
for name, _ in topologies:
    ts = float(np.mean(results[name]["task_scores"]))
    mc = float(np.mean(results[name]["msg_counts"]))
    print(f"{name:<10} {ts:>15.3f}  {mc:>14.1f}")

empty_ts = float(np.mean(results["Empty"]["task_scores"]))
ring_ts  = float(np.mean(results["Ring"]["task_scores"]))
full_ts  = float(np.mean(results["Full"]["task_scores"]))
empty_mc = float(np.mean(results["Empty"]["msg_counts"]))

print(f"\n  empty < ring:   {empty_ts:.3f} < {ring_ts:.3f}  ->  {empty_ts < ring_ts}")
print(f"  ring  <= full:  {ring_ts:.3f} <= {full_ts:.3f}  ->  {ring_ts <= full_ts}")
print(f"  empty msg = 0:  {empty_mc:.1f}  ->  {empty_mc == 0.0}")

assert empty_ts < ring_ts, (
    f"FAIL: empty ({empty_ts:.3f}) must be < ring ({ring_ts:.3f})"
)
assert ring_ts <= full_ts, (
    f"FAIL: ring ({ring_ts:.3f}) must be <= full ({full_ts:.3f})"
)
assert empty_mc == 0.0, (
    f"FAIL: empty message_count must be 0 (got {empty_mc:.1f})"
)

print("\n  All asymmetry checks passed.")
