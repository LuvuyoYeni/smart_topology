import json
import os

import numpy as np

from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome

NUM_AGENTS = 6
N_TRIALS = 30

if __name__ == "__main__":
    np.random.seed(42)

    # Ring: each agent connected to its two cyclic neighbours
    ring_mat = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    for i in range(NUM_AGENTS):
        ring_mat[i][(i + 1) % NUM_AGENTS] = True
        ring_mat[(i + 1) % NUM_AGENTS][i] = True
    ring = TopologyGenome(num_agents=NUM_AGENTS, adjacency_matrix=ring_mat)

    # Fully connected: every agent connected to every other agent
    fc_mat = np.ones((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    np.fill_diagonal(fc_mat, False)
    fc = TopologyGenome(num_agents=NUM_AGENTS, adjacency_matrix=fc_mat)

    os.makedirs("results", exist_ok=True)

    for name, genome, out_path in [
        ("Ring", ring, "results/ring_scores.json"),
        ("Fully connected", fc, "results/fc_scores.json"),
    ]:
        print(f"Evaluating {name} ({N_TRIALS} trials)...")
        scores = []
        for t in range(N_TRIALS):
            ev = FitnessEvaluator(num_trials=1, agent_mode="cooperative")
            scores.append(ev.evaluate(genome))

        with open(out_path, "w") as f:
            json.dump(scores, f)

        print(f"  mean={float(np.mean(scores)):.4f}  "
              f"std={float(np.std(scores)):.4f}  -> {out_path}")
