import os
import random
import statistics

import numpy as np

from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome

N = 1000
EDGE_PROB = 0.3
NUM_AGENTS = 6

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    evaluator = FitnessEvaluator(num_trials=5, agent_mode="cooperative")

    best_fitness = -float("inf")
    best_genome = None
    scores = []

    for i in range(N):
        genome = TopologyGenome.random(num_agents=NUM_AGENTS, edge_probability=EDGE_PROB)
        fitness = evaluator.evaluate(genome)
        scores.append(fitness)
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N}  current best: {best_fitness:.4f}")

    os.makedirs("results", exist_ok=True)
    np.savez("results/random_best_coop.npz",
             adjacency_matrix=best_genome.adjacency_matrix)

    print(f"\nRandom baseline (N={N}, edge_prob={EDGE_PROB}):")
    print(f"  best   : {best_fitness:.4f}")
    print(f"  mean   : {statistics.mean(scores):.4f}")
    print(f"  stdev  : {statistics.stdev(scores):.4f}")
    print(f"  median : {statistics.median(scores):.4f}")
    print(f"\n  Saved: results/random_best_coop.npz")
