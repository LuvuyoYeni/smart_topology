import json
import os
import random
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from evolution_engine import EvolutionEngine
from topology_genome import TopologyGenome
from fitness_evaluator import FitnessEvaluator
from strategic_evolution_fix import ConnectivityEnforcedEvaluator

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def visualise_topology(genome: TopologyGenome) -> None:
    G = nx.Graph()
    G.add_nodes_from(range(genome.num_agents))

    rows, cols = np.where(np.triu(genome.adjacency_matrix, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        G.add_edge(int(i), int(j))

    fig, ax = plt.subplots(figsize=(7, 6))

    pos = nx.spring_layout(G, seed=42)

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_colours = [degrees[n] / max(max_deg, 1) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colours,
        cmap=plt.cm.Blues,
        node_size=700,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_color="black", font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax)

    connected_str = "connected" if genome.is_connected() else "DISCONNECTED"
    ax.set_title(
        f"Best Topology  |  agents={genome.num_agents}"
        f"  density={genome.graph_density:.2f}"
        f"  fitness={genome.fitness_score:.4f}"
        f"  ({connected_str})",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def compare_topologies_extended(
    cooperative_genome: TopologyGenome, strategic_genome: TopologyGenome
) -> None:
    num_agents = cooperative_genome.num_agents

    full_mat = np.ones((num_agents, num_agents), dtype=bool)
    np.fill_diagonal(full_mat, False)

    ring_mat = np.zeros((num_agents, num_agents), dtype=bool)
    for i in range(num_agents):
        ring_mat[i][(i + 1) % num_agents] = True
        ring_mat[(i + 1) % num_agents][i] = True

    star_mat = np.zeros((num_agents, num_agents), dtype=bool)
    for i in range(1, num_agents):
        star_mat[0][i] = True
        star_mat[i][0] = True

    topologies: dict[str, TopologyGenome] = {
        "Evol-Coop": cooperative_genome,
        "Evol-Strat": strategic_genome,
        "Full":       TopologyGenome(num_agents=num_agents, adjacency_matrix=full_mat),
        "Ring":       TopologyGenome(num_agents=num_agents, adjacency_matrix=ring_mat),
        "Star":       TopologyGenome(num_agents=num_agents, adjacency_matrix=star_mat),
    }
    modes = ["cooperative", "mixed", "strategic"]

    print("\n=== Topology x Agent Mode Comparison (10 trials each) ===\n")
    col_w = 14
    print(f"  {'Topology':<13}", end="")
    for mode in modes:
        print(f"  {mode.capitalize():>{col_w}}", end="")
    print()
    print("  " + "-" * (13 + (col_w + 2) * len(modes)))

    all_results = {}
    for name, genome in topologies.items():
        all_results[name] = {}
        print(f"  {name:<13}", end="")
        for mode in modes:
            evaluator = FitnessEvaluator(num_trials=10, agent_mode=mode)
            fitness = evaluator.evaluate(genome)
            all_results[name][mode] = fitness
            print(f"  {fitness:>{col_w}.4f}", end="")
        print()

    print("\n  Key findings:")
    for mode in modes:
        best = max(topologies.keys(), key=lambda t: all_results[t][mode])
        print(f"    {mode.capitalize():<14} best topology: {best} "
              f"({all_results[best][mode]:.4f})")

    print()
    for evolved_name in ("Evol-Coop", "Evol-Strat"):
        ranks = {}
        for mode in modes:
            ordered = sorted(topologies.keys(),
                             key=lambda t: all_results[t][mode], reverse=True)
            ranks[mode] = ordered.index(evolved_name) + 1
        print(f"  {evolved_name} ranks: {ranks} (1 = best)")


def main() -> None:
    print("=" * 60)
    print("  EvoComm — Evolving Multi-Agent Communication Topologies")
    print("=" * 60)

    def run_evolution(agent_mode: str) -> tuple[TopologyGenome, EvolutionEngine]:
        eng = EvolutionEngine(agent_mode=agent_mode)
        if agent_mode == "strategic":
            eng.fitness_evaluator = ConnectivityEnforcedEvaluator(
                num_trials=5, agent_mode="strategic"
            )
        print(f"\n  agent_mode       : {agent_mode}")
        print(f"  population_size  : {eng.population_size}")
        print(f"  num_generations  : {eng.num_generations}")
        print(f"  mutation_rate    : {eng.mutation_rate}")
        print(f"  num_agents       : {eng.num_agents}")
        print()
        best = eng.run()
        print(f"\n  Best ({agent_mode}): {best}")
        eng._plot_progress()
        visualise_topology(best)
        return best, eng

    print("\n" + "=" * 60)
    print("  Run 1 — Evolving topology with cooperative agents")
    print("=" * 60)
    best_cooperative, eng_coop = run_evolution("cooperative")
    coop_fitness = best_cooperative.fitness_score

    print("\n" + "=" * 60)
    print("  Run 2 — Evolving topology with strategic agents")
    print("=" * 60)
    best_strategic, eng_strat = run_evolution("strategic")
    strat_fitness = best_strategic.fitness_score

    print("\n" + "=" * 60)
    print("  Structural comparison of evolved topologies")
    print("=" * 60)
    print(f"\n  {'Metric':<22} {'Cooperative':>14}  {'Strategic':>14}")
    print(f"  {'-' * 52}")
    print(f"  {'Fitness':<22} {best_cooperative.fitness_score:>14.4f}  "
          f"{best_strategic.fitness_score:>14.4f}")
    print(f"  {'Density':<22} {best_cooperative.graph_density:>14.4f}  "
          f"{best_strategic.graph_density:>14.4f}")
    print(f"  {'Connected':<22} "
          f"{'yes' if best_cooperative.is_connected() else 'NO':>14}  "
          f"{'yes' if best_strategic.is_connected() else 'NO':>14}")

    def max_degree(genome: TopologyGenome) -> int:
        return max(int(genome.adjacency_matrix[i].sum())
                   for i in range(genome.num_agents))

    print(f"  {'Max degree (hub?)':<22} {max_degree(best_cooperative):>14}  "
          f"{max_degree(best_strategic):>14}")

    print()
    compare_topologies_extended(best_cooperative, best_strategic)

    os.makedirs("results", exist_ok=True)
    coop_path = "results/evolved_coop.npz"
    strat_path = "results/evolved_strat.npz"
    np.savez(coop_path, adjacency_matrix=best_cooperative.adjacency_matrix)
    np.savez(strat_path, adjacency_matrix=best_strategic.adjacency_matrix)

    log = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "population_size": eng_coop.population_size,
            "num_generations": eng_coop.num_generations,
            "mutation_rate": eng_coop.mutation_rate,
            "num_agents": eng_coop.num_agents,
            "seed": SEED,
        },
        "results": {
            "cooperative_best_fitness": coop_fitness,
            "cooperative_density": best_cooperative.graph_density,
            "strategic_best_fitness": strat_fitness,
            "strategic_density": best_strategic.graph_density,
        },
        "files": {
            "evolved_coop": coop_path,
            "evolved_strat": strat_path,
        },
    }
    log_path = "results/run_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Saved: {coop_path}")
    print(f"  Saved: {strat_path}")
    print(f"  Saved: {log_path}")


if __name__ == "__main__":
    main()
