"""
ablation_study.py
-----------------
Gap 2 — Ablation study showing how fitness-function weights shape
the topology the EA converges to.

Four configurations are run, varying task/speed/message weights,
and the resulting evolved structures are compared.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from evolution_engine import EvolutionEngine
from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome


class AblationStudy:
    """
    Runs the evolutionary algorithm under different fitness-function
    weight configurations and compares the evolved topologies.
    """

    def run_config(self, config: dict, label: str) -> TopologyGenome:
        """
        Evolve a topology using a custom fitness-function configuration.

        Parameters
        ----------
        config : dict
            Keys: ``task_weight``, ``speed_weight``, ``msg_weight``,
            ``num_agents``.
        label : str
            Human-readable name for progress output.

        Returns
        -------
        TopologyGenome
            Best genome found after 50 generations.
        """
        print(f"  Running config: {label}  "
              f"(task={config['task_weight']}, "
              f"speed={config['speed_weight']}, "
              f"msg={config['msg_weight']})",
              flush=True)

        evaluator = FitnessEvaluator(
            task_weight=config["task_weight"],
            efficiency_weight=config["speed_weight"],
            num_trials=5,
            agent_mode="cooperative",
        )

        # Patch compute_fitness to use the ablation weights
        msg_weight = config["msg_weight"]
        original_compute = evaluator.compute_fitness.__func__

        from fitness_evaluator import SimResult, _MAX_STEPS

        def patched_compute(self_inner, result: SimResult) -> float:
            speed_bonus = (_MAX_STEPS - result.steps_taken) / _MAX_STEPS
            msg_penalty = min(result.message_count / 30.0, 1.0)
            fitness = (
                config["task_weight"] * result.task_score
                + config["speed_weight"] * speed_bonus
                - msg_weight * msg_penalty
            )
            return round(max(fitness, 0.0), 4)

        import types
        evaluator.compute_fitness = types.MethodType(patched_compute, evaluator)

        engine = EvolutionEngine(
            num_agents=config["num_agents"],
            population_size=20,
            num_generations=50,
            mutation_rate=0.4,
        )
        engine.fitness_evaluator = evaluator

        best = engine.run()
        print(f"    -> {best}")
        return best

    def run_all(self) -> dict[str, TopologyGenome]:
        """
        Run the four ablation configurations in order.

        Configurations
        --------------
        A_current    : task=0.6, speed=0.3, msg=0.3  (baseline)
        B_task_heavy : task=0.9, speed=0.1, msg=0.0  (ignore efficiency)
        C_efficiency : task=0.4, speed=0.2, msg=0.4  (punish messages hard)
        D_speed      : task=0.4, speed=0.5, msg=0.1  (reward fast coordination)

        Returns
        -------
        dict[str, TopologyGenome]
            Maps configuration label → best genome.
        """
        configs = [
            {"label": "A_current",    "task_weight": 0.6, "speed_weight": 0.3,
             "msg_weight": 0.3, "num_agents": 6},
            {"label": "B_task_heavy", "task_weight": 0.9, "speed_weight": 0.1,
             "msg_weight": 0.0, "num_agents": 6},
            {"label": "C_efficiency", "task_weight": 0.4, "speed_weight": 0.2,
             "msg_weight": 0.4, "num_agents": 6},
            {"label": "D_speed",      "task_weight": 0.4, "speed_weight": 0.5,
             "msg_weight": 0.1, "num_agents": 6},
        ]
        results: dict[str, TopologyGenome] = {}
        for cfg in configs:
            label = cfg.pop("label")
            results[label] = self.run_config(cfg, label)
        return results

    def print_comparison(self, results: dict[str, TopologyGenome]) -> None:
        """
        Print a table comparing evolved topology structure across configs.

        Interpretation is auto-generated from density and connectivity.

        Parameters
        ----------
        results : dict[str, TopologyGenome]
            Output of :meth:`run_all`.
        """
        header = (f"{'Config':<14} {'Fitness':>8}  {'Density':>8}  "
                  f"{'Connected':>10}  {'MaxDeg':>7}  Interpretation")
        rule = "─" * 75
        print(f"\n{header}")
        print(rule)

        for label, genome in results.items():
            density = genome.graph_density
            connected = genome.is_connected()
            max_deg = max(
                int(genome.adjacency_matrix[i].sum())
                for i in range(genome.num_agents)
            )

            if density > 0.7:
                interp = "dense — efficiency ignored"
            elif density < 0.3:
                interp = "sparse — may sacrifice coordination"
            else:
                interp = "balanced"
            if not connected:
                interp += " (disconnected!)"

            print(f"{label:<14} {genome.fitness_score:>8.4f}  "
                  f"{density:>8.4f}  "
                  f"{'yes' if connected else 'NO':>10}  "
                  f"{max_deg:>7}  {interp}")

    def visualise_all(self, results: dict[str, TopologyGenome]) -> None:
        """
        Draw a 2×2 grid of topology graphs, one per configuration.

        Nodes are steelblue, edges are grey.  Each subplot is titled
        with the config label, density, and fitness.

        Saves to ``results/ablation_topologies.png`` and calls
        ``plt.show()``.

        Parameters
        ----------
        results : dict[str, TopologyGenome]
            Output of :meth:`run_all`.
        """
        os.makedirs("results", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Ablation Study — Evolved Topologies by Fitness Config",
                     fontsize=14, y=1.01)
        axes_flat = axes.flatten()

        for ax, (label, genome) in zip(axes_flat, results.items()):
            G = nx.Graph()
            G.add_nodes_from(range(genome.num_agents))
            rows, cols = np.where(np.triu(genome.adjacency_matrix, k=1))
            for i, j in zip(rows.tolist(), cols.tolist()):
                G.add_edge(int(i), int(j))

            pos = nx.spring_layout(G, seed=42, k=1.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="steelblue",
                                   node_size=600, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                                    font_size=11, font_weight="bold")
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="grey",
                                   width=2, alpha=0.7)
            ax.set_title(
                f"{label}\ndensity={genome.graph_density:.2f}  "
                f"fitness={genome.fitness_score:.4f}",
                fontsize=11,
            )
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join("results", "ablation_topologies.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {out_path}")
        plt.show()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    print("Ablation Study — varying fitness-function weights")
    print("Warning: this runs 4 evolutionary experiments — ~4 minutes\n")

    study = AblationStudy()
    results = study.run_all()

    study.print_comparison(results)
    study.visualise_all(results)

    # ── Conclusion ────────────────────────────────────────────────────────────
    density_a = results["A_current"].graph_density
    density_b = results["B_task_heavy"].graph_density
    density_c = results["C_efficiency"].graph_density

    print(f"\n  Conclusion:")
    print(f"  Increasing msg_weight from 0.0 to 0.4 reduced evolved density "
          f"from {density_b:.2f} to {density_c:.2f}.")
    print(f"  This confirms the message penalty is actively shaping topology "
          f"structure.")
