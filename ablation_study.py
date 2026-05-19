
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from evolution_engine import EvolutionEngine
from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome


class AblationStudy:

    def run_config(self, config: dict, label: str) -> tuple[TopologyGenome, dict]:
        print(f"  Running config: {label}  "
              f"(task={config['task_weight']}, "
              f"speed={config['speed_weight']}, "
              f"msg={config['msg_weight']})",
              flush=True)

        evaluator = FitnessEvaluator(
            task_weight=config["task_weight"],
            efficiency_weight=config["speed_weight"],
            msg_weight=config["msg_weight"],
            num_trials=5,
            agent_mode="cooperative",
        )

        engine = EvolutionEngine(
            num_agents=config["num_agents"],
            population_size=20,
            num_generations=50,
            mutation_rate=0.4,
        )
        engine.fitness_evaluator = evaluator

        best = engine.run()
        print(f"    -> {best}")

        sim_eval = FitnessEvaluator(
            task_weight=config["task_weight"],
            efficiency_weight=config["speed_weight"],
            msg_weight=config["msg_weight"],
            num_trials=1,
            agent_mode="cooperative",
        )
        task_scores, msg_counts, steps_list = [], [], []
        for _ in range(10):
            res = sim_eval.run_simulation(best)
            task_scores.append(res.task_score)
            msg_counts.append(res.message_count)
            steps_list.append(res.steps_taken)

        metrics = {
            "task_score":    float(np.mean(task_scores)),
            "message_count": float(np.mean(msg_counts)),
            "steps_taken":   float(np.mean(steps_list)),
        }
        return best, metrics

    def run_all(self) -> tuple[dict[str, TopologyGenome], dict[str, dict]]:
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
        results = {}
        metrics = {}
        for cfg in configs:
            label = cfg.pop("label")
            results[label], metrics[label] = self.run_config(cfg, label)
        return results, metrics

    def print_comparison(
        self, results: dict[str, TopologyGenome], metrics: dict[str, dict]
    ) -> None:
        header = (f"{'Config':<14} {'TaskScore':>10}  {'MsgCount':>9}  "
                  f"{'Steps':>6}  {'Density':>8}  {'EvolvedFitness':>15}")
        rule = "─" * 70
        print(f"\n{header}")
        print(rule)

        for label, genome in results.items():
            m = metrics[label]
            print(f"{label:<14} {m['task_score']:>10.3f}  "
                  f"{m['message_count']:>9.1f}  "
                  f"{m['steps_taken']:>6.1f}  "
                  f"{genome.graph_density:>8.4f}  "
                  f"{genome.fitness_score:>15.4f}")

    def visualise_all(self, results: dict[str, TopologyGenome]) -> None:
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


if __name__ == "__main__":
    np.random.seed(42)
    print("Ablation Study — varying fitness-function weights")
    print("Warning: this runs 4 evolutionary experiments — ~4 minutes\n")

    study = AblationStudy()
    results, metrics = study.run_all()

    study.print_comparison(results, metrics)
    study.visualise_all(results)
