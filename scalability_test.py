"""
scalability_test.py
-------------------
Gap 4 — Scalability analysis across agent-count sizes.

Measures how fitness, evolved density, and wall-clock runtime scale
as the number of agents grows from 4 to 12.
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from evolution_engine import EvolutionEngine


class ScalabilityTester:
    """
    Runs the evolutionary algorithm at several agent-count sizes and
    records fitness, topology structure, and runtime.
    """

    def run_scale(self, num_agents: int) -> dict:
        """
        Run one evolutionary experiment at *num_agents* scale.

        Parameters
        ----------
        num_agents : int
            Number of agents (= adjacency matrix dimension).

        Returns
        -------
        dict
            Keys: ``num_agents``, ``best_fitness``, ``density``,
            ``connected``, ``runtime_seconds``, ``converged_generation``.
        """
        print(f"  Running n={num_agents} agents...", flush=True)
        engine = EvolutionEngine(
            num_agents=num_agents,
            population_size=20,
            num_generations=50,
            mutation_rate=0.4,
            agent_mode="cooperative",
        )

        t_start = time.time()
        best = engine.run()
        runtime = time.time() - t_start

        # Converged generation = first gen that reached 95 % of final fitness
        final = engine.best_per_generation[-1]
        threshold = 0.95 * final
        converged_gen = next(
            (i + 1 for i, f in enumerate(engine.best_per_generation)
             if f >= threshold),
            len(engine.best_per_generation),
        )

        result = {
            "num_agents":          num_agents,
            "best_fitness":        best.fitness_score,
            "density":             best.graph_density,
            "connected":           best.is_connected(),
            "runtime_seconds":     round(runtime, 1),
            "converged_generation": converged_gen,
        }
        print(f"    fitness={result['best_fitness']:.4f}  "
              f"density={result['density']:.2f}  "
              f"runtime={result['runtime_seconds']}s  "
              f"converged@gen{converged_gen}")
        return result

    def run_all_scales(self) -> list[dict]:
        """
        Run :meth:`run_scale` for agent counts [4, 6, 8, 10, 12].

        Returns
        -------
        list[dict]
            One result dict per agent count.
        """
        return [self.run_scale(n) for n in [4, 6, 8, 10, 12]]

    def print_table(self, results: list[dict]) -> None:
        """
        Print a formatted results table.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_all_scales`.
        """
        header = (f"{'Agents':>6}  {'Fitness':>8}  {'Density':>8}  "
                  f"{'Connected':>10}  {'Runtime(s)':>11}  {'Converged@Gen':>13}")
        rule = "─" * 70
        print(f"\n{header}")
        print(rule)
        for r in results:
            print(f"{r['num_agents']:>6}  "
                  f"{r['best_fitness']:>8.4f}  "
                  f"{r['density']:>8.4f}  "
                  f"{'yes' if r['connected'] else 'NO':>10}  "
                  f"{r['runtime_seconds']:>11.1f}  "
                  f"{r['converged_generation']:>13}")

    def plot_scaling(self, results: list[dict]) -> None:
        """
        Plot fitness and runtime against agent count.

        Left panel:  agents vs best fitness.
        Right panel: agents vs runtime in seconds.

        Saves to ``results/scaling_plot.png`` and calls ``plt.show()``.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_all_scales`.
        """
        os.makedirs("results", exist_ok=True)

        agents   = [r["num_agents"]      for r in results]
        fitness  = [r["best_fitness"]    for r in results]
        runtimes = [r["runtime_seconds"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("EvoComm Scaling Behaviour", fontsize=14)

        ax1.plot(agents, fitness, marker="o", linewidth=2,
                 color="steelblue", markersize=8)
        ax1.set_xlabel("Number of Agents", fontsize=12)
        ax1.set_ylabel("Best Fitness Score", fontsize=12)
        ax1.set_title("Fitness vs Agent Count", fontsize=12)
        ax1.set_xticks(agents)
        ax1.grid(linestyle="--", alpha=0.5)

        ax2.plot(agents, runtimes, marker="s", linewidth=2,
                 color="darkorange", markersize=8)
        ax2.set_xlabel("Number of Agents", fontsize=12)
        ax2.set_ylabel("Runtime (seconds)", fontsize=12)
        ax2.set_title("Runtime vs Agent Count", fontsize=12)
        ax2.set_xticks(agents)
        ax2.grid(linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join("results", "scaling_plot.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n  Saved: {out_path}")
        plt.show()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    print("Scalability Test — 5 agent-count experiments")
    print("Warning: running 5 evolutionary experiments — ~8 minutes\n")

    tester = ScalabilityTester()
    results = tester.run_all_scales()

    tester.print_table(results)
    tester.plot_scaling(results)

    # ── Observations ──────────────────────────────────────────────────────────
    densities = [r["density"] for r in results]
    agents    = [r["num_agents"] for r in results]
    runtimes  = [r["runtime_seconds"] for r in results]

    print(f"\n  Density trend across scales: "
          f"{[round(d, 2) for d in densities]}")

    if runtimes[0] > 0:
        ratio = runtimes[-1] / runtimes[0]
        if ratio < 4:
            growth = "sublinear"
        elif ratio <= 8:
            growth = "roughly linear"
        else:
            growth = "superlinear"
        print(f"  Runtime growth: approximately {growth} "
              f"(ratio n=12/n=4 = {ratio:.1f}x)")
