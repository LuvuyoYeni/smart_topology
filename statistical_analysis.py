"""
statistical_analysis.py
-----------------------
Gap 1 — Statistical robustness testing for EvoComm.

Evaluates topologies across many independent trials and applies
Mann-Whitney U tests to determine whether the evolved topology
significantly outperforms hand-crafted baselines.
"""

from __future__ import annotations

import os
import sys

try:
    from scipy import stats
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

from topology_genome import TopologyGenome
from fitness_evaluator import FitnessEvaluator


class StatisticalAnalyser:
    """
    Runs repeated independent fitness evaluations and performs
    non-parametric significance testing between topologies.
    """

    def run_topology_trials(
        self,
        genome: TopologyGenome,
        agent_mode: str = "cooperative",
        n_trials: int = 30,
    ) -> list[float]:
        """
        Evaluate *genome* independently *n_trials* times.

        Each trial uses a fresh FitnessEvaluator with num_trials=1 so
        every score comes from a single simulation run, giving the full
        variance of the fitness landscape rather than an averaged view.

        Parameters
        ----------
        genome : TopologyGenome
            The topology to evaluate.
        agent_mode : str
            Agent behaviour mode: ``"cooperative"``, ``"strategic"``,
            or ``"mixed"``.
        n_trials : int
            Number of independent evaluations.

        Returns
        -------
        list[float]
            Raw fitness scores, one per trial.
        """
        scores: list[float] = []
        for _ in range(n_trials):
            evaluator = FitnessEvaluator(num_trials=1, agent_mode=agent_mode)
            score = evaluator.evaluate(genome)
            scores.append(score)
        return scores

    def compare_topologies(
        self,
        topologies: dict[str, TopologyGenome],
        agent_mode: str = "cooperative",
        n_trials: int = 30,
    ) -> dict:
        """
        Run *n_trials* independent evaluations for every topology.

        Parameters
        ----------
        topologies : dict[str, TopologyGenome]
            Mapping of name → genome.
        agent_mode : str
            Agent behaviour mode used for all evaluations.
        n_trials : int
            Trials per topology.

        Returns
        -------
        dict
            For each topology name:
            ``mean``, ``std``, ``min``, ``max``, ``median``, ``scores``.
        """
        results: dict = {}
        total = len(topologies)
        for idx, (name, genome) in enumerate(topologies.items(), 1):
            print(f"  [{idx}/{total}] Evaluating {name} ({n_trials} trials)...",
                  flush=True)
            scores = self.run_topology_trials(genome, agent_mode, n_trials)
            arr = np.array(scores)
            results[name] = {
                "mean":   float(arr.mean()),
                "std":    float(arr.std()),
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "median": float(np.median(arr)),
                "scores": scores,
            }
            print(f"      mean={results[name]['mean']:.3f}  "
                  f"std={results[name]['std']:.3f}")
        return results

    def significance_table(self, results: dict) -> None:
        """
        Print a formatted significance table.

        Each topology is compared against Evol-Coop using a one-sided
        Mann-Whitney U test (alternative='less'), asking: "is this
        topology's fitness distribution shifted below Evol-Coop's?"

        Significance labels:
          p < 0.001 → ***
          p < 0.01  → **
          p < 0.05  → *
          otherwise → ns

        Parameters
        ----------
        results : dict
            Output of :meth:`compare_topologies`.
        """
        ref_key = "Evol-Coop"
        ref_scores = results[ref_key]["scores"] if ref_key in results else None

        col_name  = 14
        col_stats = 20
        col_p     = 12
        col_sig   = 5
        header = (f"{'Topology':<{col_name}} "
                  f"{'Mean ± Std':<{col_stats}} "
                  f"{'p-value':>{col_p}}    "
                  f"{'Sig':<{col_sig}}")
        rule = "─" * len(header)

        print(f"\n{header}")
        print(rule)

        for name, data in results.items():
            mean_std = f"{data['mean']:.3f} ± {data['std']:.3f}"

            if name == ref_key or ref_scores is None:
                p_str  = "baseline"
                sig    = "—"
            else:
                _, p = stats.mannwhitneyu(
                    data["scores"], ref_scores, alternative="less"
                )
                p_str = f"{p:.4f}"
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

            print(f"{name:<{col_name}} "
                  f"{mean_std:<{col_stats}} "
                  f"{p_str:>{col_p}}    "
                  f"{sig:<{col_sig}}")

    def plot_distributions(
        self, results: dict, title_suffix: str = "Cooperative Agents"
    ) -> None:
        """
        Boxplot of fitness distributions with jittered data point overlay.

        Saves to ``results/stat_boxplot.png`` and calls ``plt.show()``.

        Parameters
        ----------
        results : dict
            Output of :meth:`compare_topologies`.
        title_suffix : str
            Appended to the plot title after the em-dash.
        """
        os.makedirs("results", exist_ok=True)

        names  = list(results.keys())
        scores = [results[n]["scores"] for n in names]

        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(
            scores,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )

        colours = ["steelblue", "cornflowerblue", "salmon",
                   "sandybrown", "mediumseagreen"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)

        # Jittered scatter overlay
        rng = np.random.default_rng(seed=0)
        for i, sc in enumerate(scores, 1):
            jitter = rng.uniform(-0.1, 0.1, size=len(sc))
            ax.scatter(
                np.full(len(sc), i) + jitter,
                sc,
                color="dimgrey",
                s=18,
                alpha=0.5,
                zorder=3,
            )

        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Fitness Score", fontsize=12)
        ax.set_title(
            f"Fitness Distribution Across 30 Trials — {title_suffix}",
            fontsize=13,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = os.path.join("results", "stat_boxplot.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n  Saved: {out_path}")
        plt.show()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    EVOL_COOP_MATRIX = np.array([
        [0,0,0,1,0,1],
        [0,0,0,1,0,1],
        [0,0,0,0,1,0],
        [1,1,0,0,0,0],
        [0,0,1,0,0,1],
        [1,1,0,0,1,0],
    ], dtype=bool)

    EVOL_STRAT_MATRIX = np.array([
        [0,0,1,1,1,0],
        [0,0,1,1,1,0],
        [1,1,0,1,1,1],
        [1,1,1,0,1,1],
        [1,1,1,1,0,0],
        [0,0,1,1,0,0],
    ], dtype=bool)

    if EVOL_COOP_MATRIX is None or EVOL_STRAT_MATRIX is None:
        print("Please fill in evolved matrices from your last run.")
        print("Set EVOL_COOP_MATRIX and EVOL_STRAT_MATRIX as numpy bool arrays.")
        sys.exit(0)

    # ── Build topologies ──────────────────────────────────────────────────────
    NUM_AGENTS = 6

    full_mat = np.ones((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    np.fill_diagonal(full_mat, False)

    ring_mat = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    for i in range(NUM_AGENTS):
        ring_mat[i][(i + 1) % NUM_AGENTS] = True
        ring_mat[(i + 1) % NUM_AGENTS][i] = True

    star_mat = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    for i in range(1, NUM_AGENTS):
        star_mat[0][i] = star_mat[i][0] = True

    topologies: dict[str, TopologyGenome] = {
        "Evol-Coop":  TopologyGenome(NUM_AGENTS, EVOL_COOP_MATRIX),
        "Evol-Strat": TopologyGenome(NUM_AGENTS, EVOL_STRAT_MATRIX),
        "Full":       TopologyGenome(NUM_AGENTS, full_mat),
        "Ring":       TopologyGenome(NUM_AGENTS, ring_mat),
        "Star":       TopologyGenome(NUM_AGENTS, star_mat),
    }

    # ── Run analysis ──────────────────────────────────────────────────────────
    analyser = StatisticalAnalyser()

    print("\n=== Statistical Analysis — 30 trials, cooperative agents ===\n")
    results = analyser.compare_topologies(
        topologies, agent_mode="cooperative", n_trials=30
    )

    analyser.significance_table(results)
    analyser.plot_distributions(results)

    # ── Plain-English summary ─────────────────────────────────────────────────
    ref = results.get("Evol-Coop")
    if ref:
        sig_count = 0
        total_baselines = 0
        ref_scores = ref["scores"]
        for name, data in results.items():
            if name == "Evol-Coop":
                continue
            total_baselines += 1
            _, p = stats.mannwhitneyu(
                data["scores"], ref_scores, alternative="less"
            )
            if p < 0.05:
                sig_count += 1
        print(f"\n  Summary: Evol-Coop significantly outperformed "
              f"{sig_count} of {total_baselines} baselines (p < 0.05).")
