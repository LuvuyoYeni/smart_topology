
from __future__ import annotations

import json
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
        scores = []
        for _ in range(n_trials):
            evaluator = FitnessEvaluator(num_trials=5, agent_mode=agent_mode)
            scores.append(evaluator.evaluate(genome))
        return scores

    def compare_topologies(
        self,
        topologies: dict[str, TopologyGenome],
        agent_mode: str = "cooperative",
        n_trials: int = 30,
    ) -> dict:
        results = {}
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

    def significance_table(self, results: dict, ref_key: str = "Evol-Coop") -> None:
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
        self,
        results: dict,
        title_suffix: str = "Cooperative Agents",
        out_path: str = "results/stat_boxplot.png",
    ) -> None:
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
                   "sandybrown", "mediumseagreen", "mediumpurple"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)

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

        plt.savefig(out_path, dpi=150)
        print(f"\n  Saved: {out_path}")
        plt.show()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    for _path in ["results/evolved_coop.npz", "results/evolved_strat.npz"]:
        if not os.path.exists(_path):
            raise FileNotFoundError("Run main.py first to generate evolved genomes.")

    EVOL_COOP_MATRIX = np.load("results/evolved_coop.npz")["adjacency_matrix"]
    EVOL_STRAT_MATRIX = np.load("results/evolved_strat.npz")["adjacency_matrix"]

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

    def _build_topologies() -> dict:
        topo = {
            "Evol-Coop":  TopologyGenome(NUM_AGENTS, EVOL_COOP_MATRIX),
            "Evol-Strat": TopologyGenome(NUM_AGENTS, EVOL_STRAT_MATRIX),
            "Full":       TopologyGenome(NUM_AGENTS, full_mat),
            "Ring":       TopologyGenome(NUM_AGENTS, ring_mat),
            "Star":       TopologyGenome(NUM_AGENTS, star_mat),
        }
        rand_path = "results/random_best_coop.npz"
        if os.path.exists(rand_path):
            rand_mat = np.load(rand_path)["adjacency_matrix"]
            topo["Random-Best"] = TopologyGenome(NUM_AGENTS, rand_mat)
        else:
            print("  (random_best_coop.npz not found — run random_baseline.py to include it)")
        return topo

    analyser = StatisticalAnalyser()
    os.makedirs("results", exist_ok=True)

    def _summarise(results: dict, ref_key: str) -> None:
        ref = results.get(ref_key)
        if not ref:
            return
        sig_count = 0
        total_baselines = 0
        ref_scores = ref["scores"]
        for name, data in results.items():
            if name == ref_key:
                continue
            total_baselines += 1
            _, p = stats.mannwhitneyu(
                data["scores"], ref_scores, alternative="less"
            )
            if p < 0.05:
                sig_count += 1
        print(f"\n  Summary: {ref_key} significantly outperformed "
              f"{sig_count} of {total_baselines} baselines (p < 0.05).")

    # ── Cooperative analysis ──────────────────────────────────────────────────
    print("\n=== Statistical Analysis — 30 trials, cooperative agents ===\n")
    results_coop = analyser.compare_topologies(
        _build_topologies(), agent_mode="cooperative", n_trials=30
    )
    analyser.significance_table(results_coop, ref_key="Evol-Coop")
    analyser.plot_distributions(
        results_coop,
        title_suffix="Cooperative Agents",
        out_path="results/stat_boxplot_coop.png",
    )
    with open("results/stat_scores_coop.json", "w") as _f:
        json.dump({n: d["scores"] for n, d in results_coop.items()}, _f, indent=2)
    print("  Saved: results/stat_scores_coop.json")
    _summarise(results_coop, "Evol-Coop")

    # ── Strategic analysis ────────────────────────────────────────────────────
    print("\n=== Statistical Analysis — 30 trials, strategic agents ===\n")
    results_strat = analyser.compare_topologies(
        _build_topologies(), agent_mode="strategic", n_trials=30
    )
    analyser.significance_table(results_strat, ref_key="Evol-Strat")
    analyser.plot_distributions(
        results_strat,
        title_suffix="Strategic Agents",
        out_path="results/stat_boxplot_strat.png",
    )
    with open("results/stat_scores_strat.json", "w") as _f:
        json.dump({n: d["scores"] for n, d in results_strat.items()}, _f, indent=2)
    print("  Saved: results/stat_scores_strat.json")
    _summarise(results_strat, "Evol-Strat")
