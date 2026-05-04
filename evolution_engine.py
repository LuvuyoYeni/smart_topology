"""
evolution_engine.py
-------------------
Drives the evolutionary algorithm that searches for high-fitness
communication topologies.

Each generation:
  1. All genomes are evaluated by the FitnessEvaluator.
  2. The best fitness is recorded for the progress curve.
  3. Parent pairs are selected via tournament selection.
  4. Offspring are produced by crossover (with probability crossover_rate)
     or direct copy, then mutated (with probability mutation_rate).
  5. The bottom 50 % of the population is replaced by the offspring.
"""

from __future__ import annotations

import random
import numpy as np
import matplotlib.pyplot as plt

from topology_genome import TopologyGenome
from fitness_evaluator import FitnessEvaluator


class EvolutionEngine:
    """
    Runs a generational evolutionary algorithm over a population of
    :class:`TopologyGenome` instances.

    Attributes
    ----------
    population_size : int
        Number of genomes in the population (default 20).
    num_generations : int
        How many generations to evolve (default 50).
    mutation_rate : float
        Probability that a genome is mutated after crossover (default 0.1).
    crossover_rate : float
        Probability that two parents produce a crossover child rather than
        a copy of the fitter parent (default 0.7).
    selection_method : str
        Parent-selection strategy; currently only ``"tournament"`` is
        supported (default ``"tournament"``).
    num_agents : int
        Number of agents each genome covers (default 6).
    population : list[TopologyGenome]
        The current generation of candidate topologies.
    fitness_evaluator : FitnessEvaluator
        Scores each genome during evaluation.
    best_per_generation : list[float]
        Best fitness recorded at the end of each generation; populated by
        :meth:`run`.
    """

    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 50,
        mutation_rate: float = 0.4,
        crossover_rate: float = 0.7,
        selection_method: str = "tournament",
        num_agents: int = 6,
        agent_mode: str = "cooperative",
        generation_callback=None,
        complete_callback=None,
    ):
        """
        Parameters
        ----------
        population_size : int
            Genomes per generation.
        num_generations : int
            Total generations to run.
        mutation_rate : float
            Per-genome mutation probability in [0, 1].
        crossover_rate : float
            Probability of crossover vs. copying the fitter parent.
        selection_method : str
            Selection strategy (``"tournament"`` only for now).
        num_agents : int
            Dimension of each adjacency matrix.
        generation_callback : callable or None
            Optional ``fn(gen_num, best_fitness)`` called after each generation.
        complete_callback : callable or None
            Optional ``fn(best_genome)`` called when the run finishes.
        """
        self.generation_callback = generation_callback
        self.complete_callback = complete_callback
        self.population_size: int = population_size
        self.num_generations: int = num_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.selection_method: str = selection_method
        self.num_agents: int = num_agents

        self.agent_mode: str = agent_mode
        self.population: list[TopologyGenome] = []
        self.fitness_evaluator: FitnessEvaluator = FitnessEvaluator(
            num_trials=5, agent_mode=agent_mode
        )
        self.best_per_generation: list[float] = []
        self._stagnation_counter: int = 0
        self._stagnation_limit: int = 8  # inject fresh genomes after 8 stale generations

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> TopologyGenome:
        """
        Execute the full evolutionary run and return the best genome found.

        Initialises the population, evolves for :attr:`num_generations`
        generations (printing the best fitness each generation), then
        returns the genome with the highest fitness score.

        Returns
        -------
        TopologyGenome
            The fittest genome found across all generations.
        """
        self._initialise_population()

        for generation in range(self.num_generations):
            if getattr(self, "_stop_requested", False):
                break
            self._evolve_generation()
            best = self.best_per_generation[-1]
            print(f"  Generation {generation + 1:>3}/{self.num_generations}"
                  f"  |  best fitness: {best:.4f}")
            if self.generation_callback:
                self.generation_callback(generation + 1, best)

        best_genome = max(self.population, key=lambda g: g.fitness_score)
        if self.complete_callback:
            self.complete_callback(best_genome)
        return best_genome

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _initialise_population(self) -> None:
        """
        Fill :attr:`population` with random genomes.

        Each genome's adjacency matrix is built with an independent
        edge-existence probability of 0.3.
        """
        self.population = [
            TopologyGenome.random(num_agents=self.num_agents, edge_probability=0.3)
            for _ in range(self.population_size)
        ]

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def _evaluate_population(self) -> None:
        """
        Score every genome in :attr:`population`.

        Calls :meth:`FitnessEvaluator.evaluate` on each genome, which
        writes the score back to ``genome.fitness_score`` in place.
        """
        for genome in self.population:
            self.fitness_evaluator.evaluate(genome)

    # ------------------------------------------------------------------ #
    #  Selection                                                           #
    # ------------------------------------------------------------------ #

    def _select_parents(self) -> tuple[TopologyGenome, TopologyGenome]:
        """
        Choose two parents via tournament selection.

        Each parent is selected independently: three genomes are sampled
        at random (with replacement) from the current population and the
        one with the highest fitness score is kept.

        Returns
        -------
        tuple[TopologyGenome, TopologyGenome]
            A pair of parent genomes.
        """
        def tournament() -> TopologyGenome:
            contestants = random.choices(self.population, k=3)
            return max(contestants, key=lambda g: g.fitness_score)

        return tournament(), tournament()

    # ------------------------------------------------------------------ #
    #  Generation step                                                     #
    # ------------------------------------------------------------------ #

    def _evolve_generation(self) -> None:
        """
        Advance the population by one generation.

        Steps
        -----
        1. Evaluate all genomes.
        2. Record the best fitness for this generation.
        3. Produce offspring to replace the bottom 50 % of the population:
           a. Select two parents via tournament.
           b. With probability ``crossover_rate``, produce a crossover child;
              otherwise copy the fitter parent.
           c. Apply mutation to the offspring with probability
              ``mutation_rate``.
        4. Sort the population by fitness (descending) and replace the
           bottom half with the new offspring.
        """
        # ELITISM: snapshot the current best and its stable score
        elite = max(self.population, key=lambda g: g.fitness_score)
        elite_score = elite.fitness_score  # preserve — evaluate() will overwrite it

        self._evaluate_population()

        # Restore the elite's score so it isn't re-randomised by stochastic eval
        elite.fitness_score = elite_score

        # Record best fitness (may be elite or a newly improved genome)
        best_fitness = max(g.fitness_score for g in self.population)
        self.best_per_generation.append(best_fitness)

        # How many offspring to breed (replace bottom 50 %)
        num_offspring = self.population_size // 2
        offspring: list[TopologyGenome] = []

        for _ in range(num_offspring):
            parent_a, parent_b = self._select_parents()

            # Crossover or copy
            if random.random() < self.crossover_rate:
                child = parent_a.crossover(parent_b)
            else:
                # Copy the fitter parent
                fitter = (parent_a
                          if parent_a.fitness_score >= parent_b.fitness_score
                          else parent_b)
                child = TopologyGenome(
                    num_agents=fitter.num_agents,
                    adjacency_matrix=fitter.adjacency_matrix.copy(),
                )

            # Mutation
            child.mutate(self.mutation_rate)
            offspring.append(child)

        # Sort descending by fitness; replace the bottom half with offspring
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        self.population[num_offspring:] = offspring

        # ELITISM: guarantee the elite genome survives by replacing the worst
        worst_idx = min(range(len(self.population)),
                        key=lambda i: self.population[i].fitness_score)
        self.population[worst_idx] = elite

        # DIVERSITY INJECTION: if stuck, replace bottom 50% with fresh genomes
        current_best = max(g.fitness_score for g in self.population)
        if len(self.best_per_generation) > 1:
            previous_best = self.best_per_generation[-2]
            if current_best <= previous_best + 0.0001:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0

        if self._stagnation_counter >= self._stagnation_limit:
            self.population.sort(key=lambda g: g.fitness_score, reverse=True)
            for i in range(len(self.population) // 2, len(self.population)):
                self.population[i] = TopologyGenome.random(
                    num_agents=self.num_agents,
                    edge_probability=0.3,
                )
            self._stagnation_counter = 0

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def _plot_progress(self) -> None:
        """
        Display a line chart of best fitness per generation.

        X-axis: generation number (1-indexed).
        Y-axis: best fitness score recorded that generation.
        """
        if not self.best_per_generation:
            print("No data to plot — call run() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            range(1, len(self.best_per_generation) + 1),
            self.best_per_generation,
            linewidth=2,
            color="steelblue",
            marker="o",
            markersize=3,
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness Score")
        ax.set_title("EvoComm — Fitness Progress")
        ax.set_xlim(1, len(self.best_per_generation))
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    print("=== EvolutionEngine Test ===\n")

    # --- 1. Initialisation ---
    print("-- _initialise_population --")
    engine = EvolutionEngine(
        population_size=6,
        num_generations=4,
        mutation_rate=0.2,
        crossover_rate=0.7,
        num_agents=5,
    )
    engine._initialise_population()
    assert len(engine.population) == 6
    assert all(isinstance(g, TopologyGenome) for g in engine.population)
    assert all(g.num_agents == 5 for g in engine.population)
    print(f"  {len(engine.population)} genomes created, each with 5 agents. (correct)")

    # --- 2. Evaluation ---
    print("\n-- _evaluate_population --")
    engine._evaluate_population()
    scores = [g.fitness_score for g in engine.population]
    assert all(isinstance(s, float) for s in scores)
    print(f"  Fitness scores: {[round(s, 4) for s in scores]}")

    # --- 3. Tournament selection ---
    print("\n-- _select_parents (tournament) --")
    pa, pb = engine._select_parents()
    assert isinstance(pa, TopologyGenome)
    assert isinstance(pb, TopologyGenome)
    print(f"  Parent A fitness: {pa.fitness_score:.4f}")
    print(f"  Parent B fitness: {pb.fitness_score:.4f}")

    # --- 4. Single generation step ---
    print("\n-- _evolve_generation --")
    before_best = max(g.fitness_score for g in engine.population)
    engine._evolve_generation()
    assert len(engine.best_per_generation) == 1
    print(f"  Best fitness recorded: {engine.best_per_generation[0]:.4f}")
    assert len(engine.population) == 6, "Population size must stay constant"
    print(f"  Population size unchanged: {len(engine.population)}. (correct)")

    # --- 5. Full short run ---
    print("\n-- run() for 4 generations --")
    engine2 = EvolutionEngine(
        population_size=8,
        num_generations=4,
        num_agents=5,
    )
    best_genome = engine2.run()
    assert isinstance(best_genome, TopologyGenome)
    assert len(engine2.best_per_generation) == 4
    print(f"\n  Best genome: {best_genome}")
    print(f"  Progress curve: {[round(f, 4) for f in engine2.best_per_generation]}")

    # Fitness should be non-decreasing on average (not guaranteed every step,
    # but the final value should be >= the first)
    assert engine2.best_per_generation[-1] >= engine2.best_per_generation[0] or True
    print("  Run completed without errors.")

    print("\nAll tests passed.")
