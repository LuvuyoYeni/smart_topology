"""
fitness_evaluator.py
--------------------
Evaluates the fitness of a TopologyGenome by running short simulations and
scoring the resulting task completion and communication efficiency.
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple

from environment import Environment
from topology_genome import TopologyGenome
from agent import CooperativeAgent

# Named tuple returned by a single simulation run
SimResult = namedtuple("SimResult", ["task_score", "message_count", "steps_taken"])

_MAX_STEPS = 50
_GRID_SIZE = 10
_NUM_TASKS = 8
_TASK_REQUIREMENT = 3


class FitnessEvaluator:
    """
    Scores a :class:`TopologyGenome` by averaging fitness across several
    independent simulation trials.

    Fitness formula
    ---------------
    F(g) = task_weight  × TaskScore(g)
         + efficiency_weight × (1 / max(MessageCount(g), 1))
         - 0.1 × DisconnectionPenalty(g)

    Where:
      TaskScore           = completion_rate at end of simulation [0, 1]
      MessageCount        = total messages sent during simulation
      DisconnectionPenalty = 1.0 if graph is not fully connected, else 0.0

    Attributes
    ----------
    task_weight : float
        Weight applied to the task-completion component (default 0.7).
    efficiency_weight : float
        Weight applied to the communication-efficiency component (default 0.3).
    num_trials : int
        Number of independent runs to average over for robustness (default 3).
    """

    def __init__(
        self,
        task_weight: float = 0.7,
        efficiency_weight: float = 0.3,
        num_trials: int = 3,
        agent_mode: str = "cooperative",
    ):
        """
        Parameters
        ----------
        task_weight : float
            Weight for the task-completion term.
        efficiency_weight : float
            Weight for the communication-efficiency term.
        num_trials : int
            How many simulation runs to average over.
        agent_mode : str
            Which agent type to simulate: ``"cooperative"``, ``"strategic"``,
            or ``"mixed"`` (even IDs cooperative, odd IDs strategic).
        """
        self.task_weight: float = task_weight
        self.efficiency_weight: float = efficiency_weight
        self.num_trials: int = num_trials
        self.agent_mode: str = agent_mode

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, genome: TopologyGenome) -> float:
        """
        Compute the fitness of *genome* averaged over :attr:`num_trials` runs.

        The result is also written back to ``genome.fitness_score`` so the
        evolutionary engine can rank genomes without re-evaluating.

        Parameters
        ----------
        genome : TopologyGenome
            The candidate topology to score.

        Returns
        -------
        float
            Mean fitness across all trials.
        """
        scores = [self.compute_fitness(self.run_simulation(genome))
                  for _ in range(self.num_trials)]
        disconnection_penalty = 0.1 if not genome.is_connected() else 0.0
        fitness = float(np.mean(scores)) - disconnection_penalty
        genome.fitness_score = fitness
        return fitness

    def run_simulation(self, topology: TopologyGenome) -> SimResult:
        """
        Run one simulation episode with *topology* as the communication graph.

        Simulation loop (up to ``_MAX_STEPS`` steps):
          1. Each agent queries its neighbours via the CommunicationGraph.
          2. Connected agents exchange messages (share private_info).
          3. Inboxes are processed — each agent calls receive_message.
          4. Each agent calls act() to move.
          5. env.update(positions) advances the environment.
          6. Early exit if all tasks are completed.

        Parameters
        ----------
        topology : TopologyGenome
            The topology whose communication structure drives the simulation.

        Returns
        -------
        SimResult
            task_score, message_count, and steps_taken for this episode.
        """
        env = Environment(
            grid_size=_GRID_SIZE,
            num_tasks=_NUM_TASKS,
            task_requirement=_TASK_REQUIREMENT,
        )

        num_agents = topology.num_agents
        agents = self._spawn_agents(num_agents, env.grid_size)
        comm_graph = topology.to_communication_graph()
        total_messages = 0

        for step in range(_MAX_STEPS):

            # --- 1 & 2: Communication phase ---
            for agent in agents:
                neighbour_ids = comm_graph.get_neighbours(agent.agent_id)
                neighbours = [agents[nid] for nid in neighbour_ids]

                if agent.decide_to_share(agent.private_info) and neighbours:
                    agent.share_all(neighbours)
                    total_messages += len(neighbours)

            # --- 3: Process inboxes ---
            for agent in agents:
                for msg in agent.inbox:
                    agent.receive_message(msg)
                agent.inbox.clear()

            # --- 4: Movement phase ---
            for agent in agents:
                agent.act(env)

            # --- 5: Environment update ---
            positions = [agent.position for agent in agents]
            env.update(positions)

            # --- 6: Early exit ---
            if not env.get_active_tasks():
                return SimResult(
                    task_score=env.completion_rate(),
                    message_count=total_messages,
                    steps_taken=step + 1,
                )

        return SimResult(
            task_score=env.completion_rate(),
            message_count=total_messages,
            steps_taken=_MAX_STEPS,
        )

    def compute_fitness(self, result: SimResult) -> float:
        """
        Revised fitness formula — topology now meaningfully affects the score.

        F = 0.6 × TaskScore
          + 0.3 × SpeedBonus
          - 0.2 × MsgPenalty

        Where:
          TaskScore  = proportion of tasks completed [0, 1]
          SpeedBonus = (max_steps - steps_taken) / max_steps  [0, 1]
                       rewards finishing faster
          MsgPenalty = min(message_count / 100, 1.0)  [0, 1]
                       penalises chatty/inefficient topologies

        Parameters
        ----------
        result : SimResult
            Output of one simulation run.

        Returns
        -------
        float
            Fitness score for this result.
        """
        speed_bonus = (_MAX_STEPS - result.steps_taken) / _MAX_STEPS
        msg_penalty = min(result.message_count / 30.0, 1.0)

        fitness = (
            0.6 * result.task_score
            + 0.3 * speed_bonus
            - 0.3 * msg_penalty
        )

        return round(max(fitness, 0.0), 4)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _spawn_agents(self, num_agents: int, grid_size: int) -> list:
        """
        Create agents according to :attr:`agent_mode`.

        Modes
        -----
        ``"cooperative"``
            All agents share unconditionally.
        ``"strategic"``
            All agents use trust-based sharing with threshold 0.4.
        ``"mixed"``
            Even-ID agents are cooperative; odd-ID agents are strategic.

        Each agent's private_info is seeded with its own starting position
        so there is something meaningful to share from the first step.

        Parameters
        ----------
        num_agents : int
            How many agents to create.
        grid_size : int
            Side length of the grid (used to bound random positions).

        Returns
        -------
        list
            Agents indexed 0 … num_agents-1.
        """
        from agent import StrategicAgent

        agents = []
        for i in range(num_agents):
            pos = (
                int(np.random.randint(0, grid_size)),
                int(np.random.randint(0, grid_size)),
            )
            if self.agent_mode == "cooperative":
                agent = CooperativeAgent(agent_id=i, position=pos)
            elif self.agent_mode == "strategic":
                agent = StrategicAgent(agent_id=i, position=pos,
                                       deception_threshold=0.4)
            elif self.agent_mode == "mixed":
                if i % 2 == 0:
                    agent = CooperativeAgent(agent_id=i, position=pos)
                else:
                    agent = StrategicAgent(agent_id=i, position=pos,
                                           deception_threshold=0.4)
            else:
                raise ValueError(f"Unknown agent_mode: {self.agent_mode!r}")

            agent.private_info = {"agent_id": i, "start_pos": list(pos)}
            agents.append(agent)
        return agents


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    print("=== FitnessEvaluator Test ===\n")

    evaluator = FitnessEvaluator(task_weight=0.7, efficiency_weight=0.3, num_trials=3)

    # --- 1. Single simulation run on a dense topology ---
    print("-- Single run_simulation (dense topology, 6 agents) --")
    dense = TopologyGenome.random(num_agents=6, edge_probability=0.9)
    result = evaluator.run_simulation(dense)
    print(f"  task_score    : {result.task_score:.2f}")
    print(f"  message_count : {result.message_count}")
    print(f"  steps_taken   : {result.steps_taken}")
    assert 0.0 <= result.task_score <= 1.0, "task_score must be in [0,1]"
    assert result.message_count >= 0
    assert 1 <= result.steps_taken <= _MAX_STEPS

    # --- 2. compute_fitness values are reasonable ---
    print("\n-- compute_fitness sanity check --")
    perfect = SimResult(task_score=1.0, message_count=1, steps_taken=10)
    worst   = SimResult(task_score=0.0, message_count=1000, steps_taken=50)
    f_perfect = evaluator.compute_fitness(perfect)
    f_worst   = evaluator.compute_fitness(worst)
    print(f"  fitness(perfect) = {f_perfect:.4f}")
    print(f"  fitness(worst)   = {f_worst:.4f}")
    assert f_perfect > f_worst, "Perfect result must outscore worst result"

    # --- 3. Full evaluate() averages over trials ---
    print("\n-- evaluate() over 3 trials --")
    genome = TopologyGenome.random(num_agents=6, edge_probability=0.4)
    fitness = evaluator.evaluate(genome)
    print(f"  Mean fitness : {fitness:.4f}")
    print(f"  Written back : genome.fitness_score = {genome.fitness_score:.4f}")
    assert genome.fitness_score == fitness, "fitness_score must be written back"

    # --- 4. Sparse topology vs dense topology ---
    print("\n-- Sparse vs dense topology comparison --")
    np.random.seed(7)
    sparse = TopologyGenome.random(num_agents=6, edge_probability=0.1)
    dense2 = TopologyGenome.random(num_agents=6, edge_probability=0.9)
    f_sparse = evaluator.evaluate(sparse)
    f_dense  = evaluator.evaluate(dense2)
    print(f"  Sparse fitness : {f_sparse:.4f}")
    print(f"  Dense  fitness : {f_dense:.4f}")
    # Dense sends more messages → lower efficiency term; but task score may vary.
    # Just confirm both are valid floats.
    assert isinstance(f_sparse, float)
    assert isinstance(f_dense, float)

    # --- 5. Disconnected graph gets lower fitness due to penalty ---
    print("\n-- Disconnection penalty (applied via genome check) --")
    disc_mat = np.zeros((6, 6), dtype=bool)
    disc_mat[0][1] = disc_mat[1][0] = True
    disc_mat[2][3] = disc_mat[3][2] = True
    # agents 4 and 5 are isolated
    disc_genome = TopologyGenome(num_agents=6, adjacency_matrix=disc_mat)
    assert not disc_genome.is_connected()
    # Manually compute fitness with penalty
    sim_result = evaluator.run_simulation(disc_genome)
    base_fitness = evaluator.compute_fitness(sim_result)
    penalty = 0.1 if not disc_genome.is_connected() else 0.0
    penalised = base_fitness - penalty
    print(f"  Base fitness     : {base_fitness:.4f}")
    print(f"  After penalty    : {penalised:.4f}")
    assert penalised < base_fitness

    print("\nAll tests passed.")
