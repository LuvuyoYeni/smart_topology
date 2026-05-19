from __future__ import annotations

import numpy as np
from collections import namedtuple

from environment import Environment
from topology_genome import TopologyGenome
from agent import CooperativeAgent

SimResult = namedtuple("SimResult", ["task_score", "message_count", "steps_taken"])

_MAX_STEPS = 50
_GRID_SIZE = 10
_NUM_TASKS = 8
_TASK_REQUIREMENT = 3


class FitnessEvaluator:

    def __init__(
        self,
        task_weight: float = 0.6,
        efficiency_weight: float = 0.3,
        msg_weight: float = 0.3,
        num_trials: int = 3,
        agent_mode: str = "cooperative",
    ):
        self.task_weight: float = task_weight
        self.efficiency_weight: float = efficiency_weight
        self.msg_weight: float = msg_weight
        self.num_trials: int = num_trials
        self.agent_mode: str = agent_mode

    def evaluate(self, genome: TopologyGenome) -> float:
        scores = [self.compute_fitness(self.run_simulation(genome))
                  for _ in range(self.num_trials)]
        disconnection_penalty = 0.1 if not genome.is_connected() else 0.0
        fitness = float(np.mean(scores)) - disconnection_penalty
        genome.fitness_score = fitness
        return fitness

    def run_simulation(self, topology: TopologyGenome) -> SimResult:
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

            for agent in agents:
                agent.perceive(env)

            for agent in agents:
                neighbour_ids = comm_graph.get_neighbours(agent.agent_id)
                neighbours = [agents[nid] for nid in neighbour_ids]

                if agent.decide_to_share(agent.private_info) and neighbours:
                    agent.share_all(neighbours)
                    total_messages += len(neighbours)

            for agent in agents:
                for msg in agent.inbox:
                    agent.receive_message(msg)
                agent.inbox.clear()

            for agent in agents:
                agent.act(env)

            positions = [agent.position for agent in agents]
            env.update(positions)

            active_ids = {t["id"] for t in env.get_active_tasks()}
            for agent in agents:
                stale = [tid for tid in agent.private_info.get("known_tasks", {})
                         if tid not in active_ids]
                for tid in stale:
                    del agent.private_info["known_tasks"][tid]

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
        speed_bonus = (_MAX_STEPS - result.steps_taken) / _MAX_STEPS
        msg_penalty = min(result.message_count / 30.0, 1.0)

        fitness = (
            self.task_weight * result.task_score
            + self.efficiency_weight * speed_bonus
            - self.msg_weight * msg_penalty
        )

        return round(max(fitness, 0.0), 4)

    def _spawn_agents(self, num_agents: int, grid_size: int) -> list:
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

            agent.private_info = {"agent_id": i, "start_pos": list(pos), "known_tasks": {}}
            agents.append(agent)
        return agents


if __name__ == "__main__":
    np.random.seed(0)
    print("=== FitnessEvaluator Test ===\n")

    evaluator = FitnessEvaluator(task_weight=0.7, efficiency_weight=0.3, num_trials=3)

    print("-- Single run_simulation (dense topology, 6 agents) --")
    dense = TopologyGenome.random(num_agents=6, edge_probability=0.9)
    result = evaluator.run_simulation(dense)
    print(f"  task_score    : {result.task_score:.2f}")
    print(f"  message_count : {result.message_count}")
    print(f"  steps_taken   : {result.steps_taken}")
    assert 0.0 <= result.task_score <= 1.0, "task_score must be in [0,1]"
    assert result.message_count >= 0
    assert 1 <= result.steps_taken <= _MAX_STEPS

    print("\n-- compute_fitness sanity check --")
    perfect = SimResult(task_score=1.0, message_count=1, steps_taken=10)
    worst   = SimResult(task_score=0.0, message_count=1000, steps_taken=50)
    f_perfect = evaluator.compute_fitness(perfect)
    f_worst   = evaluator.compute_fitness(worst)
    print(f"  fitness(perfect) = {f_perfect:.4f}")
    print(f"  fitness(worst)   = {f_worst:.4f}")
    assert f_perfect > f_worst, "Perfect result must outscore worst result"

    print("\n-- evaluate() over 3 trials --")
    genome = TopologyGenome.random(num_agents=6, edge_probability=0.4)
    fitness = evaluator.evaluate(genome)
    print(f"  Mean fitness : {fitness:.4f}")
    print(f"  Written back : genome.fitness_score = {genome.fitness_score:.4f}")
    assert genome.fitness_score == fitness, "fitness_score must be written back"

    print("\n-- Sparse vs dense topology comparison --")
    np.random.seed(7)
    sparse = TopologyGenome.random(num_agents=6, edge_probability=0.1)
    dense2 = TopologyGenome.random(num_agents=6, edge_probability=0.9)
    f_sparse = evaluator.evaluate(sparse)
    f_dense  = evaluator.evaluate(dense2)
    print(f"  Sparse fitness : {f_sparse:.4f}")
    print(f"  Dense  fitness : {f_dense:.4f}")
    assert isinstance(f_sparse, float)
    assert isinstance(f_dense, float)

    print("\n-- Disconnection penalty (applied via genome check) --")
    disc_mat = np.zeros((6, 6), dtype=bool)
    disc_mat[0][1] = disc_mat[1][0] = True
    disc_mat[2][3] = disc_mat[3][2] = True
    disc_genome = TopologyGenome(num_agents=6, adjacency_matrix=disc_mat)
    assert not disc_genome.is_connected()
    sim_result = evaluator.run_simulation(disc_genome)
    base_fitness = evaluator.compute_fitness(sim_result)
    penalty = 0.1 if not disc_genome.is_connected() else 0.0
    penalised = base_fitness - penalty
    print(f"  Base fitness     : {base_fitness:.4f}")
    print(f"  After penalty    : {penalised:.4f}")
    assert penalised < base_fitness

    print("\nAll tests passed.")
