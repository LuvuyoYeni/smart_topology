"""
visualise_topology.py
---------------------
Instruments a simulation to count message traffic across every edge, then
renders the topology with edges coloured and sized by traffic volume.

Thick warm-coloured edges = heavily used communication links.
Thin cool edges           = present in topology but rarely used.
Red nodes                 = isolated (no neighbours).
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from agent import CooperativeAgent, StrategicAgent
from environment import Environment
from fitness_evaluator import _GRID_SIZE, _MAX_STEPS, _NUM_TASKS, _TASK_REQUIREMENT
from message import Message
from topology_genome import TopologyGenome


# ── Traffic measurement ───────────────────────────────────────────────────────

def measure_edge_traffic(
    genome: TopologyGenome,
    agent_mode: str = "cooperative",
    num_trials: int = 10,
) -> dict[tuple[int, int], float]:
    """
    Run *num_trials* simulations and return the average number of messages
    that traversed each directed edge.

    Parameters
    ----------
    genome : TopologyGenome
        The topology to instrument.
    agent_mode : str
        ``"cooperative"``, ``"strategic"``, or ``"mixed"``.
    num_trials : int
        Number of independent runs to average over.

    Returns
    -------
    dict[tuple[int, int], float]
        Maps (sender_id, receiver_id) → mean message count per trial.
    """
    comm_graph = genome.to_communication_graph()
    num_agents = genome.num_agents
    edge_traffic: dict[tuple[int, int], list[int]] = defaultdict(list)

    for _ in range(num_trials):
        trial_counts: dict[tuple[int, int], int] = defaultdict(int)
        agents = _spawn_agents(num_agents, _GRID_SIZE, agent_mode)
        env = Environment(
            grid_size=_GRID_SIZE,
            num_tasks=_NUM_TASKS,
            task_requirement=_TASK_REQUIREMENT,
        )

        for step in range(_MAX_STEPS):
            # Communication phase — record each message with its edge
            for agent in agents:
                neighbour_ids = comm_graph.get_neighbours(agent.agent_id)
                neighbours = [agents[nid] for nid in neighbour_ids]

                if agent.decide_to_share(agent.private_info) and neighbours:
                    for neighbour in neighbours:
                        edge = (agent.agent_id, neighbour.agent_id)
                        trial_counts[edge] += 1
                        msg = Message(
                            sender_id=agent.agent_id,
                            receiver_id=neighbour.agent_id,
                            content=dict(agent.private_info),
                        )
                        neighbour.inbox.append(msg)

            # Process inboxes
            for agent in agents:
                for msg in agent.inbox:
                    agent.receive_message(msg)
                agent.inbox.clear()

            # Movement and environment update
            for agent in agents:
                agent.act(env)
            env.update([a.position for a in agents])

            if not env.get_active_tasks():
                break

        for edge, count in trial_counts.items():
            edge_traffic[edge].append(count)

    return {edge: float(np.mean(counts))
            for edge, counts in edge_traffic.items()}


# ── Single topology visualisation ─────────────────────────────────────────────

def visualise(
    genome: TopologyGenome,
    title: str = "Topology",
    agent_mode: str = "cooperative",
    num_trials: int = 10,
    ax: plt.Axes | None = None,
) -> None:
    """
    Draw *genome* with edges coloured and sized by communication traffic.

    Parameters
    ----------
    genome : TopologyGenome
        The topology to visualise.
    title : str
        Plot title prefix.
    agent_mode : str
        Agent behaviour mode used for the traffic measurement simulation.
    num_trials : int
        Simulation runs to average traffic over.
    ax : plt.Axes or None
        If provided, draw into this axes; otherwise create a new figure.
    """
    traffic = measure_edge_traffic(genome, agent_mode, num_trials)

    G = nx.Graph()
    G.add_nodes_from(range(genome.num_agents))

    for i in range(genome.num_agents):
        for j in range(i + 1, genome.num_agents):
            if genome.adjacency_matrix[i][j]:
                flow = traffic.get((i, j), 0.0) + traffic.get((j, i), 0.0)
                G.add_edge(i, j, weight=flow)

    pos = nx.spring_layout(G, seed=42, k=1.5)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))

    connected_str = "connected" if genome.is_connected() else "DISCONNECTED"
    ax.set_title(
        f"{title}  [{agent_mode} agents]\n"
        f"density={genome.calculate_density():.2f}  "
        f"fitness={genome.fitness_score:.4f}  "
        f"({connected_str})",
        fontsize=11, pad=14,
    )
    ax.axis("off")

    # Nodes: red if isolated, steelblue otherwise
    node_colours = [
        "tomato" if G.degree(n) == 0 else "steelblue"
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                            font_size=11, font_weight="bold")

    # Edges: colour and width scaled by traffic
    if G.edges():
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        max_w = max(float(weights.max()), 1.0)
        norm = mcolors.Normalize(vmin=0, vmax=max_w)
        cmap = plt.colormaps["YlOrRd"]

        edge_colours = [cmap(norm(w)) for w in weights]
        edge_widths = [1.0 + 5.0 * (w / max_w) for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colours,
                               width=edge_widths, alpha=0.85)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02,
                     label="Avg messages per edge (across trials)")

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Side-by-side comparison ───────────────────────────────────────────────────

def compare_modes(
    genome_coop: TopologyGenome,
    genome_strat: TopologyGenome,
) -> None:
    """
    Side-by-side traffic visualisation of both evolved topologies.

    Both are rendered under cooperative agent mode so the traffic patterns
    are directly comparable — the difference you see is purely structural.

    Parameters
    ----------
    genome_coop : TopologyGenome
        Topology evolved under cooperative agents.
    genome_strat : TopologyGenome
        Topology evolved under strategic agents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Communication Flow: Cooperatively-Evolved vs Strategically-Evolved Topology\n"
        "(both rendered under cooperative agents for direct comparison)",
        fontsize=12, y=1.02,
    )
    visualise(genome_coop,  "Evol-Coop",  agent_mode="cooperative",
              num_trials=10, ax=axes[0])
    visualise(genome_strat, "Evol-Strat", agent_mode="cooperative",
              num_trials=10, ax=axes[1])
    plt.tight_layout()
    plt.show()


# ── Helper: spawn agents ──────────────────────────────────────────────────────

def _spawn_agents(num_agents: int, grid_size: int, mode: str) -> list:
    """
    Instantiate agents of the requested type at random positions.

    Parameters
    ----------
    num_agents : int
        Number of agents to create.
    grid_size : int
        Grid side length for random position bounds.
    mode : str
        ``"cooperative"``, ``"strategic"``, or ``"mixed"``.
    """
    agents = []
    for i in range(num_agents):
        pos = (int(np.random.randint(0, grid_size)),
               int(np.random.randint(0, grid_size)))
        if mode == "cooperative":
            agent = CooperativeAgent(agent_id=i, position=pos)
        elif mode == "strategic":
            agent = StrategicAgent(agent_id=i, position=pos,
                                   deception_threshold=0.4)
        elif mode == "mixed":
            if i % 2 == 0:
                agent = CooperativeAgent(agent_id=i, position=pos)
            else:
                agent = StrategicAgent(agent_id=i, position=pos,
                                       deception_threshold=0.4)
        else:
            raise ValueError(f"Unknown agent_mode: {mode!r}")
        agent.private_info = {"agent_id": i, "start_pos": list(pos)}
        agents.append(agent)
    return agents


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    from evolution_engine import EvolutionEngine
    from strategic_evolution_fix import ConnectivityEnforcedEvaluator

    np.random.seed(1)
    random.seed(1)

    print("=== visualise_topology.py Test ===\n")
    print("Evolving cooperative topology (50 generations)...")
    engine_coop = EvolutionEngine(agent_mode="cooperative")
    genome_coop = engine_coop.run()
    print(f"  Result: {genome_coop}\n")

    print("Evolving strategic topology (50 generations)...")
    engine_strat = EvolutionEngine(agent_mode="strategic")
    engine_strat.fitness_evaluator = ConnectivityEnforcedEvaluator(
        num_trials=5, agent_mode="strategic"
    )
    genome_strat = engine_strat.run()
    print(f"  Result: {genome_strat}\n")

    print("Measuring traffic and rendering visualisations...")
    print("  Window 1: Evol-Coop under cooperative agents")
    visualise(genome_coop, "Evol-Coop", agent_mode="cooperative", num_trials=10)

    print("  Window 2: Evol-Coop under strategic agents")
    visualise(genome_coop, "Evol-Coop", agent_mode="strategic", num_trials=10)

    print("  Window 3: Side-by-side comparison")
    compare_modes(genome_coop, genome_strat)

    print("\nDone.")
