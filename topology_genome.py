"""
topology_genome.py
------------------
Represents a single candidate communication topology as a binary adjacency
matrix.  Provides genetic operators (mutate, crossover) and utility methods
used by the evolutionary engine.
"""

import numpy as np
from communication_graph import CommunicationGraph


class TopologyGenome:
    """
    A candidate solution in the evolutionary algorithm.

    The topology is stored as a symmetric boolean adjacency matrix so that
    standard numpy operations can be used throughout.  Self-loops on the
    diagonal are always False (an agent never needs to 'communicate' with
    itself).

    Attributes
    ----------
    num_agents : int
        Number of agents (= matrix dimension).
    adjacency_matrix : np.ndarray[bool]
        n x n symmetric boolean array; entry [i][j] is True when agents
        i and j share a communication link.
    fitness_score : float
        Assigned by FitnessEvaluator; initialised to 0.0.
    graph_density : float
        Fraction of possible edges that are present; updated whenever the
        matrix changes.
    """

    def __init__(self, num_agents: int, adjacency_matrix: np.ndarray | None = None):
        """
        Create a genome.

        Parameters
        ----------
        num_agents : int
            How many agents this topology covers.
        adjacency_matrix : np.ndarray or None
            Pre-built boolean matrix.  If None a zero (no-edge) matrix is
            created; callers that want random initialisation should use
            :meth:`TopologyGenome.random`.
        """
        self.num_agents: int = num_agents
        self.fitness_score: float = 0.0

        if adjacency_matrix is not None:
            self.adjacency_matrix: np.ndarray = adjacency_matrix.astype(bool)
        else:
            self.adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)

        # Keep diagonal clear and recompute density
        np.fill_diagonal(self.adjacency_matrix, False)
        self.graph_density: float = self.calculate_density()

    # ------------------------------------------------------------------ #
    #  Class-level factory                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, num_agents: int, edge_probability: float = 0.3) -> "TopologyGenome":
        """
        Factory: create a genome whose edges exist with *edge_probability*.

        The matrix is forced to be symmetric (undirected topology).

        Parameters
        ----------
        num_agents : int
            Number of agents.
        edge_probability : float
            Independent probability that any given edge is present.
        """
        mat = np.zeros((num_agents, num_agents), dtype=bool)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if np.random.random() < edge_probability:
                    mat[i][j] = True
                    mat[j][i] = True
        genome = cls(num_agents=num_agents, adjacency_matrix=mat)
        return genome

    # ------------------------------------------------------------------ #
    #  Genetic operators                                                   #
    # ------------------------------------------------------------------ #

    def mutate(self, rate: float) -> None:
        """
        Apply in-place mutation to the topology.

        One mutation type is chosen uniformly at random per call:

        * **add edge**   — pick two unconnected agents and link them.
        * **remove edge** — pick an existing edge and delete it.
        * **rewire edge** — remove an existing edge and add a new one.

        The mutation is only applied if a random draw is below *rate*.

        Parameters
        ----------
        rate : float
            Probability in [0, 1] that a mutation actually occurs.
        """
        if np.random.random() >= rate:
            return

        mutation_type = np.random.choice(["add", "remove", "rewire"])

        if mutation_type == "add":
            self._add_random_edge()
        elif mutation_type == "remove":
            self._remove_random_edge()
        else:
            self._rewire_random_edge()

        np.fill_diagonal(self.adjacency_matrix, False)
        self.graph_density = self.calculate_density()

    def _add_random_edge(self) -> None:
        """Add a randomly chosen absent edge."""
        absent = self._absent_edge_pairs()
        if absent:
            i, j = absent[np.random.randint(len(absent))]
            self.adjacency_matrix[i][j] = True
            self.adjacency_matrix[j][i] = True

    def _remove_random_edge(self) -> None:
        """Remove a randomly chosen present edge."""
        present = self._present_edge_pairs()
        if present:
            i, j = present[np.random.randint(len(present))]
            self.adjacency_matrix[i][j] = False
            self.adjacency_matrix[j][i] = False

    def _rewire_random_edge(self) -> None:
        """Remove one edge and add a different absent edge."""
        self._remove_random_edge()
        self._add_random_edge()

    def _present_edge_pairs(self) -> list[tuple[int, int]]:
        """Return upper-triangle indices where an edge exists."""
        rows, cols = np.where(np.triu(self.adjacency_matrix, k=1))
        return list(zip(rows.tolist(), cols.tolist()))

    def _absent_edge_pairs(self) -> list[tuple[int, int]]:
        """Return upper-triangle indices where no edge exists."""
        upper = np.triu(np.ones((self.num_agents, self.num_agents), dtype=bool), k=1)
        absent_mask = upper & ~self.adjacency_matrix
        rows, cols = np.where(absent_mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def crossover(self, other: "TopologyGenome") -> "TopologyGenome":
        """
        Produce one offspring via row-wise recombination.

        For each row i the offspring inherits either self's row or other's
        row with equal probability.  The result is then symmetrised so the
        matrix stays undirected.

        Parameters
        ----------
        other : TopologyGenome
            The second parent; must have the same num_agents.

        Returns
        -------
        TopologyGenome
            A new genome (parents are unchanged).
        """
        child_mat = np.zeros((self.num_agents, self.num_agents), dtype=bool)

        for i in range(self.num_agents):
            if np.random.random() < 0.5:
                child_mat[i] = self.adjacency_matrix[i].copy()
            else:
                child_mat[i] = other.adjacency_matrix[i].copy()

        # Symmetrise: if either direction says True, set both
        child_mat = child_mat | child_mat.T
        np.fill_diagonal(child_mat, False)

        return TopologyGenome(num_agents=self.num_agents, adjacency_matrix=child_mat)

    # ------------------------------------------------------------------ #
    #  Graph utility methods                                               #
    # ------------------------------------------------------------------ #

    def get_neighbours(self, agent_id: int) -> list[int]:
        """
        Return the list of agents directly connected to *agent_id*.

        Parameters
        ----------
        agent_id : int
            Index of the queried agent (0-based).
        """
        return [j for j in range(self.num_agents)
                if self.adjacency_matrix[agent_id][j]]

    def is_connected(self) -> bool:
        """
        Return True if every agent can reach every other agent.

        Uses BFS from node 0; a graph is connected iff BFS visits all nodes.
        An empty or single-node graph is considered connected.
        """
        if self.num_agents <= 1:
            return True

        visited: set[int] = {0}
        queue = [0]
        while queue:
            current = queue.pop()
            for j in range(self.num_agents):
                if self.adjacency_matrix[current][j] and j not in visited:
                    visited.add(j)
                    queue.append(j)

        return len(visited) == self.num_agents

    def calculate_density(self) -> float:
        """
        Compute and return the graph density.

        Density = (number of edges) / (maximum possible edges).
        Maximum possible edges for n nodes = n*(n-1)/2.
        Returns 0.0 for graphs with fewer than 2 nodes.
        """
        max_edges = self.num_agents * (self.num_agents - 1) / 2
        if max_edges == 0:
            return 0.0
        actual_edges = np.sum(np.triu(self.adjacency_matrix, k=1))
        return float(actual_edges / max_edges)

    def to_communication_graph(self) -> CommunicationGraph:
        """
        Convert this genome into a :class:`CommunicationGraph` for use
        by agents at runtime.

        Returns
        -------
        CommunicationGraph
            Undirected graph built from the adjacency matrix.
        """
        nodes = list(range(self.num_agents))
        edges: list[tuple[int, int]] = []
        rows, cols = np.where(np.triu(self.adjacency_matrix, k=1))
        for i, j in zip(rows.tolist(), cols.tolist()):
            edges.append((i, j))
        return CommunicationGraph(nodes=nodes, edges=edges, directed=False)

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"TopologyGenome(num_agents={self.num_agents}, "
                f"density={self.graph_density:.2f}, "
                f"fitness={self.fitness_score:.4f}, "
                f"connected={self.is_connected()})")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("=== TopologyGenome Test ===\n")

    # --- 1. Random creation ---
    genome = TopologyGenome.random(num_agents=5, edge_probability=0.5)
    print("Random genome:", genome)
    print("Adjacency matrix:\n", genome.adjacency_matrix.astype(int))
    print("Neighbours of 0:", genome.get_neighbours(0))
    print("Is connected:", genome.is_connected())
    print("Density:", round(genome.graph_density, 3))

    # --- 2. Manual genome (fully connected ring) ---
    print("\n-- Ring topology (0-1-2-3-0) --")
    mat = np.zeros((4, 4), dtype=bool)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        mat[a][b] = mat[b][a] = True
    ring = TopologyGenome(num_agents=4, adjacency_matrix=mat)
    print(ring)
    print("Neighbours of 2:", ring.get_neighbours(2))

    # --- 3. Mutation ---
    print("\n-- Mutation test --")
    before_density = ring.graph_density
    # Force mutation with rate=1.0 several times
    for _ in range(5):
        ring.mutate(rate=1.0)
    print(f"Density before mutations: {before_density:.3f}")
    print(f"Density after  mutations: {ring.graph_density:.3f}")

    # --- 4. Crossover ---
    print("\n-- Crossover test --")
    parent_a = TopologyGenome.random(num_agents=6, edge_probability=0.3)
    parent_b = TopologyGenome.random(num_agents=6, edge_probability=0.7)
    child = parent_a.crossover(parent_b)
    print("Parent A:", parent_a)
    print("Parent B:", parent_b)
    print("Child   :", child)
    # Verify symmetry
    assert np.array_equal(child.adjacency_matrix, child.adjacency_matrix.T), \
        "Child matrix must be symmetric"
    print("Symmetry check passed.")

    # --- 5. Connectivity check ---
    print("\n-- Connectivity test --")
    disconnected_mat = np.zeros((4, 4), dtype=bool)
    disconnected_mat[0][1] = disconnected_mat[1][0] = True  # component {0,1}
    disconnected_mat[2][3] = disconnected_mat[3][2] = True  # component {2,3}
    disc = TopologyGenome(num_agents=4, adjacency_matrix=disconnected_mat)
    print("Disconnected genome:", disc)
    assert not disc.is_connected(), "Should be disconnected"
    print("Disconnection correctly detected.")

    # --- 6. to_communication_graph() round-trip ---
    print("\n-- CommunicationGraph round-trip --")
    g = parent_a.to_communication_graph()
    print(f"Nodes: {g.nodes}")
    print(f"Edges: {g.edges}")
    # Verify edge count matches
    expected_edges = int(np.sum(np.triu(parent_a.adjacency_matrix, k=1)))
    assert len(g.edges) == expected_edges, \
        f"Edge count mismatch: {len(g.edges)} vs {expected_edges}"
    print(f"Edge count matches adjacency matrix ({expected_edges} edges).")

    print("\nAll tests passed.")
