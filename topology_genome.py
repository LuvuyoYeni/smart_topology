from __future__ import annotations

import numpy as np
from communication_graph import CommunicationGraph


class TopologyGenome:

    def __init__(self, num_agents: int, adjacency_matrix: np.ndarray | None = None):
        self.num_agents: int = num_agents
        self.fitness_score: float = 0.0

        if adjacency_matrix is not None:
            self.adjacency_matrix: np.ndarray = adjacency_matrix.astype(bool)
        else:
            self.adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)

        np.fill_diagonal(self.adjacency_matrix, False)
        self.graph_density: float = self.calculate_density()

    @classmethod
    def random(cls, num_agents: int, edge_probability: float = 0.3) -> "TopologyGenome":
        mat = np.zeros((num_agents, num_agents), dtype=bool)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if np.random.random() < edge_probability:
                    mat[i][j] = True
                    mat[j][i] = True
        return cls(num_agents=num_agents, adjacency_matrix=mat)

    def mutate(self, rate: float) -> None:
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
        absent = self._absent_edge_pairs()
        if absent:
            i, j = absent[np.random.randint(len(absent))]
            self.adjacency_matrix[i][j] = True
            self.adjacency_matrix[j][i] = True

    def _remove_random_edge(self) -> None:
        present = self._present_edge_pairs()
        if present:
            i, j = present[np.random.randint(len(present))]
            self.adjacency_matrix[i][j] = False
            self.adjacency_matrix[j][i] = False

    def _rewire_random_edge(self) -> None:
        self._remove_random_edge()
        self._add_random_edge()

    def _present_edge_pairs(self):
        rows, cols = np.where(np.triu(self.adjacency_matrix, k=1))
        return list(zip(rows.tolist(), cols.tolist()))

    def _absent_edge_pairs(self):
        upper = np.triu(np.ones((self.num_agents, self.num_agents), dtype=bool), k=1)
        absent_mask = upper & ~self.adjacency_matrix
        rows, cols = np.where(absent_mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def crossover(self, other: "TopologyGenome") -> "TopologyGenome":
        child_mat = np.zeros((self.num_agents, self.num_agents), dtype=bool)

        for i in range(self.num_agents):
            if np.random.random() < 0.5:
                child_mat[i] = self.adjacency_matrix[i].copy()
            else:
                child_mat[i] = other.adjacency_matrix[i].copy()

        child_mat = child_mat | child_mat.T
        np.fill_diagonal(child_mat, False)

        return TopologyGenome(num_agents=self.num_agents, adjacency_matrix=child_mat)

    def get_neighbours(self, agent_id: int) -> list[int]:
        return [j for j in range(self.num_agents)
                if self.adjacency_matrix[agent_id][j]]

    def is_connected(self) -> bool:
        if self.num_agents <= 1:
            return True

        visited = {0}
        queue = [0]
        while queue:
            current = queue.pop()
            for j in range(self.num_agents):
                if self.adjacency_matrix[current][j] and j not in visited:
                    visited.add(j)
                    queue.append(j)

        return len(visited) == self.num_agents

    def calculate_density(self) -> float:
        max_edges = self.num_agents * (self.num_agents - 1) / 2
        if max_edges == 0:
            return 0.0
        actual_edges = np.sum(np.triu(self.adjacency_matrix, k=1))
        return float(actual_edges / max_edges)

    def to_communication_graph(self) -> CommunicationGraph:
        nodes = list(range(self.num_agents))
        edges = []
        rows, cols = np.where(np.triu(self.adjacency_matrix, k=1))
        for i, j in zip(rows.tolist(), cols.tolist()):
            edges.append((i, j))
        return CommunicationGraph(nodes=nodes, edges=edges, directed=False)

    def __repr__(self) -> str:
        return (f"TopologyGenome(num_agents={self.num_agents}, "
                f"density={self.graph_density:.2f}, "
                f"fitness={self.fitness_score:.4f}, "
                f"connected={self.is_connected()})")


if __name__ == "__main__":
    np.random.seed(42)
    print("=== TopologyGenome Test ===\n")

    genome = TopologyGenome.random(num_agents=5, edge_probability=0.5)
    print("Random genome:", genome)
    print("Adjacency matrix:\n", genome.adjacency_matrix.astype(int))
    print("Neighbours of 0:", genome.get_neighbours(0))
    print("Is connected:", genome.is_connected())
    print("Density:", round(genome.graph_density, 3))

    print("\n-- Ring topology (0-1-2-3-0) --")
    mat = np.zeros((4, 4), dtype=bool)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        mat[a][b] = mat[b][a] = True
    ring = TopologyGenome(num_agents=4, adjacency_matrix=mat)
    print(ring)
    print("Neighbours of 2:", ring.get_neighbours(2))

    print("\n-- Mutation test --")
    before_density = ring.graph_density
    for _ in range(5):
        ring.mutate(rate=1.0)
    print(f"Density before mutations: {before_density:.3f}")
    print(f"Density after  mutations: {ring.graph_density:.3f}")

    print("\n-- Crossover test --")
    parent_a = TopologyGenome.random(num_agents=6, edge_probability=0.3)
    parent_b = TopologyGenome.random(num_agents=6, edge_probability=0.7)
    child = parent_a.crossover(parent_b)
    print("Parent A:", parent_a)
    print("Parent B:", parent_b)
    print("Child   :", child)
    assert np.array_equal(child.adjacency_matrix, child.adjacency_matrix.T), \
        "Child matrix must be symmetric"
    print("Symmetry check passed.")

    print("\n-- Connectivity test --")
    disconnected_mat = np.zeros((4, 4), dtype=bool)
    disconnected_mat[0][1] = disconnected_mat[1][0] = True
    disconnected_mat[2][3] = disconnected_mat[3][2] = True
    disc = TopologyGenome(num_agents=4, adjacency_matrix=disconnected_mat)
    print("Disconnected genome:", disc)
    assert not disc.is_connected(), "Should be disconnected"
    print("Disconnection correctly detected.")

    print("\n-- CommunicationGraph round-trip --")
    g = parent_a.to_communication_graph()
    print(f"Nodes: {g.nodes}")
    print(f"Edges: {g.edges}")
    expected_edges = int(np.sum(np.triu(parent_a.adjacency_matrix, k=1)))
    assert len(g.edges) == expected_edges, \
        f"Edge count mismatch: {len(g.edges)} vs {expected_edges}"
    print(f"Edge count matches adjacency matrix ({expected_edges} edges).")

    print("\nAll tests passed.")
