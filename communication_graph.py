"""
communication_graph.py
----------------------
Lightweight graph structure representing which agents can communicate.
Used by agents at runtime to determine reachable neighbours.
Internal logic uses adjacency lists — networkx is reserved for visualisation only.
"""

from collections import deque


class CommunicationGraph:
    """
    A graph whose nodes are agent IDs and whose edges represent
    permitted communication links between agents.

    Attributes
    ----------
    nodes : list[int]
        All agent IDs in the graph.
    edges : list[tuple[int, int]]
        Directed or undirected communication links as (sender, receiver) pairs.
    directed : bool
        If False, every edge is treated as bidirectional.
    """

    def __init__(self, nodes: list[int], edges: list[tuple[int, int]], directed: bool = False):
        """
        Initialise the graph.

        Parameters
        ----------
        nodes : list[int]
            Agent IDs that form the vertices of the graph.
        edges : list[tuple[int, int]]
            Pairs (a, b) indicating a communication link between agent a and b.
        directed : bool
            Whether edges are one-way (default False — undirected).
        """
        self.nodes: list[int] = list(nodes)
        self.edges: list[tuple[int, int]] = list(edges)
        self.directed: bool = directed

        # Build adjacency list for O(1) neighbour lookup
        self._adj: dict[int, set[int]] = {n: set() for n in self.nodes}
        for a, b in self.edges:
            self._adj[a].add(b)
            if not self.directed:
                self._adj[b].add(a)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def can_communicate(self, a: int, b: int) -> bool:
        """
        Return True if agent *a* can send a message directly to agent *b*.

        For undirected graphs this is symmetric; for directed graphs it
        checks only the a → b direction.

        Parameters
        ----------
        a : int
            Sending agent ID.
        b : int
            Receiving agent ID.
        """
        return b in self._adj.get(a, set())

    def get_degree(self, agent_id: int) -> int:
        """
        Return the number of direct communication links for *agent_id*.

        For directed graphs this returns the out-degree only.

        Parameters
        ----------
        agent_id : int
            The agent whose degree is requested.
        """
        return len(self._adj.get(agent_id, set()))

    def get_shortest_path(self, a: int, b: int) -> list[int]:
        """
        Return the shortest path (list of agent IDs) from *a* to *b*
        using breadth-first search.

        Returns an empty list if no path exists or if either node is
        absent from the graph.

        Parameters
        ----------
        a : int
            Source agent ID.
        b : int
            Target agent ID.
        """
        if a not in self._adj or b not in self._adj:
            return []
        if a == b:
            return [a]

        visited: set[int] = {a}
        # Each queue entry is the path so far
        queue: deque[list[int]] = deque([[a]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbour in self._adj[current]:
                if neighbour == b:
                    return path + [neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        return []  # no path found

    def get_neighbours(self, agent_id: int) -> list[int]:
        """
        Return the list of agents directly reachable from *agent_id*.

        Parameters
        ----------
        agent_id : int
            The agent whose neighbours are requested.
        """
        return list(self._adj.get(agent_id, set()))


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== CommunicationGraph Test ===\n")

    # Build a simple undirected graph:
    #   0 -- 1 -- 2
    #        |
    #        3 -- 4
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    graph = CommunicationGraph(nodes=nodes, edges=edges, directed=False)

    print("Nodes:", graph.nodes)
    print("Edges:", graph.edges)

    print("\n-- can_communicate --")
    print(f"0 <-> 1 : {graph.can_communicate(0, 1)}")   # True
    print(f"1 <-> 0 : {graph.can_communicate(1, 0)}")   # True (undirected)
    print(f"0 <-> 2 : {graph.can_communicate(0, 2)}")   # False (not directly linked)
    print(f"3 <-> 4 : {graph.can_communicate(3, 4)}")   # True

    print("\n-- get_degree --")
    for n in nodes:
        print(f"  degree({n}) = {graph.get_degree(n)}")
    # Expected: 0→1, 1→3, 2→1, 3→2, 4→1

    print("\n-- get_neighbours --")
    for n in nodes:
        print(f"  neighbours({n}) = {sorted(graph.get_neighbours(n))}")

    print("\n-- get_shortest_path --")
    print(f"  0 to 4 : {graph.get_shortest_path(0, 4)}")   # [0,1,3,4]
    print(f"  2 to 4 : {graph.get_shortest_path(2, 4)}")   # [2,1,3,4]
    print(f"  0 to 0 : {graph.get_shortest_path(0, 0)}")   # [0]

    # Test directed graph
    print("\n-- directed graph test --")
    dgraph = CommunicationGraph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)], directed=True)
    print(f"  directed 0->1: {dgraph.can_communicate(0, 1)}")   # True
    print(f"  directed 1->0: {dgraph.can_communicate(1, 0)}")   # False

    # Test disconnected graph (no path between components)
    print("\n-- disconnected graph test --")
    dgraph2 = CommunicationGraph(nodes=[0, 1, 2, 3], edges=[(0, 1), (2, 3)], directed=False)
    print(f"  0 to 3 : {dgraph2.get_shortest_path(0, 3)}")   # []
    print(f"  0 to 1 : {dgraph2.get_shortest_path(0, 1)}")   # [0, 1]

    print("\nAll tests passed.")
