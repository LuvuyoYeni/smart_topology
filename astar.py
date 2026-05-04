"""
astar.py
--------
A* pathfinding on a discrete grid.

Finds the shortest path from start to goal on an open grid (no
obstacles).  Uses Manhattan distance as the heuristic — admissible
and consistent for 4-directional grid movement, guaranteeing optimality.
"""

from __future__ import annotations

import heapq


class AStarPathfinder:
    """
    A* pathfinding on a discrete grid.

    Finds the shortest path from start to goal avoiding no obstacles
    (open grid).  Uses Manhattan distance as the heuristic — admissible
    for 4-directional grid movement.

    Attributes
    ----------
    grid_size : int
        Side length of the square grid.
    """

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size

    # ------------------------------------------------------------------ #

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbours(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Return valid 4-directional neighbours within grid bounds.
        Order: up, down, left, right.
        """
        row, col = pos
        gs = self.grid_size
        candidates = [
            (row - 1, col),   # up
            (row + 1, col),   # down
            (row, col - 1),   # left
            (row, col + 1),   # right
        ]
        return [(r, c) for r, c in candidates if 0 <= r < gs and 0 <= c < gs]

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Find shortest path from start to goal using A*.

        Returns list of positions from start to goal inclusive.
        Returns [start] if start == goal.
        Returns [start, best_neighbour] if no path found
        (fallback to one step toward goal).

        Uses heapq for the open set priority queue.
        Tracks g_score (cost from start) and f_score
        (g + heuristic) for each node.  A monotonic counter breaks
        f-score ties so that the heap never compares tuples
        element-by-element — predictable ordering without
        requiring nodes to be comparable.
        """
        if start == goal:
            return [start]

        counter = 0
        open_set: list = []
        heapq.heappush(open_set, (self.heuristic(start, goal), counter, start))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {start: 0}
        closed_set: set[tuple[int, int]] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                path: list[tuple[int, int]] = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path

            for neighbour in self.get_neighbours(current):
                if neighbour in closed_set:
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + self.heuristic(neighbour, goal)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbour))

        # No path found (should not occur on an obstacle-free grid).
        # Fallback: one greedy step toward goal.
        neighbours = self.get_neighbours(start)
        if neighbours:
            best = min(neighbours, key=lambda n: self.heuristic(n, goal))
            return [start, best]
        return [start]

    def next_step(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Return only the next position to move to.
        This is what agents call each timestep.
        """
        path = self.find_path(start, goal)
        if len(path) > 1:
            return path[1]
        return start


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pf = AStarPathfinder(grid_size=10)

    # ── Long path ────────────────────────────────────────────────────────
    path = pf.find_path((0, 0), (9, 9))
    print(f"Path (0,0) -> (9,9):  {path}")
    assert path[0] == (0, 0), f"Start mismatch: {path[0]}"
    assert path[-1] == (9, 9), f"Goal mismatch: {path[-1]}"
    assert len(path) == 19, f"Expected 19 steps, got {len(path)}"

    # ── Same-cell path ───────────────────────────────────────────────────
    same = pf.find_path((5, 5), (5, 5))
    print(f"Path (5,5) -> (5,5):  {same}")
    assert same == [(5, 5)], f"Expected [(5,5)], got {same}"

    print("All A* tests passed")
