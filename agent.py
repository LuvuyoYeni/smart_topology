"""
agent.py
--------
Abstract Agent base class and two concrete subclasses:
  - CooperativeAgent  — always shares information with neighbours.
  - StrategicAgent    — shares only when the expected benefit justifies it.

Movement for both subtypes uses Manhattan distance to find the nearest
incomplete task, then advances one cell (row-first when equidistant).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from message import Message
from astar import AStarPathfinder


class Agent(ABC):
    """
    Abstract base for all agents in the multi-agent simulation.

    Subclasses must implement :meth:`act` and :meth:`decide_to_share`.

    Attributes
    ----------
    agent_id : int
        Unique identifier for this agent.
    position : tuple[int, int]
        Current (row, col) location on the grid.
    private_info : dict
        Local knowledge accumulated by this agent alone.
    inbox : list[Message]
        Messages waiting to be processed.
    """

    def __init__(self, agent_id: int, position: tuple[int, int]):
        """
        Initialise base agent state.

        Parameters
        ----------
        agent_id : int
            Unique agent identifier.
        position : tuple[int, int]
            Starting (row, col) grid position.
        """
        self.agent_id: int = agent_id
        self.position: tuple[int, int] = position
        self.private_info: dict = {}
        self.inbox: list[Message] = []
        self._pathfinder: AStarPathfinder | None = None

    # ------------------------------------------------------------------ #
    #  Pathfinder                                                         #
    # ------------------------------------------------------------------ #

    def _get_pathfinder(self, grid_size: int) -> AStarPathfinder:
        """Lazy-initialise pathfinder when grid size is known."""
        if self._pathfinder is None or self._pathfinder.grid_size != grid_size:
            self._pathfinder = AStarPathfinder(grid_size)
        return self._pathfinder

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def act(self, env) -> tuple[int, int]:
        """
        Decide and execute one movement step.

        Parameters
        ----------
        env : Environment
            The current environment (provides active task positions and
            grid bounds).

        Returns
        -------
        tuple[int, int]
            The agent's new (row, col) position after this step.
        """

    @abstractmethod
    def decide_to_share(self, info: dict) -> bool:
        """
        Return True if the agent will share *info* with a neighbour.

        Parameters
        ----------
        info : dict
            The information under consideration for sharing.
        """

    # ------------------------------------------------------------------ #
    #  Shared messaging methods                                            #
    # ------------------------------------------------------------------ #

    def send_message(self, receiver: Agent, msg: Message) -> None:
        """
        Deliver *msg* directly to *receiver*'s inbox.

        Parameters
        ----------
        receiver : Agent
            The agent that will receive the message.
        msg : Message
            The message to deliver.
        """
        receiver.inbox.append(msg)

    def receive_message(self, msg: Message) -> None:
        """
        Process one incoming message and merge its content into
        :attr:`private_info`.

        Keys from the message content overwrite existing keys; new keys
        are added.  Distorted messages are accepted but flagged in
        private_info so subclasses can inspect provenance if needed.

        Parameters
        ----------
        msg : Message
            The message to process.
        """
        for key, value in msg.content.items():
            self.private_info[key] = value
        if msg.is_distorted:
            self.private_info["_last_distorted_from"] = msg.sender_id

    # ------------------------------------------------------------------ #
    #  Shared movement helper                                              #
    # ------------------------------------------------------------------ #

    def _move_toward(
        self, target: tuple[int, int], grid_size: int
    ) -> tuple[int, int]:
        """
        Return the position one step closer to *target*.

        Prefer reducing row distance first; reduce column distance only
        when rows are already aligned.  The new position is clamped to
        [0, grid_size-1] in both dimensions.

        Parameters
        ----------
        target : tuple[int, int]
            Destination (row, col).
        grid_size : int
            Side length of the square grid (rows and columns are equal).
        """
        row, col = self.position
        tr, tc = target

        row_diff = tr - row
        col_diff = tc - col

        if row_diff != 0:
            row += 1 if row_diff > 0 else -1
        elif col_diff != 0:
            col += 1 if col_diff > 0 else -1

        # Clamp to grid bounds
        row = max(0, min(grid_size - 1, row))
        col = max(0, min(grid_size - 1, col))

        return (row, col)


# ====================================================================== #
#  CooperativeAgent                                                       #
# ====================================================================== #

class CooperativeAgent(Agent):
    """
    An agent that always shares its private information with neighbours.

    Attributes
    ----------
    sharing_probability : float
        Fraction of neighbours to share with (default 1.0 = all of them).
        Reserved for future probabilistic variants; the base behaviour is
        unconditional sharing.
    """

    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        sharing_probability: float = 1.0,
    ):
        """
        Parameters
        ----------
        agent_id : int
            Unique agent identifier.
        position : tuple[int, int]
            Starting (row, col) grid position.
        sharing_probability : float
            Probability of sharing with each neighbour (default 1.0).
        """
        super().__init__(agent_id, position)
        self.sharing_probability: float = sharing_probability

    def act(self, env) -> tuple[int, int]:
        """
        Move one step toward the nearest incomplete task
        using A* pathfinding.

        The nearest task is selected by Manhattan distance
        (same as before). A* then finds the optimal path
        and returns the next step along it.

        Parameters
        ----------
        env : Environment
            Provides active tasks and grid_size.

        Returns
        -------
        tuple[int, int]
            New (row, col) position.
        """
        active = env.get_active_tasks()
        if not active:
            return self.position

        row, col = self.position
        nearest = min(
            active,
            key=lambda t: (
                abs(t["position"][0] - row) +
                abs(t["position"][1] - col)
            ),
        )
        goal = nearest["position"]

        pathfinder = self._get_pathfinder(env.grid_size)
        next_pos = pathfinder.next_step(self.position, goal)

        self.position = next_pos
        return self.position

    def decide_to_share(self, info: dict) -> bool:
        """
        Always returns True — cooperative agents share unconditionally.

        Parameters
        ----------
        info : dict
            Information under consideration (ignored here).
        """
        return True

    def share_all(self, neighbours: list[Agent]) -> None:
        """
        Send :attr:`private_info` to every neighbour.

        A separate :class:`Message` is created for each recipient so that
        inbox entries carry the correct receiver_id.

        Parameters
        ----------
        neighbours : list[Agent]
            Agents reachable via the current communication topology.
        """
        if not self.private_info:
            return
        for neighbour in neighbours:
            msg = Message(
                sender_id=self.agent_id,
                receiver_id=neighbour.agent_id,
                content=dict(self.private_info),
            )
            self.send_message(neighbour, msg)

    # ------------------------------------------------------------------ #

    def _nearest_task(self, tasks: list[dict]) -> tuple[int, int]:
        """Return the position of the closest incomplete task."""
        row, col = self.position
        return min(
            (t["position"] for t in tasks),
            key=lambda p: abs(p[0] - row) + abs(p[1] - col),
        )


# ====================================================================== #
#  StrategicAgent                                                         #
# ====================================================================== #

class StrategicAgent(Agent):
    """
    An agent that shares information only when the expected benefit
    exceeds its deception threshold.

    Attributes
    ----------
    deception_threshold : float
        Minimum expected benefit required before sharing.  Higher values
        make the agent more guarded.
    trust_scores : dict[int, float]
        Per-neighbour trust in [0, 1].  Populated on first contact;
        defaults to 0.5.
    """

    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        deception_threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        agent_id : int
            Unique agent identifier.
        position : tuple[int, int]
            Starting (row, col) grid position.
        deception_threshold : float
            Sharing benefit must exceed this to trigger sharing (default 0.5).
        """
        super().__init__(agent_id, position)
        self.deception_threshold: float = deception_threshold
        self.trust_scores: dict[int, float] = {}

    def act(self, env) -> tuple[int, int]:
        """
        Move one step toward the nearest incomplete task
        using A* pathfinding.

        The nearest task is selected by Manhattan distance
        (same as before). A* then finds the optimal path
        and returns the next step along it.  Strategic behaviour
        is expressed only in information sharing, not movement.

        Parameters
        ----------
        env : Environment
            Provides active tasks and grid_size.

        Returns
        -------
        tuple[int, int]
            New (row, col) position.
        """
        active = env.get_active_tasks()
        if not active:
            return self.position

        row, col = self.position
        nearest = min(
            active,
            key=lambda t: (
                abs(t["position"][0] - row) +
                abs(t["position"][1] - col)
            ),
        )
        goal = nearest["position"]

        pathfinder = self._get_pathfinder(env.grid_size)
        next_pos = pathfinder.next_step(self.position, goal)

        self.position = next_pos
        return self.position

    def decide_to_share(self, info: dict) -> bool:
        """
        Share freely on the first step (no history yet), then only when
        the expected benefit exceeds :attr:`deception_threshold`.

        Parameters
        ----------
        info : dict
            Information under consideration for sharing.
        """
        if not self.trust_scores:
            return True  # bootstrap: share openly before any trust data
        return self.evaluate_sharing_benefit() > self.deception_threshold

    def evaluate_sharing_benefit(self) -> float:
        """
        Estimate the benefit of sharing as the mean trust score across
        all known neighbours.

        Returns 0.5 (neutral) when no trust data exists yet.

        Returns
        -------
        float
            Expected benefit in [0, 1].
        """
        if not self.trust_scores:
            return 0.5
        return sum(self.trust_scores.values()) / len(self.trust_scores)

    def receive_message(self, msg: Message) -> None:
        """
        Process an incoming message and update trust based on content novelty.

        Trust increases when the message contains information this agent
        did not already know (genuinely useful).  Trust decays slightly
        for redundant messages — neighbours must keep providing value to
        maintain high trust scores.

        Parameters
        ----------
        msg : Message
            The message to process.
        """
        sender = msg.sender_id
        if sender not in self.trust_scores:
            self.trust_scores[sender] = 0.5

        # Determine how much of the content is genuinely new
        new_keys = {k for k in msg.content if k not in self.private_info}
        if new_keys:
            self.trust_scores[sender] = min(1.0, self.trust_scores[sender] + 0.1)
            for k in new_keys:
                self.private_info[k] = msg.content[k]
        else:
            # Redundant message — slight decay to encourage useful sharing
            self.trust_scores[sender] = max(0.0, self.trust_scores[sender] - 0.05)

    def share_all(self, neighbours: list[Agent]) -> None:
        """
        Share private_info with neighbours, but only if :meth:`decide_to_share`
        approves.  Unlike CooperativeAgent, strategic agents may withhold.

        Parameters
        ----------
        neighbours : list[Agent]
            Agents reachable via the current communication topology.
        """
        if not self.decide_to_share(self.private_info) or not self.private_info:
            return
        for neighbour in neighbours:
            msg = Message(
                sender_id=self.agent_id,
                receiver_id=neighbour.agent_id,
                content=dict(self.private_info),
            )
            neighbour.inbox.append(msg)

    # ------------------------------------------------------------------ #

    def _nearest_task(self, tasks: list[dict]) -> tuple[int, int]:
        """Return the position of the closest incomplete task."""
        row, col = self.position
        return min(
            (t["position"] for t in tasks),
            key=lambda p: abs(p[0] - row) + abs(p[1] - col),
        )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from environment import Environment

    print("=== Agent Test ===\n")

    # --- 1. ABC enforcement ---
    print("-- ABC enforcement --")
    try:
        Agent(agent_id=0, position=(0, 0))  # type: ignore
        print("ERROR: should have raised TypeError")
    except TypeError:
        print("Cannot instantiate Agent directly. (correct)")

    # --- 2. CooperativeAgent movement ---
    print("\n-- CooperativeAgent movement --")
    env = Environment(grid_size=10, num_tasks=3, task_requirement=2)
    # Override task positions for deterministic test
    env.tasks[0]["position"] = (5, 5)
    env.tasks[1]["position"] = (8, 8)
    env.tasks[2]["position"] = (1, 1)

    ca = CooperativeAgent(agent_id=0, position=(0, 0))
    print(f"Start position: {ca.position}")
    # Nearest task is (1,1) with Manhattan dist 2
    for step in range(3):
        new_pos = ca.act(env)
        print(f"  Step {step + 1}: {new_pos}")
    # After 2 steps should be at (1,1)
    assert ca.position == (1, 1), f"Expected (1,1), got {ca.position}"
    print("Reached nearest task in 2 steps. (correct)")

    # --- 3. Row-first movement preference ---
    print("\n-- Row-first movement preference --")
    ca2 = CooperativeAgent(agent_id=1, position=(3, 3))
    env.tasks[0]["position"] = (5, 5)
    env.tasks[1]["position"] = (8, 8)
    env.tasks[2]["position"] = (9, 9)
    # nearest task to (3,3) is (5,5), dist=4
    new_pos = ca2.act(env)
    print(f"  (3,3) toward (5,5) -> {new_pos}")
    assert new_pos == (4, 3), f"Expected row-first move (4,3), got {new_pos}"
    print("Row moved first. (correct)")

    # --- 4. Grid boundary clamping ---
    print("\n-- Grid boundary clamping --")
    ca3 = CooperativeAgent(agent_id=2, position=(0, 0))
    env.tasks[0]["position"] = (0, 0)   # already there
    env.tasks[0]["completed"] = True
    env.tasks[1]["position"] = (0, 0)
    env.tasks[1]["completed"] = True
    env.tasks[2]["position"] = (0, 0)
    env.tasks[2]["completed"] = True
    pos = ca3.act(env)
    print(f"  No active tasks -> stays at {pos}")
    assert pos == (0, 0)

    # --- 5. Messaging ---
    print("\n-- send_message / receive_message --")
    sender = CooperativeAgent(agent_id=3, position=(0, 0))
    receiver = CooperativeAgent(agent_id=4, position=(1, 1))
    sender.private_info = {"task_seen": 2, "last_pos": [3, 3]}

    msg = Message(sender_id=3, receiver_id=4, content=dict(sender.private_info))
    sender.send_message(receiver, msg)
    assert len(receiver.inbox) == 1
    receiver.receive_message(receiver.inbox.pop())
    assert receiver.private_info["task_seen"] == 2
    print("  Message delivered and processed correctly.")

    # --- 6. share_all ---
    print("\n-- share_all --")
    sharer = CooperativeAgent(agent_id=5, position=(0, 0))
    sharer.private_info = {"seen_task": 1}
    neighbours = [
        CooperativeAgent(agent_id=6, position=(0, 1)),
        CooperativeAgent(agent_id=7, position=(1, 0)),
    ]
    sharer.share_all(neighbours)
    for n in neighbours:
        assert len(n.inbox) == 1
        n.receive_message(n.inbox.pop())
        assert n.private_info.get("seen_task") == 1
    print(f"  shared to {len(neighbours)} neighbours correctly.")

    # --- 7. StrategicAgent decide_to_share ---
    print("\n-- StrategicAgent decide_to_share --")
    sa = StrategicAgent(agent_id=8, position=(2, 2), deception_threshold=0.5)
    # No trust scores -> benefit = 0.0 -> will NOT share (0.0 <= 0.5)
    print(f"  No trust scores -> benefit={sa.evaluate_sharing_benefit():.1f} "
          f"-> share={sa.decide_to_share({})}")
    assert sa.decide_to_share({}) is False

    # Add high-trust neighbours -> benefit > 0.5 -> WILL share
    sa.trust_scores = {0: 0.9, 1: 0.8}
    print(f"  High trust scores -> benefit={sa.evaluate_sharing_benefit():.2f} "
          f"-> share={sa.decide_to_share({})}")
    assert sa.decide_to_share({}) is True

    # --- 8. StrategicAgent trust update via receive_message ---
    print("\n-- StrategicAgent trust update --")
    sa2 = StrategicAgent(agent_id=9, position=(0, 0), deception_threshold=0.3)
    clean_msg = Message(sender_id=1, receiver_id=9, content={"x": 1})
    sa2.receive_message(clean_msg)
    print(f"  After clean message, trust for agent 1: {sa2.trust_scores[1]:.1f}")
    assert sa2.trust_scores[1] == 0.6  # 0.5 + 0.1

    dirty_msg = Message(sender_id=1, receiver_id=9,
                        content={"x": 99}, is_distorted=True)
    sa2.receive_message(dirty_msg)
    print(f"  After distorted message, trust for agent 1: {sa2.trust_scores[1]:.1f}")
    assert sa2.trust_scores[1] == 0.5  # 0.6 - 0.1

    print("\nAll tests passed.")
