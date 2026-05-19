from __future__ import annotations

import random
from abc import ABC, abstractmethod

from message import Message
from astar import AStarPathfinder


class Agent(ABC):

    def __init__(self, agent_id: int, position: tuple[int, int]):
        self.agent_id: int = agent_id
        self.position: tuple[int, int] = position
        self.private_info: dict = {}
        self.inbox: list[Message] = []
        self._pathfinder: AStarPathfinder | None = None

    def _get_pathfinder(self, grid_size: int):
        if self._pathfinder is None or self._pathfinder.grid_size != grid_size:
            self._pathfinder = AStarPathfinder(grid_size)
        return self._pathfinder

    @abstractmethod
    def act(self, env) -> tuple[int, int]:
        pass

    @abstractmethod
    def decide_to_share(self, info: dict) -> bool:
        pass

    def perceive(self, env) -> None:
        if "known_tasks" not in self.private_info:
            self.private_info["known_tasks"] = {}
        visible = env.get_observable_tasks(self.position, radius=2)
        visible_ids = {t["id"] for t in visible}
        for task in visible:
            self.private_info["known_tasks"][task["id"]] = {
                "position": task["position"],
                "required_agents": task["required_agents"],
            }
        row, col = self.position
        to_remove = [
            tid for tid, info in self.private_info["known_tasks"].items()
            if tid not in visible_ids
            and abs(info["position"][0] - row) + abs(info["position"][1] - col) <= 2
        ]
        for tid in to_remove:
            del self.private_info["known_tasks"][tid]

    def _random_walk(self, grid_size: int) -> tuple[int, int]:
        row, col = self.position
        direction = random.choice(["stay", "up", "down", "left", "right"])
        if direction == "up":
            row = max(0, row - 1)
        elif direction == "down":
            row = min(grid_size - 1, row + 1)
        elif direction == "left":
            col = max(0, col - 1)
        elif direction == "right":
            col = min(grid_size - 1, col + 1)
        self.position = (row, col)
        return self.position

    def send_message(self, receiver: Agent, msg: Message) -> None:
        receiver.inbox.append(msg)

    def receive_message(self, msg: Message) -> None:
        for key, value in msg.content.items():
            if key == "known_tasks":
                if "known_tasks" not in self.private_info:
                    self.private_info["known_tasks"] = {}
                self.private_info["known_tasks"].update(value)
            else:
                self.private_info[key] = value
        if msg.is_distorted:
            self.private_info["_last_distorted_from"] = msg.sender_id


class CooperativeAgent(Agent):

    def __init__(self, agent_id: int, position: tuple[int, int]):
        super().__init__(agent_id, position)

    def act(self, env) -> tuple[int, int]:
        known = self.private_info.get("known_tasks", {})
        if not known:
            return self._random_walk(env.grid_size)

        row, col = self.position
        nearest_tid = min(
            known,
            key=lambda tid: (
                abs(known[tid]["position"][0] - row) +
                abs(known[tid]["position"][1] - col)
            ),
        )
        goal = known[nearest_tid]["position"]

        pathfinder = self._get_pathfinder(env.grid_size)
        self.position = pathfinder.next_step(self.position, goal)
        return self.position

    def decide_to_share(self, info: dict) -> bool:
        return True

    def share_all(self, neighbours: list[Agent]) -> None:
        known = self.private_info.get("known_tasks", {})
        if not known:
            return
        for neighbour in neighbours:
            msg = Message(
                sender_id=self.agent_id,
                receiver_id=neighbour.agent_id,
                content={"known_tasks": dict(known)},
            )
            self.send_message(neighbour, msg)


class StrategicAgent(Agent):

    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        deception_threshold: float = 0.5,
    ):
        super().__init__(agent_id, position)
        self.deception_threshold: float = deception_threshold
        self.trust_scores: dict[int, float] = {}

    def act(self, env) -> tuple[int, int]:
        known = self.private_info.get("known_tasks", {})
        if not known:
            return self._random_walk(env.grid_size)

        row, col = self.position
        nearest_tid = min(
            known,
            key=lambda tid: (
                abs(known[tid]["position"][0] - row) +
                abs(known[tid]["position"][1] - col)
            ),
        )
        goal = known[nearest_tid]["position"]

        pathfinder = self._get_pathfinder(env.grid_size)
        self.position = pathfinder.next_step(self.position, goal)
        return self.position

    def decide_to_share(self, info: dict) -> bool:
        if not self.trust_scores:
            return True
        return self.evaluate_sharing_benefit() > self.deception_threshold

    def decide_to_share_with(self, neighbour: Agent) -> bool:
        if neighbour.agent_id not in self.trust_scores:
            return True
        return self.trust_scores[neighbour.agent_id] > self.deception_threshold

    def evaluate_sharing_benefit(self) -> float:
        if not self.trust_scores:
            return 0.5
        return sum(self.trust_scores.values()) / len(self.trust_scores)

    def receive_message(self, msg: Message) -> None:
        sender = msg.sender_id
        if sender not in self.trust_scores:
            self.trust_scores[sender] = 0.5

        if msg.is_distorted:
            self.trust_scores[sender] = max(0.0, self.trust_scores[sender] - 0.1)
            return

        incoming_tasks = msg.content.get("known_tasks", {})
        existing_tasks = self.private_info.get("known_tasks", {})
        new_task_ids = set(incoming_tasks.keys()) - set(existing_tasks.keys())

        if new_task_ids:
            self.trust_scores[sender] = min(1.0, self.trust_scores[sender] + 0.1)
        else:
            self.trust_scores[sender] = max(0.0, self.trust_scores[sender] - 0.05)

        if "known_tasks" not in self.private_info:
            self.private_info["known_tasks"] = {}
        self.private_info["known_tasks"].update(incoming_tasks)
        for key, value in msg.content.items():
            if key != "known_tasks":
                self.private_info[key] = value

    def share_all(self, neighbours: list[Agent]) -> None:
        known = self.private_info.get("known_tasks", {})
        if not known:
            return
        for neighbour in neighbours:
            if not self.decide_to_share_with(neighbour):
                continue
            msg = Message(
                sender_id=self.agent_id,
                receiver_id=neighbour.agent_id,
                content={"known_tasks": dict(known)},
            )
            neighbour.inbox.append(msg)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from environment import Environment

    print("=== Agent Test ===\n")

    print("-- ABC enforcement --")
    try:
        Agent(agent_id=0, position=(0, 0))  # type: ignore
        print("ERROR: should have raised TypeError")
    except TypeError:
        print("Cannot instantiate Agent directly. (correct)")

    print("\n-- CooperativeAgent movement --")
    env = Environment(grid_size=10, num_tasks=3, task_requirement=2)
    env.tasks[0]["position"] = (5, 5)
    env.tasks[1]["position"] = (8, 8)
    env.tasks[2]["position"] = (1, 1)

    ca = CooperativeAgent(agent_id=0, position=(0, 0))
    ca.private_info["known_tasks"] = {}
    ca.perceive(env)
    print(f"Start position: {ca.position}  known: {list(ca.private_info['known_tasks'].keys())}")
    for step in range(3):
        new_pos = ca.act(env)
        print(f"  Step {step + 1}: {new_pos}")
    assert ca.position == (1, 1), f"Expected (1,1), got {ca.position}"
    print("Reached nearest known task in 2 steps. (correct)")

    print("\n-- Row-first movement preference --")
    ca2 = CooperativeAgent(agent_id=1, position=(3, 3))
    ca2.private_info["known_tasks"] = {0: {"position": (5, 5), "required_agents": 2}}
    new_pos = ca2.act(env)
    print(f"  (3,3) toward (5,5) -> {new_pos}")
    assert new_pos == (4, 3), f"Expected row-first move (4,3), got {new_pos}"
    print("Row moved first. (correct)")

    print("\n-- _random_walk boundary clamping --")
    import random as _rng
    _rng.seed(0)
    ca3 = CooperativeAgent(agent_id=2, position=(0, 0))
    for _ in range(50):
        pos = ca3._random_walk(env.grid_size)
        r, c = pos
        assert 0 <= r < env.grid_size and 0 <= c < env.grid_size, f"Out of bounds: {pos}"
    print(f"  50 random walks from corner always in bounds. (correct)")

    print("\n-- send_message / receive_message --")
    sender = CooperativeAgent(agent_id=3, position=(0, 0))
    receiver = CooperativeAgent(agent_id=4, position=(1, 1))
    receiver.private_info["known_tasks"] = {}
    sender.private_info["known_tasks"] = {7: {"position": (3, 3), "required_agents": 2}}

    msg = Message(sender_id=3, receiver_id=4,
                  content={"known_tasks": dict(sender.private_info["known_tasks"])})
    sender.send_message(receiver, msg)
    assert len(receiver.inbox) == 1
    receiver.receive_message(receiver.inbox.pop())
    assert 7 in receiver.private_info["known_tasks"]
    print("  Message delivered and known_tasks merged correctly.")

    print("\n-- share_all --")
    sharer = CooperativeAgent(agent_id=5, position=(0, 0))
    sharer.private_info["known_tasks"] = {3: {"position": (4, 4), "required_agents": 2}}
    neighbours = [
        CooperativeAgent(agent_id=6, position=(0, 1)),
        CooperativeAgent(agent_id=7, position=(1, 0)),
    ]
    for n in neighbours:
        n.private_info["known_tasks"] = {}
    sharer.share_all(neighbours)
    for n in neighbours:
        assert len(n.inbox) == 1
        n.receive_message(n.inbox.pop())
        assert 3 in n.private_info.get("known_tasks", {})
    print(f"  shared known_tasks to {len(neighbours)} neighbours correctly.")

    empty_sharer = CooperativeAgent(agent_id=10, position=(0, 0))
    empty_sharer.private_info["known_tasks"] = {}
    nb = CooperativeAgent(agent_id=11, position=(0, 1))
    empty_sharer.share_all([nb])
    assert len(nb.inbox) == 0
    print("  share_all skips when known_tasks is empty. (correct)")

    print("\n-- StrategicAgent decide_to_share --")
    sa = StrategicAgent(agent_id=8, position=(2, 2), deception_threshold=0.5)
    print(f"  No trust scores -> benefit={sa.evaluate_sharing_benefit():.1f} "
          f"-> share={sa.decide_to_share({})}")
    assert sa.decide_to_share({}) is True

    sa.trust_scores = {0: 0.9, 1: 0.8}
    print(f"  High trust scores -> benefit={sa.evaluate_sharing_benefit():.2f} "
          f"-> share={sa.decide_to_share({})}")
    assert sa.decide_to_share({}) is True

    print("\n-- StrategicAgent trust update --")
    sa2 = StrategicAgent(agent_id=9, position=(0, 0), deception_threshold=0.3)
    sa2.private_info["known_tasks"] = {}
    clean_msg = Message(sender_id=1, receiver_id=9,
                        content={"known_tasks": {0: {"position": (1, 1), "required_agents": 2}}})
    sa2.receive_message(clean_msg)
    print(f"  After clean message (new task), trust for agent 1: {sa2.trust_scores[1]:.1f}")
    assert sa2.trust_scores[1] == 0.6, f"Expected 0.6, got {sa2.trust_scores[1]}"

    dirty_msg = Message(sender_id=1, receiver_id=9,
                        content={"known_tasks": {99: {"position": (5, 5), "required_agents": 2}}},
                        is_distorted=True)
    sa2.receive_message(dirty_msg)
    print(f"  After distorted message, trust for agent 1: {sa2.trust_scores[1]:.2f}")
    assert round(sa2.trust_scores[1], 2) == 0.5
    assert 99 not in sa2.private_info.get("known_tasks", {})

    redundant_msg = Message(sender_id=1, receiver_id=9,
                            content={"known_tasks": {0: {"position": (1, 1), "required_agents": 2}}})
    sa2.receive_message(redundant_msg)
    print(f"  After redundant message (no new tasks), trust: {sa2.trust_scores[1]:.2f}")
    assert round(sa2.trust_scores[1], 2) == 0.45

    print("\n-- StrategicAgent decide_to_share_with --")
    sa3 = StrategicAgent(agent_id=10, position=(0, 0), deception_threshold=0.4)
    nb_unknown = CooperativeAgent(agent_id=11, position=(0, 1))
    assert sa3.decide_to_share_with(nb_unknown) is True
    sa3.trust_scores[11] = 0.3
    assert sa3.decide_to_share_with(nb_unknown) is False
    sa3.trust_scores[11] = 0.5
    assert sa3.decide_to_share_with(nb_unknown) is True
    print("  decide_to_share_with bootstrap and thresholding correct. (correct)")

    print("\nAll tests passed.")
