import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Environment:
    """
    A grid world where agents must coordinate to complete tasks.
    Tasks require multiple agents on the same cell simultaneously —
    this is what forces agents to communicate and coordinate.
    """

    def __init__(self, grid_size=10, num_tasks=5, task_requirement=2):
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.task_requirement = task_requirement   # agents needed per task
        self.tasks = []
        self.time_step = 0
        self._generate_tasks()

    def _generate_tasks(self):
        """Randomly place tasks on the grid."""
        self.tasks = []
        for i in range(self.num_tasks):
            task = {
                "id":               i,
                "position":         (np.random.randint(0, self.grid_size),
                                     np.random.randint(0, self.grid_size)),
                "required_agents":  self.task_requirement,
                "completed":        False
            }
            self.tasks.append(task)

    def reset(self):
        """Wipe the environment clean for a fresh simulation run."""
        self.time_step = 0
        self._generate_tasks()

    def get_active_tasks(self):
        """Return only the tasks not yet completed."""
        return [t for t in self.tasks if not t["completed"]]

    def get_observable_tasks(self, position, radius=2):
        """Return active tasks within Manhattan distance *radius* of *position*."""
        row, col = position
        return [
            t for t in self.tasks
            if not t["completed"]
            and abs(t["position"][0] - row) + abs(t["position"][1] - col) <= radius
        ]

    def update(self, agent_positions):
        """
        Called once per time step.
        agent_positions: list of (row, col) tuples, one per agent.
        Returns how many tasks were completed this step.
        """
        completed_this_step = 0

        for task in self.get_active_tasks():
            # count agents standing on this task's cell
            agents_here = sum(1 for pos in agent_positions
                              if pos == task["position"])

            if agents_here >= task["required_agents"]:
                task["completed"] = True
                completed_this_step += 1

        self.time_step += 1
        return completed_this_step

    def completion_rate(self):
        """Fraction of tasks completed — used as part of fitness score."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t["completed"])
        return completed / len(self.tasks)

    def render(self, agent_positions=None, title_suffix=""):
        """Visualise the current state of the environment."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(range(self.grid_size + 1))
        ax.set_yticks(range(self.grid_size + 1))
        ax.grid(True, color="lightgrey", lw=0.5)
        ax.set_title(f"Environment  |  Step {self.time_step}  {title_suffix}")

        # draw tasks (red = incomplete, green = done)
        for task in self.tasks:
            row, col = task["position"]
            colour = "green" if task["completed"] else "tomato"
            ax.add_patch(mpatches.Rectangle(
                (col, row), 1, 1, color=colour, alpha=0.35))
            ax.text(col + 0.5, row + 0.5, f"T{task['id']}",
                    ha="center", va="center", fontsize=9, fontweight="bold")

        # draw agents as numbered blue circles
        if agent_positions:
            for i, (row, col) in enumerate(agent_positions):
                ax.plot(col + 0.5, row + 0.5, "o",
                        color="steelblue", markersize=18, alpha=0.8)
                ax.text(col + 0.5, row + 0.5, str(i),
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        plt.tight_layout()
        plt.show()


# ── Test it immediately ───────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=== Environment Test ===\n")

    env = Environment(grid_size=8, num_tasks=4, task_requirement=2)

    task0_pos = env.tasks[0]["position"]
    agent_positions = [
        task0_pos,   # agent 0 — on task 0
        task0_pos,   # agent 1 — also on task 0 (should complete it!)
        (2, 2),      # agent 2 — somewhere random
        (5, 7),      # agent 3 — somewhere random
    ]

    print(f"Task 0 is at position: {task0_pos}")
    print(f"Active tasks before step: {len(env.get_active_tasks())}")

    # ── side-by-side: before and after in one window ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def draw_state(ax, title_suffix):
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xticks(range(env.grid_size + 1))
        ax.set_yticks(range(env.grid_size + 1))
        ax.grid(True, color="lightgrey", lw=0.5)
        ax.set_title(f"Step {env.time_step}  {title_suffix}")

        for task in env.tasks:
            row, col = task["position"]
            colour = "green" if task["completed"] else "tomato"
            ax.add_patch(mpatches.Rectangle(
                (col, row), 1, 1, color=colour, alpha=0.35))
            ax.text(col + 0.5, row + 0.5, f"T{task['id']}",
                    ha="center", va="center", fontsize=9, fontweight="bold")

        for i, (row, col) in enumerate(agent_positions):
            ax.plot(col + 0.5, row + 0.5, "o",
                    color="steelblue", markersize=18, alpha=0.8)
            ax.text(col + 0.5, row + 0.5, str(i),
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    # snapshot BEFORE
    draw_state(ax1, "(before update)")

    # run the update
    completed = env.update(agent_positions)

    # snapshot AFTER
    draw_state(ax2, "(after update)")

    plt.suptitle("Task 0 should turn green on the right — two agents met on its cell",
                 fontsize=11)
    plt.tight_layout()
    plt.show()

    print(f"\nCompleted this step: {completed}")
    print(f"Active tasks remaining: {len(env.get_active_tasks())}")
    print(f"Completion rate: {env.completion_rate():.0%}")