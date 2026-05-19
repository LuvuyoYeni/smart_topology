from __future__ import annotations

import math
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"
import random
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np

from environment import Environment
from evolution_engine import EvolutionEngine
from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome


BG_DARK   = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_INPUT  = "#313145"
FG_WHITE  = "#e0e0f0"
FG_DIM    = "#9090a0"
ACCENT    = "#4fc3f7"
GREEN     = "#43a047"
RED       = "#e53935"
ORANGE    = "#fb8c00"

_AGENT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_CELL_SIZE = 60
_MARGIN    = 20
_CANVAS_PX = 640


def _cell_to_px(r: int, c: int) -> tuple[int, int]:
    return _MARGIN + c * _CELL_SIZE + _CELL_SIZE // 2, \
           _MARGIN + r * _CELL_SIZE + _CELL_SIZE // 2


def _dim_color(hex_color: str, factor: float) -> str:
    r = min(int(int(hex_color[1:3], 16) * factor), 255)
    g = min(int(int(hex_color[3:5], 16) * factor), 255)
    b = min(int(int(hex_color[5:7], 16) * factor), 255)
    return f"#{r:02x}{g:02x}{b:02x}"


class LiveSimulationRunner:

    GRID_SIZE  = 10
    NUM_TASKS  = 8
    TASK_REQ   = 3
    NUM_AGENTS = 6

    def __init__(self) -> None:
        self.num_agents: int = self.NUM_AGENTS
        self.agent_mode: str = "cooperative"
        self.env: Environment | None = None
        self.agents: list = []
        self.comm_graph = None
        self.topology_genome: TopologyGenome | None = None
        self.step_count: int = 0
        self.total_messages: int = 0
        self.flash_edges: dict[tuple[int, int], int] = {}
        self.mission_complete: bool = False

    def load_topology(self, name: str) -> str:
        n = self.NUM_AGENTS
        if name == "Evolved (demo)":
            demo_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results", "demo_genome.npz"
            )
            if not os.path.exists(demo_path):
                return f"File not found:\n{demo_path}\n\nRun main.py first."
            data = np.load(demo_path)
            mat = data["adjacency_matrix"].astype(bool)
            n = mat.shape[0]
            self.num_agents = n
            self.topology_genome = TopologyGenome(n, mat)
        elif name == "Ring":
            mat = np.zeros((n, n), dtype=bool)
            for i in range(n):
                mat[i][(i + 1) % n] = mat[(i + 1) % n][i] = True
            self.topology_genome = TopologyGenome(n, mat)
        elif name == "Star":
            mat = np.zeros((n, n), dtype=bool)
            for i in range(1, n):
                mat[0][i] = mat[i][0] = True
            self.topology_genome = TopologyGenome(n, mat)
        elif name == "Full":
            mat = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mat, False)
            self.topology_genome = TopologyGenome(n, mat)
        elif name == "Empty":
            mat = np.zeros((n, n), dtype=bool)
            self.topology_genome = TopologyGenome(n, mat)
        else:
            return f"Unknown topology: {name!r}"
        return ""

    def reset(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        self.env = Environment(
            grid_size=self.GRID_SIZE,
            num_tasks=self.NUM_TASKS,
            task_requirement=self.TASK_REQ,
        )
        ev = FitnessEvaluator(num_trials=1, agent_mode=self.agent_mode)
        self.agents = ev._spawn_agents(self.num_agents, self.GRID_SIZE)
        if self.topology_genome is not None:
            self.comm_graph = self.topology_genome.to_communication_graph()
        self.step_count = 0
        self.total_messages = 0
        self.flash_edges = {}
        self.mission_complete = False

    def step(self) -> dict:
        if self.mission_complete or self.env is None or self.comm_graph is None:
            return self._state()

        new_flash: dict[tuple[int, int], int] = {}

        for agent in self.agents:
            agent.perceive(self.env)

        for agent in self.agents:
            neighbour_ids = self.comm_graph.get_neighbours(agent.agent_id)
            neighbours = [self.agents[nid] for nid in neighbour_ids]
            if agent.decide_to_share(agent.private_info) and neighbours:
                agent.share_all(neighbours)
                self.total_messages += len(neighbours)
                for nb in neighbours:
                    new_flash[(agent.agent_id, nb.agent_id)] = 3

        for agent in self.agents:
            for msg in agent.inbox:
                agent.receive_message(msg)
            agent.inbox.clear()

        for agent in self.agents:
            agent.act(self.env)

        positions = [agent.position for agent in self.agents]
        self.env.update(positions)

        active_ids = {t["id"] for t in self.env.get_active_tasks()}
        for agent in self.agents:
            stale = [tid for tid in agent.private_info.get("known_tasks", {})
                     if tid not in active_ids]
            for tid in stale:
                del agent.private_info["known_tasks"][tid]

        self.step_count += 1

        for edge, frames in self.flash_edges.items():
            if edge not in new_flash and frames > 1:
                new_flash[edge] = frames - 1
        self.flash_edges = new_flash

        if not self.env.get_active_tasks():
            self.mission_complete = True

        return self._state()

    def _state(self) -> dict:
        if self.env is None:
            return {}
        tasks_total = len(self.env.tasks)
        tasks_done  = sum(1 for t in self.env.tasks if t["completed"])
        adj = (
            self.topology_genome.adjacency_matrix
            if self.topology_genome is not None
            else np.zeros((self.num_agents, self.num_agents), dtype=bool)
        )
        return {
            "positions":        [a.position for a in self.agents],
            "known_tasks":      [dict(a.private_info.get("known_tasks", {}))
                                 for a in self.agents],
            "all_tasks":        list(self.env.tasks),
            "flash_edges":      dict(self.flash_edges),
            "total_messages":   self.total_messages,
            "step_count":       self.step_count,
            "tasks_done":       tasks_done,
            "tasks_total":      tasks_total,
            "mission_complete": self.mission_complete,
            "adjacency_matrix": adj,
            "num_agents":       self.num_agents,
        }


class EvoCommGUI:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EvoComm — Evolving Communication Topologies")
        self.root.geometry("1200x750")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self.engine: EvolutionEngine | None = None
        self._stop_flag = False
        self._evo_thread: threading.Thread | None = None

        self._gen_x: list[int] = []
        self._fit_y: list[float] = []

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, minsize=300, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self) -> None:
        left = tk.Frame(self.root, bg=BG_PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="EvoComm", bg=BG_PANEL, fg=ACCENT,
                 font=("Helvetica", 16, "bold")).pack(pady=(14, 0))
        tk.Label(left, text="Evolving Communication Topologies",
                 bg=BG_PANEL, fg=FG_DIM, font=("Helvetica", 8)).pack(pady=(0, 10))
        ttk.Separator(left).pack(fill="x", padx=10)

        self._section_label(left, "Configuration")

        self.var_pop    = self._slider(left, "Population Size", 10, 50, 20, resolution=1)
        self.var_gen    = self._slider(left, "Generations",     10, 100, 50, resolution=1)
        self.var_mut    = self._slider(left, "Mutation Rate",   0.1, 0.9, 0.4, resolution=0.1)
        self.var_agents = self._slider(left, "Num Agents",      4, 12, 6, resolution=1)

        mode_frame = tk.Frame(left, bg=BG_PANEL)
        mode_frame.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(mode_frame, text="Agent Mode", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
        self.var_mode = tk.StringVar(value="cooperative")
        ttk.OptionMenu(mode_frame, self.var_mode,
                       "cooperative", "cooperative", "mixed", "strategic"
                       ).pack(side="left", fill="x", expand=True)

        ttk.Separator(left).pack(fill="x", padx=10, pady=8)

        self._section_label(left, "Run Controls")

        btn_frame = tk.Frame(left, bg=BG_PANEL)
        btn_frame.pack(fill="x", padx=12, pady=4)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.btn_run = tk.Button(
            btn_frame, text="RUN EVOLUTION",
            bg=GREEN, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            command=self._start_evolution,
        )
        self.btn_run.grid(row=0, column=0, sticky="ew", padx=(0, 4), ipady=6)

        self.btn_stop = tk.Button(
            btn_frame, text="STOP",
            bg=RED, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2", state="disabled",
            command=self._stop_evolution,
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0), ipady=6)

        self.lbl_status = tk.Label(
            left, text="Ready", bg=BG_PANEL, fg=ACCENT,
            font=("Helvetica", 9, "italic"),
        )
        self.lbl_status.pack(pady=(8, 2))

        self.progress = ttk.Progressbar(left, orient="horizontal",
                                        mode="determinate", length=260)
        self.progress.pack(padx=12, pady=(2, 8))

        ttk.Separator(left).pack(fill="x", padx=10)

        self._section_label(left, "Scenario Context")
        context = (
            "This system models communication network design "
            "for drone swarms, IoT sensor networks, and distributed "
            "robotics — environments where bandwidth limits who can "
            "communicate with whom.\n\n"
            "The evolved topology represents the optimal communication "
            "graph discovered by the genetic algorithm for the given "
            "agent population and task constraints."
        )
        tk.Label(left, text=context, bg=BG_PANEL, fg=FG_DIM,
                 font=("Helvetica", 8), wraplength=265,
                 justify="left").pack(padx=12, pady=(0, 12))

        ttk.Separator(left).pack(fill="x", padx=10, pady=(8, 4))
        self._section_label(left, "Demo")
        self.btn_live_sim = tk.Button(
            left, text="LAUNCH LIVE SIMULATION",
            bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            command=self._open_live_sim_window,
        )
        self.btn_live_sim.pack(fill="x", padx=12, pady=(2, 12), ipady=8)

    def _build_right_panel(self) -> None:
        right = tk.Frame(self.root, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background=BG_DARK,  borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_DIM,
                        padding=[12, 6], font=("Helvetica", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_INPUT)],
                  foreground=[("selected", FG_WHITE)])

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._build_tab_fitness()
        self._build_tab_topology()
        self._build_tab_comparison()
        self._build_tab_about()

    def _build_tab_fitness(self) -> None:
        frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(frame, text="  Fitness Progress  ")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig_fitness = plt.Figure(figsize=(8, 5), facecolor=BG_DARK)
        self.ax_fitness  = self.fig_fitness.add_subplot(111)
        self._style_axis(self.ax_fitness, "EvoComm — Fitness Progress",
                         "Generation", "Best Fitness Score")
        self.line_fitness, = self.ax_fitness.plot([], [], color=ACCENT,
                                                   linewidth=2, marker="o",
                                                   markersize=4)
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=frame)
        self.canvas_fitness.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                                  padx=8, pady=8)

    def _build_tab_topology(self) -> None:
        frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(frame, text="  Best Topology  ")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig_topo = plt.Figure(figsize=(8, 4.5), facecolor=BG_DARK)
        self.ax_topo  = self.fig_topo.add_subplot(111)
        self._style_axis(self.ax_topo, "Best Topology", "", "")
        self.ax_topo.text(0.5, 0.5, "Run evolution to see topology",
                          ha="center", va="center", transform=self.ax_topo.transAxes,
                          color=FG_DIM, fontsize=13)

        self.canvas_topo = FigureCanvasTkAgg(self.fig_topo, master=frame)
        self.canvas_topo.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                               padx=8, pady=(8, 2))

        self.lbl_topo_info = tk.Label(
            frame, text="Agents: —  |  Edges: —  |  Density: —  |  Connected: —",
            bg=BG_DARK, fg=FG_DIM, font=("Helvetica", 9),
        )
        self.lbl_topo_info.grid(row=1, column=0, pady=(0, 8))

    def _build_tab_comparison(self) -> None:
        frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(frame, text="  Topology Comparison  ")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig_cmp = plt.Figure(figsize=(8, 4.5), facecolor=BG_DARK)
        self.ax_cmp  = self.fig_cmp.add_subplot(111)
        self._style_axis(self.ax_cmp, "Topology Comparison",
                         "Topology", "Fitness Score")
        self.ax_cmp.text(0.5, 0.5, "Run evolution to see comparison",
                         ha="center", va="center", transform=self.ax_cmp.transAxes,
                         color=FG_DIM, fontsize=13)

        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=frame)
        self.canvas_cmp.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                              padx=8, pady=(8, 2))

        self.lbl_cmp_summary = tk.Label(
            frame, text="",
            bg=BG_DARK, fg=FG_DIM, font=("Helvetica", 9),
        )
        self.lbl_cmp_summary.grid(row=1, column=0, pady=(0, 8))

    def _build_tab_about(self) -> None:
        frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(frame, text="  About EvoComm  ")

        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def heading(text: str, size: int = 13, colour: str = ACCENT) -> None:
            tk.Label(inner, text=text, bg=BG_DARK, fg=colour,
                     font=("Helvetica", size, "bold"),
                     justify="left").pack(anchor="w", padx=24, pady=(14, 2))

        def body(text: str) -> None:
            tk.Label(inner, text=text, bg=BG_DARK, fg=FG_WHITE,
                     font=("Helvetica", 10), wraplength=820,
                     justify="left").pack(anchor="w", padx=28, pady=(0, 6))

        heading("EvoComm — Evolving Communication Topologies", size=15)
        tk.Label(inner,
                 text=("An Evolutionary Algorithm Approach to Emergent "
                       "Communication Structure in Multi-Agent Systems"),
                 bg=BG_DARK, fg=FG_DIM, font=("Helvetica", 10, "italic"),
                 wraplength=820, justify="left",
                 ).pack(anchor="w", padx=28, pady=(0, 10))

        ttk.Separator(inner).pack(fill="x", padx=20, pady=6)

        heading("What It Does")
        body(
            "EvoComm evolves the communication graph structure of a multi-agent "
            "system using a genetic algorithm. Rather than designing who talks to "
            "whom by hand, the system discovers optimal communication topologies "
            "through selection, crossover, and mutation over many generations. "
            "Agents are placed on a grid and must coordinate to complete tasks "
            "that require multiple agents to occupy the same cell simultaneously."
        )

        heading("The Agents")
        body(
            "Cooperative agents share all information freely with every connected "
            "neighbour each timestep — the simplest possible communication policy.\n\n"
            "Strategic agents maintain a per-neighbour trust score updated by "
            "message usefulness. They only share when the expected benefit exceeds "
            "their deception threshold, modelling information asymmetry in "
            "competitive or resource-constrained environments."
        )

        heading("Key Findings")
        for point in (
            "Evolved topologies outperform hand-crafted baselines (Full, Ring, Star) "
            "under cooperative agent conditions.",
            "Communication structure matters as much as message content — the same "
            "agents achieve different task completion rates on different topologies.",
            "Strategic agents exploit sparse topologies differently than cooperative "
            "ones, reversing the expected hub-and-spoke preference.",
            "Fitness function design critically affects evolved structure: increasing "
            "the message-penalty weight from 0.0 to 0.4 significantly reduces "
            "the density of the evolved topology.",
        ):
            tk.Label(inner, text=f"  •  {point}", bg=BG_DARK, fg=FG_WHITE,
                     font=("Helvetica", 10), wraplength=800,
                     justify="left").pack(anchor="w", padx=28, pady=1)

        heading("Research Context")
        body(
            "University of Johannesburg\n"
            "IT08X97 Artificial Intelligence\n"
            "Academy of Computer Science and Software Engineering\n"
            "2026"
        )

    def _open_live_sim_window(self) -> None:
        if hasattr(self, "_live_sim_window") and self._live_sim_window is not None:
            try:
                self._live_sim_window.lift()
                return
            except tk.TclError:
                pass

        if hasattr(self, "_ls_after_id") and self._ls_after_id is not None:
            self.root.after_cancel(self._ls_after_id)
            self._ls_after_id = None

        win = tk.Toplevel(self.root)
        win.title("EvoComm — Live Simulation")
        win.geometry("900x700")
        win.configure(bg=BG_DARK)
        win.resizable(True, True)
        self._live_sim_window = win

        def _on_close() -> None:
            if self._ls_after_id is not None:
                self.root.after_cancel(self._ls_after_id)
                self._ls_after_id = None
            self._ls_playing = False
            self._live_sim_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        self.canvas_live = tk.Canvas(
            win, width=_CANVAS_PX, height=_CANVAS_PX,
            bg="#0d0d1a", highlightthickness=0,
        )
        self.canvas_live.pack(side="left", padx=10, pady=10)

        ctrl = tk.Frame(win, bg=BG_PANEL)
        ctrl.pack(side="right", fill="y", padx=(0, 10), pady=10)

        tk.Label(ctrl, text="LIVE SIMULATION", bg=BG_PANEL, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(pady=(14, 4))
        ttk.Separator(ctrl).pack(fill="x", padx=10, pady=4)

        self._section_label(ctrl, "Topology")
        self._ls_topo_var = tk.StringVar(value="Evolved (demo)")
        topo_options = ["Evolved (demo)", "Ring", "Star", "Full", "Empty"]
        ttk.OptionMenu(ctrl, self._ls_topo_var, "Evolved (demo)", *topo_options,
                       command=self._ls_on_topology_change).pack(
                           fill="x", padx=10, pady=(0, 6))

        self._section_label(ctrl, "Agent Mode")
        self._ls_mode_var = tk.StringVar(value="cooperative")
        mode_opts = ["cooperative", "mixed", "strategic"]
        ttk.OptionMenu(ctrl, self._ls_mode_var, "cooperative", *mode_opts,
                       command=self._ls_on_mode_change).pack(
                           fill="x", padx=10, pady=(0, 6))

        self._section_label(ctrl, "Seed")
        seed_row = tk.Frame(ctrl, bg=BG_PANEL)
        seed_row.pack(fill="x", padx=10, pady=(0, 6))
        self._ls_seed_var = tk.StringVar(value="42")
        tk.Entry(seed_row, textvariable=self._ls_seed_var,
                 bg=BG_INPUT, fg=FG_WHITE, insertbackground=FG_WHITE,
                 relief="flat", font=("Helvetica", 10), width=8,
                 ).pack(side="left")

        ttk.Separator(ctrl).pack(fill="x", padx=10, pady=6)

        btn_row = tk.Frame(ctrl, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=10, pady=(0, 4))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._ls_play_btn = tk.Button(
            btn_row, text="▶ Play",
            bg=GREEN, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            command=self._ls_toggle_play,
        )
        self._ls_play_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2), ipady=5)

        self._ls_step_btn = tk.Button(
            btn_row, text="Step",
            bg=BG_INPUT, fg=FG_WHITE, font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            command=self._ls_step_once,
        )
        self._ls_step_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0), ipady=5)

        tk.Button(
            ctrl, text="Reset",
            bg=ORANGE, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            command=self._ls_reset,
        ).pack(fill="x", padx=10, pady=(2, 6), ipady=5)

        self._section_label(ctrl, "Speed (ms / step)")
        self._ls_speed_var = tk.DoubleVar(value=300)
        tk.Scale(
            ctrl, variable=self._ls_speed_var, from_=50, to=1000, resolution=50,
            orient="horizontal", bg=BG_PANEL, fg=FG_WHITE,
            highlightthickness=0, troughcolor=BG_INPUT,
            activebackground=ACCENT, sliderrelief="flat", showvalue=True,
        ).pack(fill="x", padx=10, pady=(0, 6))

        ttk.Separator(ctrl).pack(fill="x", padx=10, pady=6)

        self._section_label(ctrl, "Status")
        self._ls_lbl_step = tk.Label(ctrl, text="Step: —", bg=BG_PANEL,
                                      fg=FG_WHITE, font=("Helvetica", 9))
        self._ls_lbl_step.pack(anchor="w", padx=12)

        self._ls_lbl_tasks = tk.Label(ctrl, text="Tasks: —", bg=BG_PANEL,
                                       fg=FG_WHITE, font=("Helvetica", 9))
        self._ls_lbl_tasks.pack(anchor="w", padx=12)

        self._ls_lbl_msgs = tk.Label(ctrl, text="Messages: —", bg=BG_PANEL,
                                      fg=FG_WHITE, font=("Helvetica", 9))
        self._ls_lbl_msgs.pack(anchor="w", padx=12)

        self._ls_lbl_topo = tk.Label(ctrl, text="Topology: —", bg=BG_PANEL,
                                      fg=FG_DIM, font=("Helvetica", 8),
                                      wraplength=200, justify="left")
        self._ls_lbl_topo.pack(anchor="w", padx=12, pady=(4, 0))

        self._ls_runner  = LiveSimulationRunner()
        self._ls_playing = False
        self._ls_after_id: str | None = None

        win.update()
        win.after(50, self._ls_reset)

    def _render_live_sim(self, state: dict) -> None:
        if not state:
            return

        cv        = self.canvas_live
        positions = state["positions"]
        adj       = state["adjacency_matrix"]
        n_agents  = state["num_agents"]

        cv.delete("all")

        grid_end = _MARGIN + 10 * _CELL_SIZE
        for i in range(11):
            x = _MARGIN + i * _CELL_SIZE
            cv.create_line(x, _MARGIN, x, grid_end, fill="#1f2030", width=1)
            cv.create_line(_MARGIN, x, grid_end, x, fill="#1f2030", width=1)

        halo_r = int(2.5 * _CELL_SIZE)
        for i, (row, col) in enumerate(positions):
            cx, cy = _cell_to_px(row, col)
            cv.create_oval(cx - halo_r, cy - halo_r,
                           cx + halo_r, cy + halo_r,
                           fill="#161628", outline="")

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if adj[i][j]:
                    x1, y1 = _cell_to_px(*positions[i])
                    x2, y2 = _cell_to_px(*positions[j])
                    cv.create_line(x1, y1, x2, y2, fill="#555566", width=1)

        _shade = {3: 1.0, 2: 0.6, 1: 0.3}
        for (sender_id, recv_id), frames in state["flash_edges"].items():
            if sender_id >= n_agents or recv_id >= n_agents:
                continue
            base  = _AGENT_COLORS[sender_id % len(_AGENT_COLORS)]
            color = _dim_color(base, _shade.get(frames, 0.3))
            width = 2 + 3 * (frames / 3)
            x1, y1 = _cell_to_px(*positions[sender_id])
            x2, y2 = _cell_to_px(*positions[recv_id])
            cv.create_line(x1, y1, x2, y2, fill=color, width=width)

        for task in state["all_tasks"]:
            tr, tc = task["position"]
            tx, ty = _cell_to_px(tr, tc)
            if task["completed"]:
                r = 14
                cv.create_oval(tx - r, ty - r, tx + r, ty + r,
                               fill="#43a047", outline="#66bb6a", width=1)
                cv.create_text(tx, ty, text="✓", fill="white",
                               font=("Helvetica", 11, "bold"))
            else:
                d = 11
                cv.create_line(tx - d, ty - d, tx + d, ty + d,
                               fill="#e53935", width=3)
                cv.create_line(tx + d, ty - d, tx - d, ty + d,
                               fill="#e53935", width=3)
                cv.create_text(tx + 14, ty + 14,
                               text=f"T{task['id']}×{task['required_agents']}",
                               fill="#ff8a80", font=("Helvetica", 8), anchor="nw")

        known_per_agent = state["known_tasks"]
        for i, (row, col) in enumerate(positions):
            color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            known = known_per_agent[i] if i < len(known_per_agent) else {}
            total_k = max(len(known), 1)
            for k, tinfo in enumerate(known.values()):
                tr, tc = tinfo["position"]
                tx, ty = _cell_to_px(tr, tc)
                angle  = k * (2 * math.pi / total_k)
                mx = tx + int(14 * math.cos(angle))
                my = ty + int(14 * math.sin(angle))
                r = 4
                cv.create_oval(mx - r, my - r, mx + r, my + r,
                               fill=color, outline="")

        for i, (row, col) in enumerate(positions):
            color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            cx, cy = _cell_to_px(row, col)
            r = 16
            cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=color, outline="white", width=2)
            cv.create_text(cx, cy, text=str(i), fill="white",
                           font=("Helvetica", 11, "bold"))

        if state.get("mission_complete"):
            mid = _CANVAS_PX // 2
            cv.create_rectangle(mid - 190, mid - 44, mid + 190, mid + 44,
                                fill="#0d0d1a", stipple="gray50",
                                outline="#43a047", width=2)
            cv.create_text(mid, mid, text="MISSION COMPLETE",
                           fill="#43a047", font=("Helvetica", 22, "bold"))

        cv.create_text(
            _CANVAS_PX // 2, 8,
            text=(f"Step {state['step_count']}  |  "
                  f"Tasks {state['tasks_done']}/{state['tasks_total']}  |  "
                  f"Messages {state['total_messages']}"),
            fill=FG_WHITE, font=("Helvetica", 9), anchor="n",
        )
        cv.update_idletasks()

    def _update_ls_status(self, state: dict) -> None:
        if not state:
            return
        self._ls_lbl_step.config(text=f"Step: {state['step_count']}")
        done  = state["tasks_done"]
        total = state["tasks_total"]
        self._ls_lbl_tasks.config(
            text=f"Tasks: {done}/{total}",
            fg=GREEN if done == total else FG_WHITE,
        )
        self._ls_lbl_msgs.config(text=f"Messages: {state['total_messages']}")

    def _ls_show_error(self, msg: str) -> None:
        self.canvas_live.delete("all")
        self.canvas_live.create_text(
            _CANVAS_PX // 2, _CANVAS_PX // 2,
            text=msg, fill="#e53935", font=("Helvetica", 11),
            width=400, anchor="center",
        )
        self._ls_play_btn.config(state="disabled")
        self._ls_step_btn.config(state="disabled")

    def _ls_toggle_play(self) -> None:
        self._ls_playing = not self._ls_playing
        if self._ls_playing:
            self._ls_play_btn.config(text="⏸ Pause", bg=ORANGE)
            self._ls_step_btn.config(state="disabled")
            self._ls_play_loop()
        else:
            self._ls_play_btn.config(text="▶ Play", bg=GREEN)
            self._ls_step_btn.config(state="normal")
            if self._ls_after_id is not None:
                self.root.after_cancel(self._ls_after_id)
                self._ls_after_id = None

    def _ls_play_loop(self) -> None:
        if not self._ls_playing:
            return
        self._ls_do_step()
        if not self._ls_runner.mission_complete:
            delay = int(self._ls_speed_var.get())
            self._ls_after_id = self.root.after(delay, self._ls_play_loop)
        else:
            self._ls_playing = False
            self._ls_play_btn.config(text="▶ Play", bg=GREEN)
            self._ls_step_btn.config(state="normal")

    def _ls_do_step(self) -> None:
        state = self._ls_runner.step()
        self._render_live_sim(state)
        self._update_ls_status(state)

    def _ls_step_once(self) -> None:
        if not self._ls_playing:
            self._ls_do_step()

    def _ls_reset(self) -> None:
        if self._ls_playing:
            self._ls_toggle_play()

        topo_name = self._ls_topo_var.get()
        err = self._ls_runner.load_topology(topo_name)
        if err:
            self._ls_show_error(err)
            return

        try:
            seed = int(self._ls_seed_var.get())
        except ValueError:
            seed = 42
            self._ls_seed_var.set("42")

        self._ls_runner.reset(seed=seed)
        self._ls_play_btn.config(state="normal")
        self._ls_step_btn.config(state="normal")

        g = self._ls_runner.topology_genome
        if g is not None:
            conn = "connected" if g.is_connected() else "DISCONNECTED"
            self._ls_lbl_topo.config(
                text=f"{topo_name} | density={g.graph_density:.2f} | {conn}",
                fg=GREEN if g.is_connected() else RED,
            )

        state = self._ls_runner._state()
        self._render_live_sim(state)
        self._update_ls_status(state)

    def _ls_on_topology_change(self, name: str) -> None:
        self._ls_reset()

    def _ls_on_mode_change(self, mode: str) -> None:
        self._ls_runner.agent_mode = mode
        self._ls_reset()

    def _start_evolution(self) -> None:
        self._stop_flag = False
        self._gen_x.clear()
        self._fit_y.clear()

        pop_size   = int(self.var_pop.get())
        n_gen      = int(self.var_gen.get())
        mut_rate   = round(float(self.var_mut.get()), 2)
        n_agents   = int(self.var_agents.get())
        agent_mode = self.var_mode.get()

        self.engine = EvolutionEngine(
            population_size=pop_size,
            num_generations=n_gen,
            mutation_rate=mut_rate,
            num_agents=n_agents,
            agent_mode=agent_mode,
            generation_callback=self._on_generation,
            complete_callback=self._on_complete,
        )
        self.engine._stop_requested = False

        self.progress["maximum"] = n_gen
        self.progress["value"]   = 0
        self._set_status("Running... Gen 0/" + str(n_gen), ACCENT)
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")

        self._reset_fitness_chart()

        self._evo_thread = threading.Thread(
            target=self._evolution_worker, daemon=True
        )
        self._evo_thread.start()

    def _evolution_worker(self) -> None:
        try:
            self.engine.run()
        except Exception as exc:
            self.root.after(0, lambda: self._set_status(f"Error: {exc}", RED))
            self.root.after(0, self._re_enable_run)

    def _stop_evolution(self) -> None:
        if self.engine:
            self.engine._stop_requested = True
        self._set_status("Stopping...", ORANGE)
        self.btn_stop.config(state="disabled")

    def _on_generation(self, gen_num: int, best_fitness: float) -> None:
        self.root.after(0, self._update_generation_ui, gen_num, best_fitness)

    def _on_complete(self, best_genome: TopologyGenome) -> None:
        self.root.after(0, self._update_complete_ui, best_genome)

    def _update_generation_ui(self, gen_num: int, best_fitness: float) -> None:
        total = self.engine.num_generations if self.engine else gen_num
        self._set_status(f"Running... Gen {gen_num}/{total}", ACCENT)
        self.progress["value"] = gen_num

        self._gen_x.append(gen_num)
        self._fit_y.append(best_fitness)

        self.line_fitness.set_data(self._gen_x, self._fit_y)
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        self.canvas_fitness.draw_idle()

    def _update_complete_ui(self, best_genome: TopologyGenome) -> None:
        self._set_status(
            f"Complete — fitness {best_genome.fitness_score:.4f}", GREEN
        )
        self.progress["value"] = self.progress["maximum"]
        self._re_enable_run()

        self._render_topology(best_genome)
        threading.Thread(
            target=self._run_comparison_worker,
            args=(best_genome,),
            daemon=True,
        ).start()

    def _render_topology(self, genome: TopologyGenome) -> None:
        self.ax_topo.clear()
        self._style_axis(self.ax_topo,
                         f"Best Topology  |  density={genome.graph_density:.2f}"
                         f"  |  fitness={genome.fitness_score:.4f}", "", "")

        G = nx.Graph()
        G.add_nodes_from(range(genome.num_agents))
        rows, cols = np.where(np.triu(genome.adjacency_matrix, k=1))
        for i, j in zip(rows.tolist(), cols.tolist()):
            G.add_edge(int(i), int(j))

        pos = nx.spring_layout(G, seed=42, k=1.5)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_colours = [degrees[n] / max(max_deg, 1) for n in G.nodes()]
        cmap = plt.cm.Blues

        nx.draw_networkx_nodes(G, pos, ax=self.ax_topo,
                               node_color=node_colours, cmap=cmap,
                               node_size=800, alpha=0.9,
                               vmin=0, vmax=1)
        nx.draw_networkx_labels(G, pos, ax=self.ax_topo,
                                font_color="white", font_size=11,
                                font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=self.ax_topo,
                               edge_color="#888", width=2, alpha=0.7)
        self.canvas_topo.draw_idle()

        num_edges = G.number_of_edges()
        connected = "yes" if genome.is_connected() else "NO"
        self.lbl_topo_info.config(
            text=(f"Agents: {genome.num_agents}  |  "
                  f"Edges: {num_edges}  |  "
                  f"Density: {genome.graph_density:.3f}  |  "
                  f"Connected: {connected}"),
            fg=GREEN if genome.is_connected() else RED,
        )

    def _run_comparison_worker(self, evolved_genome: TopologyGenome) -> None:
        try:
            n = evolved_genome.num_agents
            full_mat = np.ones((n, n), dtype=bool)
            np.fill_diagonal(full_mat, False)
            ring_mat = np.zeros((n, n), dtype=bool)
            for i in range(n):
                ring_mat[i][(i + 1) % n] = ring_mat[(i + 1) % n][i] = True
            star_mat = np.zeros((n, n), dtype=bool)
            for i in range(1, n):
                star_mat[0][i] = star_mat[i][0] = True

            mode = self.engine.agent_mode if self.engine else "cooperative"
            topologies = {
                "Evolved": evolved_genome,
                "Full":    TopologyGenome(n, full_mat),
                "Ring":    TopologyGenome(n, ring_mat),
                "Star":    TopologyGenome(n, star_mat),
            }
            scores: dict[str, float] = {}
            for name, genome in topologies.items():
                ev = FitnessEvaluator(num_trials=10, agent_mode=mode)
                scores[name] = ev.evaluate(genome)

            self.root.after(0, self._update_comparison_ui, scores)
        except Exception as exc:
            self.root.after(0, lambda: self.lbl_cmp_summary.config(
                text=f"Comparison error: {exc}", fg=RED
            ))

    def _update_comparison_ui(self, scores: dict[str, float]) -> None:
        self.ax_cmp.clear()
        self._style_axis(self.ax_cmp,
                         "Topology Comparison (10 trials each)",
                         "Topology", "Fitness Score")

        names   = list(scores.keys())
        values  = list(scores.values())
        colours = [ORANGE if n == "Evolved" else "#607d8b" for n in names]

        bars = self.ax_cmp.bar(names, values, color=colours, alpha=0.85,
                               edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            self.ax_cmp.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.005,
                             f"{val:.3f}", ha="center", va="bottom",
                             color=FG_WHITE, fontsize=9)

        self.ax_cmp.set_ylim(0, max(values) * 1.2 if values else 0.5)
        self.canvas_cmp.draw_idle()

        rank = sorted(scores, key=scores.get, reverse=True).index("Evolved") + 1
        self.lbl_cmp_summary.config(
            text=f"Evolved topology ranked {rank} of {len(scores)} topologies",
            fg=GREEN if rank == 1 else FG_DIM,
        )
        self.notebook.select(2)

    def _re_enable_run(self) -> None:
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _set_status(self, text: str, colour: str = FG_WHITE) -> None:
        self.lbl_status.config(text=text, fg=colour)

    def _reset_fitness_chart(self) -> None:
        self.ax_fitness.clear()
        self._style_axis(self.ax_fitness, "EvoComm — Fitness Progress",
                         "Generation", "Best Fitness Score")
        self.line_fitness, = self.ax_fitness.plot([], [], color=ACCENT,
                                                   linewidth=2, marker="o",
                                                   markersize=4)
        self.canvas_fitness.draw_idle()

    def _section_label(self, parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text.upper(), bg=BG_PANEL, fg=ACCENT,
                 font=("Helvetica", 8, "bold")).pack(anchor="w",
                                                      padx=12, pady=(8, 2))

    def _slider(
        self,
        parent: tk.Widget,
        label: str,
        from_: float,
        to: float,
        default: float,
        resolution: float = 1,
    ) -> tk.DoubleVar:
        var = tk.DoubleVar(value=default)
        frame = tk.Frame(parent, bg=BG_PANEL)
        frame.pack(fill="x", padx=12, pady=2)

        tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_WHITE,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")

        tk.Label(frame, textvariable=var, bg=BG_PANEL,
                 fg=ACCENT, font=("Helvetica", 9), width=5).pack(side="right")

        tk.Scale(
            parent, variable=var, from_=from_, to=to, resolution=resolution,
            orient="horizontal", bg=BG_PANEL, fg=FG_WHITE,
            highlightthickness=0, troughcolor=BG_INPUT,
            activebackground=ACCENT, sliderrelief="flat",
            showvalue=False,
        ).pack(fill="x", padx=12, pady=(0, 2))

        return var

    @staticmethod
    def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(BG_PANEL)
        ax.figure.patch.set_facecolor(BG_DARK)
        ax.set_title(title, color=FG_WHITE, fontsize=11, pad=10)
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=10)
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=10)
        ax.tick_params(colors=FG_DIM)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_INPUT)
        ax.grid(True, linestyle="--", alpha=0.3, color=FG_DIM)


if __name__ == "__main__":
    root = tk.Tk()
    app = EvoCommGUI(root)
    root.mainloop()
