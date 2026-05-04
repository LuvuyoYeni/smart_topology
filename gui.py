"""
gui.py
------
Primary demonstration interface for EvoComm.

Single-window tkinter application with live fitness chart, topology
visualisation, baseline comparison, and a contextual about page.
Evolution runs in a background daemon thread so the UI stays responsive.
"""

from __future__ import annotations

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

from evolution_engine import EvolutionEngine
from fitness_evaluator import FitnessEvaluator
from topology_genome import TopologyGenome


# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_INPUT  = "#313145"
FG_WHITE  = "#e0e0f0"
FG_DIM    = "#9090a0"
ACCENT    = "#4fc3f7"
GREEN     = "#43a047"
RED       = "#e53935"
ORANGE    = "#fb8c00"


class EvoCommGUI:
    """
    Main application window for EvoComm.

    Attributes
    ----------
    root : tk.Tk
        The top-level tkinter window.
    engine : EvolutionEngine or None
        Active engine instance while evolution is running.
    _stop_flag : bool
        Set to True when the user clicks STOP.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialise the window, lay out all widgets, and set initial state.

        Parameters
        ----------
        root : tk.Tk
            The top-level tkinter window.
        """
        self.root = root
        self.root.title("EvoComm — Evolving Communication Topologies")
        self.root.geometry("1200x750")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self.engine: EvolutionEngine | None = None
        self._stop_flag = False
        self._evo_thread: threading.Thread | None = None

        # Data for live chart
        self._gen_x: list[int] = []
        self._fit_y: list[float] = []

        self._build_layout()

    # ------------------------------------------------------------------ #
    #  Layout construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        """Construct left control panel and right tabbed visualisation panel."""
        # Root columns: 300px left, flex right
        self.root.columnconfigure(0, minsize=300, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self) -> None:
        """Build the configuration, run-controls, and context sections."""
        left = tk.Frame(self.root, bg=BG_PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        # ── Title ─────────────────────────────────────────────────────────
        tk.Label(left, text="EvoComm", bg=BG_PANEL, fg=ACCENT,
                 font=("Helvetica", 16, "bold")).pack(pady=(14, 0))
        tk.Label(left, text="Evolving Communication Topologies",
                 bg=BG_PANEL, fg=FG_DIM, font=("Helvetica", 8)).pack(pady=(0, 10))
        ttk.Separator(left).pack(fill="x", padx=10)

        # ── Configuration section ─────────────────────────────────────────
        self._section_label(left, "Configuration")

        self.var_pop   = self._slider(left, "Population Size", 10, 50, 20, resolution=1)
        self.var_gen   = self._slider(left, "Generations",     10, 100, 50, resolution=1)
        self.var_mut   = self._slider(left, "Mutation Rate",   0.1, 0.9, 0.4, resolution=0.1)
        self.var_agents= self._slider(left, "Num Agents",      4, 12, 6, resolution=1)

        # Agent mode
        mode_frame = tk.Frame(left, bg=BG_PANEL)
        mode_frame.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(mode_frame, text="Agent Mode", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
        self.var_mode = tk.StringVar(value="cooperative")
        ttk.OptionMenu(mode_frame, self.var_mode,
                       "cooperative", "cooperative", "mixed", "strategic"
                       ).pack(side="left", fill="x", expand=True)

        ttk.Separator(left).pack(fill="x", padx=10, pady=8)

        # ── Run controls ──────────────────────────────────────────────────
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

        # Status label
        self.lbl_status = tk.Label(
            left, text="Ready", bg=BG_PANEL, fg=ACCENT,
            font=("Helvetica", 9, "italic"),
        )
        self.lbl_status.pack(pady=(8, 2))

        # Progress bar
        self.progress = ttk.Progressbar(left, orient="horizontal",
                                        mode="determinate", length=260)
        self.progress.pack(padx=12, pady=(2, 8))

        ttk.Separator(left).pack(fill="x", padx=10)

        # ── Scenario context ──────────────────────────────────────────────
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

    def _build_right_panel(self) -> None:
        """Build the tabbed notebook with four visualisation tabs."""
        right = tk.Frame(self.root, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",          background=BG_DARK,  borderwidth=0)
        style.configure("TNotebook.Tab",      background=BG_PANEL, foreground=FG_DIM,
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

    # ── Tab 1: Fitness Progress ───────────────────────────────────────────

    def _build_tab_fitness(self) -> None:
        """Create the live fitness-curve tab."""
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

    # ── Tab 2: Best Topology ─────────────────────────────────────────────

    def _build_tab_topology(self) -> None:
        """Create the best-topology network graph tab."""
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

    # ── Tab 3: Topology Comparison ───────────────────────────────────────

    def _build_tab_comparison(self) -> None:
        """Create the baseline-comparison bar-chart tab."""
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

    # ── Tab 4: About ─────────────────────────────────────────────────────

    def _build_tab_about(self) -> None:
        """Create the static about / research-context tab."""
        frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(frame, text="  About EvoComm  ")

        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                   command=canvas.yview)
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

    # ------------------------------------------------------------------ #
    #  Evolution control                                                   #
    # ------------------------------------------------------------------ #

    def _start_evolution(self) -> None:
        """Read controls, build engine, and launch the background thread."""
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

        # Reset UI
        self.progress["maximum"] = n_gen
        self.progress["value"]   = 0
        self._set_status("Running... Gen 0/" + str(n_gen), ACCENT)
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")

        # Clear fitness chart
        self._reset_fitness_chart()

        self._evo_thread = threading.Thread(
            target=self._evolution_worker, daemon=True
        )
        self._evo_thread.start()

    def _evolution_worker(self) -> None:
        """Run in a background thread — calls engine.run()."""
        try:
            self.engine.run()
        except Exception as exc:
            self.root.after(0, lambda: self._set_status(
                f"Error: {exc}", RED
            ))
            self.root.after(0, self._re_enable_run)

    def _stop_evolution(self) -> None:
        """Signal the engine to stop at the next generation boundary."""
        if self.engine:
            self.engine._stop_requested = True
        self._set_status("Stopping...", ORANGE)
        self.btn_stop.config(state="disabled")

    # ------------------------------------------------------------------ #
    #  Callbacks from engine (called from background thread)              #
    # ------------------------------------------------------------------ #

    def _on_generation(self, gen_num: int, best_fitness: float) -> None:
        """Received each generation — marshals update onto the main thread."""
        self.root.after(0, self._update_generation_ui, gen_num, best_fitness)

    def _on_complete(self, best_genome: TopologyGenome) -> None:
        """Received when evolution finishes — marshals onto main thread."""
        self.root.after(0, self._update_complete_ui, best_genome)

    # ------------------------------------------------------------------ #
    #  Main-thread UI update methods                                      #
    # ------------------------------------------------------------------ #

    def _update_generation_ui(self, gen_num: int, best_fitness: float) -> None:
        """Update progress bar, status, and live fitness chart."""
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
        """Render topology graph, trigger comparison, finalize UI."""
        total = self.engine.num_generations if self.engine else "?"
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
        """Draw the best topology network graph in the topology tab."""
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
        """Build baselines and evaluate all topologies (background thread)."""
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
        """Render the comparison bar chart on the main thread."""
        self.ax_cmp.clear()
        self._style_axis(self.ax_cmp,
                         "Topology Comparison (10 trials each)",
                         "Topology", "Fitness Score")

        names  = list(scores.keys())
        values = list(scores.values())
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

        # Switch to comparison tab automatically
        self.notebook.select(2)

    # ------------------------------------------------------------------ #
    #  Helper utilities                                                    #
    # ------------------------------------------------------------------ #

    def _re_enable_run(self) -> None:
        """Restore run button and disable stop button."""
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _set_status(self, text: str, colour: str = FG_WHITE) -> None:
        """Update the status label text and colour."""
        self.lbl_status.config(text=text, fg=colour)

    def _reset_fitness_chart(self) -> None:
        """Clear the fitness axes ready for a new run."""
        self.ax_fitness.clear()
        self._style_axis(self.ax_fitness, "EvoComm — Fitness Progress",
                         "Generation", "Best Fitness Score")
        self.line_fitness, = self.ax_fitness.plot([], [], color=ACCENT,
                                                   linewidth=2, marker="o",
                                                   markersize=4)
        self.canvas_fitness.draw_idle()

    def _section_label(self, parent: tk.Widget, text: str) -> None:
        """Render a small section header inside a panel."""
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
        """
        Add a labelled Scale slider to *parent* and return its variable.

        Parameters
        ----------
        parent : tk.Widget
            Container frame.
        label : str
            Display label.
        from_ : float
            Minimum value.
        to : float
            Maximum value.
        default : float
            Initial value.
        resolution : float
            Slider step size.
        """
        var = tk.DoubleVar(value=default)
        frame = tk.Frame(parent, bg=BG_PANEL)
        frame.pack(fill="x", padx=12, pady=2)

        tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_WHITE,
                 font=("Helvetica", 9), width=16, anchor="w").pack(side="left")

        val_lbl = tk.Label(frame, textvariable=var, bg=BG_PANEL,
                           fg=ACCENT, font=("Helvetica", 9), width=5)
        val_lbl.pack(side="right")

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
        """Apply consistent dark-theme styling to a matplotlib Axes."""
        ax.set_facecolor(BG_PANEL)
        ax.figure.patch.set_facecolor(BG_DARK)
        ax.set_title(title, color=FG_WHITE, fontsize=11, pad=10)
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=10)
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=10)
        ax.tick_params(colors=FG_DIM)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_INPUT)
        ax.grid(True, linestyle="--", alpha=0.3, color=FG_DIM)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = EvoCommGUI(root)
    root.mainloop()
