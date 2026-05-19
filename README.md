# EvoComm — Evolving Communication Topologies in Multi-Agent Systems

EvoComm uses a genetic algorithm to discover optimal communication graph
structures for a multi-agent coordination task. Rather than hand-designing
who can talk to whom, the evolutionary algorithm searches a space of boolean
adjacency matrices and selects topologies that help agents complete shared
tasks quickly with minimal message overhead. Agents operate on a 10×10 grid
and must coordinate to occupy task cells simultaneously; two agent types are
supported — cooperative agents that always share information, and strategic
agents that gate sharing on a per-neighbour trust score. The system was
developed as an Honours AI research project at the University of Johannesburg
(IT08X97, 2026).

---

## Setup

### Prerequisites

- Python 3.9 or later
- pip

### Windows

```
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify tkinter is available (required for the GUI):

```
python3 -c "import tkinter; print('tkinter OK')"
```

If that fails on macOS, install the Tcl/Tk-enabled Python from
[python.org](https://www.python.org/downloads/) rather than the system Python.

---

## Running the project

### Main experiment (`main.py`)

Runs two full evolutionary experiments (cooperative agents, then strategic
agents), prints generation-by-generation progress, displays the best evolved
topology as a network graph, and produces a cross-topology comparison table.

```
python main.py
```

Expected runtime: 5–15 minutes depending on hardware.

---

### GUI demonstration (`gui.py`)

Interactive tkinter window. Configure population size, number of generations,
mutation rate, agent count, and agent mode with sliders, then click
**RUN EVOLUTION** to watch the fitness curve build live. Results appear in
four tabs: Fitness Progress, Best Topology, Topology Comparison, and About.

```
python gui.py
```

---

### Statistical significance testing (`statistical_analysis.py`)

Evaluates five topologies (Evol-Coop, Evol-Strat, Full, Ring, Star) over 30
independent trials each and applies Mann-Whitney U tests to determine whether
the evolved topology significantly outperforms the baselines.

Before running, open the file and replace `EVOL_COOP_MATRIX` and
`EVOL_STRAT_MATRIX` with the adjacency matrices printed by `main.py`.

```
python statistical_analysis.py
```

Output: significance table printed to terminal; boxplot saved to
`results/stat_boxplot.png`.

---

### Ablation study (`ablation_study.py`)

Runs the EA under four fitness-weight configurations to show how changing
`task_weight` and `efficiency_weight` shapes the topology the algorithm
converges to.

```
python ablation_study.py
```

Expected runtime: ~4 minutes (4 × 50-generation runs).
Output: comparison table printed to terminal; topology grid saved to
`results/ablation_topologies.png`.

---

### Strategic evolution fix (`strategic_evolution_fix.py`)

Compares unconstrained strategic evolution (which can exploit disconnected
subgraphs) against `ConnectivityEnforcedEvaluator`, which hard-zeros any
disconnected genome.

```
python strategic_evolution_fix.py
```

Expected runtime: ~3 minutes (2 × 50-generation runs).

---

### Scalability test (`scalability_test.py`)

Runs one evolutionary experiment at each agent count in [4, 6, 8, 10, 12]
and records best fitness, evolved density, and wall-clock runtime.

```
python scalability_test.py
```

Expected runtime: ~8 minutes.
Output: results table printed to terminal; scaling plot saved to
`results/scaling_plot.png`.

---

### Edge-traffic topology visualiser (`visualise_topology.py`)

Instruments a simulation to count messages per edge, then draws the topology
with edges coloured by traffic volume. Run as standalone to see cooperative
and strategic evolved topologies side by side.

```
python visualise_topology.py
```

---

### Report section generator (`generate_report_section.py`)

Checks that the expected result images exist in `results/` and prints a
fully-written findings, limitations, and future-work section with placeholder
values to fill in from your actual output.

```
python generate_report_section.py
```

---

### Module-level tests

Each core module has a standalone test block. Run any of these to verify
that module works correctly in isolation:

```
python communication_graph.py
python message.py
python topology_genome.py
python astar.py
python environment.py
python agent.py
python fitness_evaluator.py
python evolution_engine.py
```

All should print `All tests passed.` with no errors.

---

## Fitness formula

```
F = task_weight × TaskScore
  + efficiency_weight × SpeedBonus
  − 0.3 × MsgPenalty
  − 0.1 × DisconnectionPenalty
```

| Term | Definition | Range |
|---|---|---|
| `TaskScore` | Proportion of tasks completed by end of simulation | [0, 1] |
| `SpeedBonus` | `(max_steps − steps_taken) / max_steps` | [0, 1] |
| `MsgPenalty` | `min(message_count / 30, 1.0)` | [0, 1] |
| `DisconnectionPenalty` | 1.0 if the graph is not fully connected, else 0.0 | {0, 1} |

Default weights: `task_weight = 0.6`, `efficiency_weight = 0.3`.
Both are configurable via the `FitnessEvaluator` constructor.
The message penalty weight (0.3) and disconnection penalty (0.1) are fixed.

Scores are averaged over `num_trials` independent simulation runs (default 5
inside `EvolutionEngine`, 3 when constructing `FitnessEvaluator` directly).

---

## Default EA parameters

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 50 |
| Mutation rate | 0.4 |
| Crossover rate | 0.7 |
| Selection | Tournament (k = 3) |
| Elitism | Best genome preserved each generation |
| Diversity injection | Bottom 50% replaced after 8 stale generations |
| Agents | 6 |
| Grid size | 10 × 10 |
| Tasks | 8 (require 3 agents on same cell) |
| Max steps per simulation | 50 |

---

## Research findings

1. **EA outperforms hand-crafted baselines.** Under cooperative agent
   conditions, the evolved topology consistently scores higher than Full,
   Ring, and Star topologies across 30 independent trials. Statistical
   significance confirmed via Mann-Whitney U test (p < 0.05).

2. **Topology and agent strategy interact.** The best topology for cooperative
   agents is not the best topology for strategic agents, and vice versa.
   Evaluating an evolved topology under a different agent mode than it was
   trained on degrades performance.

3. **Strategic agents cause fitness exploitation without a connectivity
   constraint.** Evolving under strategic agents with the standard evaluator
   produces disconnected subgraphs that score well by reducing message count
   rather than completing tasks. `ConnectivityEnforcedEvaluator` eliminates
   this by hard-zeroing any disconnected genome.

4. **Fitness improves with more agents; density decreases.** As agent count
   grows from 4 to 12, evolved fitness increases (more agents means easier
   task completion) while evolved graph density decreases (larger populations
   need less dense communication to coordinate effectively).

5. **Fitness-function weights shape evolved structure.** Ablation experiments
   show that reducing `task_weight` and increasing the message penalty
   produces measurably sparser evolved topologies, confirming that the
   penalty term actively influences the search rather than acting as a
   post-hoc correction.
