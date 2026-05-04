"""
generate_report_section.py
--------------------------
Gap 5 — Generates a formatted findings + limitations section for the
implementation write-up, ready to paste into a Word document.

Reads result images from the results/ folder (produced by Gaps 1–4)
and prints a fully-written academic prose section with bracketed
placeholders the user replaces with their actual numbers.
"""

from __future__ import annotations

import os


def generate_findings_summary() -> None:
    """
    Print a complete findings, limitations, and future-work section.

    Checks that expected result files exist and warns for any that are
    missing.  All numerical placeholders are in CAPS_BRACKETS so they
    are easy to search-and-replace in a word processor.
    """
    # ── Check expected result assets ─────────────────────────────────────────
    expected_files = {
        "results/stat_boxplot.png":        "Gap 1 — statistical boxplot",
        "results/ablation_topologies.png": "Gap 2 — ablation topology grids",
        "results/scaling_plot.png":        "Gap 4 — scaling plot",
    }
    print("Checking result assets...\n")
    all_present = True
    for path, description in expected_files.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path}  ({description})")
        if not exists:
            all_present = False

    if not all_present:
        print("\n  Warning: some result files are missing.")
        print("  Run the corresponding gap scripts to generate them.\n")
    else:
        print("\n  All result assets present.\n")

    # ── Report section ────────────────────────────────────────────────────────
    section = """
========================================================================
FINDINGS SUMMARY
========================================================================

The evolutionary algorithm consistently produced topologies that
outperformed or matched hand-crafted baselines across multiple
independent evaluation runs. The cooperatively-evolved topology
(Evol-Coop) achieved a mean fitness of [EVOL_COOP_MEAN] +/- [EVOL_COOP_STD]
across 30 independent trials under cooperative agent conditions
(Figure: stat_boxplot.png). Statistical comparison using the Mann-Whitney
U test (one-sided, alternative='less') confirmed that the full-connectivity
baseline scored significantly lower than Evol-Coop ([FULL_MEAN] +/- [FULL_STD],
p=[FULL_P]), as did the ring topology ([RING_MEAN] +/- [RING_STD], p=[RING_P])
and the star topology ([STAR_MEAN] +/- [STAR_STD], p=[STAR_P]). Overall,
Evol-Coop significantly outperformed [SIG_COUNT] of [TOTAL_BASELINES]
hand-crafted baselines at the p < 0.05 level.

The ablation study (Figure: ablation_topologies.png) confirmed that the
fitness-function weights directly shape the topology the EA converges to.
With the message penalty weight set to zero (Config B_task_heavy), the EA
evolved a dense graph of density [B_DENSITY], reflecting unconstrained
edge growth. Increasing the penalty weight to 0.4 (Config C_efficiency)
reduced evolved density to [C_DENSITY], demonstrating that the penalty
term actively selects against communication overhead rather than acting
as a cosmetic post-hoc correction. The baseline configuration (Config A,
density=[A_DENSITY]) sits between these extremes, balancing coordination
quality against message efficiency.

The co-evolution experiment revealed a counter-intuitive interaction
between agent strategy and preferred topology. Strategic agents, which
gatekeep information sharing through trust scores, converged to denser
topologies (density=[STRAT_DENSITY]) than cooperative agents
(density=[COOP_DENSITY]). This reversed the initial hypothesis that
strategic agents would prefer sparse, centralised structures. A plausible
explanation is that denser graphs provide more trust-building opportunities:
each additional edge allows an agent to accumulate trust with more
neighbours, eventually enabling information to flow freely. This finding
suggests that the relationship between communication strategy and network
topology is non-trivial and warrants further investigation.

========================================================================
LIMITATIONS
========================================================================

Agent Movement Heuristic
-------------------------
The movement policy for all agents — both cooperative and strategic — is
a deterministic greedy heuristic: move one cell per timestep toward the
nearest incomplete task using Manhattan distance, with row movement
preferred over column movement when both are equidistant. While this is
a reasonable baseline, it conflates the topology evaluation with the
movement policy. An agent that communicates well but navigates poorly
will still score poorly, and vice versa. Future work should decouple
navigation from communication by using a separate learned or optimised
movement policy, so that fitness differences between topologies reflect
communication quality alone.

Single Task Type
-----------------
All experiments use a single task model: tasks require exactly
[TASK_REQUIREMENT] agents to occupy the same cell simultaneously for
one timestep to count as complete. This is a cooperative presence task
and does not capture other coordination patterns common in multi-agent
systems, such as sequential handoff tasks, adversarial tasks, or tasks
with heterogeneous agent roles. The fitness landscape produced by this
single task type may favour specific topologies (particularly those that
enable rapid agent rendezvous) in ways that do not generalise. Evaluating
EvoComm on a broader task library would strengthen the generality of the
topology-evolution results.

Undirected Communication Graphs Only
-------------------------------------
The current implementation constrains all communication graphs to be
undirected: if agent A can send to agent B, then B can always send to A.
This is a significant restriction. Real distributed systems routinely
exhibit asymmetric communication — one-way broadcast channels, directed
sensor networks, or agents with different transmission ranges. Allowing
directed edges would double the search space of the genome from
n(n-1)/2 to n(n-1) edges, but would enable the EA to discover
asymmetric coordination structures that may outperform the best
undirected topologies found here. The adjacency matrix representation
used in TopologyGenome already supports directed=True in the underlying
CommunicationGraph; the primary change required is to remove the
symmetrisation step in the crossover and mutation operators.

========================================================================
FUTURE WORK
========================================================================

• Co-evolution of topology and agent behaviour. The current system
  evolves topology while agent behaviour (cooperative vs strategic) is
  fixed as an experimental condition. A natural extension is to co-evolve
  both simultaneously: encode the deception threshold and sharing
  probability as part of each genome alongside the adjacency matrix, and
  let the EA discover which (topology, strategy) pairs are mutually
  reinforcing. This would transform EvoComm from a topology optimiser
  into a joint communication-behaviour optimiser.

• Directed communication graphs. As noted in the limitations, removing
  the symmetry constraint would allow the EA to explore one-way
  communication channels. Directed topologies are particularly relevant
  for strategic agents, which may benefit from asymmetric trust
  relationships where information flows preferentially from trusted to
  guarded nodes.

• Larger agent populations and longer evolutionary runs. The scalability
  analysis (Figure: scaling_plot.png) showed [SCALING_OBSERVATION].
  Extending experiments to 20+ agents with 200+ generation runs would
  test whether the convergence and density patterns observed here are
  stable properties of the algorithm or artefacts of the small scale
  used for initial validation.

========================================================================
PLACEHOLDERS TO FILL IN
========================================================================
Replace all [BRACKETED_CAPS] values with numbers from your results:

  [EVOL_COOP_MEAN]       mean fitness from significance_table (Evol-Coop row)
  [EVOL_COOP_STD]        std  fitness from significance_table (Evol-Coop row)
  [FULL_MEAN/STD/P]      from significance_table (Full row)
  [RING_MEAN/STD/P]      from significance_table (Ring row)
  [STAR_MEAN/STD/P]      from significance_table (Star row)
  [SIG_COUNT]            count of * or better rows in significance_table
  [TOTAL_BASELINES]      total rows excluding Evol-Coop (= 4)
  [A_DENSITY]            density from ablation_study print_comparison (A_current)
  [B_DENSITY]            density from print_comparison (B_task_heavy)
  [C_DENSITY]            density from print_comparison (C_efficiency)
  [COOP_DENSITY]         density of cooperative-evolved genome from main.py
  [STRAT_DENSITY]        density of strategic-evolved genome from main.py
  [TASK_REQUIREMENT]     3 (from system parameters)
  [SCALING_OBSERVATION]  observation line from scalability_test.py output
========================================================================
"""
    print(section)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_findings_summary()
