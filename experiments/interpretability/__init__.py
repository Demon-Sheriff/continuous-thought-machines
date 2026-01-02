"""
Mechanistic Interpretability Experiments for Continuous Thought Machines.

This package contains experiments to investigate how CTMs encode
positional information when solving 2D mazes without positional embeddings.

Modules:
    probe_experiment: Place cell analysis - correlating neuron activations with maze positions
    teleport_experiment: Activation patching to verify causal role of position-encoding neurons
    modal_runner: Cloud GPU runner using Modal

Usage:
    # Run probe experiment
    python -m experiments.interpretability.probe_experiment --maze-size medium

    # Run teleport experiment
    python -m experiments.interpretability.teleport_experiment --patch-type goal

    # Run on Modal cloud GPU
    modal run experiments/interpretability/modal_runner.py::run_probe_experiment
"""

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Default output directory
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
