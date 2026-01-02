"""
Probe Experiment: Mechanistic Interpretability of CTM on Maze Navigation

This script implements the "Probe" experiment to investigate whether CTM neurons
develop "Place Cell" behavior - firing patterns correlated with specific (x,y)
coordinates in the maze.

Hypothesis: Since CTM solves mazes without positional embeddings, the model must
construct a "Virtual Coordinate System" dynamically within its internal states.

Key Questions:
1. Do specific neurons correlate with maze positions?
2. Does the synchronization matrix encode spatial information?
3. How does positional information emerge across internal ticks?

Author: Research Engineer (Mechanistic Interpretability)
Date: 2026-01-01
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ctm import ContinuousThoughtMachine
from data.custom_datasets import MazeImageFolder

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with `pip install wandb` for experiment tracking.")


@dataclass
class ProbeConfig:
    """Configuration for the probe experiment."""
    # Model settings
    checkpoint_path: str = "checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data settings
    data_root: str = "data/mazes"
    maze_size: str = "medium"  # "small", "medium", or "large"
    maze_route_length: int = 100
    num_samples: int = 100  # Number of mazes to analyze
    batch_size: int = 16

    # Analysis settings
    num_top_neurons: int = 50  # Top neurons to analyze for place cell behavior
    num_ticks_to_analyze: Optional[int] = None  # None = all ticks

    # Output settings
    output_dir: str = "experiments/interpretability/outputs"
    save_raw_data: bool = True

    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "ctm-interpretability"
    wandb_run_name: Optional[str] = None


def parse_args() -> ProbeConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CTM Place Cell Probe Experiment")

    parser.add_argument("--checkpoint", type=str, default=ProbeConfig.checkpoint_path,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=ProbeConfig.device,
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--data-root", type=str, default=ProbeConfig.data_root,
                        help="Root directory for maze data")
    parser.add_argument("--maze-size", type=str, default=ProbeConfig.maze_size,
                        choices=["small", "medium", "large"],
                        help="Size of mazes to analyze")
    parser.add_argument("--num-samples", type=int, default=ProbeConfig.num_samples,
                        help="Number of mazes to analyze")
    parser.add_argument("--batch-size", type=int, default=ProbeConfig.batch_size,
                        help="Batch size for inference")
    parser.add_argument("--num-top-neurons", type=int, default=ProbeConfig.num_top_neurons,
                        help="Number of top neurons to analyze")
    parser.add_argument("--output-dir", type=str, default=ProbeConfig.output_dir,
                        help="Output directory for results")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=ProbeConfig.wandb_project,
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name")

    args = parser.parse_args()

    config = ProbeConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        data_root=args.data_root,
        maze_size=args.maze_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_top_neurons=args.num_top_neurons,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb and WANDB_AVAILABLE,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

    return config


def load_model(config: ProbeConfig) -> ContinuousThoughtMachine:
    """
    Load the CTM model from checkpoint.

    The model architecture for maze solving:
    - Backbone: ResNet-based feature extractor
    - No positional embeddings (key for our hypothesis!)
    - Synapse model: U-Net style MLP for cross-neuron communication
    - NLMs: Private MLPs per neuron processing activation history
    - Output: Synchronization-based predictions

    Returns:
        ContinuousThoughtMachine: Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {config.checkpoint_path}")

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {config.checkpoint_path}\n"
            f"Download from: https://drive.google.com/drive/folders/1vSg8T7FqP-guMDk1LU7_jZaQtXFP9sZg"
        )

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
    model_args = checkpoint['args']

    # Handle legacy arguments
    if not hasattr(model_args, 'backbone_type') and hasattr(model_args, 'resnet_type'):
        model_args.backbone_type = f'{model_args.resnet_type}-{getattr(model_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(model_args, 'neuron_select_type'):
        model_args.neuron_select_type = 'first-last'
    if not hasattr(model_args, 'n_random_pairing_self'):
        model_args.n_random_pairing_self = 0

    prediction_reshaper = [model_args.out_dims // 5, 5] if hasattr(model_args, 'out_dims') else None

    print(f"Model config:")
    print(f"  - d_model: {model_args.d_model}")
    print(f"  - iterations (ticks): {model_args.iterations}")
    print(f"  - memory_length: {model_args.memory_length}")
    print(f"  - positional_embedding_type: {model_args.positional_embedding_type}")

    model = ContinuousThoughtMachine(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.d_input,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=model_args.n_synch_action,
        synapse_depth=model_args.synapse_depth,
        memory_length=model_args.memory_length,
        deep_nlms=model_args.deep_memory,
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=model_args.do_normalisation,
        backbone_type=model_args.backbone_type,
        positional_embedding_type=model_args.positional_embedding_type,
        out_dims=model_args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=0,
        neuron_select_type=model_args.neuron_select_type,
        n_random_pairing_self=model_args.n_random_pairing_self,
    ).to(config.device)

    # Handle different checkpoint formats
    state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
    load_result = model.load_state_dict(checkpoint[state_dict_key], strict=False)
    print(f"Loaded state dict. Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}")

    model.eval()
    return model


def load_maze_data(config: ProbeConfig) -> torch.utils.data.DataLoader:
    """
    Load maze dataset for analysis.

    Dataset structure:
    - Input: RGB maze image (H, W, 3) → tensor (3, H, W)
    - Target: Sequence of moves [0=Up, 1=Down, 2=Left, 3=Right, 4=Wait]
    - Maze encoding: Red=start, Green=goal, Black=walls, White=path
    """
    data_path = f"{config.data_root}/{config.maze_size}/test"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Maze data not found: {data_path}\n"
            f"Download from: https://drive.google.com/file/d/1cBgqhaUUtsrll8-o2VY42hPpyBcfFv86/view"
        )

    print(f"Loading maze data from: {data_path}")

    dataset = MazeImageFolder(
        root=data_path,
        which_set='test',
        maze_route_length=config.maze_route_length,
        expand_range=True,  # Scale to [-1, 1]
        trunc=True if config.num_samples < 1000 else False
    )

    # Limit to num_samples
    if len(dataset) > config.num_samples:
        indices = list(range(config.num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Loaded {len(dataset)} mazes")
    return loader


def get_maze_positions_from_solution(maze_image: np.ndarray, solution: np.ndarray) -> List[Tuple[int, int]]:
    """
    Trace the solution path through the maze to get (x, y) positions.

    Args:
        maze_image: RGB maze image of shape (H, W, 3) in range [0, 1]
        solution: Sequence of moves [0=Up, 1=Down, 2=Left, 3=Right, 4=Wait]

    Returns:
        List of (row, col) positions visited during solution
    """
    # Find start position (red pixel)
    start_coords = np.argwhere((maze_image == [1, 0, 0]).all(axis=2))
    if len(start_coords) == 0:
        # Try with normalized values
        start_coords = np.argwhere(
            (maze_image[:,:,0] > 0.9) &
            (maze_image[:,:,1] < 0.1) &
            (maze_image[:,:,2] < 0.1)
        )

    if len(start_coords) == 0:
        return []

    current_pos = list(start_coords[0])
    positions = [tuple(current_pos)]

    # Direction mappings: 0=Up, 1=Down, 2=Left, 3=Right, 4=Wait
    direction_delta = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0),   # Wait
    }

    for move in solution:
        if move == 4:  # Wait/Stop
            positions.append(tuple(current_pos))
            continue

        delta = direction_delta.get(move, (0, 0))
        current_pos[0] += delta[0]
        current_pos[1] += delta[1]
        positions.append(tuple(current_pos))

    return positions


@dataclass
class InternalState:
    """Container for CTM internal states at each tick."""
    pre_activations: np.ndarray   # Shape: (T, B, D) - pre-activation at each tick
    post_activations: np.ndarray  # Shape: (T, B, D) - post-activation (z_t) at each tick
    synch_out: np.ndarray         # Shape: (T, B, S) - output synchronization
    synch_action: np.ndarray      # Shape: (T, B, S) - action synchronization
    attention: np.ndarray         # Shape: (T, B, H, Hf, Wf) - attention over features
    predictions: np.ndarray       # Shape: (B, R*5, T) - raw predictions
    certainties: np.ndarray       # Shape: (B, 2, T) - certainty values


def collect_internal_states(
    model: ContinuousThoughtMachine,
    dataloader: torch.utils.data.DataLoader,
    config: ProbeConfig
) -> Tuple[List[InternalState], List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
    """
    Run model on maze data and collect internal states.

    This is the core data collection for the probe experiment.
    We track:
    1. Post-activations (z_t) - the key representation for each neuron
    2. Synchronization matrices - pairwise neuron correlations
    3. Predictions and maze positions - for ground truth mapping

    Returns:
        Tuple of:
        - states: List of InternalState per batch
        - mazes: List of maze images
        - solutions: List of ground truth solutions
        - positions: List of (x,y) position sequences
    """
    all_states = []
    all_mazes = []
    all_solutions = []
    all_positions = []

    print("Collecting internal states...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            # Run model with tracking enabled
            # Returns: predictions, certainties, (synch_out, synch_action), pre_act, post_act, attention
            predictions, certainties, synch_tracking, pre_activations, post_activations, attention = model(
                inputs, track=True
            )

            synch_out, synch_action = synch_tracking

            # Store internal states
            state = InternalState(
                pre_activations=pre_activations,      # Shape: (T, B, D)
                post_activations=post_activations,    # Shape: (T, B, D)
                synch_out=synch_out,                  # Shape: (T, B, S_out)
                synch_action=synch_action,            # Shape: (T, B, S_action)
                attention=attention,                  # Shape: (T, B, H, Hf, Wf)
                predictions=predictions.cpu().numpy(),
                certainties=certainties.cpu().numpy()
            )
            all_states.append(state)

            # Store maze images and solutions
            mazes_np = ((inputs.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)  # (B, H, W, 3)
            all_mazes.append(mazes_np)
            all_solutions.append(targets.cpu().numpy())

            # Compute positions for each maze in batch
            batch_positions = []
            for i in range(inputs.size(0)):
                positions = get_maze_positions_from_solution(mazes_np[i], targets[i].cpu().numpy())
                batch_positions.append(positions)
            all_positions.append(batch_positions)

    return all_states, all_mazes, all_solutions, all_positions


def analyze_place_cells(
    states: List[InternalState],
    positions: List[List[List[Tuple[int, int]]]],
    config: ProbeConfig
) -> Dict:
    """
    Analyze neuron activations for place cell behavior.

    For each neuron, we compute:
    1. Activation strength at each maze position
    2. Spatial selectivity (variance across positions vs. within positions)
    3. Place field maps (where does each neuron fire strongest?)

    A "Place Cell" neuron would show:
    - High activation at specific (x, y) locations
    - Low activation elsewhere
    - Consistent across different mazes

    Returns:
        Dict containing analysis results
    """
    print("Analyzing place cell behavior...")

    # Aggregate all post-activations and positions
    # post_activations shape per state: (T, B, D)
    all_activations = []  # Will be (N_samples, T, D)
    all_pos_flat = []     # Will be (N_samples, T, 2) - (x, y) at each tick

    for state_idx, state in enumerate(states):
        T, B, D = state.post_activations.shape
        batch_positions = positions[state_idx]

        for sample_idx in range(B):
            pos_list = batch_positions[sample_idx]

            # Get activations for this sample: (T, D)
            activations = state.post_activations[:, sample_idx, :]

            # Map ticks to positions
            # The model predicts a sequence of 100 moves over T ticks
            # We need to map each tick to the "imagined" position
            # Simplification: Use position at tick t (assuming linear mapping)
            for t in range(T):
                # Map tick to position index (model outputs 100 moves over T ticks)
                pos_idx = min(t * len(pos_list) // T, len(pos_list) - 1)
                if pos_idx < len(pos_list):
                    pos = pos_list[pos_idx]
                    all_activations.append(activations[t])
                    all_pos_flat.append(pos)

    all_activations = np.array(all_activations)  # (N, D)
    all_pos_flat = np.array(all_pos_flat)        # (N, 2)

    print(f"Collected {len(all_activations)} activation samples")
    print(f"Position range: x=[{all_pos_flat[:,0].min()}, {all_pos_flat[:,0].max()}], "
          f"y=[{all_pos_flat[:,1].min()}, {all_pos_flat[:,1].max()}]")

    # Compute spatial information for each neuron
    D = all_activations.shape[1]

    # Create position-to-activation mapping for each neuron
    position_activations = defaultdict(lambda: defaultdict(list))

    for i in range(len(all_activations)):
        pos = tuple(all_pos_flat[i])
        for neuron_idx in range(D):
            position_activations[neuron_idx][pos].append(all_activations[i, neuron_idx])

    # Compute spatial selectivity metrics
    neuron_metrics = {}

    for neuron_idx in tqdm(range(D), desc="Computing neuron metrics"):
        pos_means = {}
        pos_stds = {}

        for pos, acts in position_activations[neuron_idx].items():
            pos_means[pos] = np.mean(acts)
            pos_stds[pos] = np.std(acts) if len(acts) > 1 else 0

        if len(pos_means) > 0:
            mean_activations = list(pos_means.values())

            # Spatial variance (how much does mean activation vary across positions)
            spatial_variance = np.var(mean_activations)

            # Within-position variance (average variance within same position)
            within_variance = np.mean([s**2 for s in pos_stds.values()]) if pos_stds else 0

            # Place cell score: spatial_variance / (within_variance + epsilon)
            # High score = consistent activation at specific locations
            place_cell_score = spatial_variance / (within_variance + 1e-6)

            # Peak position (where does neuron fire strongest)
            peak_pos = max(pos_means.keys(), key=lambda p: pos_means[p])
            peak_activation = pos_means[peak_pos]

            neuron_metrics[neuron_idx] = {
                'spatial_variance': spatial_variance,
                'within_variance': within_variance,
                'place_cell_score': place_cell_score,
                'peak_position': peak_pos,
                'peak_activation': peak_activation,
                'position_means': pos_means
            }

    # Rank neurons by place cell score
    ranked_neurons = sorted(
        neuron_metrics.keys(),
        key=lambda n: neuron_metrics[n]['place_cell_score'],
        reverse=True
    )

    top_neurons = ranked_neurons[:config.num_top_neurons]

    print(f"\nTop {config.num_top_neurons} Place Cell Neurons:")
    for i, neuron_idx in enumerate(top_neurons[:10]):
        metrics = neuron_metrics[neuron_idx]
        print(f"  {i+1}. Neuron {neuron_idx}: "
              f"score={metrics['place_cell_score']:.4f}, "
              f"peak_pos={metrics['peak_position']}, "
              f"peak_act={metrics['peak_activation']:.4f}")

    return {
        'neuron_metrics': neuron_metrics,
        'ranked_neurons': ranked_neurons,
        'top_neurons': top_neurons,
        'all_activations': all_activations,
        'all_positions': all_pos_flat
    }


def create_place_field_plots(
    analysis_results: Dict,
    config: ProbeConfig,
    maze_size: int = 39
):
    """
    Create visualization of place fields for top neurons.

    For each top neuron, we create a heatmap showing:
    - X-axis: Maze column (x coordinate)
    - Y-axis: Maze row (y coordinate)
    - Color: Mean neuron activation at that position
    """
    print("Creating place field visualizations...")

    os.makedirs(config.output_dir, exist_ok=True)

    top_neurons = analysis_results['top_neurons']
    neuron_metrics = analysis_results['neuron_metrics']

    # Create grid for place fields
    n_cols = 5
    n_rows = (len(top_neurons) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for idx, neuron_idx in enumerate(top_neurons):
        ax = axes[idx]
        metrics = neuron_metrics[neuron_idx]

        # Create place field heatmap
        place_field = np.zeros((maze_size, maze_size))
        counts = np.zeros((maze_size, maze_size))

        for pos, mean_act in metrics['position_means'].items():
            row, col = pos
            if 0 <= row < maze_size and 0 <= col < maze_size:
                place_field[row, col] = mean_act
                counts[row, col] = 1

        # Mask positions with no data
        place_field = np.ma.masked_where(counts == 0, place_field)

        im = ax.imshow(place_field, cmap='hot', aspect='equal')
        ax.set_title(f"Neuron {neuron_idx}\nScore: {metrics['place_cell_score']:.2f}")
        ax.set_xlabel("Column (x)")
        ax.set_ylabel("Row (y)")
        plt.colorbar(im, ax=ax, label="Activation")

    # Hide unused subplots
    for idx in range(len(top_neurons), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(config.output_dir, "place_fields.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved place field plot to: {save_path}")

    # Log to wandb if enabled
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({"place_fields": wandb.Image(save_path)})

    return save_path


def create_neuron_position_correlation_plot(
    analysis_results: Dict,
    config: ProbeConfig
):
    """
    Create scatter plot of neuron activation vs. maze position.
    """
    print("Creating neuron-position correlation plots...")

    os.makedirs(config.output_dir, exist_ok=True)

    all_activations = analysis_results['all_activations']
    all_positions = analysis_results['all_positions']
    top_neurons = analysis_results['top_neurons'][:10]  # Top 10 for visualization

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, neuron_idx in enumerate(top_neurons):
        ax = axes[idx]

        activations = all_activations[:, neuron_idx]
        x_coords = all_positions[:, 1]  # Column
        y_coords = all_positions[:, 0]  # Row

        # Create 2D histogram
        scatter = ax.scatter(x_coords, y_coords, c=activations, cmap='viridis',
                           alpha=0.5, s=1)
        ax.set_xlabel("X (column)")
        ax.set_ylabel("Y (row)")
        ax.set_title(f"Neuron {neuron_idx}")
        plt.colorbar(scatter, ax=ax, label="Activation")

    plt.tight_layout()

    save_path = os.path.join(config.output_dir, "neuron_position_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot to: {save_path}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({"neuron_position_correlation": wandb.Image(save_path)})

    return save_path


def run_experiment(config: ProbeConfig):
    """
    Main experiment runner.
    """
    print("=" * 60)
    print("CTM Place Cell Probe Experiment")
    print("=" * 60)
    print(f"Config: {config}")
    print()

    # Initialize wandb if enabled
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"probe_{config.maze_size}",
            config=vars(config)
        )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    try:
        # Step 1: Load model
        print("\n[Step 1/4] Loading model...")
        model = load_model(config)

        # Step 2: Load data
        print("\n[Step 2/4] Loading maze data...")
        dataloader = load_maze_data(config)

        # Step 3: Collect internal states
        print("\n[Step 3/4] Collecting internal states...")
        states, mazes, solutions, positions = collect_internal_states(model, dataloader, config)

        # Step 4: Analyze place cell behavior
        print("\n[Step 4/4] Analyzing place cell behavior...")
        analysis_results = analyze_place_cells(states, positions, config)

        # Create visualizations
        print("\n[Visualization] Creating plots...")

        # Determine maze size from data
        maze_size = 39 if config.maze_size in ["small", "medium"] else 99

        place_field_path = create_place_field_plots(analysis_results, config, maze_size)
        correlation_path = create_neuron_position_correlation_plot(analysis_results, config)

        # Save results
        if config.save_raw_data:
            results_path = os.path.join(config.output_dir, "probe_results.npz")
            np.savez_compressed(
                results_path,
                top_neurons=analysis_results['top_neurons'],
                ranked_neurons=analysis_results['ranked_neurons'],
                all_positions=analysis_results['all_positions'],
                # Note: neuron_metrics contains dicts, need special handling
            )
            print(f"Saved results to: {results_path}")

        # Summary
        print("\n" + "=" * 60)
        print("Experiment Complete!")
        print("=" * 60)
        print(f"\nKey Findings:")
        print(f"  - Analyzed {len(analysis_results['ranked_neurons'])} neurons")
        print(f"  - Top place cell neuron: {analysis_results['top_neurons'][0]}")

        top_metrics = analysis_results['neuron_metrics'][analysis_results['top_neurons'][0]]
        print(f"  - Best place cell score: {top_metrics['place_cell_score']:.4f}")
        print(f"  - Peak firing position: {top_metrics['peak_position']}")
        print(f"\nOutputs saved to: {config.output_dir}")

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "top_place_cell_neuron": analysis_results['top_neurons'][0],
                "top_place_cell_score": top_metrics['place_cell_score'],
                "num_neurons_analyzed": len(analysis_results['ranked_neurons'])
            })
            wandb.finish()

        return analysis_results

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo run this experiment, you need to download:")
        print("1. Model checkpoints from: https://drive.google.com/drive/folders/1vSg8T7FqP-guMDk1LU7_jZaQtXFP9sZg")
        print("2. Maze data from: https://drive.google.com/file/d/1cBgqhaUUtsrll8-o2VY42hPpyBcfFv86/view")

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish(exit_code=1)

        return None


if __name__ == "__main__":
    config = parse_args()
    run_experiment(config)
