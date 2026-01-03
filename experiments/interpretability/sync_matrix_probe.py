"""
Synchronization Matrix Position Decoding Experiment

Compares position decoding accuracy from:
- Z_t: Individual neuron activations (2048 dims)
- S_t: Synchronization matrix (neuron correlations)

Hypothesis: Position may be encoded in neuron CORRELATIONS (S_t) rather than
individual neuron activations (Z_t), explaining why "place cell" scores are weak.

Usage:
    # Local (quick test)
    python experiments/interpretability/sync_matrix_probe.py --num-samples 20 --local

    # Modal (full experiment)
    modal run experiments/interpretability/sync_matrix_probe.py --num-samples 100
"""

import argparse
from pathlib import Path

# For Modal
import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Modal app setup
app = modal.App("ctm-sync-matrix-probe")

ctm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "Pillow",
        "scikit-learn",
        "scipy",
        "rich",
        "huggingface_hub",
        "safetensors",
        "datasets",
    ])
    .add_local_dir(PROJECT_ROOT / "models", remote_path="/app/models")
    .add_local_dir(PROJECT_ROOT / "utils", remote_path="/app/utils")
    .add_local_file(PROJECT_ROOT / "data" / "custom_datasets.py", remote_path="/app/data/custom_datasets.py")
)

output_volume = modal.Volume.from_name("ctm-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("ctm-data", create_if_missing=True)


def run_sync_matrix_experiment(
    num_samples: int = 100,
    maze_size: str = "medium",
    checkpoint_path: str = "",
    data_root: str = "",
    output_dir: str = "",
    device: str = "cuda",
):
    """
    Run the synchronization matrix position decoding experiment.

    Compares linear probes trained on:
    - Z_t: Neuron activations at tick t
    - S_t: Synchronization matrix (correlations) at tick t

    Returns:
        dict with R² scores and analysis results
    """
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from tqdm import tqdm
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import os

    console = Console(force_terminal=True)

    # Default paths - use large checkpoint since that's what we have
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"
    if not output_dir:
        output_dir = "/outputs/sync_matrix"

    os.makedirs(output_dir, exist_ok=True)

    console.print(Panel(
        f"[bold]Synchronization Matrix Position Decoding[/]\n"
        f"Samples: {num_samples} | Maze: {maze_size} | Device: {device}",
        border_style="blue"
    ))

    # =========================================================================
    # Step 1: Load model and data
    # =========================================================================
    console.print("\n[bold blue]Step 1:[/] Loading model and data...")

    from models.ctm import ContinuousThoughtMachine as CTM
    from data.custom_datasets import MazeImageFolder

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']

    # Handle legacy arguments
    if not hasattr(model_args, 'backbone_type') and hasattr(model_args, 'resnet_type'):
        model_args.backbone_type = f'{model_args.resnet_type}-{getattr(model_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(model_args, 'neuron_select_type'):
        model_args.neuron_select_type = 'first-last'
    if not hasattr(model_args, 'n_random_pairing_self'):
        model_args.n_random_pairing_self = 0

    prediction_reshaper = [model_args.out_dims // 5, 5] if hasattr(model_args, 'out_dims') else None

    # Initialize model with args from checkpoint
    model = CTM(
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
    ).to(device)

    # Load state dict
    state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_dict_key], strict=False)
    model.eval()

    hidden_size = model_args.d_model
    iterations = model_args.iterations
    console.print(f"  [green]✓[/] Model loaded: {hidden_size} neurons, {iterations} iterations")

    # Load dataset
    data_path = f"{data_root}/{maze_size}/test"
    dataset = MazeImageFolder(
        root=data_path,
        which_set='test',
        maze_route_length=100,
        expand_range=True,
        trunc=True
    )

    # Random subset
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    console.print(f"  [green]✓[/] Dataset loaded: {len(dataset)} samples")

    # =========================================================================
    # Step 2: Collect Z_t and S_out (model's synch_out) - SYNCHRONIZED WITH POSITION
    # =========================================================================
    console.print("\n[bold blue]Step 2:[/] Collecting activations and synch_out (time-aligned)...")

    all_Z = []  # (N, D) - neuron activations
    all_S_out = []  # (N, D_out) - model's actual synchronization output
    all_positions = []  # (N, 2) - x, y positions
    all_ticks = []  # (N,) - which tick

    MAZE_SIZE_PIXELS = {"medium": 39, "large": 59, "small": 19}
    maze_pixels = MAZE_SIZE_PIXELS.get(maze_size, 39)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Processing mazes")):
            images = images.to(device)

            # Run model with tracking
            predictions, certainties, synch_tuple, pre_act, post_act, attn = model(images, track=True)

            # synch_tuple is (synch_out_tracking, synch_action_tracking) - extract synch_out
            synch_out = synch_tuple[0]  # Shape: (iterations, batch, synch_representation_size_out)

            # post_act shape: (iterations, batch, hidden_size)
            # synch_out shape: (iterations, batch, synch_representation_size_out)

            # Get ground truth directions from the solution path
            # targets is a list of direction indices: 0=up, 1=down, 2=left, 3=right
            directions = np.array(targets[0]) if isinstance(targets, (list, tuple)) else targets.cpu().numpy().flatten()

            # Map directions to position deltas: 0=Up, 1=Down, 2=Left, 3=Right
            dx = [0, 0, -1, 1]  # Up, Down, Left, Right
            dy = [-1, 1, 0, 0]  # Up, Down, Left, Right

            # CRITICAL FIX: Sync ticks with path steps!
            # Model output at tick t corresponds to the decision made for step t
            path_length = min(len(directions), iterations)
            curr_x, curr_y = 1, 1  # Start position

            for t in range(path_length):
                # Get model state at THIS specific tick
                # post_act is [Iterations, Batch, D] -> We want [t, 0, :]
                Z_t_raw = post_act[t, 0]
                if hasattr(Z_t_raw, 'cpu'):
                    Z_t = Z_t_raw.cpu().numpy()
                else:
                    Z_t = np.array(Z_t_raw)

                # Get synch_out at THIS specific tick (the model's actual S_t representation)
                # synch_out is [Iterations, Batch, D_out] -> We want [t, 0, :]
                S_out_raw = synch_out[t, 0]
                if hasattr(S_out_raw, 'cpu'):
                    S_out = S_out_raw.cpu().numpy()
                else:
                    S_out = np.array(S_out_raw)

                # Store data for THIS step
                all_Z.append(Z_t)
                all_S_out.append(S_out)
                all_positions.append([curr_x / maze_pixels, curr_y / maze_pixels])
                all_ticks.append(t)

                # Move to next position for the NEXT tick
                d = int(directions[t])
                if d < 4:  # If not 'Wait'
                    curr_x += dx[d] * 2  # Maze has wall between each cell
                    curr_y += dy[d] * 2

    # Convert to arrays
    all_Z = np.array(all_Z)
    all_S_out = np.array(all_S_out)
    all_positions = np.array(all_positions)
    all_ticks = np.array(all_ticks)

    console.print(f"  [green]✓[/] Collected {len(all_Z)} samples (tick-aligned)")
    console.print(f"      Z_t shape: {all_Z.shape} (neuron activations)")
    console.print(f"      S_out shape: {all_S_out.shape} (synch_out - model's learned representation)")

    # =========================================================================
    # Step 3: Train linear probes for position decoding
    # =========================================================================
    console.print("\n[bold blue]Step 3:[/] Training position decoders...")

    # Split data
    X_Z = all_Z  # Raw neuron activations
    X_S_out = all_S_out  # Model's synch_out (learned synchronization representation)
    y = all_positions

    # Train/test split
    (X_Z_train, X_Z_test,
     X_S_out_train, X_S_out_test,
     y_train, y_test) = train_test_split(
        X_Z, X_S_out, y,
        test_size=0.2, random_state=42
    )

    results = {}

    # Probe 1: Z_t -> position (raw neuron activations)
    console.print("  Training Z_t decoder (neuron activations)...")
    probe_Z = Ridge(alpha=1.0)
    probe_Z.fit(X_Z_train, y_train)
    y_pred_Z = probe_Z.predict(X_Z_test)
    r2_Z = r2_score(y_test, y_pred_Z)
    results['Z_t'] = {'r2': r2_Z, 'dims': X_Z.shape[1]}
    console.print(f"    R² = {r2_Z:.4f} (dims: {X_Z.shape[1]})")

    # Probe 2: S_out -> position (model's learned synchronization representation)
    console.print("  Training S_out decoder (model's synch_out)...")
    probe_S_out = Ridge(alpha=1.0)
    probe_S_out.fit(X_S_out_train, y_train)
    y_pred_S_out = probe_S_out.predict(X_S_out_test)
    r2_S_out = r2_score(y_test, y_pred_S_out)
    results['S_out'] = {'r2': r2_S_out, 'dims': X_S_out.shape[1]}
    console.print(f"    R² = {r2_S_out:.4f} (dims: {X_S_out.shape[1]})")

    # Probe 3: Combined Z_t + S_out -> position
    console.print("  Training combined decoder (Z_t + S_out)...")
    X_combined = np.concatenate([X_Z, X_S_out], axis=1)
    X_combined_train = np.concatenate([X_Z_train, X_S_out_train], axis=1)
    X_combined_test = np.concatenate([X_Z_test, X_S_out_test], axis=1)
    probe_combined = Ridge(alpha=1.0)
    probe_combined.fit(X_combined_train, y_train)
    y_pred_combined = probe_combined.predict(X_combined_test)
    r2_combined = r2_score(y_test, y_pred_combined)
    results['combined'] = {'r2': r2_combined, 'dims': X_combined.shape[1]}
    console.print(f"    R² = {r2_combined:.4f} (dims: {X_combined.shape[1]})")

    # =========================================================================
    # Step 4: Analyze by tick
    # =========================================================================
    console.print("\n[bold blue]Step 4:[/] Analyzing decoding accuracy by tick...")

    tick_results = {}
    unique_ticks = np.unique(all_ticks)

    for tick in unique_ticks:
        mask = all_ticks == tick
        X_tick = all_Z[mask]
        y_tick = all_positions[mask]

        if len(X_tick) > 50:  # Need enough samples
            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                X_tick, y_tick, test_size=0.2, random_state=42
            )
            probe = Ridge(alpha=1.0)
            probe.fit(X_train_t, y_train_t)
            y_pred = probe.predict(X_test_t)
            r2 = r2_score(y_test_t, y_pred)
            tick_results[int(tick)] = r2
            console.print(f"    Tick {tick}: R² = {r2:.4f}")

    # =========================================================================
    # Step 5: Create visualizations
    # =========================================================================
    console.print("\n[bold blue]Step 5:[/] Creating visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: R² comparison bar chart
    ax = axes[0]
    methods = ['Z_t\n(activations)', 'S_out\n(synch_out)', 'Combined\n(Z_t + S_out)']
    r2_values = [results['Z_t']['r2'], results['S_out']['r2'], results['combined']['r2']]
    colors = ['#3498db', '#e74c3c', '#9b59b6']

    bars = ax.bar(methods, r2_values, color=colors, alpha=0.8)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Position Decoding Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, r2_values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    # Plot 2: Decoding by tick
    ax = axes[1]
    if tick_results:
        ticks = sorted(tick_results.keys())
        r2_by_tick = [tick_results[t] for t in ticks]
        ax.plot(ticks, r2_by_tick, 'o-', linewidth=2, markersize=4, color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Tick', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Z_t Decoding Across Ticks', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Plot 3: Predicted vs Actual positions
    ax = axes[2]
    ax.scatter(y_test[:, 0], y_pred_Z[:, 0], alpha=0.3, s=10, c='blue', label='X (Z_t)')
    ax.scatter(y_test[:, 1], y_pred_Z[:, 1], alpha=0.3, s=10, c='orange', label='Y (Z_t)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Actual Position (normalized)', fontsize=12)
    ax.set_ylabel('Predicted Position', fontsize=12)
    ax.set_title(f'Z_t Predictions (R²={r2_Z:.3f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sync_matrix_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    console.print(f"  [green]✓[/] Saved plot to {plot_path}")

    # =========================================================================
    # Step 6: Print summary
    # =========================================================================
    table = Table(title="Position Decoding Results (Tick-Aligned)", show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("Dimensions", justify="right")
    table.add_column("R² Score", justify="right", style="yellow")
    table.add_column("Interpretation", style="green")

    best_method = max(results.keys(), key=lambda k: results[k]['r2'])

    for method, data in results.items():
        is_best = "★ BEST" if method == best_method else ""
        table.add_row(method, str(data['dims']), f"{data['r2']:.4f}", is_best)

    console.print()
    console.print(table)

    # Save results
    np.savez(
        os.path.join(output_dir, 'sync_matrix_results.npz'),
        results=results,
        tick_results=tick_results,
        r2_Z=r2_Z,
        r2_S_out=r2_S_out,
        r2_combined=r2_combined,
    )
    console.print(f"\n  [green]✓[/] Saved results to {output_dir}/sync_matrix_results.npz")

    # Key insight
    if r2_S_out > r2_Z:
        console.print(Panel(
            f"[bold green]KEY FINDING:[/] Synchronization (S_out) encodes position better than activations (Z_t)!\n"
            f"R² improvement: {r2_S_out - r2_Z:.3f}\n"
            f"This supports the hypothesis that CTM builds a 'Virtual Coordinate System' in correlation space.",
            border_style="green"
        ))
    elif r2_Z > r2_S_out:
        console.print(Panel(
            f"[bold yellow]FINDING:[/] Activations (Z_t) encode position better than synch_out (S_out).\n"
            f"R² difference: {r2_Z - r2_S_out:.3f}\n"
            f"Position may be encoded in individual neurons ('place cells').",
            border_style="yellow"
        ))

    return {
        'status': 'success',
        'results': results,
        'tick_results': tick_results,
        'output_dir': output_dir,
    }


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/outputs": output_volume, "/data": data_volume},
)
def run_on_modal(
    num_samples: int = 100,
    maze_size: str = "medium",
):
    """Run the sync matrix experiment on Modal."""
    result = run_sync_matrix_experiment(
        num_samples=num_samples,
        maze_size=maze_size,
        checkpoint_path="/data/checkpoints/mazes/ctm_mazesmedium_D=2048_T=50_M=25.pt" if maze_size == "medium"
                        else "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt",
        data_root="/data/mazes",
        output_dir="/outputs/sync_matrix",
        device="cuda",
    )

    # Commit volume
    output_volume.commit()

    return result


@app.local_entrypoint()
def main(
    num_samples: int = 100,
    maze_size: str = "medium",
    local: bool = False,
):
    """
    Run sync matrix position decoding experiment.

    Args:
        num_samples: Number of mazes to process
        maze_size: Size of mazes ('medium' or 'large')
        local: If True, run locally instead of on Modal
    """
    if local:
        print("Running locally...")
        result = run_sync_matrix_experiment(
            num_samples=num_samples,
            maze_size=maze_size,
            checkpoint_path=str(PROJECT_ROOT / "checkpoints" / "mazes" / f"ctm_mazes{maze_size}_D=2048_T={'50' if maze_size == 'medium' else '75'}_M=25.pt"),
            data_root=str(PROJECT_ROOT / "mazes"),
            output_dir=str(PROJECT_ROOT / "experiments" / "interpretability" / "outputs" / "sync_matrix"),
            device="cuda" if __import__('torch').cuda.is_available() else "cpu",
        )
    else:
        result = run_on_modal.remote(
            num_samples=num_samples,
            maze_size=maze_size,
        )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Output: {result.get('output_dir', 'N/A')}")
    for method, data in result.get('results', {}).items():
        print(f"  {method}: R² = {data['r2']:.4f}")
