"""
Vector Steering Experiment for CTM Position Encoding

This experiment provides CAUSAL evidence that position is encoded in the
synchronization representation by using linear probe coefficients as steering vectors.

Key insight: The trained probe Position ≈ W · S_out + b means W encodes position directions.
By adding α · W[x] to S_out, we can make the model "think" it's at a different position.

Expected result: Adding the "rightward" vector should cause early right turns / crashes.
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

app = modal.App("ctm-vector-steering")

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install([
        "torch>=2.0.0",
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/outputs": output_volume, "/data": data_volume},
)
def run_vector_steering(
    num_samples: int = 100,
    alpha_values: list = None,
    intervention_ticks: list = None,
    maze_size: str = "large",
    checkpoint_path: str = None,
    data_root: str = None,
    output_dir: str = None,
):
    """
    Run vector steering experiment using probe coefficients as steering vectors.

    Steps:
    1. Collect activations to train position probe
    2. Extract steering vectors from probe coefficients
    3. Apply steering during inference and measure behavioral changes
    """
    import sys
    sys.path.insert(0, '/app')

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from tqdm import tqdm
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import os
    from typing import Callable, Optional

    console = Console(force_terminal=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults
    if alpha_values is None:
        alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    if intervention_ticks is None:
        intervention_ticks = [5, 10, 15, 20]
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"
    if not output_dir:
        output_dir = "/outputs/vector_steering"

    os.makedirs(output_dir, exist_ok=True)

    console.print(Panel(
        "[bold cyan]Vector Steering Experiment[/]\n"
        "Using probe coefficients as steering vectors for causal intervention",
        border_style="blue"
    ))

    # =========================================================================
    # Step 1: Load model and data
    # =========================================================================
    console.print("\n[bold blue]Step 1:[/] Loading model and data...")

    from models.ctm import ContinuousThoughtMachine as CTM
    from data.custom_datasets import MazeImageFolder

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
        trunc=False,
        expand_range=True,
    )

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    console.print(f"  [green]✓[/] Dataset loaded: {len(dataset)} samples")

    MAZE_SIZE_PIXELS = {"medium": 39, "large": 59, "small": 19}
    maze_pixels = MAZE_SIZE_PIXELS.get(maze_size, 59)

    # =========================================================================
    # Step 2: Collect data and train position probe
    # =========================================================================
    console.print("\n[bold blue]Step 2:[/] Collecting activations and training position probe...")

    all_synch_out = []
    all_positions = []
    all_ticks = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Collecting activations")):
            images = images.to(device)

            # Run model with tracking
            predictions, certainties, synch_tuple, pre_act, post_act, attn = model(images, track=True)
            synch_out = synch_tuple[0]  # Shape: (iterations, batch, synch_size)

            # Get ground truth path
            directions = np.array(targets[0]) if isinstance(targets, (list, tuple)) else targets.cpu().numpy().flatten()

            dx = [0, 0, -1, 1]  # Up, Down, Left, Right
            dy = [-1, 1, 0, 0]

            path_length = min(len(directions), iterations)
            curr_x, curr_y = 1, 1

            for t in range(path_length):
                S_out_raw = synch_out[t, 0]
                if hasattr(S_out_raw, 'cpu'):
                    S_out = S_out_raw.cpu().numpy()
                else:
                    S_out = np.array(S_out_raw)

                all_synch_out.append(S_out)
                all_positions.append([curr_x / maze_pixels, curr_y / maze_pixels])
                all_ticks.append(t)

                d = int(directions[t])
                if d < 4:
                    curr_x += dx[d] * 2
                    curr_y += dy[d] * 2

    all_synch_out = np.array(all_synch_out)
    all_positions = np.array(all_positions)
    all_ticks = np.array(all_ticks)

    console.print(f"  [green]✓[/] Collected {len(all_synch_out)} samples")
    console.print(f"      synch_out shape: {all_synch_out.shape}")

    # Train probe
    X_train, X_test, y_train, y_test = train_test_split(
        all_synch_out, all_positions, test_size=0.2, random_state=42
    )

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    console.print(f"  [green]✓[/] Probe trained: R² = {r2:.4f}")

    # =========================================================================
    # Step 3: Extract steering vectors from probe coefficients
    # =========================================================================
    console.print("\n[bold blue]Step 3:[/] Extracting steering vectors from probe...")

    # probe.coef_ has shape (2, synch_size) for (x, y) outputs
    v_x = probe.coef_[0]  # "Eastward" direction (increasing x)
    v_y = probe.coef_[1]  # "Southward" direction (increasing y)

    # Normalize to unit vectors
    v_x_norm = v_x / np.linalg.norm(v_x)
    v_y_norm = v_y / np.linalg.norm(v_y)

    # Also create random control vector
    np.random.seed(123)
    v_random = np.random.randn(len(v_x))
    v_random = v_random / np.linalg.norm(v_random)

    console.print(f"  [green]✓[/] Steering vectors extracted")
    console.print(f"      v_x norm: {np.linalg.norm(v_x):.4f}")
    console.print(f"      v_y norm: {np.linalg.norm(v_y):.4f}")
    console.print(f"      v_x · v_y: {np.dot(v_x_norm, v_y_norm):.4f} (orthogonality)")

    # Convert to tensors
    v_x_tensor = torch.tensor(v_x_norm, dtype=torch.float32, device=device)
    v_y_tensor = torch.tensor(v_y_norm, dtype=torch.float32, device=device)
    v_random_tensor = torch.tensor(v_random, dtype=torch.float32, device=device)

    # =========================================================================
    # Step 4: Create modified model with synch_out intervention
    # =========================================================================
    console.print("\n[bold blue]Step 4:[/] Setting up intervention framework...")

    class ModifiedCTMWithSynchSteering(nn.Module):
        """CTM wrapper that allows steering the synch_out representation."""

        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.steering_vector = None
            self.steering_alpha = 0.0
            self.steering_ticks = None  # Which ticks to apply steering

        def set_steering(self, vector, alpha, ticks=None):
            """Set steering parameters."""
            self.steering_vector = vector
            self.steering_alpha = alpha
            self.steering_ticks = ticks

        def forward(self, x):
            """Forward pass with synch_out steering."""
            B = x.size(0)
            device = x.device
            m = self.base

            # Featurize input
            kv = m.compute_features(x)

            # Initialize state
            state_trace = m.start_trace.unsqueeze(0).expand(B, -1, -1)
            activated_state = m.start_activated_state.unsqueeze(0).expand(B, -1)

            predictions = torch.empty(B, m.out_dims, m.iterations, device=device, dtype=torch.float32)
            certainties = torch.empty(B, 2, m.iterations, device=device, dtype=torch.float32)

            decay_alpha_action, decay_beta_action = None, None
            m.decay_params_action.data = torch.clamp(m.decay_params_action, 0, 15)
            m.decay_params_out.data = torch.clamp(m.decay_params_out, 0, 15)
            r_action = torch.exp(-m.decay_params_action).unsqueeze(0).repeat(B, 1)
            r_out = torch.exp(-m.decay_params_out).unsqueeze(0).repeat(B, 1)

            _, decay_alpha_out, decay_beta_out = m.compute_synchronisation(
                activated_state, None, None, r_out, synch_type='out'
            )

            for stepi in range(m.iterations):
                # Synchronization for attention
                synchronisation_action, decay_alpha_action, decay_beta_action = m.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
                )

                # Attention
                q = m.q_proj(synchronisation_action).unsqueeze(1)
                attn_out, _ = m.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
                attn_out = attn_out.squeeze(1)
                pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

                # Synapse model
                state = m.synapses(pre_synapse_input)
                state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

                # NLM
                activated_state = m.trace_processor(state_trace)

                # Output synchronization
                synchronisation_out, decay_alpha_out, decay_beta_out = m.compute_synchronisation(
                    activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
                )

                # >>> STEERING INTERVENTION <<<
                if self.steering_vector is not None and self.steering_alpha != 0:
                    should_steer = (self.steering_ticks is None or stepi in self.steering_ticks)
                    if should_steer:
                        synchronisation_out = synchronisation_out + self.steering_alpha * self.steering_vector.unsqueeze(0)

                # Predictions (uses potentially steered synch_out)
                current_prediction = m.output_projector(synchronisation_out)
                current_certainty = m.compute_certainty(current_prediction)

                predictions[..., stepi] = current_prediction
                certainties[..., stepi] = current_certainty

            return predictions, certainties

    steered_model = ModifiedCTMWithSynchSteering(model)
    console.print(f"  [green]✓[/] Steering framework ready")

    # =========================================================================
    # Step 5: Run steering experiments
    # =========================================================================
    console.print("\n[bold blue]Step 5:[/] Running steering experiments...")

    # Direction mapping: 0=Up, 1=Down, 2=Left, 3=Right, 4=Wait
    direction_names = ['Up', 'Down', 'Left', 'Right', 'Wait']

    def get_direction_counts(predictions):
        """Get counts of each direction from model predictions."""
        # predictions shape: (B, out_dims, iterations)
        # For maze: out_dims = 50 = 10 steps × 5 directions
        pred_dirs = predictions.argmax(dim=1)  # Shape: (B, iterations)

        # Count per direction
        counts = {d: 0 for d in range(5)}
        for d in range(5):
            counts[d] = (pred_dirs == d).sum().item()
        return counts

    def compute_direction_bias(counts):
        """Compute directional bias metrics."""
        total = sum(counts.values())
        if total == 0:
            return {'lr_bias': 0, 'ud_bias': 0}

        # Left-Right bias: positive = more right
        lr_bias = (counts[3] - counts[2]) / total
        # Up-Down bias: positive = more down
        ud_bias = (counts[1] - counts[0]) / total

        return {'lr_bias': lr_bias, 'ud_bias': ud_bias}

    results = {
        'v_x': {},
        'v_y': {},
        'v_random': {}
    }

    steering_configs = [
        ('v_x', v_x_tensor, "X-direction (Eastward)"),
        ('v_y', v_y_tensor, "Y-direction (Southward)"),
        ('v_random', v_random_tensor, "Random (Control)")
    ]

    # Run baseline first
    console.print("  Running baseline (no steering)...")
    baseline_counts = {d: 0 for d in range(5)}

    with torch.no_grad():
        steered_model.set_steering(None, 0.0)
        for images, _ in tqdm(dataloader, desc="Baseline", leave=False):
            images = images.to(device)
            predictions, _ = steered_model(images)
            counts = get_direction_counts(predictions)
            for d in range(5):
                baseline_counts[d] += counts[d]

    baseline_bias = compute_direction_bias(baseline_counts)
    console.print(f"  Baseline: LR bias = {baseline_bias['lr_bias']:.4f}, UD bias = {baseline_bias['ud_bias']:.4f}")

    # Run steering experiments
    for vec_name, vec_tensor, vec_desc in steering_configs:
        console.print(f"\n  Testing {vec_desc}...")
        results[vec_name] = {'alphas': [], 'lr_bias': [], 'ud_bias': [], 'counts': []}

        for alpha in alpha_values:
            all_counts = {d: 0 for d in range(5)}

            with torch.no_grad():
                steered_model.set_steering(vec_tensor, alpha, ticks=set(intervention_ticks))

                for images, _ in tqdm(dataloader, desc=f"α={alpha:.1f}", leave=False):
                    images = images.to(device)
                    predictions, _ = steered_model(images)
                    counts = get_direction_counts(predictions)
                    for d in range(5):
                        all_counts[d] += counts[d]

            bias = compute_direction_bias(all_counts)
            results[vec_name]['alphas'].append(alpha)
            results[vec_name]['lr_bias'].append(bias['lr_bias'])
            results[vec_name]['ud_bias'].append(bias['ud_bias'])
            results[vec_name]['counts'].append(all_counts)

            console.print(f"    α={alpha:+.1f}: LR={bias['lr_bias']:+.4f}, UD={bias['ud_bias']:+.4f}")

    # =========================================================================
    # Step 6: Create visualizations
    # =========================================================================
    console.print("\n[bold blue]Step 6:[/] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vector Steering: Causal Test of Position Encoding', fontsize=14, fontweight='bold')

    # Plot 1: LR bias vs alpha for X-direction steering
    ax = axes[0, 0]
    ax.plot(results['v_x']['alphas'], results['v_x']['lr_bias'], 'o-', color='#e74c3c',
            linewidth=2, markersize=8, label='X-direction steering')
    ax.plot(results['v_random']['alphas'], results['v_random']['lr_bias'], 's--', color='gray',
            linewidth=2, markersize=6, alpha=0.7, label='Random (control)')
    ax.axhline(y=baseline_bias['lr_bias'], color='black', linestyle=':', alpha=0.5, label='Baseline')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax.set_xlabel('Steering Strength (α)', fontsize=11)
    ax.set_ylabel('Left-Right Bias\n(positive = more Right)', fontsize=11)
    ax.set_title('X-Direction Steering → Left-Right Bias', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 2: UD bias vs alpha for Y-direction steering
    ax = axes[0, 1]
    ax.plot(results['v_y']['alphas'], results['v_y']['ud_bias'], 'o-', color='#3498db',
            linewidth=2, markersize=8, label='Y-direction steering')
    ax.plot(results['v_random']['alphas'], results['v_random']['ud_bias'], 's--', color='gray',
            linewidth=2, markersize=6, alpha=0.7, label='Random (control)')
    ax.axhline(y=baseline_bias['ud_bias'], color='black', linestyle=':', alpha=0.5, label='Baseline')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax.set_xlabel('Steering Strength (α)', fontsize=11)
    ax.set_ylabel('Up-Down Bias\n(positive = more Down)', fontsize=11)
    ax.set_title('Y-Direction Steering → Up-Down Bias', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 3: Cross-effect check (X steering shouldn't affect UD bias much)
    ax = axes[1, 0]
    ax.plot(results['v_x']['alphas'], results['v_x']['ud_bias'], 'o-', color='#e74c3c',
            linewidth=2, markersize=8, alpha=0.7, label='X steering → UD bias')
    ax.plot(results['v_y']['alphas'], results['v_y']['lr_bias'], 's-', color='#3498db',
            linewidth=2, markersize=8, alpha=0.7, label='Y steering → LR bias')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax.set_xlabel('Steering Strength (α)', fontsize=11)
    ax.set_ylabel('Cross-Axis Bias', fontsize=11)
    ax.set_title('Orthogonality Check\n(cross-effects should be small)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    # Compare effect at α=2.0 vs α=-2.0
    x_effect = results['v_x']['lr_bias'][-1] - results['v_x']['lr_bias'][0]  # α=2 - α=-2
    y_effect = results['v_y']['ud_bias'][-1] - results['v_y']['ud_bias'][0]
    rand_x_effect = results['v_random']['lr_bias'][-1] - results['v_random']['lr_bias'][0]
    rand_y_effect = results['v_random']['ud_bias'][-1] - results['v_random']['ud_bias'][0]

    categories = ['X→LR', 'Y→UD', 'Rand→LR', 'Rand→UD']
    effects = [x_effect, y_effect, rand_x_effect, rand_y_effect]
    colors = ['#e74c3c', '#3498db', 'gray', 'gray']

    bars = ax.bar(categories, effects, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel('Bias Change (α=+2 minus α=-2)', fontsize=11)
    ax.set_title('Steering Effect Size', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, effects):
        ax.annotate(f'{val:+.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'vector_steering_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    console.print(f"  [green]✓[/] Saved plot to {plot_path}")

    # =========================================================================
    # Step 7: Statistical analysis and summary
    # =========================================================================
    console.print("\n[bold blue]Step 7:[/] Statistical analysis...")

    from scipy import stats

    # Test if X-direction steering has significant effect on LR bias
    x_alphas = np.array(results['v_x']['alphas'])
    x_lr_bias = np.array(results['v_x']['lr_bias'])
    slope_x, intercept_x, r_value_x, p_value_x, _ = stats.linregress(x_alphas, x_lr_bias)

    y_alphas = np.array(results['v_y']['alphas'])
    y_ud_bias = np.array(results['v_y']['ud_bias'])
    slope_y, intercept_y, r_value_y, p_value_y, _ = stats.linregress(y_alphas, y_ud_bias)

    rand_alphas = np.array(results['v_random']['alphas'])
    rand_lr_bias = np.array(results['v_random']['lr_bias'])
    slope_rand, _, r_value_rand, p_value_rand, _ = stats.linregress(rand_alphas, rand_lr_bias)

    # Print summary table
    table = Table(title="Vector Steering Results", show_header=True, header_style="bold magenta")
    table.add_column("Steering", style="cyan")
    table.add_column("Target Bias", justify="center")
    table.add_column("Slope", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("p-value", justify="right", style="yellow")
    table.add_column("Significant?", justify="center")

    table.add_row("X → LR", "LR bias", f"{slope_x:.4f}", f"{r_value_x**2:.3f}",
                  f"{p_value_x:.2e}", "✓ Yes" if p_value_x < 0.05 else "✗ No")
    table.add_row("Y → UD", "UD bias", f"{slope_y:.4f}", f"{r_value_y**2:.3f}",
                  f"{p_value_y:.2e}", "✓ Yes" if p_value_y < 0.05 else "✗ No")
    table.add_row("Random → LR", "LR bias", f"{slope_rand:.4f}", f"{r_value_rand**2:.3f}",
                  f"{p_value_rand:.2e}", "✓ Yes" if p_value_rand < 0.05 else "✗ No")

    console.print(table)

    # Save results
    np.savez(
        os.path.join(output_dir, 'vector_steering_results.npz'),
        results=results,
        baseline_bias=baseline_bias,
        probe_r2=r2,
        v_x=v_x_norm,
        v_y=v_y_norm,
        v_random=v_random,
        stats={
            'x_slope': slope_x, 'x_r2': r_value_x**2, 'x_p': p_value_x,
            'y_slope': slope_y, 'y_r2': r_value_y**2, 'y_p': p_value_y,
            'rand_slope': slope_rand, 'rand_r2': r_value_rand**2, 'rand_p': p_value_rand,
        },
        alpha_values=alpha_values,
        intervention_ticks=intervention_ticks,
    )
    console.print(f"\n  [green]✓[/] Results saved to {output_dir}/vector_steering_results.npz")

    # Key findings
    if p_value_x < 0.05 and abs(slope_x) > abs(slope_rand):
        console.print(Panel(
            f"[bold green]KEY FINDING: Vector Steering Works![/]\n\n"
            f"X-direction steering significantly affects Left-Right bias:\n"
            f"  Slope: {slope_x:.4f} (p = {p_value_x:.2e})\n"
            f"  Random control slope: {slope_rand:.4f}\n\n"
            f"This provides CAUSAL evidence that the position representation\n"
            f"is functionally used by the model for navigation.",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]Result: Steering effect not significant[/]\n\n"
            f"X-direction slope: {slope_x:.4f} (p = {p_value_x:.2e})\n"
            f"Random slope: {slope_rand:.4f}\n\n"
            f"The position representation may be read but not causally used,\n"
            f"or the steering magnitude may need adjustment.",
            border_style="yellow"
        ))

    console.print("\n[bold green]EXPERIMENT COMPLETE[/]")

    return {
        'status': 'success',
        'probe_r2': r2,
        'x_effect': {'slope': slope_x, 'p': p_value_x, 'r2': r_value_x**2},
        'y_effect': {'slope': slope_y, 'p': p_value_y, 'r2': r_value_y**2},
        'control_effect': {'slope': slope_rand, 'p': p_value_rand, 'r2': r_value_rand**2},
    }


@app.local_entrypoint()
def main(
    num_samples: int = 100,
    maze_size: str = "large",
):
    """Run the vector steering experiment."""
    result = run_vector_steering.remote(
        num_samples=num_samples,
        maze_size=maze_size,
    )
    print(f"\nResult: {result}")
