"""
Robust Experiment Runner for CTM Place Cell Analysis

This module runs experiments with:
- Larger sample sizes (200+ mazes)
- Multiple random seeds for reproducibility
- Multiple maze sizes (medium, large)
- More control trials (20+ random neuron sets)
- Statistical analysis with confidence intervals
- Wandb terminal logging for live progress tracking

Usage (with --detach to prevent timeout on client disconnect):
    modal run --detach experiments/interpretability/robust/robust_runner.py --experiment all
    modal run --detach experiments/interpretability/robust/robust_runner.py --experiment probe_robust
    modal run --detach experiments/interpretability/robust/robust_runner.py --experiment control_robust
    modal run --detach experiments/interpretability/robust/robust_runner.py --experiment tick_sweep_robust
"""

from pathlib import Path
import modal

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def setup_wandb_with_logging(project: str, name: str, config: dict, tags: list = None):
    """
    Setup wandb with terminal logging enabled.
    Uses service mode for better streaming of logs to terminal.
    """
    import os
    import wandb

    # Force wandb to use service mode for better terminal logging
    os.environ["WANDB_CONSOLE"] = "wrap"  # Capture all stdout/stderr
    os.environ["WANDB_SILENT"] = "false"  # Show wandb status messages

    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags or [],
        settings=wandb.Settings(
            console="wrap",  # Wrap console output
            _disable_stats=False,
            _disable_meta=False,
        ),
        reinit=True,
    )

    # Log initial status
    print(f"\n{'='*60}")
    print(f"WANDB RUN INITIALIZED")
    print(f"  Project: {project}")
    print(f"  Run name: {name}")
    print(f"  Run URL: {run.url}")
    print(f"{'='*60}\n")

    return run


def log_progress(wandb_run, step: int, metrics: dict, prefix: str = ""):
    """Log metrics with progress info to both wandb and terminal."""
    import wandb

    # Add prefix to metric names
    prefixed_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

    # Log to wandb
    wandb.log(prefixed_metrics, step=step)

    # Also print to terminal for live tracking
    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items()])
    print(f"[Step {step}] {prefix}: {metrics_str}")


# ============================================================================
# Rich Terminal UI Helpers
# ============================================================================

def create_rich_console():
    """Create a rich console for beautiful terminal output."""
    from rich.console import Console
    return Console(force_terminal=True)


def print_experiment_header(console, title: str, config: dict):
    """Print a styled experiment header panel."""
    from rich.panel import Panel
    from rich.text import Text

    config_text = " | ".join([f"[cyan]{k}[/]: [yellow]{v}[/]" for k, v in config.items()])
    content = Text.from_markup(f"[bold white]{title}[/]\n{config_text}")

    panel = Panel(
        content,
        border_style="blue",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


def create_results_table(title: str, columns: list, rows: list):
    """Create a formatted results table."""
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style="bold magenta")

    for col in columns:
        table.add_column(col["name"], justify=col.get("justify", "left"), style=col.get("style", ""))

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    return table


def print_step_header(console, step_num: int, description: str):
    """Print a styled step header."""
    console.print(f"\n[bold blue]━━━ Step {step_num}: {description} ━━━[/]")


def print_seed_progress(console, seed_idx: int, total_seeds: int, seed: int):
    """Print seed progress info."""
    console.print(f"\n[bold green]▶ Seed {seed_idx+1}/{total_seeds}[/] [dim](seed={seed})[/]")


def print_metric(console, name: str, value, style: str = ""):
    """Print a single metric with optional styling."""
    if isinstance(value, float):
        value_str = f"{value:.4f}"
    else:
        value_str = str(value)

    if style:
        console.print(f"  {name}: [{style}]{value_str}[/]")
    else:
        console.print(f"  {name}: {value_str}")


def print_success_banner(console, message: str):
    """Print a success banner."""
    from rich.panel import Panel
    console.print()
    console.print(Panel(
        f"[bold green]✓ {message}[/]",
        border_style="green",
    ))


def print_statistics_summary(console, title: str, stats: dict):
    """Print a statistics summary panel."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(Panel(table, title=f"[bold]{title}[/]", border_style="blue"))


# Modal app setup
app = modal.App("ctm-robust-interpretability")

# Docker image with dependencies
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
        "wandb",
        "opencv-python-headless",
        "imageio",
        "huggingface_hub",
        "safetensors",
        "datasets",
        "scipy",  # For statistical tests
        "rich",   # For beautiful terminal output
    ])
    .add_local_dir(PROJECT_ROOT / "models", remote_path="/app/models")
    .add_local_dir(PROJECT_ROOT / "utils", remote_path="/app/utils")
    .add_local_file(PROJECT_ROOT / "data" / "custom_datasets.py", remote_path="/app/data/custom_datasets.py")
    .add_local_dir(PROJECT_ROOT / "experiments", remote_path="/app/experiments")
)

# Volumes
output_volume = modal.Volume.from_name("ctm-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("ctm-data", create_if_missing=True)


@app.function(
    image=ctm_image,
    timeout=7200,  # 2 hours for upload
    volumes={"/data": data_volume},
)
def upload_maze_data(maze_size: str = "large", local_path: str = ""):
    """
    Upload maze data to Modal volume.

    Usage:
        modal run robust_runner.py::upload_maze_data --maze-size large --local-path /path/to/mazes/large
    """
    import os
    import shutil

    if not local_path:
        print("ERROR: --local-path is required")
        print("Usage: modal run robust_runner.py::upload_maze_data --maze-size large --local-path /path/to/mazes/large")
        return {"status": "error", "message": "local_path required"}

    target_dir = f"/data/mazes/{maze_size}"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Uploading {local_path} to {target_dir}...")

    # Check what's in the local path
    if os.path.exists(local_path):
        for item in os.listdir(local_path):
            src = os.path.join(local_path, item)
            dst = os.path.join(target_dir, item)
            if os.path.isdir(src):
                print(f"  Copying directory: {item}")
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                print(f"  Copying file: {item}")
                shutil.copy2(src, dst)

    data_volume.commit()
    print(f"\nUpload complete! Contents of {target_dir}:")
    for item in os.listdir(target_dir):
        print(f"  {item}")

    return {"status": "success", "target_dir": target_dir}


@app.function(
    image=ctm_image,
    timeout=300,
    volumes={"/data": data_volume},
)
def list_volume_contents(path: str = "/data"):
    """List contents of the data volume."""
    import os

    def list_dir_recursive(path, prefix=""):
        items = []
        try:
            for item in sorted(os.listdir(path)):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    items.append(f"{prefix}{item}/")
                    items.extend(list_dir_recursive(full_path, prefix + "  "))
                else:
                    size = os.path.getsize(full_path)
                    items.append(f"{prefix}{item} ({size:,} bytes)")
        except PermissionError:
            items.append(f"{prefix}[Permission denied]")
        return items

    print(f"Contents of {path}:")
    for line in list_dir_recursive(path):
        print(line)

    return {"status": "success"}


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=28800,  # 8 hour timeout (increased for robust experiments)
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_robust_probe(
    maze_size: str = "medium",
    num_samples: int = 200,
    num_top_neurons: int = 50,
    seeds: str = "42,123,456,789,1234",
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Robust probe experiment with multiple seeds and larger sample size.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from scipy import stats
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

    # Import rich helpers
    from experiments.interpretability.robust.robust_runner import (
        create_rich_console, print_experiment_header, print_step_header,
        print_seed_progress, create_results_table, print_success_banner,
        print_statistics_summary
    )

    console = create_rich_console()

    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    # Print experiment header
    print_experiment_header(console, "CTM Robust Probe Experiment", {
        "Maze": maze_size,
        "Samples": num_samples,
        "Seeds": len(seed_list),
        "Top Neurons": num_top_neurons,
    })

    from experiments.interpretability.probe_experiment import (
        ProbeConfig, load_model, collect_internal_states, analyze_place_cells,
        create_place_field_plots
    )

    # Maze size in pixels for place field visualization
    MAZE_SIZE_PIXELS = {"medium": 39, "large": 59, "small": 19}
    from data.custom_datasets import MazeImageFolder

    wandb_run = None
    if use_wandb:
        from experiments.interpretability.robust.robust_runner import setup_wandb_with_logging
        wandb_run = setup_wandb_with_logging(
            project="ctm-interpretability",
            name=f"robust_probe_{maze_size}_{num_samples}samples",
            config={
                "experiment": "robust_probe",
                "maze_size": maze_size,
                "num_samples": num_samples,
                "seeds": seed_list,
                "num_top_neurons": num_top_neurons,
            },
            tags=["robust", "probe", maze_size],
        )

    # Load model once
    print_step_header(console, 1, "Loading model")
    with console.status("[bold green]Loading CTM model...[/]"):
        config = ProbeConfig(
            checkpoint_path=checkpoint_path,
            device="cuda",
            data_root=data_root,
            maze_size=maze_size,
            num_samples=num_samples,
        )
        model = load_model(config)
    console.print("  [green]✓[/] Model loaded successfully")

    # Aggregate results across seeds
    all_neuron_scores = {}  # neuron_id -> list of scores across seeds
    last_analysis = None  # Store last analysis for place field visualization

    print_step_header(console, 2, "Running probe analysis across seeds")

    # Create progress bar for seeds
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        seed_task = progress.add_task("Processing seeds", total=len(seed_list))

        for seed_idx, seed in enumerate(seed_list):
            progress.update(seed_task, description=f"[bold blue]Seed {seed_idx+1}/{len(seed_list)} (seed={seed})")

            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load data with this seed's shuffling
            data_path = f"{data_root}/{maze_size}/test"
            dataset = MazeImageFolder(
                root=data_path,
                which_set='test',
                maze_route_length=100,
                expand_range=True,
                trunc=True
            )

            # Random subset based on seed
            all_indices = list(range(len(dataset)))
            np.random.shuffle(all_indices)
            indices = all_indices[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )

            # Run probe - collect internal states and analyze place cells
            states, mazes, solutions, sample_positions = collect_internal_states(model, dataloader, config)
            analysis = analyze_place_cells(states, sample_positions, config)
            last_analysis = analysis  # Keep for place field visualization

            # Extract scores from analysis
            neuron_metrics = analysis['neuron_metrics']
            ranked_neurons = analysis['ranked_neurons']

            # Aggregate scores - neuron_metrics is a dict with neuron_id -> metrics
            for neuron_id, metrics in neuron_metrics.items():
                score = metrics['place_cell_score']
                if neuron_id not in all_neuron_scores:
                    all_neuron_scores[neuron_id] = []
                all_neuron_scores[neuron_id].append(score)

            # Get top 5 for logging
            top_5 = ranked_neurons[:5]
            top_5_scores = [neuron_metrics[n]['place_cell_score'] for n in top_5]

            # Log to wandb
            if use_wandb:
                import wandb
                wandb.log({
                    "seed": seed,
                    "seed_idx": seed_idx,
                    "top_neuron_id": int(top_5[0]),
                    "top_score": float(top_5_scores[0]),
                    "mean_top5_score": float(np.mean(top_5_scores)),
                    "progress": (seed_idx + 1) / len(seed_list),
                }, step=seed_idx)

            progress.advance(seed_task)

    # Compute statistics across seeds
    print_step_header(console, 3, "Computing cross-seed statistics")

    neuron_ids = list(all_neuron_scores.keys())
    mean_scores = np.array([np.mean(all_neuron_scores[n]) for n in neuron_ids])
    std_scores = np.array([np.std(all_neuron_scores[n]) for n in neuron_ids])

    # Rank by mean score
    ranked_indices = np.argsort(mean_scores)[::-1]

    # Create and display results table
    top_n_display = min(20, num_top_neurons, len(ranked_indices))
    table_rows = []
    for i in range(top_n_display):
        idx = ranked_indices[i]
        neuron_id = neuron_ids[idx]
        table_rows.append([
            i + 1,
            neuron_id,
            f"{mean_scores[idx]:.3f}",
            f"±{std_scores[idx]:.3f}",
        ])

    results_table = create_results_table(
        f"Top {top_n_display} Place Cell Neurons",
        [
            {"name": "Rank", "justify": "right", "style": "cyan"},
            {"name": "Neuron ID", "justify": "right", "style": "green"},
            {"name": "Score", "justify": "right", "style": "yellow"},
            {"name": "Std", "justify": "right", "style": "dim"},
        ],
        table_rows
    )
    console.print(results_table)

    # Print summary statistics
    print_statistics_summary(console, "Score Distribution", {
        "Total Neurons": len(mean_scores),
        "95th Percentile": np.percentile(mean_scores, 95),
        "Top 10 Mean": np.mean([mean_scores[i] for i in ranked_indices[:10]]),
        "Baseline (1.0)": "Random selectivity",
    })

    # Create visualizations
    print_step_header(console, 4, "Creating visualizations")
    output_dir = "/outputs/robust/probe"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Score distribution
    axes[0, 0].hist(mean_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=np.percentile(mean_scores, 95), color='red', linestyle='--',
                       label=f'95th percentile: {np.percentile(mean_scores, 95):.2f}')
    axes[0, 0].set_xlabel('Mean Spatial Selectivity Score', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title(f'Score Distribution (n={len(mean_scores)} neurons, {len(seed_list)} seeds)', fontsize=14)
    axes[0, 0].legend()

    # Plot 2: Top neurons with error bars
    top_n = 20
    top_indices = ranked_indices[:top_n]
    top_means = [mean_scores[i] for i in top_indices]
    top_stds = [std_scores[i] for i in top_indices]
    top_ids = [neuron_ids[i] for i in top_indices]

    x_pos = np.arange(top_n)
    axes[0, 1].bar(x_pos, top_means, yerr=top_stds, capsize=3, alpha=0.7, color='steelblue')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([str(n) for n in top_ids], rotation=45, ha='right')
    axes[0, 1].set_xlabel('Neuron ID', fontsize=12)
    axes[0, 1].set_ylabel('Mean Score ± Std', fontsize=12)
    axes[0, 1].set_title(f'Top {top_n} Place Cells (with cross-seed variance)', fontsize=14)

    # Plot 3: Score consistency (mean vs std)
    axes[1, 0].scatter(mean_scores, std_scores, alpha=0.5, s=10)
    # Highlight top neurons
    for i in ranked_indices[:10]:
        axes[1, 0].scatter(mean_scores[i], std_scores[i], color='red', s=50, zorder=5)
        axes[1, 0].annotate(str(neuron_ids[i]), (mean_scores[i], std_scores[i]), fontsize=8)
    axes[1, 0].set_xlabel('Mean Score', fontsize=12)
    axes[1, 0].set_ylabel('Std Score', fontsize=12)
    axes[1, 0].set_title('Score Consistency Across Seeds', fontsize=14)

    # Plot 4: Cumulative distribution
    sorted_means = np.sort(mean_scores)[::-1]
    axes[1, 1].plot(range(len(sorted_means)), sorted_means, 'b-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Score = 1.0 (baseline)')
    axes[1, 1].set_xlabel('Neuron Rank', fontsize=12)
    axes[1, 1].set_ylabel('Mean Score', fontsize=12)
    axes[1, 1].set_title('Ranked Score Distribution', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 200)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"robust_probe_{maze_size}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save data
    np.savez(
        os.path.join(output_dir, f"robust_probe_{maze_size}.npz"),
        neuron_ids=np.array(neuron_ids),
        mean_scores=mean_scores,
        std_scores=std_scores,
        ranked_indices=ranked_indices,
        all_scores=np.array([all_neuron_scores[n] for n in neuron_ids]),
        seeds=np.array(seed_list),
        config={
            "maze_size": maze_size,
            "num_samples": num_samples,
            "num_seeds": len(seed_list),
        }
    )

    console.print(f"  [green]✓[/] Saved plot to {save_path}")

    # Create place field visualizations
    print_step_header(console, 5, "Creating place field heatmaps")
    place_field_path = None
    if last_analysis is not None:
        # Update config output_dir for place field function
        config.output_dir = output_dir
        maze_pixels = MAZE_SIZE_PIXELS.get(maze_size, 39)
        try:
            place_field_path = create_place_field_plots(
                last_analysis,
                config,
                maze_size=maze_pixels
            )
            console.print(f"  [green]✓[/] Saved place field heatmaps to {place_field_path}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/] Could not create place fields: {e}")

    if use_wandb:
        log_data = {
            "top_neuron_1_id": int(neuron_ids[ranked_indices[0]]),
            "top_neuron_1_score": float(mean_scores[ranked_indices[0]]),
            "top_10_mean_score": float(np.mean([mean_scores[i] for i in ranked_indices[:10]])),
            "score_95th_percentile": float(np.percentile(mean_scores, 95)),
            "robust_probe_plot": wandb.Image(save_path),
        }
        if place_field_path:
            log_data["place_fields"] = wandb.Image(place_field_path)
        wandb.log(log_data)
        wandb.finish()

    output_volume.commit()

    print_success_banner(console, f"Probe experiment complete! Top neuron: {neuron_ids[ranked_indices[0]]} (score: {mean_scores[ranked_indices[0]]:.3f})")

    return {
        "status": "success",
        "top_neurons": [int(neuron_ids[i]) for i in ranked_indices[:num_top_neurons]],
        "top_scores": [float(mean_scores[i]) for i in ranked_indices[:num_top_neurons]],
        "output_dir": output_dir,
    }


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=43200,  # 12 hour timeout for extensive control experiment
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_robust_control(
    maze_size: str = "medium",
    num_samples: int = 200,
    num_random_trials: int = 20,
    seeds: str = "42,123,456",
    intervention_tick: int = 5,
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Robust control experiment with more random trials and multiple seeds.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from scipy import stats
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

    # Import rich helpers
    from experiments.interpretability.robust.robust_runner import (
        create_rich_console, print_experiment_header, print_step_header,
        create_results_table, print_success_banner, print_statistics_summary
    )

    console = create_rich_console()

    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    # Print experiment header
    print_experiment_header(console, "CTM Robust Control Experiment", {
        "Maze": maze_size,
        "Samples": num_samples,
        "Seeds": len(seed_list),
        "Random Trials": num_random_trials,
        "Intervention Tick": intervention_tick,
    })

    from experiments.interpretability.teleport_experiment import (
        TeleportConfig, load_model, identify_position_neurons,
        run_teleport_experiment, analyze_teleport_results,
    )
    from data.custom_datasets import MazeImageFolder

    wandb_run = None
    if use_wandb:
        from experiments.interpretability.robust.robust_runner import setup_wandb_with_logging
        wandb_run = setup_wandb_with_logging(
            project="ctm-interpretability",
            name=f"robust_control_{maze_size}_{num_samples}samples_{num_random_trials}trials",
            config={
                "experiment": "robust_control",
                "maze_size": maze_size,
                "num_samples": num_samples,
                "num_random_trials": num_random_trials,
                "seeds": seed_list,
                "intervention_tick": intervention_tick,
            },
            tags=["robust", "control", maze_size],
        )

    # Load model once
    print_step_header(console, 1, "Loading model")
    with console.status("[bold green]Loading CTM model...[/]"):
        config = TeleportConfig(
            checkpoint_path=checkpoint_path,
            device="cuda",
            data_root=data_root,
            maze_size=maze_size,
            num_samples=num_samples,
            intervention_tick=intervention_tick,
            patch_type="goal",
        )
        model = load_model(config)
    console.print("  [green]✓[/] Model loaded successfully")
    d_model = model.d_model

    # Results storage
    all_position_diffs = []
    all_random_diffs = []
    seed_results = {}

    print_step_header(console, 2, "Running control experiments")

    total_trials = len(seed_list) * (1 + num_random_trials)  # position + random trials per seed

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("Running experiments", total=total_trials)

        for seed_idx, seed in enumerate(seed_list):
            progress.update(main_task, description=f"[bold blue]Seed {seed_idx+1}/{len(seed_list)} - Position neurons")

            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load data
            data_path = f"{data_root}/{maze_size}/test"
            dataset = MazeImageFolder(
                root=data_path,
                which_set='test',
                maze_route_length=100,
                expand_range=True,
                trunc=True
            )

            # Random subset
            all_indices = list(range(len(dataset)))
            np.random.shuffle(all_indices)
            indices = all_indices[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )

            # Identify position neurons for this seed
            start_neurons, goal_neurons, _ = identify_position_neurons(model, dataloader, config)
            num_neurons_to_patch = len(start_neurons)

            # Run with position neurons
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            position_results = run_teleport_experiment(
                model, dataloader, start_neurons, goal_neurons, config
            )
            position_analysis = analyze_teleport_results(position_results, config)
            position_diff = position_analysis['avg_move_diff_ratio']
            all_position_diffs.append(position_diff)

            progress.advance(main_task)

            # Run random trials
            random_diffs_this_seed = []

            for trial in range(num_random_trials):
                progress.update(main_task, description=f"[bold blue]Seed {seed_idx+1}/{len(seed_list)} - Random trial {trial+1}/{num_random_trials}")

                # Select random neurons
                all_neurons = set(range(d_model))
                excluded = set(start_neurons) | set(goal_neurons)
                available = list(all_neurons - excluded)

                random_start = list(np.random.choice(available, size=num_neurons_to_patch, replace=False))
                random_goal = list(np.random.choice(
                    [n for n in available if n not in random_start],
                    size=num_neurons_to_patch, replace=False
                ))

                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0
                )

                random_results = run_teleport_experiment(
                    model, dataloader, random_start, random_goal, config
                )
                random_analysis = analyze_teleport_results(random_results, config)
                random_diff = random_analysis['avg_move_diff_ratio']

                random_diffs_this_seed.append(random_diff)
                all_random_diffs.append(random_diff)

                # Log each trial to wandb
                if use_wandb:
                    import wandb
                    global_step = seed_idx * num_random_trials + trial
                    wandb.log({
                        "seed": seed,
                        "trial": trial,
                        "random_diff": random_diff,
                        "position_diff": position_diff,
                        "running_random_mean": np.mean(random_diffs_this_seed),
                        "progress": (seed_idx * num_random_trials + trial + 1) / (len(seed_list) * num_random_trials),
                    }, step=global_step)

                progress.advance(main_task)

            seed_results[seed] = {
                'position_diff': position_diff,
                'random_diffs': random_diffs_this_seed,
                'random_mean': np.mean(random_diffs_this_seed),
                'random_std': np.std(random_diffs_this_seed),
            }

    # Compute aggregate statistics
    print_step_header(console, 3, "Computing aggregate statistics")

    position_mean = np.mean(all_position_diffs)
    position_std = np.std(all_position_diffs)
    random_mean = np.mean(all_random_diffs)
    random_std = np.std(all_random_diffs)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((position_std**2 + random_std**2) / 2)
    cohens_d = (position_mean - random_mean) / pooled_std if pooled_std > 0 else float('inf')

    # T-test
    t_stat, p_value = stats.ttest_ind(all_position_diffs, all_random_diffs)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(all_position_diffs, all_random_diffs, alternative='greater')

    # Display results table
    comparison_table = create_results_table(
        "Position vs Random Neurons Comparison",
        [
            {"name": "Metric", "style": "cyan"},
            {"name": "Position Neurons", "justify": "right", "style": "green"},
            {"name": "Random Neurons", "justify": "right", "style": "yellow"},
        ],
        [
            ["Mean", f"{position_mean:.4f} ({position_mean*100:.2f}%)", f"{random_mean:.4f} ({random_mean*100:.2f}%)"],
            ["Std", f"{position_std:.4f}", f"{random_std:.4f}"],
            ["N", str(len(all_position_diffs)), str(len(all_random_diffs))],
        ]
    )
    console.print(comparison_table)

    # Effect size display
    effect_color = "green" if cohens_d > 0.8 else "yellow" if cohens_d > 0.5 else "red"
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    print_statistics_summary(console, "Statistical Analysis", {
        "Cohen's d": f"{cohens_d:.3f}",
        "Effect Ratio": f"{position_mean/random_mean:.2f}x" if random_mean > 0 else "inf",
        "T-test p-value": f"{p_value:.2e} {sig_text}",
        "Mann-Whitney p": f"{u_pvalue:.2e}",
    })

    # Create visualizations
    print_step_header(console, 4, "Creating visualizations")
    output_dir = "/outputs/robust/control"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Distribution comparison
    axes[0, 0].hist(all_random_diffs, bins=30, alpha=0.7, label='Random neurons',
                    color='blue', edgecolor='black', density=True)
    for pos_diff in all_position_diffs:
        axes[0, 0].axvline(x=pos_diff, color='red', alpha=0.5, linewidth=2)
    axes[0, 0].axvline(x=position_mean, color='darkred', linewidth=3,
                       label=f'Position mean: {position_mean:.3f}')
    axes[0, 0].axvline(x=random_mean, color='darkblue', linewidth=3, linestyle='--',
                       label=f'Random mean: {random_mean:.3f}')
    axes[0, 0].set_xlabel('Move Difference Ratio', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title(f'Position vs Random Neurons\n(p < {p_value:.2e})', fontsize=14)
    axes[0, 0].legend()

    # Plot 2: Box plot by seed
    seed_data = []
    seed_labels = []
    for seed in seed_list:
        seed_data.append([seed_results[seed]['position_diff']])
        seed_labels.append(f'Pos\nS{seed}')
        seed_data.append(seed_results[seed]['random_diffs'])
        seed_labels.append(f'Rand\nS{seed}')

    bp = axes[0, 1].boxplot(seed_data, labels=seed_labels, patch_artist=True)
    colors = ['red', 'blue'] * len(seed_list)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0, 1].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[0, 1].set_title('Results by Seed', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Effect size visualization
    categories = ['Position\nNeurons', 'Random\nNeurons']
    means = [position_mean, random_mean]
    stds = [position_std, random_std]
    x_pos = [0, 1]

    bars = axes[1, 0].bar(x_pos, means, yerr=stds, capsize=10, color=['red', 'blue'],
                          alpha=0.7, width=0.5)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[1, 0].set_title(f"Effect Size: Cohen's d = {cohens_d:.2f}", fontsize=14)

    # Add significance annotation
    y_max = max(means) + max(stds) + 0.02
    axes[1, 0].plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
    axes[1, 0].plot([0, 0], [y_max-0.005, y_max], 'k-', linewidth=1)
    axes[1, 0].plot([1, 1], [y_max-0.005, y_max], 'k-', linewidth=1)
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    axes[1, 0].text(0.5, y_max + 0.005, sig_text, ha='center', fontsize=14)

    # Plot 4: Summary table
    axes[1, 1].axis('off')
    table_data = [
        ['Metric', 'Position', 'Random'],
        ['Mean', f'{position_mean:.4f}', f'{random_mean:.4f}'],
        ['Std', f'{position_std:.4f}', f'{random_std:.4f}'],
        ['N', str(len(all_position_diffs)), str(len(all_random_diffs))],
        ["Cohen's d", f'{cohens_d:.3f}', '-'],
        ['T-test p', f'{p_value:.2e}', '-'],
        ['Ratio', f'{position_mean/random_mean:.2f}x' if random_mean > 0 else 'inf', '-'],
    ]
    table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='center',
                              colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    axes[1, 1].set_title('Statistical Summary', fontsize=14, pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"robust_control_{maze_size}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save data
    np.savez(
        os.path.join(output_dir, f"robust_control_{maze_size}.npz"),
        position_diffs=np.array(all_position_diffs),
        random_diffs=np.array(all_random_diffs),
        position_mean=position_mean,
        position_std=position_std,
        random_mean=random_mean,
        random_std=random_std,
        cohens_d=cohens_d,
        t_stat=t_stat,
        p_value=p_value,
        seeds=np.array(seed_list),
        seed_results=seed_results,
    )

    console.print(f"  [green]✓[/] Saved plot to {save_path}")

    if use_wandb:
        wandb.log({
            "position_mean": position_mean,
            "random_mean": random_mean,
            "cohens_d": cohens_d,
            "p_value": p_value,
            "effect_ratio": position_mean / random_mean if random_mean > 0 else float('inf'),
            "robust_control_plot": wandb.Image(save_path),
        })
        wandb.finish()

    output_volume.commit()

    print_success_banner(console, f"Control experiment complete! Effect: {position_mean/random_mean:.2f}x (p={p_value:.2e})")

    return {
        "status": "success",
        "position_mean": float(position_mean),
        "random_mean": float(random_mean),
        "cohens_d": float(cohens_d),
        "p_value": float(p_value),
        "output_dir": output_dir,
    }


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=43200,  # 12 hours for extensive tick sweep
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_robust_tick_sweep(
    maze_size: str = "medium",
    num_samples: int = 200,
    seeds: str = "42,123,456",
    tick_values: str = "5,15,25,35,45,55,65,74",
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Robust tick sweep with multiple seeds and larger sample size.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from scipy import stats
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

    # Import rich helpers
    from experiments.interpretability.robust.robust_runner import (
        create_rich_console, print_experiment_header, print_step_header,
        create_results_table, print_success_banner, print_statistics_summary
    )

    console = create_rich_console()

    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    ticks = [int(t.strip()) for t in tick_values.split(",")]

    # Print experiment header
    print_experiment_header(console, "CTM Robust Tick Sweep Experiment", {
        "Maze": maze_size,
        "Samples": num_samples,
        "Seeds": len(seed_list),
        "Ticks": len(ticks),
    })

    from experiments.interpretability.teleport_experiment import (
        TeleportConfig, load_model, identify_position_neurons,
        run_teleport_experiment, analyze_teleport_results,
    )
    from data.custom_datasets import MazeImageFolder

    wandb_run = None
    if use_wandb:
        from experiments.interpretability.robust.robust_runner import setup_wandb_with_logging
        wandb_run = setup_wandb_with_logging(
            project="ctm-interpretability",
            name=f"robust_tick_sweep_{maze_size}_{num_samples}samples",
            config={
                "experiment": "robust_tick_sweep",
                "maze_size": maze_size,
                "num_samples": num_samples,
                "seeds": seed_list,
                "ticks": ticks,
            },
            tags=["robust", "tick_sweep", maze_size],
        )

    # Load model
    print_step_header(console, 1, "Loading model")
    with console.status("[bold green]Loading CTM model...[/]"):
        config = TeleportConfig(
            checkpoint_path=checkpoint_path,
            device="cuda",
            data_root=data_root,
            maze_size=maze_size,
            num_samples=num_samples,
            patch_type="goal",
        )
        model = load_model(config)
    console.print("  [green]✓[/] Model loaded successfully")

    # Results: tick -> list of diffs across seeds
    tick_results = {t: [] for t in ticks}

    print_step_header(console, 2, "Running tick sweep experiments")

    total_experiments = len(seed_list) * len(ticks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("Running experiments", total=total_experiments)

        for seed_idx, seed in enumerate(seed_list):
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load data
            data_path = f"{data_root}/{maze_size}/test"
            dataset = MazeImageFolder(
                root=data_path,
                which_set='test',
                maze_route_length=100,
                expand_range=True,
                trunc=True
            )

            all_indices = list(range(len(dataset)))
            np.random.shuffle(all_indices)
            indices = all_indices[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )

            # Identify neurons once per seed
            start_neurons, goal_neurons, _ = identify_position_neurons(model, dataloader, config)

            for tick in ticks:
                progress.update(main_task, description=f"[bold blue]Seed {seed_idx+1}/{len(seed_list)} - Tick {tick}")
                config.intervention_tick = tick

                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0
                )

                results = run_teleport_experiment(
                    model, dataloader, start_neurons, goal_neurons, config
                )
                analysis = analyze_teleport_results(results, config)
                move_diff = analysis['avg_move_diff_ratio']
                tick_results[tick].append(move_diff)

                # Log to wandb
                if use_wandb:
                    import wandb
                    tick_idx = ticks.index(tick)
                    global_step = seed_idx * len(ticks) + tick_idx
                    wandb.log({
                        "seed": seed,
                        "tick": tick,
                        "move_diff": move_diff,
                        "progress": (global_step + 1) / (len(seed_list) * len(ticks)),
                    }, step=global_step)

                progress.advance(main_task)

    # Compute statistics
    print_step_header(console, 3, "Computing statistics")
    tick_means = [np.mean(tick_results[t]) for t in ticks]
    tick_stds = [np.std(tick_results[t]) for t in ticks]
    tick_sems = [s / np.sqrt(len(seed_list)) for s in tick_stds]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(ticks, tick_means)

    # Display results table
    table_rows = [[t, f"{tick_means[i]:.4f}", f"±{tick_stds[i]:.4f}"] for i, t in enumerate(ticks)]
    results_table = create_results_table(
        "Tick Sweep Results",
        [
            {"name": "Tick", "justify": "right", "style": "cyan"},
            {"name": "Mean Diff", "justify": "right", "style": "yellow"},
            {"name": "Std", "justify": "right", "style": "dim"},
        ],
        table_rows
    )
    console.print(results_table)

    # Linear trend summary
    trend_dir = "decreasing" if slope < 0 else "increasing"
    print_statistics_summary(console, "Linear Trend Analysis", {
        "Slope": f"{slope:.6f}",
        "R²": f"{r_value**2:.3f}",
        "p-value": f"{p_value:.4f}",
        "Trend": f"{trend_dir} (earlier interventions → larger effect)",
    })

    # Visualizations
    print_step_header(console, 4, "Creating visualizations")
    output_dir = "/outputs/robust/tick_sweep"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Tick sweep with error bars
    axes[0].errorbar(ticks, tick_means, yerr=tick_sems, fmt='bo-', capsize=5,
                     linewidth=2, markersize=8, label='Mean ± SEM')
    # Add regression line
    reg_line = [slope * t + intercept for t in ticks]
    axes[0].plot(ticks, reg_line, 'r--', linewidth=2,
                 label=f'Linear fit (r²={r_value**2:.3f})')
    axes[0].set_xlabel('Intervention Tick', fontsize=12)
    axes[0].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[0].set_title(f'Behavior Change vs Intervention Timing\n({len(seed_list)} seeds × {num_samples} samples)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Individual seed traces
    for seed_idx, seed in enumerate(seed_list):
        seed_diffs = [tick_results[t][seed_idx] for t in ticks]
        axes[1].plot(ticks, seed_diffs, 'o-', alpha=0.5, label=f'Seed {seed}')
    axes[1].plot(ticks, tick_means, 'ko-', linewidth=3, markersize=10, label='Mean')
    axes[1].set_xlabel('Intervention Tick', fontsize=12)
    axes[1].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[1].set_title('Individual Seed Trajectories', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"robust_tick_sweep_{maze_size}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save data
    np.savez(
        os.path.join(output_dir, f"robust_tick_sweep_{maze_size}.npz"),
        ticks=np.array(ticks),
        tick_means=np.array(tick_means),
        tick_stds=np.array(tick_stds),
        tick_results={str(t): tick_results[t] for t in ticks},
        slope=slope,
        r_squared=r_value**2,
        p_value=p_value,
        seeds=np.array(seed_list),
    )

    console.print(f"  [green]✓[/] Saved plot to {save_path}")

    if use_wandb:
        wandb.log({
            "tick_sweep_slope": slope,
            "tick_sweep_r_squared": r_value**2,
            "robust_tick_sweep_plot": wandb.Image(save_path),
        })
        wandb.finish()

    output_volume.commit()

    print_success_banner(console, f"Tick sweep complete! Slope: {slope:.6f}, R²: {r_value**2:.3f}")

    return {
        "status": "success",
        "tick_means": tick_means,
        "slope": float(slope),
        "r_squared": float(r_value**2),
        "output_dir": output_dir,
    }


@app.local_entrypoint()
def main(
    experiment: str = "all",
    maze_size: str = "medium",
    num_samples: int = 200,
    seeds: str = "42,123,456",
    num_random_trials: int = 20,
    use_wandb: bool = True,
):
    """
    Run robust experiments.

    Usage:
        modal run robust_runner.py --experiment probe_robust
        modal run robust_runner.py --experiment control_robust
        modal run robust_runner.py --experiment tick_sweep_robust
        modal run robust_runner.py --experiment all
    """
    print("=" * 60)
    print("CTM Robust Interpretability Experiments")
    print("=" * 60)
    print(f"\nExperiment: {experiment}")
    print(f"Maze size: {maze_size}")
    print(f"Samples: {num_samples}")
    print(f"Seeds: {seeds}")
    print()

    if experiment == "probe_robust" or experiment == "all":
        print("Running ROBUST PROBE experiment...")
        result = run_robust_probe.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            seeds=seeds,
            use_wandb=use_wandb,
        )
        print(f"\nProbe result: {result}")

    if experiment == "control_robust" or experiment == "all":
        print("\nRunning ROBUST CONTROL experiment...")
        result = run_robust_control.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            num_random_trials=num_random_trials,
            seeds=seeds,
            use_wandb=use_wandb,
        )
        print(f"\nControl result: {result}")

    if experiment == "tick_sweep_robust" or experiment == "all":
        print("\nRunning ROBUST TICK SWEEP experiment...")
        result = run_robust_tick_sweep.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            seeds=seeds,
            use_wandb=use_wandb,
        )
        print(f"\nTick sweep result: {result}")

    if experiment not in ["probe_robust", "control_robust", "tick_sweep_robust", "all"]:
        print(f"Unknown experiment: {experiment}")
        print("Available: probe_robust, control_robust, tick_sweep_robust, all")
