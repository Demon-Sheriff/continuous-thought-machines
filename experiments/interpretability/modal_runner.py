"""
Modal Runner for CTM Interpretability Experiments

This script provides a Modal-based runner for GPU experimentation.
Modal (https://modal.com) allows running GPU workloads in the cloud
without managing infrastructure.

Usage:
    # First time: Upload data and checkpoints to Modal volume
    modal run experiments/interpretability/modal_runner.py --experiment upload

    # Run the probe experiment on Modal with GPU
    modal run experiments/interpretability/modal_runner.py --experiment probe

Requirements:
    pip install modal wandb
    modal setup  # One-time authentication
    wandb login  # One-time authentication
"""

import modal
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Create Modal app
app = modal.App("ctm-interpretability")

# Define the container image with all dependencies
# Only include source code, not data/checkpoints (use volumes for those)
ctm_image = (
    modal.Image.debian_slim(python_version="3.12")
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
    ])
    # Only add Python source files, not data
    .add_local_dir(PROJECT_ROOT / "models", remote_path="/app/models")
    .add_local_dir(PROJECT_ROOT / "utils", remote_path="/app/utils")
    .add_local_file(PROJECT_ROOT / "data" / "custom_datasets.py", remote_path="/app/data/custom_datasets.py")
    .add_local_dir(PROJECT_ROOT / "experiments", remote_path="/app/experiments")
)

# Create volumes for persistent storage
output_volume = modal.Volume.from_name("ctm-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("ctm-data", create_if_missing=True)


@app.function(
    image=ctm_image,
    volumes={"/data": data_volume},
    timeout=3600,
)
def upload_data():
    """
    Upload local data and checkpoints to Modal volume.
    This needs to be run once before running experiments.
    """
    import os
    import shutil

    print("Checking /data volume contents...")

    # List current contents
    if os.path.exists("/data"):
        for root, dirs, files in os.walk("/data"):
            for f in files:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                print(f"  {path}: {size:,} bytes")
    else:
        print("  (empty)")

    return {"status": "checked", "message": "See logs for volume contents"}


@app.function(
    image=ctm_image,
    gpu="A10G",  # Use A10G GPU (24GB VRAM)
    timeout=3600,  # 1 hour timeout
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_probe_experiment(
    maze_size: str = "medium",
    num_samples: int = 100,
    num_top_neurons: int = 50,
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Run the probe experiment on Modal GPU.

    Args:
        maze_size: Size of mazes ("small", "medium", "large")
        num_samples: Number of mazes to analyze
        num_top_neurons: Number of top neurons to analyze
        use_wandb: Whether to log to wandb
        checkpoint_path: Path to model checkpoint (on Modal volume or local)
        data_root: Path to maze data directory
    """
    import sys
    sys.path.insert(0, "/app")

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use provided paths or defaults
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    print(f"\nCheckpoint path: {checkpoint_path}")
    print(f"Data root: {data_root}")

    # Import and run the probe experiment
    from experiments.interpretability.probe_experiment import ProbeConfig, run_experiment

    config = ProbeConfig(
        checkpoint_path=checkpoint_path,
        device="cuda",
        data_root=data_root,
        maze_size=maze_size,
        num_samples=num_samples,
        num_top_neurons=num_top_neurons,
        output_dir="/outputs/probe",
        use_wandb=use_wandb,
        wandb_project="ctm-interpretability"
    )

    print(f"\nRunning probe experiment with config:")
    print(f"  maze_size: {maze_size}")
    print(f"  num_samples: {num_samples}")
    print(f"  num_top_neurons: {num_top_neurons}")
    print(f"  use_wandb: {use_wandb}")
    print()

    results = run_experiment(config)

    # Save results summary
    if results:
        output_volume.commit()
        return {
            "status": "success",
            "top_neurons": results['top_neurons'][:10] if 'top_neurons' in results else None,
            "output_dir": config.output_dir
        }
    else:
        return {"status": "error", "message": "Experiment failed"}


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_teleport_experiment(
    maze_size: str = "medium",
    num_samples: int = 50,
    intervention_tick: int = 5,
    patch_type: str = "goal",
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Run the teleport experiment on Modal GPU.
    """
    import sys
    sys.path.insert(0, "/app")

    # Use provided paths or defaults
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    from experiments.interpretability.teleport_experiment import TeleportConfig, run_experiment

    config = TeleportConfig(
        checkpoint_path=checkpoint_path,
        device="cuda",
        data_root=data_root,
        maze_size=maze_size,
        num_samples=num_samples,
        intervention_tick=intervention_tick,
        patch_type=patch_type,
        output_dir="/outputs/teleport",
        use_wandb=use_wandb,
        wandb_project="ctm-interpretability"
    )

    results, analysis = run_experiment(config)

    if results:
        output_volume.commit()
        return {
            "status": "success",
            "analysis": analysis,
            "output_dir": config.output_dir
        }
    else:
        return {"status": "error", "message": "Experiment failed"}


@app.function(
    image=ctm_image,
    volumes={"/outputs": output_volume, "/data": data_volume},
)
def list_outputs():
    """List contents of the output and data volumes."""
    import os

    def list_dir(path, indent=0, max_files=50):
        count = 0
        if os.path.exists(path):
            items = sorted(os.listdir(path))
            if not items:
                print("  " * indent + "(empty)")
            for item in items:
                if count >= max_files:
                    remaining = len(items) - count
                    print("  " * indent + f"... and {remaining} more items")
                    break
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    print("  " * indent + f"[DIR] {item}/")
                    if indent < 2:
                        list_dir(full_path, indent + 1, max_files=10)
                else:
                    size = os.path.getsize(full_path)
                    print("  " * indent + f"{item} ({size:,} bytes)")
                count += 1
        else:
            print(f"Path does not exist: {path}")

    print("=" * 50)
    print("Contents of /data:")
    print("=" * 50)
    list_dir("/data")

    print("\n" + "=" * 50)
    print("Contents of /outputs:")
    print("=" * 50)
    list_dir("/outputs")


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=7200,  # 2 hour timeout for sweep
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_tick_sweep(
    maze_size: str = "medium",
    num_samples: int = 50,
    tick_values: str = "5,10,20,30,40,50,60,70",  # Comma-separated tick values
    patch_type: str = "goal",
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Run teleport experiment across multiple intervention ticks.
    Returns sweep results for plotting.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Use provided paths or defaults
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    # Parse tick values
    ticks = [int(t.strip()) for t in tick_values.split(",")]
    print(f"Running tick sweep with ticks: {ticks}")

    from experiments.interpretability.teleport_experiment import (
        TeleportConfig, load_model, identify_position_neurons,
        run_teleport_experiment, analyze_teleport_results,
        ModifiedCTMForIntervention
    )
    from data.custom_datasets import MazeImageFolder

    # Initialize wandb for sweep
    if use_wandb:
        import wandb
        wandb.init(
            project="ctm-interpretability",
            name=f"tick_sweep_{patch_type}",
            config={
                "experiment": "tick_sweep",
                "ticks": ticks,
                "patch_type": patch_type,
                "num_samples": num_samples,
                "maze_size": maze_size,
            }
        )

    # Load model once
    print("\n[Step 1] Loading model...")
    config = TeleportConfig(
        checkpoint_path=checkpoint_path,
        device="cuda",
        data_root=data_root,
        maze_size=maze_size,
        num_samples=num_samples,
        patch_type=patch_type,
    )
    model = load_model(config)

    # Load data once
    print("\n[Step 2] Loading data...")
    data_path = f"{data_root}/{maze_size}/test"
    dataset = MazeImageFolder(
        root=data_path,
        which_set='test',
        maze_route_length=100,
        expand_range=True,
        trunc=True
    )
    if len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Identify position neurons once
    print("\n[Step 3] Identifying position-encoding neurons...")
    start_neurons, goal_neurons, _ = identify_position_neurons(model, dataloader, config)

    # Run sweep
    sweep_results = {
        'ticks': ticks,
        'move_diff_ratios': [],
        'wait_increases': [],
        'stop_increase_counts': [],
    }

    for tick in ticks:
        print(f"\n[Tick {tick}] Running teleport experiment...")

        # Reload dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Update config for this tick
        config.intervention_tick = tick

        # Run experiment
        results = run_teleport_experiment(
            model, dataloader, start_neurons, goal_neurons, config
        )
        analysis = analyze_teleport_results(results, config)

        sweep_results['move_diff_ratios'].append(analysis['avg_move_diff_ratio'])
        sweep_results['wait_increases'].append(analysis['avg_wait_increase'])
        sweep_results['stop_increase_counts'].append(analysis['stop_increase_count'])

        if use_wandb:
            wandb.log({
                f"tick_{tick}/move_diff_ratio": analysis['avg_move_diff_ratio'],
                f"tick_{tick}/wait_increase": analysis['avg_wait_increase'],
                f"tick_{tick}/stop_increase_count": analysis['stop_increase_count'],
            })

    # Create sweep visualization
    print("\n[Step 4] Creating sweep visualization...")
    output_dir = "/outputs/tick_sweep"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Move difference ratio vs tick
    axes[0].plot(ticks, sweep_results['move_diff_ratios'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Intervention Tick', fontsize=12)
    axes[0].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[0].set_title('Behavior Change vs Intervention Timing', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(sweep_results['move_diff_ratios']) * 1.1)

    # Plot 2: Wait increase vs tick
    axes[1].plot(ticks, sweep_results['wait_increases'], 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Intervention Tick', fontsize=12)
    axes[1].set_ylabel('Average Wait Increase', fontsize=12)
    axes[1].set_title('Wait/Stop Change vs Intervention Timing', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 3: Samples with increased stopping
    axes[2].bar(ticks, sweep_results['stop_increase_counts'], color='purple', alpha=0.7, width=3)
    axes[2].set_xlabel('Intervention Tick', fontsize=12)
    axes[2].set_ylabel('Samples with Increased Stopping', fontsize=12)
    axes[2].set_title('Stopping Behavior vs Intervention Timing', fontsize=14)
    axes[2].set_ylim(0, num_samples)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "tick_sweep.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save raw data
    np.savez(
        os.path.join(output_dir, "tick_sweep_results.npz"),
        **sweep_results
    )

    if use_wandb:
        wandb.log({"tick_sweep_plot": wandb.Image(save_path)})
        wandb.finish()

    output_volume.commit()

    print("\n" + "=" * 60)
    print("Tick Sweep Complete!")
    print("=" * 60)
    print(f"\nResults:")
    for i, tick in enumerate(ticks):
        print(f"  Tick {tick:2d}: {sweep_results['move_diff_ratios'][i]:.2%} move diff, "
              f"{sweep_results['wait_increases'][i]:+.2f} wait change")

    return {
        "status": "success",
        "sweep_results": sweep_results,
        "output_dir": output_dir
    }


@app.function(
    image=ctm_image,
    gpu="A10G",
    timeout=7200,  # 2 hour timeout
    volumes={"/outputs": output_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("api-keys")],
)
def run_control_experiment(
    maze_size: str = "medium",
    num_samples: int = 50,
    intervention_tick: int = 5,
    num_random_trials: int = 5,  # Number of random neuron sets to test
    use_wandb: bool = True,
    checkpoint_path: str = "",
    data_root: str = "",
):
    """
    Control experiment: Compare patching position-encoding neurons vs random neurons.

    This tests the specificity of the place cell hypothesis by showing that
    random neurons don't cause the same behavior disruption as position-encoding neurons.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Use provided paths or defaults
    if not checkpoint_path:
        checkpoint_path = "/data/checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    if not data_root:
        data_root = "/data/mazes"

    from experiments.interpretability.teleport_experiment import (
        TeleportConfig, load_model, identify_position_neurons,
        run_teleport_experiment, analyze_teleport_results,
        ModifiedCTMForIntervention
    )
    from data.custom_datasets import MazeImageFolder

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="ctm-interpretability",
            name=f"control_random_neurons",
            config={
                "experiment": "control",
                "intervention_tick": intervention_tick,
                "num_random_trials": num_random_trials,
                "num_samples": num_samples,
                "maze_size": maze_size,
            }
        )

    # Load model
    print("\n[Step 1] Loading model...")
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
    d_model = model.d_model  # Total number of neurons

    # Load data
    print("\n[Step 2] Loading data...")
    data_path = f"{data_root}/{maze_size}/test"
    dataset = MazeImageFolder(
        root=data_path,
        which_set='test',
        maze_route_length=100,
        expand_range=True,
        trunc=True
    )
    if len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Identify position neurons (the "real" place cells)
    print("\n[Step 3] Identifying position-encoding neurons...")
    start_neurons, goal_neurons, _ = identify_position_neurons(model, dataloader, config)
    num_neurons_to_patch = len(start_neurons)

    print(f"  Position neurons identified: {num_neurons_to_patch} start, {len(goal_neurons)} goal")

    # Results storage
    results = {
        'position_neurons': {'move_diff': None, 'wait_change': None},
        'random_neurons': {'move_diffs': [], 'wait_changes': []},
    }

    # === Run with position-encoding neurons (baseline) ===
    print("\n[Step 4] Running intervention with POSITION neurons...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    position_results = run_teleport_experiment(
        model, dataloader, start_neurons, goal_neurons, config
    )
    position_analysis = analyze_teleport_results(position_results, config)

    results['position_neurons']['move_diff'] = position_analysis['avg_move_diff_ratio']
    results['position_neurons']['wait_change'] = position_analysis['avg_wait_increase']

    print(f"  Position neurons: {position_analysis['avg_move_diff_ratio']:.2%} move diff")

    # === Run with random neurons (control) ===
    print(f"\n[Step 5] Running {num_random_trials} trials with RANDOM neurons...")

    np.random.seed(42)  # For reproducibility

    for trial in range(num_random_trials):
        # Select random neurons (excluding the position neurons we identified)
        all_neurons = set(range(d_model))
        excluded = set(start_neurons) | set(goal_neurons)
        available_neurons = list(all_neurons - excluded)

        # Randomly select same number of neurons as position neurons
        random_start = list(np.random.choice(available_neurons, size=num_neurons_to_patch, replace=False))
        random_goal = list(np.random.choice(
            [n for n in available_neurons if n not in random_start],
            size=num_neurons_to_patch,
            replace=False
        ))

        print(f"  Trial {trial + 1}/{num_random_trials}...")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        random_results = run_teleport_experiment(
            model, dataloader, random_start, random_goal, config
        )
        random_analysis = analyze_teleport_results(random_results, config)

        results['random_neurons']['move_diffs'].append(random_analysis['avg_move_diff_ratio'])
        results['random_neurons']['wait_changes'].append(random_analysis['avg_wait_increase'])

        print(f"    Random trial {trial + 1}: {random_analysis['avg_move_diff_ratio']:.2%} move diff")

    # Calculate statistics
    random_move_mean = np.mean(results['random_neurons']['move_diffs'])
    random_move_std = np.std(results['random_neurons']['move_diffs'])
    random_wait_mean = np.mean(results['random_neurons']['wait_changes'])

    position_move = results['position_neurons']['move_diff']

    # Effect size (how many std devs above random)
    if random_move_std > 0:
        effect_size = (position_move - random_move_mean) / random_move_std
    else:
        effect_size = float('inf') if position_move > random_move_mean else 0

    print(f"\n" + "=" * 60)
    print("Control Experiment Results")
    print("=" * 60)
    print(f"\nPosition Neurons:")
    print(f"  Move difference: {position_move:.2%}")
    print(f"\nRandom Neurons ({num_random_trials} trials):")
    print(f"  Move difference: {random_move_mean:.2%} ± {random_move_std:.2%}")
    print(f"\nEffect Size: {effect_size:.2f} standard deviations")
    print(f"Position neurons cause {position_move/random_move_mean:.1f}x more disruption than random")

    # === Create visualizations ===
    print("\n[Step 6] Creating visualizations...")
    output_dir = "/outputs/control"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Bar comparison
    categories = ['Position\nNeurons'] + [f'Random\nTrial {i+1}' for i in range(num_random_trials)]
    values = [position_move] + results['random_neurons']['move_diffs']
    colors = ['red'] + ['blue'] * num_random_trials

    bars = axes[0].bar(categories, values, color=colors, alpha=0.7)
    axes[0].axhline(y=random_move_mean, color='blue', linestyle='--',
                    label=f'Random mean: {random_move_mean:.2%}')
    axes[0].set_ylabel('Move Difference Ratio', fontsize=12)
    axes[0].set_title('Position Neurons vs Random Neurons\n(Control Experiment)', fontsize=14)
    axes[0].legend()
    axes[0].set_ylim(0, max(values) * 1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Distribution comparison with effect size
    axes[1].hist(results['random_neurons']['move_diffs'], bins=max(3, num_random_trials//2),
                 alpha=0.7, color='blue', label='Random neurons', edgecolor='black')
    axes[1].axvline(x=position_move, color='red', linewidth=3,
                    label=f'Position neurons: {position_move:.2%}')
    axes[1].axvline(x=random_move_mean, color='blue', linestyle='--', linewidth=2,
                    label=f'Random mean: {random_move_mean:.2%}')
    axes[1].set_xlabel('Move Difference Ratio', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'Effect Size: {effect_size:.2f} σ above random', fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "control_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save raw data
    np.savez(
        os.path.join(output_dir, "control_results.npz"),
        position_move_diff=position_move,
        position_wait_change=results['position_neurons']['wait_change'],
        random_move_diffs=results['random_neurons']['move_diffs'],
        random_wait_changes=results['random_neurons']['wait_changes'],
        effect_size=effect_size,
        start_neurons=start_neurons,
        goal_neurons=goal_neurons,
    )

    if use_wandb:
        wandb.log({
            "position_neurons_move_diff": position_move,
            "random_neurons_move_diff_mean": random_move_mean,
            "random_neurons_move_diff_std": random_move_std,
            "effect_size": effect_size,
            "disruption_ratio": position_move / random_move_mean if random_move_mean > 0 else float('inf'),
            "control_comparison": wandb.Image(save_path),
        })
        wandb.finish()

    output_volume.commit()

    return {
        "status": "success",
        "position_move_diff": float(position_move),
        "random_move_diff_mean": float(random_move_mean),
        "random_move_diff_std": float(random_move_std),
        "effect_size": float(effect_size),
        "output_dir": output_dir
    }


@app.local_entrypoint()
def main(
    experiment: str = "probe",
    maze_size: str = "medium",
    num_samples: int = 50,
    num_top_neurons: int = 20,
    use_wandb: bool = False,
):
    """
    Main entrypoint for running experiments.

    Usage:
        # List volume contents
        modal run modal_runner.py --experiment list

        # Run probe experiment
        modal run modal_runner.py --experiment probe --maze-size medium --num-samples 50

        # Run teleport experiment
        modal run modal_runner.py --experiment teleport --maze-size medium

    Note: Before running experiments, you need to upload data to the Modal volume.
    Use: modal volume put ctm-data ./checkpoints /checkpoints
         modal volume put ctm-data ./data/mazes /mazes
    """
    print("=" * 60)
    print("CTM Interpretability Experiments - Modal Runner")
    print("=" * 60)
    print()

    if experiment == "probe":
        print(f"Running PROBE experiment...")
        print(f"  maze_size: {maze_size}")
        print(f"  num_samples: {num_samples}")
        print(f"  num_top_neurons: {num_top_neurons}")
        print(f"  use_wandb: {use_wandb}")
        print()

        result = run_probe_experiment.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            num_top_neurons=num_top_neurons,
            use_wandb=use_wandb,
        )

        print("\n" + "=" * 60)
        print("Result:")
        print(result)

    elif experiment == "teleport":
        print(f"Running TELEPORT experiment...")
        result = run_teleport_experiment.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            use_wandb=use_wandb,
        )
        print("\n" + "=" * 60)
        print("Result:")
        print(result)

    elif experiment == "list":
        print("Listing volume contents...")
        list_outputs.remote()

    elif experiment == "upload":
        print("Checking data volume...")
        result = upload_data.remote()
        print(result)

    elif experiment == "tick_sweep":
        print(f"Running TICK SWEEP experiment...")
        print(f"  maze_size: {maze_size}")
        print(f"  num_samples: {num_samples}")
        print(f"  use_wandb: {use_wandb}")
        print()

        result = run_tick_sweep.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            use_wandb=use_wandb,
        )

        print("\n" + "=" * 60)
        print("Result:")
        print(result)

    elif experiment == "control":
        print(f"Running CONTROL experiment (random neuron patching)...")
        print(f"  maze_size: {maze_size}")
        print(f"  num_samples: {num_samples}")
        print(f"  use_wandb: {use_wandb}")
        print()

        result = run_control_experiment.remote(
            maze_size=maze_size,
            num_samples=num_samples,
            use_wandb=use_wandb,
        )

        print("\n" + "=" * 60)
        print("Result:")
        print(result)

    else:
        print(f"Unknown experiment: {experiment}")
        print("Available: probe, teleport, tick_sweep, control, list, upload")
