"""
Teleport Experiment: Activation Patching for CTM Maze Navigation

This script implements the "Teleport" experiment to causally verify
that CTM neurons encode positional information.

Experiment Design:
1. Identify neurons associated with "Start" position
2. Identify neurons associated with "Goal" position
3. Mid-inference, patch the "Start" neuron activations with "Goal" activations
4. Observe if the model's behavior changes (stops moving, reverses direction, etc.)

This is an intervention experiment to establish causality, not just correlation.

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
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import copy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ctm import ContinuousThoughtMachine
from data.custom_datasets import MazeImageFolder

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TeleportConfig:
    """Configuration for the teleport experiment."""
    # Model settings
    checkpoint_path: str = "checkpoints/mazes/ctm_mazeslarge_D=2048_T=75_M=25.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data settings
    data_root: str = "data/mazes"
    maze_size: str = "medium"
    maze_route_length: int = 100
    num_samples: int = 50
    batch_size: int = 1  # Process one maze at a time for intervention

    # Intervention settings
    intervention_tick: int = 5  # Tick at which to apply the patch
    num_neurons_to_patch: int = 10  # Number of top position-encoding neurons to patch
    patch_type: str = "goal"  # "goal", "start", "random", or "zero"

    # Analysis settings
    probe_results_path: str = "experiments/interpretability/outputs/probe_results.npz"

    # Output settings
    output_dir: str = "experiments/interpretability/outputs/teleport"
    save_videos: bool = True

    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "ctm-interpretability"
    wandb_run_name: Optional[str] = None


class ActivationPatcher:
    """
    Hook-based activation patching for CTM.

    This class allows us to:
    1. Record activations during a clean forward pass
    2. Inject modified activations during a patched forward pass

    The patching targets the post-activation state (z_t), which is
    the output of the Neuron-Level Models and the key representation
    that feeds into synchronization computation.
    """

    def __init__(
        self,
        model: ContinuousThoughtMachine,
        target_neurons: List[int],
        intervention_tick: int
    ):
        self.model = model
        self.target_neurons = target_neurons
        self.intervention_tick = intervention_tick

        # Storage for recorded activations
        self.recorded_activations: Dict[int, torch.Tensor] = {}
        self.patch_values: Optional[torch.Tensor] = None

        # Hook handles
        self.hooks = []
        self.current_tick = 0
        self.is_patching = False

    def _get_activated_state_hook(self):
        """
        Create a hook for the trace_processor (NLM) output.

        The activated_state is computed as:
            activated_state = self.trace_processor(state_trace)

        We hook into this to record and modify post-activations.
        """
        def hook(module, input, output):
            # output shape: (B, D) - post-activations for all neurons
            if self.is_patching and self.current_tick == self.intervention_tick:
                # Apply patch to target neurons
                if self.patch_values is not None:
                    patched_output = output.clone()
                    patched_output[:, self.target_neurons] = self.patch_values
                    return patched_output
            else:
                # Record activations
                self.recorded_activations[self.current_tick] = output.clone().detach()

            return output

        return hook

    def register_hooks(self):
        """Register forward hooks on the model."""
        # Hook the trace_processor (NLM) output
        hook = self.model.trace_processor.register_forward_hook(
            self._get_activated_state_hook()
        )
        self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def record_clean_pass(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a clean forward pass and record activations.

        Returns:
            Tuple of (predictions, certainties)
        """
        self.is_patching = False
        self.current_tick = 0
        self.recorded_activations = {}

        # We need to manually track ticks since the model doesn't expose this
        # Solution: Create a modified forward that tracks ticks

        with torch.no_grad():
            predictions, certainties, _ = self.model(inputs)

        return predictions, certainties

    def run_patched_pass(
        self,
        inputs: torch.Tensor,
        patch_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass with activation patching.

        Args:
            inputs: Input tensor
            patch_values: Values to patch into target neurons at intervention_tick

        Returns:
            Tuple of (predictions, certainties)
        """
        self.is_patching = True
        self.patch_values = patch_values
        self.current_tick = 0

        with torch.no_grad():
            predictions, certainties, _ = self.model(inputs)

        self.is_patching = False
        return predictions, certainties


class ModifiedCTMForIntervention(nn.Module):
    """
    Modified CTM that allows tick-by-tick intervention.

    This wrapper exposes the internal loop to allow patching at specific ticks.
    """

    def __init__(self, base_model: ContinuousThoughtMachine):
        super().__init__()
        self.base = base_model
        self.intervention_hook: Optional[Callable] = None

    def set_intervention_hook(self, hook: Callable):
        """
        Set a hook function that's called after each tick.

        Hook signature: hook(tick: int, activated_state: Tensor) -> Tensor
        """
        self.intervention_hook = hook

    def forward(self, x, track=False):
        """
        Forward pass with intervention support.

        This is a modified version of ContinuousThoughtMachine.forward()
        that allows patching activated_state at any tick.
        """
        B = x.size(0)
        device = x.device
        model = self.base

        # Tracking initialization
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # Featurize input data
        kv = model.compute_features(x)

        # Initialize recurrent state
        state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = model.start_activated_state.unsqueeze(0).expand(B, -1)

        # Prepare storage
        predictions = torch.empty(B, model.out_dims, model.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, model.iterations, device=device, dtype=torch.float32)

        # Initialize synchronization
        decay_alpha_action, decay_beta_action = None, None
        model.decay_params_action.data = torch.clamp(model.decay_params_action, 0, 15)
        model.decay_params_out.data = torch.clamp(model.decay_params_out, 0, 15)
        r_action = torch.exp(-model.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-model.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        # Recurrent loop with intervention support
        for stepi in range(model.iterations):
            # Synchronization for attention
            synchronisation_action, decay_alpha_action, decay_beta_action = model.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )

            # Attention
            q = model.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = model.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # Synapse model
            state = model.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # NLM (Neuron-Level Models)
            activated_state = model.trace_processor(state_trace)

            # >>> INTERVENTION POINT <<<
            if self.intervention_hook is not None:
                activated_state = self.intervention_hook(stepi, activated_state)

            # Output synchronization
            synchronisation_out, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )

            # Predictions
            current_prediction = model.output_projector(synchronisation_out)
            current_certainty = model.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return (predictions, certainties,
                   (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                   np.array(pre_activations_tracking),
                   np.array(post_activations_tracking),
                   np.array(attention_tracking))

        return predictions, certainties, synchronisation_out


def load_model(config: TeleportConfig) -> ContinuousThoughtMachine:
    """Load the CTM model from checkpoint."""
    print(f"Loading checkpoint from: {config.checkpoint_path}")

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

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

    state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_dict_key], strict=False)
    model.eval()

    return model


def load_probe_results(config: TeleportConfig) -> Dict:
    """Load results from the probe experiment."""
    if not os.path.exists(config.probe_results_path):
        print(f"Warning: Probe results not found at {config.probe_results_path}")
        print("Using random neuron selection instead")
        return None

    results = np.load(config.probe_results_path, allow_pickle=True)
    return dict(results)


def identify_position_neurons(
    model: ContinuousThoughtMachine,
    dataloader: torch.utils.data.DataLoader,
    config: TeleportConfig
) -> Tuple[List[int], List[int], Dict]:
    """
    Identify neurons that encode start vs goal positions.

    Strategy:
    1. Run model on mazes and record activations at each tick
    2. For early ticks (near start), identify highly active neurons
    3. For late ticks (near goal), identify highly active neurons
    4. Select neurons that differ most between start and goal

    Returns:
        Tuple of (start_neurons, goal_neurons, analysis_dict)
    """
    print("Identifying position-encoding neurons...")

    # Collect activations
    early_tick_activations = []  # Ticks 0-10 (near start)
    late_tick_activations = []   # Ticks 50-75 (near goal)

    wrapped_model = ModifiedCTMForIntervention(model)

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(config.device)

            # Run with tracking
            results = wrapped_model(inputs, track=True)
            post_activations = results[4]  # Shape: (T, B, D)

            # Collect early and late activations
            T = post_activations.shape[0]
            early_range = range(0, min(10, T))
            late_range = range(max(0, T-25), T)

            for t in early_range:
                early_tick_activations.append(post_activations[t])  # (B, D)

            for t in late_range:
                late_tick_activations.append(post_activations[t])  # (B, D)

    # Stack and analyze
    early_acts = np.concatenate(early_tick_activations, axis=0)  # (N_early, D)
    late_acts = np.concatenate(late_tick_activations, axis=0)   # (N_late, D)

    # Mean activation per neuron at start vs goal
    early_mean = np.mean(early_acts, axis=0)  # (D,)
    late_mean = np.mean(late_acts, axis=0)    # (D,)

    # Difference score: how much does each neuron differ between start and goal?
    diff_score = np.abs(early_mean - late_mean)

    # Identify start-preferring neurons (higher early, lower late)
    start_preference = early_mean - late_mean
    start_neurons = np.argsort(start_preference)[-config.num_neurons_to_patch:]

    # Identify goal-preferring neurons (higher late, lower early)
    goal_preference = late_mean - early_mean
    goal_neurons = np.argsort(goal_preference)[-config.num_neurons_to_patch:]

    print(f"Start neurons: {start_neurons.tolist()}")
    print(f"Goal neurons: {goal_neurons.tolist()}")

    analysis = {
        'early_mean': early_mean,
        'late_mean': late_mean,
        'diff_score': diff_score,
        'start_preference': start_preference,
        'goal_preference': goal_preference
    }

    return start_neurons.tolist(), goal_neurons.tolist(), analysis


def run_teleport_experiment(
    model: ContinuousThoughtMachine,
    dataloader: torch.utils.data.DataLoader,
    start_neurons: List[int],
    goal_neurons: List[int],
    config: TeleportConfig
) -> Dict:
    """
    Run the teleport experiment.

    For each maze:
    1. Run clean forward pass, record goal-position activations
    2. Run patched forward pass, injecting goal activations at intervention_tick
    3. Compare predictions before and after patch
    4. Analyze behavior change

    Returns:
        Dict with experiment results
    """
    print(f"\nRunning teleport experiment...")
    print(f"  Intervention tick: {config.intervention_tick}")
    print(f"  Patch type: {config.patch_type}")

    results = {
        'clean_predictions': [],
        'patched_predictions': [],
        'behavior_changes': [],
        'move_distributions': {'clean': [], 'patched': []},
    }

    wrapped_model = ModifiedCTMForIntervention(model)

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(config.device)
        targets = targets.numpy()

        # === Clean forward pass ===
        wrapped_model.set_intervention_hook(None)
        with torch.no_grad():
            clean_results = wrapped_model(inputs, track=True)
            clean_predictions = clean_results[0]  # (B, out_dims, T)
            clean_post_acts = clean_results[4]    # (T, B, D)

        # Get clean move sequence (at most certain tick)
        clean_certainties = clean_results[1]  # (B, 2, T)
        most_certain_tick = clean_certainties[0, 1].argmax().item()

        clean_moves = clean_predictions[0, :, most_certain_tick].reshape(-1, 5).argmax(-1)
        clean_moves = clean_moves.cpu().numpy()

        # === Prepare patch values ===
        if config.patch_type == "goal":
            # Use activations from late ticks (near goal)
            T = clean_post_acts.shape[0]
            goal_tick = T - 1
            # Shape: (B, num_neurons) - index batch first, then select neurons
            patch_values = torch.from_numpy(clean_post_acts[goal_tick][:, goal_neurons]).to(config.device)
            target_neurons = start_neurons
        elif config.patch_type == "start":
            # Use activations from early ticks (near start)
            patch_values = torch.from_numpy(clean_post_acts[0][:, start_neurons]).to(config.device)
            target_neurons = goal_neurons
        elif config.patch_type == "zero":
            patch_values = torch.zeros(1, len(start_neurons), device=config.device)
            target_neurons = start_neurons
        else:  # random
            patch_values = torch.randn(1, len(start_neurons), device=config.device)
            target_neurons = start_neurons

        # === Patched forward pass ===
        def intervention_hook(tick: int, activated_state: torch.Tensor) -> torch.Tensor:
            if tick == config.intervention_tick:
                modified_state = activated_state.clone()
                modified_state[:, target_neurons] = patch_values
                return modified_state
            return activated_state

        wrapped_model.set_intervention_hook(intervention_hook)
        with torch.no_grad():
            patched_results = wrapped_model(inputs, track=True)
            patched_predictions = patched_results[0]

        # Get patched move sequence
        patched_certainties = patched_results[1]
        most_certain_tick_patched = patched_certainties[0, 1].argmax().item()

        patched_moves = patched_predictions[0, :, most_certain_tick_patched].reshape(-1, 5).argmax(-1)
        patched_moves = patched_moves.cpu().numpy()

        # === Analyze behavior change ===
        # Compare move distributions
        move_diff = (clean_moves != patched_moves).sum()
        move_diff_ratio = move_diff / len(clean_moves)

        # Check for specific behavior changes
        wait_moves_clean = (clean_moves == 4).sum()
        wait_moves_patched = (patched_moves == 4).sum()

        results['clean_predictions'].append(clean_moves)
        results['patched_predictions'].append(patched_moves)
        results['behavior_changes'].append({
            'move_diff_count': int(move_diff),
            'move_diff_ratio': float(move_diff_ratio),
            'wait_moves_clean': int(wait_moves_clean),
            'wait_moves_patched': int(wait_moves_patched),
            'most_certain_tick_clean': most_certain_tick,
            'most_certain_tick_patched': most_certain_tick_patched
        })

        # Move distribution analysis
        for move_type in range(5):
            results['move_distributions']['clean'].append((clean_moves == move_type).sum())
            results['move_distributions']['patched'].append((patched_moves == move_type).sum())

    return results


def analyze_teleport_results(results: Dict, config: TeleportConfig) -> Dict:
    """Analyze and summarize teleport experiment results."""
    print("\nAnalyzing teleport results...")

    behavior_changes = results['behavior_changes']

    # Aggregate metrics
    avg_move_diff = np.mean([b['move_diff_ratio'] for b in behavior_changes])
    avg_wait_increase = np.mean([
        b['wait_moves_patched'] - b['wait_moves_clean']
        for b in behavior_changes
    ])

    # Did patching cause the model to "stop" more often?
    stop_increase_count = sum(
        1 for b in behavior_changes
        if b['wait_moves_patched'] > b['wait_moves_clean']
    )

    analysis = {
        'avg_move_diff_ratio': float(avg_move_diff),
        'avg_wait_increase': float(avg_wait_increase),
        'stop_increase_count': stop_increase_count,
        'total_samples': len(behavior_changes)
    }

    print(f"\nTeleport Experiment Results:")
    print(f"  Average move difference: {avg_move_diff:.2%}")
    print(f"  Average wait/stop increase: {avg_wait_increase:.2f} moves")
    print(f"  Samples with increased stopping: {stop_increase_count}/{len(behavior_changes)}")

    return analysis


def create_teleport_visualizations(
    results: Dict,
    analysis: Dict,
    config: TeleportConfig
):
    """Create visualizations for the teleport experiment."""
    print("\nCreating visualizations...")

    os.makedirs(config.output_dir, exist_ok=True)

    # Plot 1: Move distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    move_names = ['Up', 'Down', 'Left', 'Right', 'Wait']

    # Clean distribution
    clean_dist = np.zeros(5)
    patched_dist = np.zeros(5)

    for i, (clean_moves, patched_moves) in enumerate(zip(
        results['clean_predictions'],
        results['patched_predictions']
    )):
        for m in range(5):
            clean_dist[m] += (clean_moves == m).sum()
            patched_dist[m] += (patched_moves == m).sum()

    x = np.arange(5)
    width = 0.35

    axes[0].bar(x - width/2, clean_dist, width, label='Clean', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, patched_dist, width, label='Patched', color='red', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(move_names)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Move Distribution: Clean vs Patched')
    axes[0].legend()

    # Difference plot
    diff = patched_dist - clean_dist
    colors = ['green' if d > 0 else 'red' for d in diff]
    axes[1].bar(x, diff, color=colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(move_names)
    axes[1].set_ylabel('Change in Count')
    axes[1].set_title('Move Distribution Change (Patched - Clean)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "move_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved move distribution plot to: {save_path}")

    # Plot 2: Move difference per sample
    fig, ax = plt.subplots(figsize=(10, 5))

    move_diffs = [b['move_diff_ratio'] for b in results['behavior_changes']]
    ax.bar(range(len(move_diffs)), move_diffs, color='purple', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Move Difference Ratio')
    ax.set_title(f'Behavior Change per Sample (Intervention at tick {config.intervention_tick})')
    ax.axhline(y=np.mean(move_diffs), color='red', linestyle='--',
               label=f'Mean: {np.mean(move_diffs):.2%}')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "behavior_change.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved behavior change plot to: {save_path}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "move_distribution": wandb.Image(os.path.join(config.output_dir, "move_distribution.png")),
            "behavior_change": wandb.Image(os.path.join(config.output_dir, "behavior_change.png"))
        })


def run_experiment(config: TeleportConfig):
    """Main experiment runner."""
    print("=" * 60)
    print("CTM Teleport Experiment")
    print("=" * 60)

    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"teleport_{config.patch_type}",
            config=vars(config)
        )

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        # Load model
        print("\n[Step 1/5] Loading model...")
        model = load_model(config)

        # Load data
        print("\n[Step 2/5] Loading data...")
        data_path = f"{config.data_root}/{config.maze_size}/test"
        dataset = MazeImageFolder(
            root=data_path,
            which_set='test',
            maze_route_length=config.maze_route_length,
            expand_range=True,
            trunc=True
        )

        if len(dataset) > config.num_samples:
            indices = list(range(config.num_samples))
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        # Identify position neurons
        print("\n[Step 3/5] Identifying position-encoding neurons...")
        start_neurons, goal_neurons, neuron_analysis = identify_position_neurons(
            model, dataloader, config
        )

        # Reload dataloader (consumed by identification)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        # Run teleport experiment
        print("\n[Step 4/5] Running teleport experiment...")
        results = run_teleport_experiment(
            model, dataloader, start_neurons, goal_neurons, config
        )

        # Analyze results
        print("\n[Step 5/5] Analyzing results...")
        analysis = analyze_teleport_results(results, config)

        # Create visualizations
        create_teleport_visualizations(results, analysis, config)

        # Save results
        np.savez(
            os.path.join(config.output_dir, "teleport_results.npz"),
            start_neurons=start_neurons,
            goal_neurons=goal_neurons,
            analysis=analysis
        )

        # Summary
        print("\n" + "=" * 60)
        print("Teleport Experiment Complete!")
        print("=" * 60)
        print(f"\nKey Findings:")
        print(f"  - Intervention tick: {config.intervention_tick}")
        print(f"  - Patch type: {config.patch_type}")
        print(f"  - Average behavior change: {analysis['avg_move_diff_ratio']:.2%}")
        print(f"  - Average wait increase: {analysis['avg_wait_increase']:.2f} moves")
        print(f"\nOutputs saved to: {config.output_dir}")

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log(analysis)
            wandb.finish()

        return results, analysis

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish(exit_code=1)
        return None, None


def parse_args() -> TeleportConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CTM Teleport Experiment")

    parser.add_argument("--checkpoint", type=str, default=TeleportConfig.checkpoint_path)
    parser.add_argument("--device", type=str, default=TeleportConfig.device)
    parser.add_argument("--data-root", type=str, default=TeleportConfig.data_root)
    parser.add_argument("--maze-size", type=str, default=TeleportConfig.maze_size)
    parser.add_argument("--num-samples", type=int, default=TeleportConfig.num_samples)
    parser.add_argument("--intervention-tick", type=int, default=TeleportConfig.intervention_tick)
    parser.add_argument("--num-neurons", type=int, default=TeleportConfig.num_neurons_to_patch)
    parser.add_argument("--patch-type", type=str, default=TeleportConfig.patch_type,
                       choices=["goal", "start", "zero", "random"])
    parser.add_argument("--output-dir", type=str, default=TeleportConfig.output_dir)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=TeleportConfig.wandb_project)

    args = parser.parse_args()

    return TeleportConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        data_root=args.data_root,
        maze_size=args.maze_size,
        num_samples=args.num_samples,
        intervention_tick=args.intervention_tick,
        num_neurons_to_patch=args.num_neurons,
        patch_type=args.patch_type,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb and WANDB_AVAILABLE,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    config = parse_args()
    run_experiment(config)
