"""
Scaling Comparison Chart: Medium vs Large Maze Results

This script creates visualizations comparing how CTM's position-encoding
behavior scales with maze complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent

# Data from experiments
# Medium maze results (from observations.md - Experiment 6b)
medium_data = {
    'maze_size': 'Medium (39x39)',
    'maze_pixels': 39,
    'position_mean': 0.0100,  # 1.00%
    'position_std': 0.0,  # Not available, approximate
    'random_mean': 0.0021,  # 0.21%
    'random_std': 0.0,
    'cohens_d': 2.53,
    'p_value': 0.0033,
    'effect_ratio': 4.8,
    # Probe data
    'top_neuron_score': 0.290,
    'top_10_mean_score': 0.220,
    'percentile_95': 0.104,
}

# Large maze results (from robust_control_large.npz)
large_data = {
    'maze_size': 'Large (59x59)',
    'maze_pixels': 59,
    'position_mean': 0.0265,  # 2.65%
    'position_std': 0.0037,
    'random_mean': 0.0082,  # 0.82%
    'random_std': 0.0051,
    'cohens_d': 4.14,
    'p_value': 7.75e-08,
    'effect_ratio': 3.26,
    # Probe data
    'top_neuron_score': 0.594,
    'top_10_mean_score': 0.500,
    'percentile_95': 0.442,
}

# Try to load actual data from npz files if available
try:
    large_npz = np.load(OUTPUT_DIR / 'large' / 'robust_control_large.npz', allow_pickle=True)
    large_data['position_mean'] = float(large_npz['position_mean'])
    large_data['position_std'] = float(large_npz['position_std'])
    large_data['random_mean'] = float(large_npz['random_mean'])
    large_data['random_std'] = float(large_npz['random_std'])
    large_data['cohens_d'] = float(large_npz['cohens_d'])
    large_data['p_value'] = float(large_npz['p_value'])
    print("Loaded large maze data from npz")
except Exception as e:
    print(f"Using hardcoded large maze data: {e}")

try:
    medium_npz = np.load(OUTPUT_DIR / 'robust' / 'control' / 'robust_control_medium.npz', allow_pickle=True)
    medium_data['position_mean'] = float(medium_npz['position_mean'])
    medium_data['position_std'] = float(medium_npz['position_std'])
    medium_data['random_mean'] = float(medium_npz['random_mean'])
    medium_data['random_std'] = float(medium_npz['random_std'])
    medium_data['cohens_d'] = float(medium_npz['cohens_d'])
    medium_data['p_value'] = float(medium_npz['p_value'])
    print("Loaded medium maze data from npz")
except Exception as e:
    print(f"Using hardcoded medium maze data: {e}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('CTM Position Encoding: Scaling with Maze Complexity', fontsize=16, fontweight='bold')

# Color scheme
POSITION_COLOR = '#e74c3c'  # Red
RANDOM_COLOR = '#3498db'    # Blue
MEDIUM_COLOR = '#9b59b6'    # Purple
LARGE_COLOR = '#27ae60'     # Green

maze_sizes = ['Medium\n(39×39)', 'Large\n(59×59)']
x_positions = np.array([0, 1])

# ============================================
# Plot 1: Position vs Random Move Difference
# ============================================
ax1 = fig.add_subplot(2, 2, 1)

width = 0.35
position_means = [medium_data['position_mean'] * 100, large_data['position_mean'] * 100]
random_means = [medium_data['random_mean'] * 100, large_data['random_mean'] * 100]
position_stds = [medium_data['position_std'] * 100, large_data['position_std'] * 100]
random_stds = [medium_data['random_std'] * 100, large_data['random_std'] * 100]

bars1 = ax1.bar(x_positions - width/2, position_means, width,
                label='Position Neurons', color=POSITION_COLOR,
                yerr=position_stds, capsize=5, alpha=0.8)
bars2 = ax1.bar(x_positions + width/2, random_means, width,
                label='Random Neurons', color=RANDOM_COLOR,
                yerr=random_stds, capsize=5, alpha=0.8)

ax1.set_ylabel('Move Difference (%)', fontsize=12)
ax1.set_title('Intervention Effect by Maze Size', fontsize=14)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(maze_sizes)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars1, position_means):
    ax1.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
for bar, val in zip(bars2, random_means):
    ax1.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

# ============================================
# Plot 2: Cohen's d Effect Size
# ============================================
ax2 = fig.add_subplot(2, 2, 2)

cohens_d_values = [medium_data['cohens_d'], large_data['cohens_d']]
colors = [MEDIUM_COLOR, LARGE_COLOR]

bars = ax2.bar(x_positions, cohens_d_values, color=colors, alpha=0.8, width=0.5)
ax2.axhline(y=0.8, color='gray', linestyle='--', label='Large effect threshold (0.8)')
ax2.axhline(y=0.5, color='gray', linestyle=':', label='Medium effect threshold (0.5)')

ax2.set_ylabel("Cohen's d", fontsize=12)
ax2.set_title('Effect Size Scaling', fontsize=14)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(maze_sizes)
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, cohens_d_values):
    ax2.annotate(f'd = {val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')

# ============================================
# Plot 3: Scaling Trends (Line Plot)
# ============================================
ax3 = fig.add_subplot(2, 2, 3)

maze_pixels = [medium_data['maze_pixels'], large_data['maze_pixels']]

# Plot multiple metrics on same axes with different y-scales
ax3_twin = ax3.twinx()

line1, = ax3.plot(maze_pixels, position_means, 'o-', color=POSITION_COLOR,
                  linewidth=2, markersize=10, label='Position Effect (%)')
line2, = ax3.plot(maze_pixels, random_means, 's--', color=RANDOM_COLOR,
                  linewidth=2, markersize=10, label='Random Effect (%)')
line3, = ax3_twin.plot(maze_pixels, cohens_d_values, '^-', color='#f39c12',
                       linewidth=2, markersize=10, label="Cohen's d")

ax3.set_xlabel('Maze Size (pixels)', fontsize=12)
ax3.set_ylabel('Move Difference (%)', fontsize=12, color='black')
ax3_twin.set_ylabel("Cohen's d", fontsize=12, color='#f39c12')
ax3.set_title('Scaling Trends with Maze Complexity', fontsize=14)

# Combine legends
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left')
ax3.grid(alpha=0.3)
ax3.set_xticks(maze_pixels)
ax3.set_xticklabels(['39×39\n(Medium)', '59×59\n(Large)'])

# ============================================
# Plot 4: Summary Statistics Table
# ============================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# Create table data
table_data = [
    ['Metric', 'Medium (39×39)', 'Large (59×59)', 'Change'],
    ['Position Effect', f'{medium_data["position_mean"]*100:.2f}%', f'{large_data["position_mean"]*100:.2f}%', f'+{((large_data["position_mean"]/medium_data["position_mean"])-1)*100:.0f}%'],
    ['Random Effect', f'{medium_data["random_mean"]*100:.2f}%', f'{large_data["random_mean"]*100:.2f}%', f'+{((large_data["random_mean"]/medium_data["random_mean"])-1)*100:.0f}%'],
    ["Cohen's d", f'{medium_data["cohens_d"]:.2f}', f'{large_data["cohens_d"]:.2f}', f'+{((large_data["cohens_d"]/medium_data["cohens_d"])-1)*100:.0f}%'],
    ['P-value', f'{medium_data["p_value"]:.2e}', f'{large_data["p_value"]:.2e}', 'More sig.'],
    ['Effect Ratio', f'{medium_data["effect_ratio"]:.1f}x', f'{large_data["effect_ratio"]:.1f}x', f'{((large_data["effect_ratio"]/medium_data["effect_ratio"])-1)*100:+.0f}%'],
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Style data rows
for row in range(1, len(table_data)):
    for col in range(4):
        if col == 3:  # Change column
            table[(row, col)].set_facecolor('#e8f6e8' if '+' in table_data[row][col] else '#fff3e8')
        elif col == 2:  # Large column
            table[(row, col)].set_facecolor('#e8f4e8')
        elif col == 1:  # Medium column
            table[(row, col)].set_facecolor('#f4e8f4')

ax4.set_title('Scaling Summary', fontsize=14, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = OUTPUT_DIR / 'scaling_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved scaling comparison to: {output_path}")

# Also save a simplified version for the observations
plt.figure(figsize=(10, 6))

# Simple bar chart comparing key metrics
metrics = ['Position\nEffect (%)', "Cohen's d", 'Effect\nRatio']
medium_values = [medium_data['position_mean']*100, medium_data['cohens_d'], medium_data['effect_ratio']]
large_values = [large_data['position_mean']*100, large_data['cohens_d'], large_data['effect_ratio']]

x = np.arange(len(metrics))
width = 0.35

fig2, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, medium_values, width, label='Medium (39×39)', color=MEDIUM_COLOR, alpha=0.8)
bars2 = ax.bar(x + width/2, large_values, width, label='Large (59×59)', color=LARGE_COLOR, alpha=0.8)

ax.set_ylabel('Value', fontsize=12)
ax.set_title('CTM Position Encoding Scales with Maze Complexity', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
simple_path = OUTPUT_DIR / 'scaling_simple.png'
plt.savefig(simple_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved simple comparison to: {simple_path}")

plt.close('all')

# ============================================
# Create comprehensive scaling chart with probe data
# ============================================
fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('CTM Mechanistic Interpretability: Scaling with Maze Complexity', fontsize=16, fontweight='bold')

maze_sizes = ['Medium\n(39×39)', 'Large\n(59×59)']
x_pos = np.array([0, 1])

# Plot 1: Intervention Effect (Position vs Random)
ax = axes[0, 0]
width = 0.35
position_means = [medium_data['position_mean'] * 100, large_data['position_mean'] * 100]
random_means = [medium_data['random_mean'] * 100, large_data['random_mean'] * 100]
bars1 = ax.bar(x_pos - width/2, position_means, width, label='Position Neurons', color=POSITION_COLOR, alpha=0.8)
bars2 = ax.bar(x_pos + width/2, random_means, width, label='Random Neurons', color=RANDOM_COLOR, alpha=0.8)
ax.set_ylabel('Move Difference (%)', fontsize=11)
ax.set_title('Intervention Effect', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(maze_sizes)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, position_means):
    ax.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Plot 2: Cohen's d
ax = axes[0, 1]
cohens_values = [medium_data['cohens_d'], large_data['cohens_d']]
bars = ax.bar(x_pos, cohens_values, color=[MEDIUM_COLOR, LARGE_COLOR], alpha=0.8, width=0.5)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='Large effect (0.8)')
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('Effect Size', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(maze_sizes)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cohens_values):
    ax.annotate(f'd={val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Place Cell Scores (Probe Results)
ax = axes[0, 2]
top_scores = [medium_data['top_neuron_score'], large_data['top_neuron_score']]
top10_scores = [medium_data['top_10_mean_score'], large_data['top_10_mean_score']]
width = 0.35
bars1 = ax.bar(x_pos - width/2, top_scores, width, label='Top Neuron', color='#e67e22', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, top10_scores, width, label='Top 10 Mean', color='#16a085', alpha=0.8)
ax.set_ylabel('Place Cell Score', fontsize=11)
ax.set_title('Place Cell Strength (Probe)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(maze_sizes)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, top_scores):
    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Plot 4: Scaling Trend Lines
ax = axes[1, 0]
maze_pixels = [39, 59]
ax.plot(maze_pixels, position_means, 'o-', color=POSITION_COLOR, linewidth=2, markersize=10, label='Position Effect (%)')
ax.plot(maze_pixels, random_means, 's--', color=RANDOM_COLOR, linewidth=2, markersize=10, label='Random Effect (%)')
ax.set_xlabel('Maze Size (pixels)', fontsize=11)
ax.set_ylabel('Move Difference (%)', fontsize=11)
ax.set_title('Effect Scaling Trend', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(maze_pixels)

# Plot 5: Percentage Change Summary
ax = axes[1, 1]
metrics_change = ['Position\nEffect', "Cohen's d", 'Top\nNeuron', 'Top 10\nMean']
pct_changes = [
    ((large_data['position_mean'] / medium_data['position_mean']) - 1) * 100,
    ((large_data['cohens_d'] / medium_data['cohens_d']) - 1) * 100,
    ((large_data['top_neuron_score'] / medium_data['top_neuron_score']) - 1) * 100,
    ((large_data['top_10_mean_score'] / medium_data['top_10_mean_score']) - 1) * 100,
]
colors_change = ['#27ae60' if p > 0 else '#e74c3c' for p in pct_changes]
bars = ax.bar(range(len(metrics_change)), pct_changes, color=colors_change, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('% Change (Medium → Large)', fontsize=11)
ax.set_title('Scaling Summary', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(metrics_change)))
ax.set_xticklabels(metrics_change, fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, pct_changes):
    ax.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# Plot 6: Summary Table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Metric', 'Medium', 'Large', 'Change'],
    ['Position Effect', f'{medium_data["position_mean"]*100:.2f}%', f'{large_data["position_mean"]*100:.2f}%', f'+{((large_data["position_mean"]/medium_data["position_mean"])-1)*100:.0f}%'],
    ["Cohen's d", f'{medium_data["cohens_d"]:.2f}', f'{large_data["cohens_d"]:.2f}', f'+{((large_data["cohens_d"]/medium_data["cohens_d"])-1)*100:.0f}%'],
    ['Top Neuron Score', f'{medium_data["top_neuron_score"]:.3f}', f'{large_data["top_neuron_score"]:.3f}', f'+{((large_data["top_neuron_score"]/medium_data["top_neuron_score"])-1)*100:.0f}%'],
    ['Top 10 Mean', f'{medium_data["top_10_mean_score"]:.3f}', f'{large_data["top_10_mean_score"]:.3f}', f'+{((large_data["top_10_mean_score"]/medium_data["top_10_mean_score"])-1)*100:.0f}%'],
    ['P-value', f'{medium_data["p_value"]:.2e}', f'{large_data["p_value"]:.2e}', 'More sig.'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.28, 0.22, 0.22, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')
ax.set_title('Complete Comparison', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
comprehensive_path = OUTPUT_DIR / 'scaling_comprehensive.png'
plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved comprehensive scaling chart to: {comprehensive_path}")

plt.close('all')
print("\nDone! All charts created successfully.")
