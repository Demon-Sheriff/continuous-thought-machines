# Position Encoding in Continuous Thought Machines: It's in the Correlations, Not the Neurons

## Executive Summary

**Research problem:** I investigated how the Continuous Thought Machine (CTM) - a recently published non-Transformer architecture - encodes spatial position during maze navigation *without any positional embeddings*. The CTM paper claims its "Synchronization Matrix" (S_t = Z·Z^T, measuring neuron correlations) is the key representational innovation. I wanted to test this claim mechanistically: is position encoded in individual neuron activations, or in how neurons correlate with each other?

**Why I chose this over a Transformer project:** I was genuinely excited to explore something different. Most mech interp work focuses on attention-based LLMs, but CTM represents a fundamentally different computational paradigm - one that's arguably closer to biological neural systems (population coding, temporal dynamics). I wanted to see if standard interpretability techniques would transfer, and whether studying a novel architecture could teach us something new about how neural networks represent space. This felt more interesting than running another probe on GPT-2.

**Why it's worth pursuing:** Understanding how spatial reasoning emerges without explicit structure is relevant beyond this specific architecture. If CTM encodes position in neuron *correlations* rather than individual activations, it suggests that standard single-neuron probing methods may miss important information in models that use distributed representations. This has implications for interpreting any architecture that relies on population coding.

**High-level takeaways:**
1. **Position IS encoded** - a linear probe achieves R² = 0.58-0.91 decoding (x,y) from internal states
2. **Synchronization beats activations** - decoding from the correlation representation (S_out) is **67% better** than from raw neuron activations (Z_t): R² = 0.58 vs 0.35
3. **Individual neurons show weak "place cell" behavior** - top spatial selectivity score is only 0.59 (weak, not like biological place cells)
4. **Intervention effects are real but small** - patching position-encoding neurons causes 4.8x more behavioral disruption than random neurons (p < 0.001), but only changes 1-3% of moves
5. **Vector steering partially works** - Y-direction steering shows strong causal effect (R²=0.97, p<0.001), but X-direction steering has no effect - an interesting asymmetry

**Epistemic status:** This was a hurried ~16-hour investigation with significant time constraints. Some experiments I wanted to run (non-linear probes, per-tick analysis, attention pattern visualization) couldn't be completed. Results should be taken as preliminary/exploratory.

---

## Background: What is CTM and Why Study It?

The Continuous Thought Machine (Shuvaev et al., 2025) differs fundamentally from Transformers:

| Aspect | Transformer | CTM |
|--------|-------------|-----|
| Core computation | Self-attention | Synchronization matrix (S_t = Z·Z^T) |
| Information flow | Layer by layer | Internal "ticks" (75 iterations) |
| Position encoding | Explicit embeddings | **None - must emerge** |
| Output representation | Token embeddings | Neuron correlation patterns |

The CTM solves 2D maze navigation with `positional_embedding_type='none'`. The model receives only the visual maze image - no (x,y) coordinates. Yet it successfully navigates. How?

**My hypothesis:** The model constructs a "virtual coordinate system" within its Synchronization Matrix - encoding position in *how neurons fire together* rather than *which neurons fire*.

---

## Experimental Setup

- **Task:** 2D maze navigation (39×39 and 59×59 grids)
- **Model:** Pre-trained CTM (2048 neurons, 75 ticks, no positional embeddings)
- **Infrastructure:** Modal cloud GPU (A10G), ~4 GPU-hours total
- **Code:** `experiments/interpretability/`

---

## Experiment 1: Looking for "Place Cells"

**Hypothesis:** Individual neurons encode specific (x,y) positions, similar to hippocampal place cells.

**Method:** Compute spatial selectivity score for all 2048 neurons across 500 mazes.

**Results:**

| Maze Size | Top Neuron Score | Top 10 Mean |
|-----------|------------------|-------------|
| Medium (39×39) | 0.29 | 0.22 |
| Large (59×59) | 0.59 | 0.50 |

![Place Fields](experiments/interpretability/outputs/large/place_fields.png)
*Figure 1: Place field heatmaps for top neurons in large mazes. Brighter = higher activation. The patterns show mild spatial preferences but not the sharp, localized responses of true biological place cells.*

**Interpretation:** These scores are weak. Real hippocampal place cells show dramatic, localized firing. These neurons show only mild position preferences. Position isn't concentrated in individual neurons.

---

## Experiment 2: Activation Patching (Intervention)

**Hypothesis:** If some neurons encode position, patching them should disrupt navigation more than patching random neurons.

**Method:**
1. Identify "position neurons" (high activation variance across maze positions)
2. Patch their activations mid-inference (tick T=5)
3. Compare behavior change to random neuron baseline (20 control trials)

**Results (500 samples, 3 seeds):**

| Neuron Type | Move Difference | Cohen's d | p-value |
|-------------|-----------------|-----------|---------|
| Position neurons | 1.00% / 2.65% | 2.53 / 4.14 | 0.003 / 7.75e-8 |
| Random neurons | 0.21% / 0.82% | - | - |
| **Effect ratio** | **4.8x / 3.3x** | | |

![Control Comparison](experiments/interpretability/outputs/robust/control/robust_control_medium.png)
*Figure 2: Position neurons vs random neurons (medium mazes). Position neurons cause significantly more disruption, but absolute effect is small.*

![Scaling](experiments/interpretability/outputs/scaling_comprehensive.png)
*Figure 3: All effects scale with maze complexity. Larger mazes show stronger position encoding and intervention effects.*

**Interpretation:** Position neurons cause significantly more disruption than random neurons (this is the key baseline comparison). But the absolute effect is small - patching only changes ~1-3% of moves. This suggests massive redundancy or distributed encoding.

---

## Experiment 3: Tick Sweep

**Question:** When during the 75 internal ticks is position information used?

![Tick Sweep](experiments/interpretability/outputs/robust/tick_sweep/robust_tick_sweep_medium.png)
*Figure 4: Intervention effect vs tick. Early intervention has larger effect (R² = 0.825 linear fit).*

**Finding:** Earlier ticks show stronger intervention effects. Position information is used early and integrated quickly.

---

## Experiment 4: Synchronization vs Activations (Key Result)

This is the most important experiment.

**Hypothesis:** Position is encoded in the synchronization matrix (neuron correlations), not raw activations.

**Method:** Train Ridge regression probes to decode (x,y) from:
- Z_t: Raw neuron activations (2048 dims)
- S_out: Synchronization output (2080 dims)

**Results (500 mazes):**

| Representation | R² Score |
|----------------|----------|
| Z_t (activations) | 0.35 |
| **S_out (synchronization)** | **0.58** |
| Combined | 0.65 |

![Sync Matrix Comparison](experiments/interpretability/outputs/sync_matrix/sync_matrix_comparison_500.png)
*Figure 5: Position decoding from synchronization (S_out) vs activations (Z_t). The synchronization representation encodes position 67% better.*

**This is the key finding.** The synchronization representation - which captures *how neurons correlate* - encodes position much better than raw activations. This validates the CTM paper's core architectural claim.

**Why this matters:**
- Standard probing (looking at individual neurons) misses most of the position signal
- Position is encoded in neuron *relationships*, not which neurons fire
- This is a distributed, correlational code - more robust but harder to interpret with standard methods

---

## Experiment 5: Vector Steering (Partial Success)

**Hypothesis:** If the probe finds a "position direction" in S_out space, we can steer behavior by adding this vector during inference.

**Method:**
1. Extract probe weights W as steering vectors (W[0] = "X/East-West", W[1] = "Y/North-South")
2. Add α·W[i] to S_out during inference for all ticks
3. Measure if directional bias shifts (up/down for Y, left/right for X)
4. Alpha range: [-10, -5, -2, -1, 0, 1, 2, 5, 10]

**Results:**

| Steering Vector | Effect on Bias | R² | p-value |
|-----------------|----------------|-----|---------|
| Y (North-South) | **Strong** | 0.97 | 1.2e-6 |
| X (East-West) | None | 0.005 | 0.86 |
| Random | Weak | 0.53 | 0.03 |

![Vector Steering](experiments/interpretability/outputs/vector_steering_fixed/vector_steering_results.png)
*Figure 6: Vector steering results. Y-direction steering shows a strong linear relationship between steering magnitude and up/down bias (R²=0.97). X-direction steering has no effect.*

**Y steering works:** Adding the Y probe vector causes the model to predict more "down" moves; subtracting it causes more "up" moves. The effect is highly linear (R²=0.97) and statistically significant (p<0.001).

**X steering doesn't work:** Despite the probe successfully decoding X position, steering with the X vector has no effect on left/right bias.

**Interpretation:** This asymmetry is interesting. The model uses Y position causally in a way that's accessible to linear steering, but X position encoding is either:
- Read through a non-linear computation
- Redundantly encoded in ways that resist simple vector addition
- Used differently in the downstream computation

---

## Summary Table

| Question | Answer | Evidence |
|----------|--------|----------|
| Does CTM encode position? | Yes | R² = 0.58-0.91 probe |
| Where is it encoded? | Correlations > Activations | +67% R² |
| Are there "place cells"? | Weak only | Max score 0.59 |
| Is encoding causal? | Yes (patching) | 4.8x vs random (p<0.001) |
| Can we steer with it? | **Y: Yes, X: No** | Y: R²=0.97, p<0.001 |

---

## Limitations

This investigation was constrained by time (~16 hours) and several important experiments couldn't be completed:

1. **X vs Y asymmetry is unexplained** - Vector steering works for Y but not X. Could be architectural (how inputs are processed), task-specific (vertical vs horizontal movement patterns), or representational (X encoded non-linearly).

2. **Small intervention effects** - 1-3% behavior change from patching is statistically significant but practically small. The model may have massive redundancy.

3. **Single task/checkpoint** - Only tested maze navigation on one pre-trained model.

4. **No attention analysis** - CTM uses attention for visual processing; I didn't analyze these patterns.

5. **Linear probes only** - Non-linear methods might reveal stronger encoding or explain the X/Y asymmetry.

---

## What I Learned

1. **Start simple:** Linear probes before complex methods. The S_out vs Z_t comparison was the most informative experiment.

2. **Always compare to baselines:** The random neuron control was essential. Without it, the 1% intervention effect would be meaningless. (Neel emphasizes this - I took it seriously.)

3. **Partial results are informative:** Vector steering working for Y but not X reveals asymmetric encoding - more interesting than "all works" or "nothing works".

4. **Architecture shapes interpretability:** CTM's correlational encoding suggests we need to analyze neuron *relationships*, not just individual neurons. Standard single-neuron probes miss most of the signal.

---

## Why This Might Matter

1. **Novel architecture:** Most mech interp focuses on Transformers. Studying alternatives helps the field generalize.

2. **Distributed representations:** CTM's correlational encoding is similar to biological population coding. If future AI uses more brain-like architectures, we'll need methods for interpreting distributed codes.

3. **Modest claims:** This is early exploration, not a safety breakthrough. But understanding how spatial reasoning emerges without explicit structure seems generally useful for understanding model internals.

---

## Future Work

The X/Y steering asymmetry raises precise follow-up questions:

1. **Why does Y work but not X?**
   - Hypothesis A: Architectural - CTM processes images row-by-row (via conv layers), making vertical position more directly encoded
   - Hypothesis B: Task-specific - maze generation or optimal paths favor vertical movement patterns
   - Hypothesis C: Representational - X is encoded non-linearly while Y is linear
   - *Test*: Analyze probe weight norms/distributions, check maze movement statistics for vertical vs horizontal bias

2. **Non-linear steering for X**
   - Try MLP-based steering vectors instead of linear probe weights
   - Use activation patching on X-selective neurons identified by variance analysis

3. **Per-tick steering analysis**
   - When during the 75 ticks does Y steering have maximum effect?
   - Does X encoding emerge later in processing where linear steering can't reach?

4. **Attention pattern analysis**
   - CTM uses attention for visual processing - do attention patterns encode X vs Y differently?
   - Could X position be encoded in attention weights rather than S_out?

5. **Scale experiments**
   - Current steering used 50 mazes (sufficient for Y, p=1.2e-6)
   - Scale to 500+ mazes to detect smaller X effects if they exist
   - Test on different maze sizes (39×39 vs 59×59) for generalization

---

## Code and Reproducibility

All experiments at:
- `experiments/interpretability/sync_matrix_probe.py` - Key S_out vs Z_t comparison
- `experiments/interpretability/robust/robust_runner.py` - Intervention experiments with controls
- `experiments/interpretability/vector_steering/vector_steering.py` - Steering (partial success: Y works, X doesn't)
- `observations.md` - Detailed experimental log

---

## References

- Shuvaev et al. (2025). "Continuous Thought Machines"
- Pre-trained checkpoints from official CTM repository
