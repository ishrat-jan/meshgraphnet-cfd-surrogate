# MeshGraphNet for CFD Surrogate Modeling

A PyTorch implementation of [**MeshGraphNet**](https://arxiv.org/abs/2010.03409) (Pfaff et al., ICLR 2021) — a graph neural network that learns to simulate mesh-based physics by passing messages over the simulation mesh itself.

Trained on DeepMind's **cylinder_flow** dataset (2D incompressible flow around a cylinder), the model achieves stable **300-step autoregressive rollouts** at **~59 steps/s** on a single NVIDIA T4 GPU, with a mean velocity error of **0.13** at the final frame.

---

## Why This Matters

Traditional CFD solvers (finite-element, finite-volume) are accurate but computationally expensive — a single simulation can take hours to days. Neural surrogates like MeshGraphNet learn to approximate the solver's output in milliseconds, enabling:

- **Real-time digital twins** for engineering systems
- **Design-space exploration** (thousands of what-if scenarios)
- **Hybrid solvers** where the neural model provides fast coarse predictions that a classical solver refines

This is directly relevant to AI-for-science applications in climate modeling, weather prediction, and ocean/atmosphere simulation, where mesh-based PDEs are the core computational bottleneck.

---

## Architecture

MeshGraphNet uses an **Encode → Process → Decode** paradigm operating on the simulation mesh:

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   Encoder   │     │   Processor (×15)    │     │   Decoder   │
│             │     │                      │     │             │
│ Node MLP    │────▶│  GraphNetBlock:      │────▶│ Node MLP    │
│ Edge MLP    │     │  • Edge update MLP   │     │ (latent→2D) │
│             │     │  • Aggregation       │     │             │
│ (features   │     │  • Node update MLP   │     │ (velocity   │
│  → latent)  │     │  • Residual skip     │     │  increment) │
└─────────────┘     └──────────────────────┘     └─────────────┘
```

**Key design choices:**

| Component | Detail |
|---|---|
| **Node features** | Current velocity (2D) + one-hot node type (9 classes: interior, wall, inflow, outflow, etc.) |
| **Edge features** | Relative position vector + Euclidean distance (3D total) |
| **Latent dimension** | 128 |
| **Message-passing blocks** | 15 |
| **Normalization** | Running mean/std (online accumulation during training) |
| **Training noise** | Gaussian noise (σ = 3×10⁻⁴) injected into input velocities to stabilize autoregressive rollout |
| **Loss** | MSE on velocity increments, computed only on interior (non-boundary) nodes |

---

## Dataset

[**cylinder_flow**](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) from DeepMind's MeshGraphNets release:

- 2D incompressible flow around a cylinder (Re ≈ varies across trajectories)
- **1,923 nodes**, **3,612 triangular cells**, **5,535 edges** per mesh
- **600 time steps** per trajectory
- Stored as TFRecords; parsed with TensorFlow and converted to PyTorch tensors for training

The dataset is automatically downloaded from Google Cloud Storage when running the notebook.

---

## Results

### Rollout Visualization (Frame 200 of 300)

| Prediction (Rollout) | Ground Truth |
|---|---|
| Model's autoregressive output after 200 steps | Direct simulation reference |

### Error Curve

The mean per-node velocity error stays low through the first ~150 steps and grows gradually — characteristic of autoregressive rollout error accumulation, but remains stable (no blow-up) through all 300 steps.

| Metric | Value |
|---|---|
| Rollout steps | 300 |
| Inference speed | ~59 steps/s (T4 GPU) |
| Final-frame mean velocity error | 0.13 |
| Training time | ~5.8 hours (2× T4, Kaggle) |
| Training epochs | 25 |

---

## Repository Structure

```
meshgraphnet-cfd-surrogate/
├── meshgraphnet_cylinder_flow.py   # Self-contained training + evaluation script
├── notebook/
│   └── cylinder_flow_meshnet.ipynb # Full Kaggle notebook with outputs
├── results/
│   └── mgn_result.png              # Rollout vs ground truth + error curve
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Train + Evaluate

```bash
python meshgraphnet_cylinder_flow.py
```

This will:
1. Download the cylinder_flow dataset (~1.3 GB) from Google Cloud Storage
2. Build mesh graphs (nodes, edges, features) from TFRecord trajectories
3. Train MeshGraphNet for 25 epochs (~5–6 hours on a T4)
4. Run a 300-step autoregressive rollout on a test trajectory
5. Save `mgn_result.png` with prediction vs ground truth panels and an error curve

### Run on Kaggle

The easiest way to reproduce is to run the notebook directly on Kaggle with GPU acceleration enabled. The notebook is self-contained and downloads all data automatically.

---

## Implementation Notes

- **TensorFlow is used only for data loading** (parsing DeepMind's TFRecord format). All modeling and training is in PyTorch.
- **Boundary pinning**: during rollout, boundary node velocities are reset to ground truth at each step — only interior node predictions are autoregressive. This follows the original paper's protocol.
- **Noise injection** during training is critical for rollout stability. Without it, small single-step errors compound rapidly and the rollout diverges within ~50 steps.
- **Running normalization** (online mean/std accumulation) is used instead of batch/layer norm, following the original implementation.

---

## References

- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. (2021). *Learning Mesh-Based Simulation with Graph Networks*. ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., & Battaglia, P. (2020). *Learning to Simulate Complex Physics with Graph Networks*. ICML 2020. [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)
- DeepMind MeshGraphNets dataset: [github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

---

## License

MIT
