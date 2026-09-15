"""
MeshGraphNet for CFD Surrogate Modeling
========================================
PyTorch implementation of MeshGraphNet (Pfaff et al., ICLR 2021) trained on
DeepMind's cylinder_flow dataset.

Learns to predict velocity fields on irregular triangular meshes via
message-passing graph neural networks, achieving stable 300-step
autoregressive rollouts.

Reference: https://arxiv.org/abs/2010.03409
"""

import os
import json
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
LATENT_DIM = 128          # Latent feature dimension for all MLPs
N_BLOCKS = 15             # Number of message-passing blocks
EPOCHS = 25               # Training epochs
STEPS_PER_TRAJ = 50       # Random time-step samples per trajectory per epoch
NOISE_STD = 3e-4          # Gaussian noise injected into input velocities
LR = 1e-4                 # Initial learning rate (Adam)
ROLLOUT_STEPS = 300       # Autoregressive rollout length at evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download dataset
# ──────────────────────────────────────────────────────────────────────────────
def download_dataset(data_dir="cf"):
    """Download DeepMind's cylinder_flow dataset from Google Cloud Storage."""
    base_url = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/"
    os.makedirs(data_dir, exist_ok=True)
    for fname in ["meta.json", "train.tfrecord", "test.tfrecord"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            subprocess.run(
                ["wget", "-q", "-O", fpath, base_url + fname],
                check=True,
            )
        print(f"  {fname}: {os.path.getsize(fpath) // 1024} KB")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TFRecord parsing (TensorFlow used only here)
# ──────────────────────────────────────────────────────────────────────────────
def load_tfrecord_dataset(path, meta):
    """Parse a TFRecord file using the metadata schema."""
    import tensorflow as tf

    def parse_fn(proto):
        feats = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
        parsed = tf.io.parse_single_example(proto, feats)
        out = {}
        for key, field in meta["features"].items():
            raw = tf.io.decode_raw(parsed[key].values, getattr(tf, field["dtype"]))
            raw = tf.reshape(raw, field["shape"])
            if field["type"] == "static":
                raw = tf.tile(raw, [meta["trajectory_length"], 1, 1])
            out[key] = raw
        return out

    return tf.data.TFRecordDataset(path).map(parse_fn)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Graph construction
# ──────────────────────────────────────────────────────────────────────────────
def build_graph(traj, device):
    """
    Convert a single trajectory dict into PyTorch tensors for training.

    Returns:
        vel_T   – velocity field at all time steps       [T, N, 2]
        edge_idx – bidirectional edge index               [2, E]
        edge_feat – relative position + distance          [E, 3]
        node_oh  – one-hot encoded node type              [N, 9]
        mask     – boolean mask for interior (moving) nodes [N]
        pos      – 2D node positions                      [N, 2]
        cells    – triangle cell indices (for visualization) [C, 3]
    """
    pos = torch.tensor(traj["mesh_pos"][0].numpy())
    cells = traj["cells"][0].numpy().astype("int64")
    ntype = torch.tensor(traj["node_type"][0].numpy()).squeeze(-1).long()
    vel_T = torch.tensor(traj["velocity"].numpy()).to(device)

    # Build bidirectional edges from triangular cells
    c = torch.tensor(cells)
    edges = torch.cat([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]], dim=0)
    edges = torch.cat([edges, edges.flip(1)], dim=0).unique(dim=0)
    edge_idx = edges.t().contiguous().to(device)

    # Edge features: relative position + Euclidean distance
    src, dst = edge_idx
    pos = pos.to(device)
    rel = pos[src] - pos[dst]
    edge_feat = torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)

    # Node features: one-hot node type
    node_oh = torch.nn.functional.one_hot(ntype.to(device), 9).float()

    # Interior node mask (nodes with non-trivial velocity variation)
    mask = vel_T.std(0).norm(dim=1) > 1e-8

    return vel_T, edge_idx, edge_feat, node_oh, mask, pos, cells


# ──────────────────────────────────────────────────────────────────────────────
# 4. Model components
# ──────────────────────────────────────────────────────────────────────────────
def make_mlp(in_dim, out_dim, hidden_dim=LATENT_DIM):
    """Two-layer MLP with ReLU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class GraphNetBlock(nn.Module):
    """Single message-passing block with residual connections."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Edge update: concatenate [edge, sender, receiver] → updated edge
        self.edge_mlp = make_mlp(3 * latent_dim, latent_dim)
        # Node update: concatenate [node, aggregated messages] → updated node
        self.node_mlp = make_mlp(2 * latent_dim, latent_dim)

    def forward(self, h_node, h_edge, edge_index):
        src, dst = edge_index

        # Edge update
        e_input = torch.cat([h_edge, h_node[src], h_node[dst]], dim=1)
        e_updated = self.edge_mlp(e_input)

        # Aggregate messages at destination nodes
        agg = torch.zeros_like(h_node).index_add_(0, dst, e_updated)

        # Node update
        v_input = torch.cat([h_node, agg], dim=1)
        v_updated = self.node_mlp(v_input)

        # Residual connections
        return h_node + v_updated, h_edge + e_updated


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet: Encode-Process-Decode architecture for mesh-based simulation.

    Encoder:  project node/edge features into latent space
    Processor: N message-passing blocks with residual connections
    Decoder:  project latent node features to velocity increment (2D)
    """

    def __init__(self, n_node_feat, n_edge_feat, latent_dim=LATENT_DIM,
                 n_blocks=N_BLOCKS):
        super().__init__()
        self.node_encoder = make_mlp(n_node_feat, latent_dim)
        self.edge_encoder = make_mlp(n_edge_feat, latent_dim)
        self.processor = nn.ModuleList(
            [GraphNetBlock(latent_dim) for _ in range(n_blocks)]
        )
        self.decoder = make_mlp(latent_dim, 2)  # Output: 2D velocity increment

    def forward(self, node_feats, edge_feats, edge_index):
        h_node = self.node_encoder(node_feats)
        h_edge = self.edge_encoder(edge_feats)

        for block in self.processor:
            h_node, h_edge = block(h_node, h_edge, edge_index)

        return self.decoder(h_node)


class RunningNorm(nn.Module):
    """Online running mean/std normalization (replaces batch/layer norm)."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("sum_sq", torch.zeros(size))
        self.register_buffer("count", torch.zeros(1))

    def _stats(self):
        mean = self.sum / self.count.clamp(min=1)
        std = (self.sum_sq / self.count.clamp(min=1) - mean * mean).clamp(min=1e-8).sqrt()
        return mean, std

    def forward(self, x, accumulate=True):
        if accumulate:
            self.sum += x.sum(0)
            self.sum_sq += (x * x).sum(0)
            self.count += x.shape[0]
        mean, std = self._stats()
        return (x - mean) / std

    def inverse(self, x):
        mean, std = self._stats()
        return x * std + mean


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(model, node_norm, edge_norm, out_norm, meta, device):
    """Train MeshGraphNet on the full training dataset."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    loss_fn = nn.MSELoss()

    train_ds = load_tfrecord_dataset("cf/train.tfrecord", meta)

    print(f"Training for {EPOCHS} epochs on {device}...")
    print(f"  Latent dim: {LATENT_DIM}, Blocks: {N_BLOCKS}, Noise: {NOISE_STD}")
    print()

    for epoch in range(EPOCHS):
        last_loss = 0.0
        for traj in train_ds:
            vel_T, edge_idx, edge_feat, node_oh, mask, _, _ = build_graph(traj, device)

            for _ in range(STEPS_PER_TRAJ):
                # Sample a random time step
                t = np.random.randint(0, vel_T.shape[0] - 1)

                # Add noise to input velocity (critical for rollout stability)
                v = vel_T[t] + torch.randn(vel_T.shape[1], 2, device=device) * NOISE_STD

                # Normalize inputs
                nf = node_norm(torch.cat([v, node_oh], dim=1))
                ef = edge_norm(edge_feat)

                # Target: normalized velocity increment
                target = out_norm(vel_T[t + 1] - v)

                # Forward pass + loss on interior nodes only
                pred = model(nf, ef, edge_idx)
                loss = loss_fn(pred[mask], target[mask])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                last_loss = loss.item()

        print(f"  Epoch {epoch:3d}   loss {last_loss:.6f}")

    # Save trained weights
    torch.save(model.state_dict(), "mgn.pt")
    print("\nSaved model weights to mgn.pt")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Autoregressive rollout
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout(model, node_norm, edge_norm, out_norm, meta, device):
    """Run autoregressive rollout on a test trajectory."""
    test_ds = load_tfrecord_dataset("cf/test.tfrecord", meta)
    traj = next(iter(test_ds))
    vel_T, edge_idx, edge_feat, node_oh, mask, pos, cells = build_graph(traj, device)

    model.eval()
    t0 = time.time()

    v = vel_T[0].clone()
    predictions = [v.clone()]

    for t in range(1, ROLLOUT_STEPS):
        nf = node_norm(torch.cat([v, node_oh], dim=1), accumulate=False)
        ef = edge_norm(edge_feat, accumulate=False)

        # Predict velocity increment and update
        v = v + out_norm.inverse(model(nf, ef, edge_idx))

        # Pin boundary nodes to ground truth (following the paper's protocol)
        v[~mask] = vel_T[t][~mask]

        predictions.append(v.clone())

    predictions = torch.stack(predictions)
    elapsed = time.time() - t0

    print(f"\nRolled {ROLLOUT_STEPS} steps in {elapsed:.2f}s "
          f"({ROLLOUT_STEPS / elapsed:.0f} steps/s)")

    return predictions, vel_T[:ROLLOUT_STEPS], pos, cells, mask


# ──────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ──────────────────────────────────────────────────────────────────────────────
def visualize(predictions, ground_truth, pos, cells):
    """Plot rollout vs ground truth velocity fields and error curve."""
    # Per-frame mean node error
    error = (predictions - ground_truth).norm(dim=2).mean(dim=1).cpu().numpy()

    # Triangulation for plotting
    tri = mtri.Triangulation(pos[:, 0].cpu(), pos[:, 1].cpu(), cells)

    # Pick a frame to visualize
    frame = min(200, ROLLOUT_STEPS - 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    # Predicted velocity magnitude
    pred_speed = predictions[frame].norm(dim=1).cpu().numpy()
    axes[0].tripcolor(tri, pred_speed, shading="gouraud")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Prediction (rollout), frame {frame}")

    # Ground truth velocity magnitude
    gt_speed = ground_truth[frame].norm(dim=1).cpu().numpy()
    axes[1].tripcolor(tri, gt_speed, shading="gouraud")
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Ground truth, frame {frame}")

    # Error over time
    axes[2].plot(error)
    axes[2].set_xlabel("Rollout step")
    axes[2].set_ylabel("Mean velocity error")
    axes[2].set_title("Rollout error accumulation")

    plt.tight_layout()
    plt.savefig("mgn_result.png", dpi=150)
    plt.show()

    print(f"Final-frame mean error: {error[-1]:.4f}")
    print("Saved visualization to mgn_result.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MeshGraphNet — CFD Surrogate for Cylinder Flow")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    # Download data
    download_dataset()

    # Load metadata
    with open("cf/meta.json") as f:
        meta = json.load(f)

    # Initialize model and normalizers
    # Node features: velocity (2) + one-hot node type (9) = 11
    # Edge features: relative position (2) + distance (1) = 3
    model = MeshGraphNet(n_node_feat=11, n_edge_feat=3).to(DEVICE)
    node_norm = RunningNorm(11).to(DEVICE)
    edge_norm = RunningNorm(3).to(DEVICE)
    out_norm = RunningNorm(2).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}\n")

    # Train
    train(model, node_norm, edge_norm, out_norm, meta, DEVICE)

    # Rollout + visualize
    predictions, ground_truth, pos, cells, mask = rollout(
        model, node_norm, edge_norm, out_norm, meta, DEVICE
    )
    visualize(predictions, ground_truth, pos, cells)


if __name__ == "__main__":
    main()
