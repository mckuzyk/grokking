from pathlib import Path
import copy
import math
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from grokking.model import Transformer, MultiHeadAttention
from grokking.config import TransformerConfig
from grokking.data import ModPDataset
from scipy.signal import find_peaks


rc = Path().home() / ".config/matplotlib/matplotlibrc"
if rc.exists():
    plt.style.use(rc)

_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_PRIMARY = _colors[0]
COLOR_SECONDARY = _colors[1]
COLOR_HIGHLIGHT = _colors[2]


class MultiHeadAttentionWithCache(MultiHeadAttention):
    """
    Class used to pick out attention scores on forward pass for analysis
    """

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dim (d_model)

        # Split into n_heads with view and move the head dimension left one so all batch dimensions come first
        # Resulting tensors have shape (B, n_heads, T, d_head == d_model // n_heads)
        q = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))

        # Softmax is applied to norm weights of a given query dotted w/ all keys, so dim=-1
        att_scores = F.softmax(att, dim=-1)

        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        att = att_scores @ v

        # Concat
        att = att.transpose(1, 2).reshape(B, T, C)
        att = self.o_proj(att)

        return att, att_scores


class TransformerWithCache(Transformer):
    def __init__(self, cfg, attention_cls=MultiHeadAttentionWithCache):
        self.attention_cls = attention_cls
        super().__init__(cfg)

    def forward(self, t):
        cache = {}

        positions = torch.arange(3, device=t.device)
        x = self.w_embed(t) + self.w_pos_emb(positions)
        cache["embed"] = x.detach()

        for i in range(self.config.n_blocks):
            att, att_scores = self.attention_heads[i](x)
            cache[f"att_scores_{i}"] = att_scores
            x = x + att
            cache[f"post_att_{i}"] = x.detach()
            x = x + self.mlps[i](x)
            cache[f"post_mlp_{i}"] = x.detach()
        # Only return vectors of "=" token
        # Since x: (batch, sequence, embedding), and sequence = 3,
        # the return value of interest is:
        return self.w_unembed(x)[:, 2, :], cache


class Results:
    def __init__(self, path):
        self.path = Path(path)
        self.cfg = None
        self.model = None
        self.checkpoints = None
        self.current_checkpoint = None
        self.current_epoch = None

        self._load_run()
        # Load last checkpoint by default
        self.load_checkpoint(-1)

    def _load_run(self):
        with open(self.path / "metrics.json", "r") as f:
            self.metrics = json.load(f)
        self.cfg = TransformerConfig.load(self.path / "model_config.json")
        self.model = TransformerWithCache(self.cfg)
        self.checkpoints = sorted(
            list(self.path.glob("*.pt")),
            key=lambda x: int(re.search(r"(\d+)", x.stem).group(1)),
        )

    def load_checkpoint(self, idx):
        self.current_checkpoint = self.checkpoints[idx]
        self.current_epoch = int(
            re.search(r"(\d+)", self.current_checkpoint.stem).group(1)
        )
        self.model.load_state_dict(torch.load(self.checkpoints[idx], weights_only=True))

    @property
    def n_checkpoints(self):
        return len(self.checkpoints)


def attention_scores(r: Results, block: int = 0):
    """
    Get the attention scores for all heads accross all inputs (a,b).
    block is zero-indexed attention block. Default model only has 1 block.
    return value has t.size() = (P**2, n_heads, n_context, n_context) where
    P**2 is the number of (a,b) pairs computed. Since the P**2 pairs are ordered
    by meshgrid, the results here can be reshaped to (113,113) to get value a varying
    along 1 axis, and b along the other.
    """
    data = ModPDataset(r.cfg.P)
    inputs, _ = data[:]
    with torch.no_grad():
        out, cache = r.model.forward(inputs)
    att_scores = cache[f"att_scores_{block}"]
    return att_scores


def frequency_contribution(logits, k, P):
    fft_logits = np.fft.fft(logits)
    component = np.zeros(P, dtype=complex)
    component[k] = fft_logits[k]
    component[P - k] = fft_logits[P - k]  # conjugate pair
    return np.real(np.fft.ifft(component))


def find_fft_peaks(fft_marginal, distance=2, height=0.2):
    normed = fft_marginal / fft_marginal.max()
    peaks, _ = find_peaks(normed, distance=distance, height=height)
    return peaks[peaks > 0]  # exclude DC


def softmax(logits):
    exp = np.exp(logits - logits.max())  # Subtract max for numerical stabilitiy
    return exp / exp.sum()


if __name__ == "__main__":
    from sklearn.decomposition import PCA

    path = Path("runs") / "baseline_weight_init_seed_137"
    r = Results(path)

    dset = ModPDataset(P=113)
    test_inputs, test_labels = dset[:]
    nums = test_inputs.detach().numpy().reshape(113, 113, 3)[:, 0, 0]
    print(nums)

    w_embed = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft = np.fft.fft(w_embed, axis=0)
    print(w_embed_fft.shape)

    peaks = find_fft_peaks((np.abs(w_embed_fft) ** 2).mean(axis=1)[: r.cfg.P // 2])
    print(peaks)

    with torch.no_grad():
        vecs = r.model.w_embed(test_inputs).numpy()
    print(vecs.shape)
    vecs = vecs.reshape(113, 113, 3, 128)[:, 0, 0, :]
    print(vecs.shape)
    pca = PCA(n_components=10)
    x = pca.fit_transform(vecs)
    print(x.shape)
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.cm.get_cmap("viridis", 5)
    plt.scatter(x[:, 0], x[:, 1], alpha=0.9, marker=".", color=cmap(0))
    plt.scatter(x[:, 2], x[:, 3], alpha=0.9, marker=".", color=cmap(1))
    plt.scatter(x[:, 4], x[:, 5], alpha=0.9, marker=".", color=cmap(2))
    plt.scatter(x[:, 6], x[:, 7], alpha=0.9, marker=".", color=cmap(3))
    plt.scatter(x[:, 8], x[:, 9], alpha=0.9, marker=".", color=cmap(4))

    plt.figure()
    plt.bar(np.arange(w_embed.shape[0]), np.mean(np.abs(w_embed_fft) ** 2, axis=1))
    for peak in peaks:
        plt.axvline(peak)
    plt.xlim(0, r.cfg.P // 2)

    a, b = (50, 37)
    t = torch.tensor([[a, b, 113]])
    with torch.no_grad():
        logits, _ = r.model(t)
    logits = logits.numpy().squeeze()
    print(logits)
    cons = []
    for peak in peaks:
        freq_con = frequency_contribution(logits, peak, r.cfg.P)
        cons.append(freq_con)

    fig2, ax2 = plt.subplots(3, 1)
    ax2[0].plot(np.array(cons).sum(axis=0))
    ax2[0].plot(logits, "o")
    ax2[0].axvline(87, color=COLOR_HIGHLIGHT, linestyle="--", linewidth=1.5)

    con = frequency_contribution(logits, 1, r.cfg.P)
    for k in range(1, 113 // 2):
        con += frequency_contribution(logits, k, r.cfg.P)
    ax2[1].plot(con)
    ax2[1].plot(logits, "o")

    #    ax2[2].bar(np.arange(len(logits)), softmax(logits))
    ax2[2].plot(softmax(logits), ".")

    plt.show()
