from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from scripts import analysis as a
from scripts.analysis import Results
from sklearn.decomposition import PCA
from grokking.data import ModPDataset
import matplotlib.ticker as ticker


rc = Path().home() / ".config/matplotlib/matplotlibrc"
if rc.exists():
    plt.style.use(rc)

_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_PRIMARY = _colors[0]
COLOR_SECONDARY = _colors[1]
COLOR_HIGHLIGHT = _colors[2]


def training_dynamics(r: Results):
    """
    Generate panels for the training dynamics of a single run, including
    * train/test loss
    * train/test acc
    * weight norm
    *grad norm
    """
    with open(r.path / "metrics.json") as f:
        metrics = json.load(f)

    fig, ax = plt.subplots(2, 2, sharex=True)
    ax[0, 0].plot(metrics["train_loss"], label="train")
    ax[0, 0].plot(metrics["test_loss"], label="test")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel("CE Loss")
    ax[0, 0].legend()

    ax[0, 1].plot(metrics["train_acc"], label="train")
    ax[0, 1].plot(metrics["test_acc"], label="test")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()

    ax[1, 0].plot(metrics["weight_norm"], label="weight norm")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("Weight Norm")
    ax[1, 0].legend()

    ax[1, 1].plot(metrics["grad_norm"], label="grad norm")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("Grad Norm")
    ax[1, 1].legend()

    fig.tight_layout()
    return fig, ax


def implementation_details_comparisons(
    r_correct: Results, r_no_weight_init: Results, r_no_float64: Results
):
    with open(r_correct.path / "metrics.json") as f:
        m_correct = json.load(f)
    with open(r_no_weight_init.path / "metrics.json") as f:
        m_no_weight = json.load(f)
    with open(r_no_float64.path / "metrics.json") as f:
        m_no_float = json.load(f)

    fig, ax = plt.subplots(2, 1, sharey=True)
    ax[0].plot(m_no_float["test_loss"], label="float32", alpha=0.7)
    ax[0].plot(m_no_weight["test_loss"], label="float64", alpha=0.7)

    ax[1].plot(m_no_weight["test_loss"], label="no weight init", alpha=0.7)
    ax[1].plot(m_correct["test_loss"], label="weight init", alpha=0.7)
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_ylabel("CE Loss")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_xlim(None, 40000)
    ax[0].legend()
    ax[1].legend()

    return fig, ax


def embedding_fft(r: Results):
    w_embed = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft = np.fft.fft(w_embed, axis=0)
    fft_powers = np.mean(np.abs(w_embed_fft) ** 2, axis=1)
    peaks = a.find_fft_peaks(fft_powers[: r.cfg.P // 2], height=0.3)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(w_embed.shape[0]), fft_powers)
    for peak in peaks:
        ax.axvline(peak, color=COLOR_HIGHLIGHT)
    ax.set_xlim(0, r.cfg.P // 2)
    ax.set_xlabel("k (Frequency Index)")
    ax.set_ylabel("Fourier Power")

    return fig, ax


def embedding_fft_before_after(r: Results):
    w_embed_final = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft_final = np.fft.fft(w_embed_final, axis=0)
    fft_cos_final = np.mean(np.abs(np.real(w_embed_fft_final)), axis=1)
    fft_sin_final = np.mean(np.abs(np.imag(w_embed_fft_final)), axis=1)

    r.load_checkpoint(1)
    init_epoch = r.current_epoch
    w_embed_init = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft_init = np.fft.fft(w_embed_init, axis=0)
    fft_cos_init = np.mean(np.abs(np.real(w_embed_fft_init)), axis=1)
    fft_sin_init = np.mean(np.abs(np.imag(w_embed_fft_init)), axis=1)

    r.load_checkpoint(-1)  # So downstream functions using r don't get screwed up

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4)
    ax[0].bar(
        np.arange(w_embed_init.shape[0]),
        fft_cos_init,
        align="edge",
        width=0.4,
        label="cos",
    )
    ax[0].bar(
        np.arange(w_embed_init.shape[0]),
        fft_sin_init,
        align="edge",
        width=-0.4,
        label="sin",
    )
    ax[1].bar(
        np.arange(w_embed_final.shape[0]),
        fft_cos_final,
        align="edge",
        width=0.4,
        label="cos",
    )
    ax[1].bar(
        np.arange(w_embed_final.shape[0]),
        fft_sin_final,
        align="edge",
        width=-0.4,
        label="sin",
    )
    ax[1].set_xlim(0, r.cfg.P // 2)
    ax[1].set_xlabel("k (Frequency Index)")
    ax[1].set_ylabel("Fourier Amplitude")
    ax[0].set_ylabel("Fourier Amplitude")

    ax[0].set_title(f"Epoch {init_epoch} (memorization)")
    ax[1].set_title(f"Epoch {r.current_epoch} (generalization)")

    ax[0].legend()
    ax[1].legend()

    return fig, ax


def embedding_pca(r: Results):
    dset = ModPDataset(r.cfg.P)
    test_inputs, test_labels = dset[:]
    with torch.no_grad():
        vecs = r.model.w_embed(test_inputs).numpy()  # (P**2, 3, d_model)
    vecs = vecs.reshape(113, 113, 3, 128)[:, 0, 0, :]
    pca = PCA(n_components=10)
    x = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = plt.get_cmap("viridis", 5)
    ax.scatter(x[:, 0], x[:, 1], alpha=0.9, marker=".", color=cmap(0))
    ax.scatter(x[:, 2], x[:, 3], alpha=0.9, marker=".", color=cmap(1))
    ax.scatter(x[:, 4], x[:, 5], alpha=0.9, marker=".", color=cmap(2))
    ax.scatter(x[:, 6], x[:, 7], alpha=0.9, marker=".", color=cmap(3))
    ax.scatter(x[:, 8], x[:, 9], alpha=0.9, marker=".", color=cmap(4))
    ax.axis("off")

    return fig, ax


def attention_heatmaps(r: Results):
    att_scores = a.attention_scores(r)
    _, n_heads, n_context, _ = att_scores.size()
    att_scores = att_scores.reshape(r.cfg.P, r.cfg.P, n_heads, n_context, n_context)

    plt.rcParams["figure.autolayout"] = False
    fig, ax = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    ax[0, 0].imshow(att_scores[:, :, 0, 2, 0])
    ax[0, 1].imshow(att_scores[:, :, 1, 2, 0])
    ax[1, 0].imshow(att_scores[:, :, 2, 2, 0])
    ax[1, 1].imshow(att_scores[:, :, 3, 2, 0])

    for _ax in ax.flat:
        _ax.set_xticks([0, 50, 100])
        _ax.set_yticks([0, 50, 100])

    for _ax in ax[1]:
        _ax.set_xlabel("a")
    for _ax in ax[:, 0]:
        _ax.set_ylabel("b")

    for _ax in ax[0]:
        _ax.set_xticklabels([])

    for _ax in ax[:, 1]:
        _ax.set_yticklabels([])

    return fig, ax


def attention_fft_2d(r: Results):
    peaks = get_peaks(r)

    # Compute attention scores and fft
    att_scores = a.attention_scores(r).detach().numpy()
    _, n_heads, n_context, _ = att_scores.shape
    att_scores = att_scores.reshape(r.cfg.P, r.cfg.P, n_heads, n_context, n_context)
    att_scores = att_scores - att_scores.mean(axis=(0, 1), keepdims=True)

    fft0 = np.abs(np.fft.fft2(att_scores[:, :, 0, 2, 0])) ** 2
    fft1 = np.abs(np.fft.fft2(att_scores[:, :, 1, 2, 0])) ** 2
    fft2 = np.abs(np.fft.fft2(att_scores[:, :, 2, 2, 0])) ** 2
    fft3 = np.abs(np.fft.fft2(att_scores[:, :, 3, 2, 0])) ** 2

    plt.rcParams["figure.autolayout"] = False
    fig, ax = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    ax[0, 0].imshow(np.log1p(fft0), extent=[-0.5, r.cfg.P - 0.5, r.cfg.P - 0.5, -0.5])
    ax[0, 1].imshow(np.log1p(fft1), extent=[-0.5, r.cfg.P - 0.5, r.cfg.P - 0.5, -0.5])
    ax[1, 0].imshow(np.log1p(fft2), extent=[-0.5, r.cfg.P - 0.5, r.cfg.P - 0.5, -0.5])
    ax[1, 1].imshow(np.log1p(fft3), extent=[-0.5, r.cfg.P - 0.5, r.cfg.P - 0.5, -0.5])
    for i in (0, 1):
        for j in (0, 1):
            for peak in peaks:
                ax[i, j].axvline(peak, color=COLOR_HIGHLIGHT, linewidth=0.8)
                ax[i, j].axhline(peak, color=COLOR_HIGHLIGHT, linewidth=0.8)

    for _ax in ax.flat:
        _ax.set_xticks([0, 20, 40])
        _ax.set_yticks([0, 20, 40])
        _ax.set_xlim(0, r.cfg.P // 2)
        _ax.set_ylim(r.cfg.P // 2, 0)

    for _ax in ax[1]:
        _ax.set_xlabel(r"$k_a$")
    for _ax in ax[:, 0]:
        _ax.set_ylabel(r"$k_b$")

    for _ax in ax[0]:
        _ax.set_xticklabels([])

    for _ax in ax[:, 1]:
        _ax.set_yticklabels([])

    return fig, ax


def post_attention_fft(r: Results):
    peaks = get_peaks(r)

    dset = ModPDataset(P=r.cfg.P)
    test_inputs, test_labels = dset[:]

    logits, cache = r.model(test_inputs)
    x = cache["post_att_0"][:, 2, :].reshape(r.cfg.P, r.cfg.P, r.cfg.d_model)

    fft_a = np.fft.fft(x - x.mean(axis=0, keepdims=True), axis=0)
    power_a = (np.abs(fft_a) ** 2).mean(axis=(1, 2))[: r.cfg.P // 2]

    fft_b = np.fft.fft(x - x.mean(axis=1, keepdims=True), axis=1)
    power_b = (np.abs(fft_b) ** 2).mean(axis=(0, 2))[: r.cfg.P // 2]

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(power_a)), power_a)
    for peak in peaks:
        ax.axvline(peak, color=COLOR_HIGHLIGHT)
    ax.set_xlabel("k (Frequency Index)")
    ax.set_ylabel("Fourier Power")

    return fig, ax


def post_mlp_fft(r: Results):
    peaks = get_peaks(r)

    dset = ModPDataset(P=r.cfg.P)
    test_inputs, test_labels = dset[:]

    logits, cache = r.model(test_inputs)
    x = cache["post_mlp_0"][:, 2, :].reshape(r.cfg.P, r.cfg.P, r.cfg.d_model)

    fft_a = np.fft.fft(x - x.mean(axis=0, keepdims=True), axis=0)
    power_a = (np.abs(fft_a) ** 2).mean(axis=(1, 2))[: r.cfg.P // 2]

    fft_b = np.fft.fft(x - x.mean(axis=1, keepdims=True), axis=1)
    power_b = (np.abs(fft_b) ** 2).mean(axis=(0, 2))[: r.cfg.P // 2]

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(power_a)), power_a)
    for peak in peaks:
        ax.axvline(peak, color=COLOR_HIGHLIGHT)
    ax.set_xlabel("k (Frequency Index)")
    ax.set_ylabel("Fourier Power")

    return fig, ax


def wl_fft(r: Results):
    peaks = get_peaks(r)

    W_U = r.model.w_unembed.weight.detach().numpy()
    W_out = r.model.mlps[0].mlp[2].weight.detach().numpy()
    W_L = W_U @ W_out
    wfft = np.fft.fft(W_L, axis=0)
    power = np.mean(np.abs(wfft) ** 2, axis=1)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(power)), power)
    for peak in peaks:
        ax.axvline(peak, color=COLOR_HIGHLIGHT)
    ax.set_xlim(0, r.cfg.P // 2)
    ax.set_xlabel("k (Frequency Index)")
    ax.set_ylabel("Fourier Power")

    return fig, ax


def cascaded_ffts(r: Results):
    peaks = get_peaks(r)

    dset = ModPDataset(P=r.cfg.P)
    test_inputs, test_labels = dset[:]

    logits, cache = r.model(test_inputs)

    # W_embed
    w_embed = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft = np.fft.fft(w_embed, axis=0)
    w_powers = np.mean(np.abs(w_embed_fft) ** 2, axis=1)

    # Residual post attention
    x_att = cache["post_att_0"][:, 2, :].reshape(r.cfg.P, r.cfg.P, r.cfg.d_model)

    fft_a = np.fft.fft(x_att - x_att.mean(axis=0, keepdims=True), axis=0)
    att_powers = (np.abs(fft_a) ** 2).mean(axis=(1, 2))[: r.cfg.P // 2]

    # Residual post mlp
    x_mlp = cache["post_mlp_0"][:, 2, :].reshape(r.cfg.P, r.cfg.P, r.cfg.d_model)

    fft_m = np.fft.fft(x_mlp - x_mlp.mean(axis=0, keepdims=True), axis=0)
    mlp_powers = (np.abs(fft_m) ** 2).mean(axis=(1, 2))[: r.cfg.P // 2]

    # W_L
    W_U = r.model.w_unembed.weight.detach().numpy()
    W_out = r.model.mlps[0].mlp[2].weight.detach().numpy()
    W_L = W_U @ W_out
    wfft = np.fft.fft(W_L, axis=0)
    wl_powers = np.mean(np.abs(wfft) ** 2, axis=1)

    # Plot
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax[0].bar(np.arange(len(w_powers)), w_powers)
    ax[0].set_ylabel(r"$W_E$")
    ax[1].bar(np.arange(len(att_powers)), att_powers)
    ax[1].set_ylabel(r"post-attn")
    ax[2].bar(np.arange(len(mlp_powers)), mlp_powers)
    ax[2].set_ylabel(r"post-mlp")
    ax[3].bar(np.arange(len(wl_powers)), wl_powers)
    ax[3].set_ylabel(r"$W_L$")

    ax[3].set_xlim(0, r.cfg.P // 2)
    ax[3].set_xlabel("k (Frequency Index)")

    for _ax in ax:
        for peak in peaks:
            _ax.axvline(peak, color=COLOR_HIGHLIGHT, linewidth=1.5, alpha=0.7)

    return fig, ax


def output_logits(r: Results):
    peaks = get_peaks(r)

    A, B = (50, 37)
    t = torch.tensor([[A, B, 113]])
    with torch.no_grad():
        logits, _ = r.model(t)
    logits = logits.numpy().squeeze()

    cons = []
    for peak in peaks:
        freq_con = a.frequency_contribution(logits, peak, r.cfg.P)
        cons.append(freq_con)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(cons).sum(axis=0), label="key freq.\ncontributions")
    ax[0].plot(logits, "o", label="logits")
    ax[0].axvline(87, color=COLOR_HIGHLIGHT, linewidth=1.0, alpha=0.8)
    ax[0].set_ylabel("Logit")

    con = a.frequency_contribution(logits, 1, r.cfg.P)
    for k in range(1, 113 // 2):
        con += a.frequency_contribution(logits, k, r.cfg.P)
    ax[1].plot(con, label="all freq.\ncontributions")
    ax[1].plot(logits, "o", label="logits")
    ax[1].axvline(87, color=COLOR_HIGHLIGHT, linewidth=1.0, alpha=0.8)
    ax[1].set_ylabel("Logit")

    ax[2].plot(a.softmax(np.array(cons).sum(axis=0)), ".")
    ax[2].axvline(87, color=COLOR_HIGHLIGHT, linewidth=1.0, alpha=0.8)

    ax[2].set_xlabel("Output Token c")
    ax[2].set_ylabel("Probability")

    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return fig, ax


def get_peaks(r: Results):
    # Get key peak frequencies
    w_embed = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft = np.fft.fft(w_embed, axis=0)
    fft_powers = np.mean(np.abs(w_embed_fft) ** 2, axis=1)
    peaks = a.find_fft_peaks(fft_powers[: r.cfg.P // 2], height=0.3)
    return peaks


if __name__ == "__main__":
    path = Path("runs") / "baseline_weight_init_seed_137"
    r = Results(path)
    save_path = (
        Path().home()
        / "mckuzyk.com/content/posts/transformers-grokking-modular-arithmetic/images/"
    )

    fig, ax = training_dynamics(r)
    #    fig.savefig("figures/training_dynamics.png")
    fig.savefig(save_path / "training_dynamics.svg")
    fig2, ax2 = implementation_details_comparisons(
        r,
        Results(Path("runs") / "baseline_seed_137"),
        Results(Path("runs") / "full_test_60k"),
    )
    #    fig2.savefig("figures/implementation_details.png")
    fig2.savefig(save_path / "implementation_details.svg")

    fig3, ax3 = embedding_fft(r)
    #    fig3.savefig("figures/embedding_fft.png")
    fig3.savefig(save_path / "embedding_fft.svg")

    fig4, ax4 = embedding_pca(r)
    #    fig4.savefig("figures/embedding_pca.png")
    fig4.savefig(save_path / "embedding_pca.svg")

    fig5, ax5 = attention_heatmaps(r)
    fig5.savefig(save_path / "attention_heatmaps.png")

    fig6, ax6 = attention_fft_2d(r)
    fig6.savefig(save_path / "attention_fft_2d.png")

    fig10, ax10 = cascaded_ffts(r)
    #    fig10.savefig("figures/cascaded_ffts.png")
    fig10.savefig(save_path / "cascaded_ffts.svg")

    fig11, ax11 = output_logits(r)
    #    fig11.savefig("figures/output_logits.png")
    fig11.savefig(save_path / "output_logits.svg")

    fig12, ax12 = embedding_fft_before_after(r)
    fig12.savefig(save_path / "embedding_fft_before_after.svg")

    plt.show()
