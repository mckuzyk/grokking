from pathlib import Path
import numpy as np
from scripts.analysis import Results
from scripts import analysis as a
from grokking.data import ModPDataset
import matplotlib.pyplot as plt


def get_peaks(r: Results):
    # Get key peak frequencies
    w_embed = r.model.w_embed.weight.detach().numpy()  # (P, d_model)
    w_embed_fft = np.fft.fft(w_embed, axis=0)
    fft_powers = np.mean(np.abs(w_embed_fft) ** 2, axis=1)
    peaks = a.find_fft_peaks(fft_powers[: r.cfg.P // 2], height=0.3)
    return peaks


def power_fraction(fft_power, peaks):
    return np.sum(fft_power[peaks]) / np.sum(fft_power)


def relative_power(r: Results):

    peaks = get_peaks(r)
    print(peaks)

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

    w_power_fraction = power_fraction(w_powers[: r.cfg.P // 2], peaks)
    print(w_power_fraction)

    wl_power_fraction = power_fraction(wl_powers[: r.cfg.P // 2], peaks)
    print(wl_power_fraction)

    x_att_power_fraction = power_fraction(att_powers, peaks)
    print(x_att_power_fraction)

    x_mlp_power_fraction = power_fraction(mlp_powers, peaks)
    print(x_mlp_power_fraction)


def wl_rank(r: Results):
    W_U = r.model.w_unembed.weight.detach().numpy()
    W_out = r.model.mlps[0].mlp[2].weight.detach().numpy()
    W_L = W_U @ W_out
    print(np.linalg.matrix_rank(W_L))
    U, s, Vh = np.linalg.svd(W_L)
    plt.plot(s)
    plt.show()


if __name__ == "__main__":
    path = Path("runs") / "baseline_weight_init_seed_137"
    r = Results(path)
    relative_power(r)
    wl_rank(r)
