from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_validation_waveforms(waveforms, results, delta_t, duration,
                              merger_time, output_dir):
    """Plot clean time-domain waveforms alongside basic QA outcomes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cases = len(waveforms)
    n_cols = 5
    n_rows = (n_cases + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.7 * n_rows),
                             sharex=True, squeeze=False)

    for ax, clean, result in zip(axes.flat, waveforms, results):
        time = np.arange(len(clean)) * delta_t
        i = result["case"]
        chirp_mass = result["chirp_mass_Msun"]
        q = result["q"]
        distance = result["distance_Mpc"]
        passed = result["passed_basic_QA"]
        ax.plot(time, np.asarray(clean) / 1e-22, linewidth=0.9)
        ax.axvline(merger_time, linestyle="--", linewidth=0.8,
                   color="gray", alpha=0.7)
        ax.set_xlim(0, duration)
        ax.set_title(
            f"{i:02d}: Mc={chirp_mass:.1f}, q={q:.2f}, "
            f"D={distance:.0f}\n{'PASS' if passed else 'CHECK'}",
            fontsize=9,
        )
        ax.grid(alpha=0.2)

    for ax in list(axes.flat)[n_cases:]:
        ax.set_visible(False)
    fig.supxlabel("Time (s)")
    fig.supylabel(r"Clean plus-polarization strain ($10^{-22}$)")
    fig.suptitle(
        f"{n_cases} IMRPhenomD time-domain waveform validation cases",
        fontsize=15,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    path = output_dir / "waveforms_20.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_noise_psd(frequencies, band, target_psd, mean_psd,
                   n_realizations, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(frequencies[band], target_psd[band],
              label="Target aLIGOZeroDetHighPower PSD", lw=2)
    ax.loglog(frequencies[band], mean_psd[band],
              label=f"Average of {n_realizations} noise PSDs", alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("One-sided PSD (strain²/Hz)")
    ax.set_title("Noise PSD validation")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "noise_psd.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_whitening_psd(frequencies, band, mean_whitened_psd,
                       n_realizations, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frequencies[band], mean_whitened_psd[band], lw=0.9,
            label=f"Average of {n_realizations} whitened noise PSDs")
    ax.axhline(1.0, color="black", linestyle="--", lw=1,
               label="Ideal whitened PSD = 1")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("One-sided whitened-noise PSD")
    ax.set_title("Whitening validation (noise only)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "whitening_psd.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_snr(results, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        [row["distance_Mpc"] for row in results],
        [row["optimal_snr_4s"] for row in results],
        c=[row["chirp_mass_Msun"] for row in results],
        cmap="viridis", s=55, edgecolors="black", linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="Chirp mass (solar masses)")
    ax.set_xlabel("Luminosity distance (Mpc)")
    ax.set_ylabel("Optimal SNR (4 s h_plus, simplified detector)")
    ax.set_title("Optimal SNR across 20 validation systems")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(output_dir) / "snr_vs_distance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_truncation(results, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array([r["case"] for r in results])
    retained_pct = 100 * np.array(
        [r["retained_snr2_fraction_approx"] for r in results]
    )
    ax.scatter(x, retained_pct)
    ax.axhline(100, color="black", linestyle="--", lw=1,
               label="100% (comparison reference)")
    ax.set_xticks(x)
    ax.set_xlabel("Validation case")
    ax.set_ylabel("Approximate retained optimal SNR² (%)")
    ax.set_title("Truncation/taper impact (approximate comparison)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "truncation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path