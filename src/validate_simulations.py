import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycbc.psd import aLIGOZeroDetHighPower

#importing simulations functions
from simulations import generate_waveform, generate_noise, whiten_data


def main():
    # Same fixed configuration as simulations.py
    duration = 4.0
    sample_rate = 1024
    f_lower = 20.0
    n_samples = int(duration * sample_rate)
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / duration
    flen = n_samples // 2 + 1
    merger_time = 3.0
    time = np.arange(n_samples) * delta_t

    # One persistent RNG: reproducible run, different noise for each case
    rng = np.random.default_rng(42)
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    # Test the 8 boundary combinations
    cases = [
        (mc, q, distance)
        for mc in (18.0, 35.0)
        for q in (0.5, 1.0)
        for distance in (300.0, 1500.0)]

    # Also test of 12 random cases from the existing parameter ranges
    for _ in range(12):
        cases.append((
            float(rng.uniform(18.0, 35.0)),
            float(rng.uniform(0.5, 1.0)),
            float(rng.uniform(300.0, 1500.0)),
        ))

    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 5, figsize=(18, 11), sharex=True)
    results = []
    edge_n = max(1, int(round(0.05 / delta_t)))  # inspect first/last 50 ms

    for i, (chirp_mass, q, distance) in enumerate(cases, start=1):
        # Waveform with merger near 3 seconds
        hp_time = generate_waveform(
            chirp_mass, q, distance, f_lower, delta_t, n_samples
        )

        # Test the complete signal -> noise -> whitening pathway
        noise = generate_noise(psd, n_samples, delta_t, rng)
        d_t = hp_time + noise
        whitened = whiten_data(d_t, psd, f_lower)

        clean = np.asarray(hp_time.numpy())
        noisy = np.asarray(d_t.numpy())
        white = np.asarray(whitened.numpy())
        noise_values = np.asarray(noise.numpy())

        correct_length = (
            len(clean) == n_samples
            and len(noisy) == n_samples
            and len(white) == n_samples
        )
        correct_dt = np.isclose(float(hp_time.delta_t), delta_t)
        finite = (
            np.isfinite(clean).all()
            and np.isfinite(noisy).all()
            and np.isfinite(white).all()
            and np.isfinite(noise_values).all()
        )

        if not finite:
            raise ValueError(f"Nonfinite data in case {i}: "
                             f"Mc={chirp_mass}, q={q}, D={distance}")

        max_amp = float(np.max(np.abs(clean)))
        peak_index = int(np.argmax(np.abs(clean)))
        peak_time = peak_index * delta_t
        edge_amp = float(max(np.max(np.abs(clean[:edge_n])),
            np.max(np.abs(clean[-edge_n:])),))
        edge_fraction = edge_amp / max_amp if max_amp > 0 else np.inf

        # Simple QA thresholds
        good_timing = abs(peak_time - merger_time) <= 0.05
        quiet_edges = edge_fraction <= 0.01
        passed = (correct_length and correct_dt and finite
                  and max_amp > 0 and good_timing and quiet_edges)

        results.append({
            "case": i,
            "chirp_mass_Msun": chirp_mass,
            "q": q,
            "distance_Mpc": distance,
            "n_samples": len(clean),
            "peak_time_s": peak_time,
            "edge_to_peak_ratio": edge_fraction,
            "whitened_std": float(np.std(white)),
            "passed_basic_QA": passed,
        })

        print(f"{i:02d}/20  Mc={chirp_mass:5.2f} Msun  "
              f"q={q:.3f}  D={distance:6.1f} Mpc  "
              f"peak={peak_time:.4f} s  "
              f"edge/peak={edge_fraction:.2e}  "
              f"{'PASS' if passed else 'CHECK'}")

        ax = axes.flat[i - 1]
        ax.plot(time, clean / 1e-22, linewidth=0.9)
        ax.axvline(merger_time, linestyle="--", linewidth=0.8, color="gray", alpha=0.7)
        ax.set_xlim(0, duration)
        ax.set_title(f"{i:02d}: Mc={chirp_mass:.1f}, q={q:.2f}, "
                     f"D={distance:.0f}\n{'PASS' if passed else 'CHECK'}",fontsize=9)
        ax.grid(alpha=0.2)

    fig.supxlabel("Time (s)")
    fig.supylabel(r"Clean plus-polarization strain ($10^{-22}$)")
    fig.suptitle("20 IMRPhenomD time-domain waveform validation cases", fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    figure_path = output_dir / "waveforms_20.png"
    fig.savefig(figure_path, dpi=160)
    plt.show()

    csv_path = output_dir / "validation_metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    n_passed = sum(row["passed_basic_QA"] for row in results)
    print(f"\nBasic QA: {n_passed}/{len(results)} passed")
    print(f"Plot saved to: {figure_path}")
    print(f"Metrics saved to: {csv_path}")
    if n_passed != len(results):
        print("Inspect cases marked CHECK before generating the dataset.")
    else:
        print("Basic numerical checks passed. Inspect plots for physical issues.")


if __name__ == "__main__":
    main()
