import csv
from pathlib import Path

import numpy as np
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.waveform import get_td_waveform

from simulations import generate_waveform, generate_noise, whiten_data
from visualize import (
    plot_validation_waveforms,
    plot_noise_psd,
    plot_whitening_psd,
    plot_snr,
    plot_truncation,
)

# Keep these settings consistent with simulations.py
DURATION = 4.0
SAMPLE_RATE = 1024
F_LOWER = 20.0
MERGER_TIME = 3.0
N_SAMPLES = int(DURATION * SAMPLE_RATE)
DELTA_T = 1.0 / SAMPLE_RATE
DELTA_F = 1.0 / DURATION
FLEN = N_SAMPLES // 2 + 1
N_NOISE = 100


def make_cases(rng):
    """Eight boundary cases plus twelve random systems; shared by all tests."""
    cases = [
        (mc, q, distance)
        for mc in (18.0, 35.0)
        for q in (0.5, 1.0)
        for distance in (300.0, 1500.0)
    ]
    for _ in range(12):
        cases.append((
            float(rng.uniform(18.0, 35.0)),
            float(rng.uniform(0.5, 1.0)),
            float(rng.uniform(300.0, 1500.0)),
        ))
    return cases


def save_csv(rows, path):
    """Save dictionaries as a CSV table."""
    if not rows:
        raise ValueError("Cannot save an empty metrics table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def periodogram(values, delta_t):
    """One-sided periodogram using the standard density normalization."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    p = (2.0 * delta_t / n) * np.abs(np.fft.rfft(values))**2
    p[0] *= 0.5
    if n % 2 == 0:
        p[-1] *= 0.5
    return p


def valid_band(frequencies, psd_values, delta_t):
    """Keep frequencies at/above 20 Hz and below Nyquist with valid PSD."""
    return (
        (frequencies >= F_LOWER)
        & (frequencies < 0.5 / delta_t)
        & np.isfinite(psd_values)
        & (psd_values > 0)
    )


def optimal_snr(time_series):
    """Idealized PSD-weighted SNR for a clean h_plus waveform.

    Not a recovered SNR, and not a realistic antenna-projected detector SNR.
    """
    samples = np.asarray(time_series.numpy(), dtype=np.float64)
    delta_t = float(time_series.delta_t)
    n = len(samples)
    delta_f = 1.0 / (n * delta_t)
    frequencies = np.fft.rfftfreq(n, delta_t)
    psd = aLIGOZeroDetHighPower(len(frequencies), delta_f, F_LOWER)
    band = valid_band(frequencies, psd.numpy(), delta_t)
    htilde = np.fft.rfft(samples) * delta_t
    rho_squared = 4.0 * delta_f * np.sum(
        np.abs(htilde[band])**2 / psd.numpy()[band]
    )
    return float(np.sqrt(max(0.0, rho_squared)))


def raw_waveform(chirp_mass, q, distance):
    """Unpositioned TD waveform with the same physical source parameters."""
    eta = q / (1.0 + q)**2
    total_mass = chirp_mass / eta**(3.0 / 5.0)
    m1 = total_mass / (1.0 + q)
    m2 = total_mass * q / (1.0 + q)
    return get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1, mass2=m2,
        spin1z=0.0, spin2z=0.0,
        distance=distance,
        f_lower=F_LOWER, delta_t=DELTA_T,
    )


def validate_waveforms(cases, psd, rng, output_dir):
    """Current 20-waveform checks, moved out of the old main()."""
    print("\n=== 1. WAVEFORM VALIDATION ===")
    waveforms = []
    positioned_series = []
    results = []
    edge_n = max(1, int(round(0.05 / DELTA_T)))

    for i, (chirp_mass, q, distance) in enumerate(cases, start=1):
        hp_time = generate_waveform(
            chirp_mass, q, distance, F_LOWER, DELTA_T, N_SAMPLES
        )
        noise = generate_noise(psd, N_SAMPLES, DELTA_T, rng)
        d_t = hp_time + noise
        whitened = whiten_data(d_t, psd, F_LOWER)

        clean = np.asarray(hp_time.numpy())
        noisy = np.asarray(d_t.numpy())
        white = np.asarray(whitened.numpy())
        noise_values = np.asarray(noise.numpy())

        correct_length = (len(clean) == N_SAMPLES
                          and len(noisy) == N_SAMPLES
                          and len(white) == N_SAMPLES)
        correct_dt = np.isclose(float(hp_time.delta_t), DELTA_T)
        finite = all(np.isfinite(arr).all() for arr in
                     (clean, noisy, white, noise_values))
        if not finite:
            raise ValueError(f"Nonfinite data in waveform case {i}")

        max_amp = float(np.max(np.abs(clean)))
        peak_index = int(np.argmax(np.abs(clean)))
        peak_time = peak_index * DELTA_T
        edge_amp = float(max(np.max(np.abs(clean[:edge_n])),
                             np.max(np.abs(clean[-edge_n:]))))
        edge_fraction = edge_amp / max_amp if max_amp > 0 else np.inf
        passed = (correct_length and correct_dt and finite and max_amp > 0
                  and abs(peak_time - MERGER_TIME) <= 0.05
                  and edge_fraction <= 0.01)

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
        waveforms.append(clean.copy())
        positioned_series.append(hp_time)
        print(f"{i:02d}/{len(cases)}  Mc={chirp_mass:5.2f} "
              f"q={q:.3f} D={distance:6.1f}  "
              f"peak={peak_time:.4f}s  "
              f"edge/peak={edge_fraction:.2e}  "
              f"{'PASS' if passed else 'CHECK'}")

    figure_path = plot_validation_waveforms(
        waveforms, results, DELTA_T, DURATION, MERGER_TIME, output_dir
    )
    csv_path = save_csv(results, output_dir / "validation_metrics.csv")
    n_passed = sum(row["passed_basic_QA"] for row in results)
    print(f"Basic waveform QA: {n_passed}/{len(results)} passed")
    print(f"Saved: {figure_path}, {csv_path}")
    return positioned_series


def validate_noise(psd, rng, output_dir):
    """Generate noise realizations and compare their mean PSD to the target."""
    print("\n=== 2. NOISE PSD VALIDATION ===")
    frequencies = np.fft.rfftfreq(N_SAMPLES, DELTA_T)
    target_psd = psd.numpy()
    band = valid_band(frequencies, target_psd, DELTA_T)
    noises = []
    psd_estimates = []

    for _ in range(N_NOISE):
        noise = generate_noise(psd, N_SAMPLES, DELTA_T, rng)
        values = np.asarray(noise.numpy())
        if not np.isfinite(values).all():
            raise ValueError("Non-finite samples in generated noise")
        noises.append(noise)
        psd_estimates.append(periodogram(values, DELTA_T))

    mean_estimated_psd = np.mean(psd_estimates, axis=0)
    ratio = mean_estimated_psd[band] / target_psd[band]
    print(f"Noise realizations: {N_NOISE}")
    print(f"Median estimated/target PSD ratio: {np.median(ratio):.3f}")
    path = plot_noise_psd(
        frequencies, band, target_psd, mean_estimated_psd,
        N_NOISE, output_dir,
    )
    print(f"Saved: {path}")
    return noises


def validate_whitening(noises, psd, output_dir):
    """Whiten those same noise realizations and inspect their mean PSD."""
    print("\n=== 3. WHITENING VALIDATION ===")
    frequencies = np.fft.rfftfreq(N_SAMPLES, DELTA_T)
    band = valid_band(frequencies, psd.numpy(), DELTA_T)
    white_periodograms = []
    white_std = []

    for noise in noises:
        whitened = whiten_data(noise, psd, F_LOWER)
        values = np.asarray(whitened.numpy())
        if not np.isfinite(values).all():
            raise ValueError("Non-finite samples after whitening")
        white_periodograms.append(periodogram(values, DELTA_T))
        white_std.append(float(np.std(values)))

    mean_white_psd = np.mean(white_periodograms, axis=0)
    print(f"Median whitened PSD in band: "
          f"{np.median(mean_white_psd[band]):.3f} (ideal around 1)")
    print(f"Mean whitened-noise standard deviation: "
          f"{np.mean(white_std):.3f} (approximate ideal: "
          f"{np.sqrt(np.count_nonzero(band) * DELTA_F):.2f})")
    path = plot_whitening_psd(
        frequencies, band, mean_white_psd, len(noises), output_dir
    )
    print(f"Saved: {path}")


def validate_snr(cases, waveforms, output_dir):
    """Compute optimal SNR of each positioned 4-second clean waveform."""
    print("\n=== 4. SNR VALIDATION ===")
    rows = []
    for i, ((chirp_mass, q, distance), hp_time) in enumerate(
        zip(cases, waveforms), start=1
    ):
        rho = optimal_snr(hp_time)
        rows.append({
            "case": i,
            "chirp_mass_Msun": chirp_mass,
            "q": q,
            "distance_Mpc": distance,
            "optimal_snr_4s": rho,
        })
        print(f"Case {i:02d}: Mc={chirp_mass:.1f}, q={q:.2f}, "
              f"D={distance:.0f} Mpc, optimal SNR={rho:.2f}")

    # First 8 cases are consecutive distance pairs with the same masses.
    for idx in (0, 2, 4, 6):
        near = rows[idx]["optimal_snr_4s"]
        far = rows[idx + 1]["optimal_snr_4s"]
        print(f"Distance scaling {idx+1}/{idx+2}: "
              f"SNR(300)/SNR(1500)={near / far:.3f} (expect ~5)")

    path = plot_snr(rows, output_dir)
    print(f"Saved: {path}")
    return rows


def validate_truncation(cases, positioned_waveforms, snr_rows, output_dir):
    """Compare raw and positioned waveforms with *approximate* SNR² ratio.

    Different time segments have different FFT grids. Treat retained-SNR²
    fractions as diagnostics, not definitive measurements of signal loss.
    """
    print("\n=== 5. TRUNCATION VALIDATION ===")
    rows = []
    target_index = int(round(MERGER_TIME / DELTA_T))

    for (chirp_mass, q, distance), positioned, snr_row in zip(cases, positioned_waveforms, snr_rows):
        raw_hp, raw_hc = raw_waveform(chirp_mass, q, distance)
        peak_index = int(np.argmax(np.hypot(raw_hp.numpy(), raw_hc.numpy())))
        discarded_before = max(0, peak_index - target_index)
        discarded_after = max(
            0, len(raw_hp) - peak_index - (N_SAMPLES - target_index)
        )
        #----------------TEST--------------#
        # Calculate the original waveform amplitude
        amplitude = np.hypot(raw_hp.numpy(), raw_hc.numpy())

        # Determine the retained interval
        start = discarded_before
        end = len(raw_hp) - discarded_after

        # Calculate the squared amplitude discarded
        discarded_squared = (np.sum(amplitude[:start]**2) + np.sum(amplitude[end:]**2))

        # Calculate total squared amplitude
        total_squared = np.sum(amplitude**2)

        # Fraction discarded
        fraction = discarded_squared / total_squared if total_squared > 0 else np.nan
        print(f"Discarded squared strain: {100 * fraction:.6f}%")
        #----------------TEST--------------#

        raw_snr = optimal_snr(raw_hp)
        kept_snr = snr_row["optimal_snr_4s"]
        retained = ((kept_snr / raw_snr)**2 if raw_snr > 0
                    else float("nan"))
        row = dict(snr_row)
        row.update({
            "optimal_snr_raw": raw_snr,
            "retained_snr2_fraction_approx": retained,
            "raw_duration_s": len(raw_hp) * DELTA_T,
            "discarded_before_peak_s": discarded_before * DELTA_T,
            "discarded_after_peak_s": discarded_after * DELTA_T,
            "discarded_strain_squared_pct": 100 * fraction,  #test line
        })
        rows.append(row)
        print(f"Case {row['case']:02d}: SNR_4s={kept_snr:.2f}, "
              f"retained SNR²≈{100 * retained:.1f}%, "
              f"early discarded={row['discarded_before_peak_s']:.3f}s")

    csv_path = save_csv(rows, output_dir / "snr_truncation_metrics.csv")
    plot_path = plot_truncation(rows, output_dir)
    print(f"Saved: {plot_path}, {csv_path}")
    print("NOTE: The SNR² comparison is approximate. Also inspect the "
          "waveform boundaries and taper; this is not a proof of no truncation.")


def main():
    """Run all five checks in order and save outputs to results/validation/."""
    output_dir = Path(__file__).resolve().parent.parent / "results" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate RNG streams for reproducible cases and independent noise tests.
    cases = make_cases(np.random.default_rng(42))
    waveform_noise_rng = np.random.default_rng(43)
    bulk_noise_rng = np.random.default_rng(44)
    psd = aLIGOZeroDetHighPower(FLEN, DELTA_F, F_LOWER)

    waveforms = validate_waveforms(cases, psd, waveform_noise_rng, output_dir)
    noises = validate_noise(psd, bulk_noise_rng, output_dir)
    validate_whitening(noises, psd, output_dir)
    snr_rows = validate_snr(cases, waveforms, output_dir)
    validate_truncation(cases, waveforms, snr_rows, output_dir)

    print(f"\nAll validation routines finished. Results: {output_dir}")
    print("Review the figures and diagnostic metrics before dataset creation.")


if __name__ == "__main__":
    main()