"""Generate reproducible GW simulation datasets for the SBI prototype.

Run from the repository root or from src/:
    python src/generate_dataset.py --n-examples 20   # quick smoke test
    python src/generate_dataset.py --n-examples 2000 # prototype

Requires the *time-domain* simulations.py module in the same src/ directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pycbc.psd import aLIGOZeroDetHighPower

from simulations import generate_example  # time-domain implementation


def generate_dataset(n_examples, seed, psd, n_samples, delta_t, delta_f, flen, f_lower):
    
    rng = np.random.default_rng(seed)
    X = np.empty((n_examples, n_samples), dtype=np.float32)
    theta = np.empty((n_examples, 3), dtype=np.float32)

    for i in range(n_examples):
        # generate_example returns (whitened strain, source parameters, raw strain)
        x_i, theta_i, _, _ = generate_example(rng, psd, n_samples, delta_t, delta_f, flen, f_lower)

        if x_i.shape != (n_samples,) or theta_i.shape != (3,):
            raise ValueError(f"Unexpected shape at example {i}: X={x_i.shape}, theta={theta_i.shape}")
        if not np.isfinite(x_i).all() or not np.isfinite(theta_i).all():
            raise ValueError(f"Non-finite values in example {i}")

        X[i] = x_i
        theta[i] = theta_i

        if (i + 1) % 100 == 0 or i + 1 == n_examples:
            print(f"Generated {i + 1}/{n_examples} simulations")

    return X, theta


def main():
    parser = argparse.ArgumentParser(description="Create a GW-SBI training dataset")
    parser.add_argument("--n-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_examples < 10:
        raise ValueError("Use at least 10 examples to create training, validation and test splits")

    # Same validated settings as simulations.py
    duration = 4.0
    sample_rate = 1024
    f_lower = 20.0
    n_samples = int(duration * sample_rate)
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / duration
    flen = n_samples // 2 + 1

    # Reusing one analytic PSD for all simulated noise and whitening operations.
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    X, theta = generate_dataset(
        args.n_examples, args.seed, psd, n_samples,
        delta_t, delta_f, flen, f_lower
    )

    # Independent, reproducible random partition of complete examples.
    split_rng = np.random.default_rng(args.seed + 1)
    indices = split_rng.permutation(args.n_examples)
    n_train = int(0.80 * args.n_examples)
    n_val = int(0.10 * args.n_examples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"dataset_{args.n_examples}.npz"
    np.savez_compressed(dataset_path,
        X_train=X[train_idx], theta_train=theta[train_idx],
        X_val=X[val_idx], theta_val=theta[val_idx],
        X_test=X[test_idx], theta_test=theta[test_idx])

    # Recording all assumptions needed to interpret and reproduce the dataset.
    metadata = {
        "n_examples": args.n_examples,
        "seed": args.seed,
        "split_seed": args.seed + 1,
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "duration_s": duration,
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "f_lower_hz": f_lower,
        "delta_t_s": delta_t,
        "delta_f_hz": delta_f,
        "waveform_approximant": "IMRPhenomD (PyCBC get_td_waveform)",
        "noise_psd": "aLIGOZeroDetHighPower",
        "source_parameter_names": ["chirp_mass_Msun", "q", "distance_Mpc"],
        "source_parameter_ranges": {
            "chirp_mass_Msun": [18, 35],
            "q": [0.5, 1.0],
            "distance_Mpc": [300, 1500],
        },
        "aligned_spins": [0.0, 0.0],
        "merger_peak_time_s": 3.0,
        "edge_taper_s": 0.1,
        "detector_model": "plus polarization only (no antenna projection)",
        "preprocessing": "Whitened using the known PSD; no per-example normalization",
        "input_dtype": "float32",
    }
    metadata_path = output_dir / f"dataset_{args.n_examples}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\nX shape: {X.shape}; theta shape: {theta.shape}")
    print(f"Train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()