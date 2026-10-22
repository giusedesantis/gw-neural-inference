import matplotlib.pyplot as plt
import numpy as np

from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform


def sample_parameters(rng):
    # Draw random binary parameters
    chirp_mass = rng.uniform(18, 35)
    q = rng.uniform(0.5, 1.0)
    distance = rng.uniform(300, 1500)

    return chirp_mass, q, distance


def position_waveform(hp, hc, n_samples, delta_t,
                      merger_time=3.0, taper_seconds=0.1):
    
    peak_amplitude = np.hypot(hp.numpy(), hc.numpy())
    peak_index = int(np.argmax(peak_amplitude))
    target_index = int(round(merger_time / delta_t))

    if not 0 < target_index < n_samples:
        raise ValueError("merger_time must lie strictly inside the segment")

    # Source and destination intervals, with the peak placed at target_index.
    source_start = max(0, peak_index - target_index)
    source_end = min(len(hp), peak_index + (n_samples - target_index))
    destination_start = target_index - (peak_index - source_start)
    count = source_end - source_start

    if count <= 0:
        raise ValueError("No waveform samples overlap the target segment")

    segment = np.zeros(n_samples, dtype=np.float64)
    segment[destination_start:destination_start + count] = (
        hp.numpy()[source_start:source_end]
    )

    # Soften both nonzero interval boundaries. Taper both ends even if the
    # original waveform is naturally short, to avoid discontinuities.
    fade_samples = min(int(round(taper_seconds / delta_t)), count // 2)
    if fade_samples >= 2:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        sl = slice(destination_start, destination_start + fade_samples)
        sr = slice(destination_start + count - fade_samples,
                   destination_start + count)
        segment[sl] *= ramp
        segment[sr] *= ramp[::-1]

    return TimeSeries(segment, delta_t=delta_t, epoch=0)


def generate_waveform(chirp_mass, q, distance,
                      f_lower, delta_t, n_samples):

    # Convert chirp mass and mass ratio to component masses
    eta = q / (1 + q) ** 2
    total_mass = chirp_mass / (eta**(3 / 5))
    m1 = total_mass / (1 + q)
    m2 = m1 * q

    # IMRPhenomD is an FD model; PyCBC handles its FD to TD conversion.
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        spin1z=0.0,
        spin2z=0.0,
        distance=distance,
        f_lower=f_lower,
        delta_t=delta_t,
    )

    # The raw TimeSeries length varies with the binary parameters.
    return position_waveform(hp, hc, n_samples, delta_t, merger_time=3.0)


def generate_noise(psd, n_samples, delta_t, rng):

    # Use a different random noise seed for each simulation
    seed = int(rng.integers(0, 2**31 - 1))
    noise = noise_from_psd(n_samples, delta_t, psd, seed=seed)
    noise.start_time = 0

    return noise


def whiten_data(d_t, psd, f_lower):

    # Convert detector data into frequency domain
    d_f = d_t.to_frequencyseries()

    # Create an empty frequency-domain array
    d_whiten_f = d_f.copy()
    d_whiten_f.data[:] = 0

    kmin = int(np.ceil(f_lower / float(d_f.delta_f)))
    psd_values = psd.numpy()
    valid_bins = (
        (np.arange(len(psd)) >= kmin)
        & np.isfinite(psd_values)
        & (psd_values > 0)
    )

    # Whiten the valid bins
    d_whiten_f.data[valid_bins] = d_f.data[valid_bins] / np.sqrt(psd.data[valid_bins])
    # Convert back to time domain
    d_whiten_t = d_whiten_f.to_timeseries()
    d_whiten_t.start_time = 0

    return d_whiten_t


def generate_example(rng, psd, n_samples,
                     delta_t, delta_f, flen, f_lower):

    # Draw source parameters
    chirp_mass, q, distance = sample_parameters(rng)

    # Generate gravitational waveform
    hp_time = generate_waveform(
        chirp_mass, q, distance, f_lower, delta_t, n_samples)

    # Generate detector noise
    noise = generate_noise(psd, n_samples, delta_t, rng)

    # Combine signal and noise
    d_t = hp_time + noise

    # Whiten detector data
    d_whiten_t = whiten_data(d_t, psd, f_lower)

    # Prepare ML input and labels
    X = np.asarray(d_whiten_t.numpy(), dtype=np.float32)
    theta = np.array([chirp_mass, q, distance], dtype=np.float32)

    return X, theta, d_t, hp_time


def main():
    # Simulation settings
    duration = 4.0
    sample_rate = 1024
    f_lower = 20.0

    n_samples = int(duration * sample_rate)
    delta_t = 1 / sample_rate
    delta_f = 1 / duration
    flen = n_samples // 2 + 1

    # Create random generator
    rng = np.random.default_rng(42)
    # Generate PSD once
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
    # Generate one example
    X, theta, d_t, hp_time = generate_example(
        rng, psd, n_samples, delta_t, delta_f, flen, f_lower
    )

    print("Input shape:", X.shape)
    print("Parameters:", theta)
    print("Finite values:", np.isfinite(X).all())
    assert X.shape == (n_samples,)
    assert np.isfinite(X).all()

    # Plot clean, positioned GW waveform
    plt.figure(figsize=(10, 5))

    plt.plot(
        hp_time.sample_times,
        hp_time,
        color='blue'
    )

    plt.xlim(2.85, 3.15)

    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.title("IMRPhenomD Merger and Ringdown")

    plt.grid()
    plt.show()

    # unwhitened vs whitened plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(d_t.sample_times, d_t,
             color="blue", alpha=0.6, label="Unwhitened detector data")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Strain", color="blue")

    ax2 = ax1.twinx()
    ax2.plot(d_t.sample_times, X,
             color="orange", alpha=0.6, label="Whitened detector data")
    ax2.set_ylabel("Whitened amplitude", color="orange")

    plt.title("Unwhitened vs Whitened Detector Data (TD IMRPhenomD)")
    fig.legend(loc="upper center")
    plt.show()


if __name__ == "__main__":
    main()
