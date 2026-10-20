from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
import matplotlib.pyplot as plt
import numpy as np

def sample_parameters(rng):
    # Draw random binary parameters
    chirp_mass = rng.uniform(18, 35)
    q = rng.uniform(0.5, 1.0)
    distance = rng.uniform(300, 1500)

    return chirp_mass, q, distance


def generate_waveform(chirp_mass, q, distance,
                      f_lower, delta_f, flen):

    # Convert chirp mass and mass ratio to component masses
    eta = q / (1 + q)**2
    total_mass = chirp_mass / eta**(3/5)

    m1 = total_mass / (1 + q)
    m2 = m1 * q

    # Generate frequency-domain waveform
    hp, hc = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        spin1z=0.0,
        spin2z=0.0,
        distance=distance,
        f_lower=f_lower,
        delta_f=delta_f
    )

    # Convert to the required time grid
    hp.resize(flen)
    hp_time = hp.to_timeseries()
    hp_time.start_time = 0

    return hp_time


def generate_noise(psd, n_samples, delta_t, rng):

    # Use a different random noise seed for each simulation
    seed = int(rng.integers(0, 2**31 - 1))

    noise = noise_from_psd(
        n_samples,
        delta_t,
        psd,
        seed=seed
    )

    noise.start_time = 0

    return noise


def whiten_data(d_t, psd, f_lower):

    # Convert detector data into frequency domain
    d_f = d_t.to_frequencyseries()

    # Create an empty frequency-domain array
    d_whiten_f = d_f.copy()
    d_whiten_f.data[:] = 0

    # Identify valid frequency bins
    kmin = int(np.ceil(f_lower / float(d_f.delta_f)))

    valid_bins = (
        (np.arange(len(psd)) >= kmin) &
        (psd.numpy() > 0) &
        np.isfinite(psd.numpy())
    )

    # Whiten the valid bins
    d_whiten_f.data[valid_bins] = (
        d_f.data[valid_bins] /
        np.sqrt(psd.data[valid_bins])
    )

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
        chirp_mass, q, distance,
        f_lower, delta_f, flen
    )

    # Generate detector noise
    noise = generate_noise(
        psd, n_samples, delta_t, rng
    )

    # Combine signal and noise
    d_t = hp_time + noise

    # Whiten detector data
    d_whiten_t = whiten_data(d_t, psd, f_lower)

    # Prepare ML input and labels
    X = np.asarray(d_whiten_t.numpy(), dtype=np.float32)

    theta = np.array(
        [chirp_mass, q, distance],
        dtype=np.float32
    )

    return X, theta


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
    psd = aLIGOZeroDetHighPower(
        flen, delta_f, f_lower
    )

    # Generate one example
    X, theta = generate_example(
        rng, psd, n_samples,
        delta_t, delta_f, flen, f_lower
    )

    # Verify the result
    print("Input shape:", X.shape)
    print("Parameters:", theta)
    print("Finite values:", np.isfinite(X).all())

    # Plot the whitened detector data
    time = np.arange(n_samples) * delta_t

    plt.figure(figsize=(10, 6))
    plt.plot(time, X)
    plt.xlabel("Time (s)")
    plt.ylabel("Whitened amplitude")
    plt.title("Whitened Detector Data")
    plt.show()


if __name__ == "__main__":
    main()

# # Generation of a random simulated event
# chirp_mass = np.random.uniform(18, 35)  # Random chirp mass between 18 and 35 solar masses
# q = np.random.uniform(0.5, 1)  # Random mass ratio between 0.5 and 1
# distance = np.random.uniform(300, 1500)  # Random distance between 300 and 1500 Mpc

# # Convert chirp mass and mass ratio to individual masses required for waveform generation
# eta = q / (1 + q)**2  # Symmetric mass ratio
# total_mass = chirp_mass / (eta**(3/5))  # Total mass from chirp mass and symmetric mass ratio
# m1 = total_mass / (1 + q)  # Mass of the first object
# m2 = m1 * q  # Mass of the second object

# # Initial simulation parameters
# spin1z = 0.0  # Spin of the first object along the z-axis
# spin2z = 0.0  # Spin of the second object along the z-axis
# f_lower = 20  # Lower frequency cutoff for the waveform
# duration  = 4.0  # Duration of the waveform in seconds
# delta_f = 1 / duration  # Frequency resolution for the waveform
# n_samples  = 4096  # Number of samples for the waveform
# sample_rate = n_samples / duration  # Sample rate for the waveform
# delta_t = 1 / sample_rate # Time resolution for the waveform
# flen = n_samples // 2 + 1  # Length of the frequency domain waveform


# # Two polarizations of the waveform
# hp, hc = get_fd_waveform(
#     approximant="IMRPhenomD",
#     mass1=m1,
#     mass2=m2,
#     spin1z=spin1z,
#     spin2z=spin2z,
#     distance=distance,
#     f_lower=f_lower,
#     delta_f=delta_f
#     )

# hp.resize(flen) # Resize the waveform to the desired length
# hp_time = hp.to_timeseries() # Convert the frequency domain waveform to time domain (inverse Fourier transform)

# # Generate PSD
# psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

# # Generate noise from the PSD
# noise = noise_from_psd(n_samples, delta_t, psd, seed=42)

# hp_time.start_time = 0
# noise.start_time = 0
# # Simulated detector data (signal + noise)
# d_t = hp_time + noise

# # Converting simulated detector data to frequency domain
# d_f = d_t.to_frequencyseries()

# # Creating an empty frequency-domain array for the whitened data
# d_whiten_f = d_f.copy()
# d_whiten_f[:] = 0

# # Identifying valid frequency bins
# kmin = int(f_lower / delta_f)
# valid_bins = (
#     (np.arange(len(psd)) >= kmin) &
#     (psd.numpy() > 0) &
#     (np.isfinite(psd.numpy()))
# )

# # Whitening only the valid frequency bins
# d_whiten_f.data[valid_bins] = d_f.data[valid_bins] / np.sqrt(psd.data[valid_bins]) 

# # Converting whitened data back to time domain
# d_whiten_t = d_whiten_f.to_timeseries()
# d_whiten_t.start_time = 0

# #plotting the simulated detector data
# plt.figure(figsize=(10, 6))
# # plt.plot(noise.sample_times, noise, color='green', label='Simulated noise', alpha=0.8)
# # plt.plot(hp_time.sample_times, hp_time, color='red', label='Simulated waveform (signal)', alpha=0.8)
# #plt.plot(d_t.sample_times, d_t, color='blue', label='Simulated detector data (signal + noise)', alpha=0.9, linewidth=0.7)
# plt.plot(d_whiten_t.sample_times, d_whiten_t, color='orange', label='Whitened detector data',alpha=0.6, linewidth=0.8)
# plt.xlabel('Time (s)')
# plt.ylabel('Strain')
# plt.legend()
# plt.show()
