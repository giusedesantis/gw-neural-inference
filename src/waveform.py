from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
import matplotlib.pyplot as plt
import numpy as np

# Generation of a random simulated event
chirp_mass = np.random.uniform(18, 35)  # Random chirp mass between 18 and 35 solar masses
q = np.random.uniform(0.5, 1)  # Random mass ratio between 0.5 and 1
distance = np.random.uniform(300, 1500)  # Random distance between 300 and 1500 Mpc

# Convert chirp mass and mass ratio to individual masses required for waveform generation
eta = q / (1 + q)**2  # Symmetric mass ratio
total_mass = chirp_mass / (eta**(3/5))  # Total mass from chirp mass and symmetric mass ratio
m1 = total_mass / (1 + q)  # Mass of the first object
m2 = m1 * q  # Mass of the second object

# Initial simulation parameters
spin1z = 0.0  # Spin of the first object along the z-axis
spin2z = 0.0  # Spin of the second object along the z-axis
f_lower = 20  # Lower frequency cutoff for the waveform
duration  = 4.0  # Duration of the waveform in seconds
delta_f = 1 / duration  # Frequency resolution for the waveform
n_samples  = 4096  # Number of samples for the waveform
sample_rate = n_samples / duration  # Sample rate for the waveform
delta_t = 1 / sample_rate # Time resolution for the waveform
flen = n_samples // 2 + 1  # Length of the frequency domain waveform


# Two polarizations of the waveform
hp, hc = get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1,
    mass2=m2,
    spin1z=spin1z,
    spin2z=spin2z,
    distance=distance,
    f_lower=f_lower,
    delta_f=delta_f
    )

hp.resize(flen) # Resize the waveform to the desired length
hp_time = hp.to_timeseries() # Convert the frequency domain waveform to time domain (inverse Fourier transform)

# Generate PSD
psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

# Generate noise from the PSD
noise = noise_from_psd(n_samples, delta_t, psd, seed=42)

hp_time.start_time = 0
noise.start_time = 0
# Simulated detector data (signal + noise)
d_t = hp_time + noise

#plotting the simulated detector data
plt.figure(figsize=(10, 6))
plt.plot(noise.sample_times, noise, color='green', label='Simulated noise', alpha=0.8)
plt.plot(hp_time.sample_times, hp_time, color='red', label='Simulated waveform (signal)', alpha=0.8)
plt.plot(d_t.sample_times, d_t, color='blue', label='Simulated detector data (signal + noise)', alpha=0.55)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.show()
