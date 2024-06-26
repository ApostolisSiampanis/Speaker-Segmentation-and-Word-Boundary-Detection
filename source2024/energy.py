import librosa
import numpy as np
from skimage.filters import threshold_otsu


def calculate_energy_and_threshold(audio, window_size, hop_size):
    """
    Calculates energy and threshold of audio.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    #stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)
    # Compute the magnitude of the STFT
    #magnitude = np.abs(stft)
    # Compute the energy from the magnitude
    #energy = np.sum(magnitude ** 2, axis=0)
    # Normalize the energy
    #energy = energy / np.max(energy)

    #median_energy = np.median(energy)
    #mad_energy = np.median(np.abs(energy - median_energy))
    #mean_energy = np.mean(energy)
    #std_energy = np.std(energy)

    #print(f' Mean Energy: {mean_energy}')
    #print(f' Std Energy: {std_energy}')

    #threshold = median_energy - 0.99 * mad_energy
    #threshold = mean_energy - 0.4 * std_energy
    #threshold = threshold_otsu(energy)

    energy = np.array([
        sum(abs(audio[i:i + window_size] ** 2))
        for i in range(0, len(audio), hop_size)
    ])
    # Normalize the energy
    energy = energy / np.max(energy)
    threshold = 0.0005

    return energy, threshold