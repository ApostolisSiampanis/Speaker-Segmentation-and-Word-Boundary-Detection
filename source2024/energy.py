import numpy as np


def calculate_energy_and_threshold(audio, window_size, hop_size):
    """
    Calculates energy and threshold of audio.
    """
    energy = np.array([
        sum(abs(audio[i:i + window_size] ** 2))
        for i in range(0, len(audio), hop_size)
    ])
    # Normalize the energy
    energy = energy / np.max(energy)
    threshold = 0.0005

    return energy, threshold