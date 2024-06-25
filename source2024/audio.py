import librosa
import numpy as np
import matplotlib.pyplot as plt

from source2024.feature_extraction import extract_features


def load_and_preprocess_audio(filepath, offset, duration):
    """
    Loads audio file, extracts mel-spectrogram and converts to dB.
    """
    audio, sample_rate = librosa.load(filepath, offset=offset, duration=duration)
    # Extract Features
    melspectrogram = extract_features(audio, sample_rate)
    # Convert to dB
    db_melspectrogram = db_conversion(melspectrogram)
    return db_melspectrogram, sample_rate


def db_conversion(melspectrogram):
    """
    Convert the mel-spectrogram to dB
    """
    return librosa.power_to_db(melspectrogram, ref=np.max)


def calculate_num_frames(duration, sample_rate, hop_size=256):
    """
    Calculate the number of frames.
    """
    return int(np.ceil(duration * sample_rate / hop_size))


def plot_spectrogram(db_melspectrogram, sample_rate, title):
    """
    Plots the mel-spectrogram
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(db_melspectrogram, sr=sample_rate, hop_length=256, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()