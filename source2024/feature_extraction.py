import librosa

def extract_features(audio, sample_rate):
    """
    Extract features from audio using moving window method.
    """
    window_size = 512  # Window length
    hop_size = 256  # Step size
    mel_bands = 96  # Number of Mel bands (filter bank size)
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=window_size, hop_length=hop_size,
                                                    n_mels=mel_bands)

    return melspectrogram