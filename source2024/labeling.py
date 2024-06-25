import numpy as np


def binarize_predictions(predictions, threshold=0.5):
    """
    Binarize the predictions based on a given threshold.
    """
    return (predictions >= threshold).astype(int)


def label_audio(melspectrogram, directory):
    """
    Label the audio file if it is a foreground sound or if it is a background sound.
    """
    if "foreground" in directory:
        labels = np.ones(melspectrogram.shape[1])
    else:
        labels = np.zeros(melspectrogram.shape[1])
    return labels