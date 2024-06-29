import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import tensorflow as tf

from source2024.audio import db_conversion
from source2024.energy import calculate_energy_and_threshold
from source2024.feature_extraction import extract_features
from source2024.post_processing import apply_median_filter


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


def predict_audio_labels(filepath, classifier_or_theta):
    """
    Predict the labels of the audio file using the given classifier.
    """
    audio, sample_rate = librosa.load(filepath)
    melspectrogram = extract_features(audio, sample_rate)
    db_melspectrogram = db_conversion(melspectrogram)
    features = db_melspectrogram.T # Transpose to shape (frames, n_mels)
    if isinstance(classifier_or_theta, LinearSVC) or isinstance(classifier_or_theta, MLPClassifier):
        # Predict using SVM or MLP classifier
        predictions = classifier_or_theta.predict(features)
    elif isinstance(classifier_or_theta, tf.Tensor):
        # Predict using Least Squares theta
        # Add bias term to features
        features_bias = np.c_[np.ones((features.shape[0], 1)), features]
        features_bias = tf.constant(features_bias, dtype=tf.float32)
        # Compute predictions using the provided theta (least squares solution)
        predictions = tf.matmul(features_bias, classifier_or_theta)
        predictions = binarize_predictions(predictions.numpy(),0.5)
    elif isinstance(classifier_or_theta, tf.keras.models.Sequential):
        # Reshape features to fit RNN input shape
        features = features.reshape(1, -1, 96) # mel_bands = 96
        # Predict using the RNN model
        predictions = classifier_or_theta.predict(features)
        predictions = predictions.flatten()
        predictions = binarize_predictions(predictions, 0.5)

    # Convert predictions to list and flatten if necessary
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        predictions = predictions.flatten().tolist()
    else:
        predictions = predictions.tolist()

    # Apply median filter to the predictions
    filtered_predictions = apply_median_filter(predictions, kernel_size=5)

    return filtered_predictions, sample_rate


def label_audio_based_on_energy(energy, threshold):
    """
    Labels the audio file if it is a foreground or if it is a background sound based on a given energy threshold.
    """
    labels = [0 if e < threshold else 1 for e in energy]
    return labels


def label_test_audio(audio_filepath):
    """
    Add labels to the user's audio file.
    """
    window_size = 512  # Window length
    hop_size = 256  # Step size
    audio, sample_rate = librosa.load(audio_filepath)
    energy, threshold = calculate_energy_and_threshold(audio, window_size, hop_size)
    labels = label_audio_based_on_energy(energy, threshold)
    labels = apply_median_filter(labels, kernel_size=7)

    # Plot the energy and labels
    plt.figure(figsize=(10, 6))
    plt.plot(energy, label='Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.plot(labels, label='Labels', linestyle='-', marker='o')
    plt.xlabel('Frame Index')
    plt.ylabel('Energy')
    plt.title('Audio Energy and Labels')
    plt.legend()
    plt.show()

    return labels