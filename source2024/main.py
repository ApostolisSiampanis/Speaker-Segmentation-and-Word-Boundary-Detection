import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

# CONST
window_size = 512  # Window length
hop_size = 256  # Step size
mel_bands = 96  # Number of Mel bands (filter bank size)


def load_and_preprocess_audio(filepath, window_size, hop_size, mel_bands, offset=None, duration=None):
    """
    Loads audio file, extracts mel-spectrogram and converts to dB.
    """
    audio, sample_rate = librosa.load(filepath, offset=offset, duration=duration)
    # Extract Features
    melspectrogram = extract_features(audio, sample_rate, window_size, hop_size, mel_bands)
    # Convert to dB
    db_melspectrogram = db_conversion(melspectrogram)
    return db_melspectrogram


def extract_features(audio, sample_rate, window_size, hop_size, mel_bands):
    """
    Extract features from audio using moving window method.
    """
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=window_size, hop_length=hop_size,
                                                    n_mels=mel_bands)

    return melspectrogram


def db_conversion(melspectrogram):
    """
    Convert the mel-spectrogram to dB
    """
    return librosa.power_to_db(melspectrogram, ref=np.max)


def label_audio(melspectrogram, directory):
    """
    Label the audio file if it is a foreground sound or if it is a background sound.
    """
    if "foreground" in directory:
        labels = np.ones(melspectrogram.shape[1])
    else:
        labels = np.zeros(melspectrogram.shape[1])
    return labels


def plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title):
    """
    Plots the mel-spectrogram
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(db_melspectrogram, sr=sample_rate, hop_length=hop_size, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def binarize_predictions(predictions, threshold=0.5):
    """
    Binarize the predictions based on a given threshold.
    """
    return (predictions >= threshold).astype(int)


def train_evaluate_svc(features_train, features_test, labels_train, labels_test):
    """
    Trains and evaluates the LinearSVC classifier.
    """
    svc_clf = LinearSVC()
    svc_clf.fit(features_train, labels_train)

    # Make predictions on the test set
    labels_pred = svc_clf.predict(features_test)

    # Evaluate the classifier
    print("SVM Classifier")
    accuracy = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))

    return svc_clf


def train_evaluate_mlp(features_train, features_test, labels_train, labels_test):
    """
    Trains and evaluates the MLP Classifier.
    """
    mlp_clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=100, random_state=1, early_stopping=True)
    mlp_clf.fit(features_train, labels_train)

    # Make predictions on the test set
    labels_pred = mlp_clf.predict(features_test)

    # Evaluate the classifier
    print("MLP Classifier")
    print(f"Number of features used: {mlp_clf.n_features_in_}")
    accuracy = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))

    return mlp_clf


def train_evaluate_least_squares(features_train, features_test, labels_train, labels_test):
    """
    Trains and evaluates the Least Squares classifier.
    """
    # Add a bias term to the features
    features_train_b = np.c_[np.ones((features_train.shape[0], 1)), features_train]
    features_test_b = np.c_[np.ones((features_test.shape[0], 1)), features_test]
    # Convert to TensorFlow tensors
    features_train_b = tf.constant(features_train_b, dtype=tf.float32)
    labels_train = tf.constant(labels_train.reshape(-1, 1), dtype=tf.float32)
    features_test_b = tf.constant(features_test_b, dtype=tf.float32)
    # Compute the least squares solution
    theta = tf.linalg.lstsq(features_train_b, labels_train)
    # Make predictions on the test set
    labels_pred = tf.matmul(features_test_b, theta)
    labels_pred_binarized = binarize_predictions(labels_pred.numpy(), 0.5)
    print("Least Squares Classifier")
    accuracy = accuracy_score(labels_test, labels_pred_binarized)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred_binarized))

    return theta


def apply_median_filter(predictions, kernel_size):
    """
    Apply median filter to the predictions.
    """
    return scipy.signal.medfilt(predictions, kernel_size=kernel_size)


def predict_audio_labels(filepath, classifier_or_theta, window_size, hop_size, mel_bands):
    """
    Predict the labels of the audio file using the given classifier.
    """
    audio, sample_rate = librosa.load(filepath, sr=None)
    melspectrogram = extract_features(audio, sample_rate, window_size, hop_size, mel_bands)
    db_melspectrogram = db_conversion(melspectrogram)
    plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title=f'Mel-Spectrogram')
    features = db_melspectrogram.T  # Transpose to shape (frames, n_mels)
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
        predictions = binarize_predictions(predictions.numpy())

        # Convert predictions to list and flatten if necessary
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        predictions = predictions.flatten().tolist()
    else:
        predictions = predictions.tolist()

        # Apply median filter to the predictions
    filtered_predictions = apply_median_filter(predictions, kernel_size=3)

    return filtered_predictions


def process_directory(directory, is_rnn=False):
    """
    Read the files from the directory and extract all the audio files.
    """
    features = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_duration = librosa.get_duration(path=filepath)
        if file_duration < 4.0 and is_rnn:
            continue
        if is_rnn:
            duration = 2.0
        else:
            duration = file_duration

        # Load files and find spectrogram
        db_melspectrogram = load_and_preprocess_audio(filepath, window_size, hop_size, mel_bands, offset=0.5,
                                                      duration=duration)

        # Label each frame
        frame_labels = label_audio(db_melspectrogram, directory)
        features.append(db_melspectrogram.T)
        labels.append(frame_labels)
        # Plot spectrogram
        #plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title=f'Mel-Spectrogram of {filename}')

    return np.vstack(features), np.hstack(labels)


if __name__ == '__main__':
    # Load files
    foreground_features, foreground_labels = process_directory("../auxiliary2024/dataset/foreground", is_rnn=False)
    background_features, background_labels = process_directory("../auxiliary2024/dataset/background", is_rnn=False)

    # Combine features and labels to the same structures
    features = np.vstack((foreground_features, background_features))
    labels = np.hstack((foreground_labels, background_labels))

    # Split features and labels to train and test batches
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    # Train and Evaluate the classifiers.
    svm_classifier = train_evaluate_svc(features_train, features_test, labels_train, labels_test)
    mlp_classifier = train_evaluate_mlp(features_train, features_test, labels_train, labels_test)
    least_squares_theta = train_evaluate_least_squares(features_train, features_test, labels_train, labels_test)

    new_audio_filepath = "../84-121123-0015.flac"

    # Predict using the SVM classifier
    svm_predictions = predict_audio_labels(new_audio_filepath, svm_classifier, window_size, hop_size, mel_bands)
    print(f"SVM Predictions: {svm_predictions}")

    # Predict using the MLP classifier
    mlp_predictions = predict_audio_labels(new_audio_filepath, mlp_classifier, window_size, hop_size, mel_bands)
    print(f"MLP Predictions: {mlp_predictions}")

    # Predict using the Least Squares classifier
    least_squares_predictions = predict_audio_labels(new_audio_filepath, least_squares_theta, window_size, hop_size, mel_bands)
    print(f"Least Squares Predictions: {least_squares_predictions}")