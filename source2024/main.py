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

# CONST
window_size = 512  # Window length
hop_size = 256  # Step size
mel_bands = 96  # Number of Mel bands (filter bank size)
offset = 0.5  # Offset in seconds
duration = 2.0  # Duration in seconds


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
    Plot the mel-spectrogram
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(db_melspectrogram, sr=sample_rate, hop_length=hop_size, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def train_evaluate_svc(features_train, features_test, labels_train, labels_test):
    """
    Train and evaluate the LinearSVC classifier.
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
    Train and evaluate the MLP Classifier.
    """
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=1)
    mlp_clf.fit(features_train, labels_train)

    # Make predictions on the test set
    labels_pred = mlp_clf.predict(features_test)

    # Evaluate the classifier
    print("MLP Classifier")
    accuracy = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))

    return mlp_clf


def apply_median_filter(predictions, kernel_size):
    """
    Apply median filter to the predictions.
    """
    return scipy.signal.medfilt(predictions, kernel_size=kernel_size)

def predict_audio_class(filepath, classifier, window_size, hop_size, mel_bands):
    """
    Predict the class of the audio file using the given classifier.
    """
    audio, sample_rate = librosa.load(filepath, sr=None)
    melspectrogram = extract_features(audio, sample_rate, window_size, hop_size, mel_bands)
    db_melspectrogram = db_conversion(melspectrogram)
    plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title=f'Mel-Spectrogram')
    features = db_melspectrogram.T  # Transpose to shape (frames, n_mels)

    # Predict the class for each frame
    predictions = classifier.predict(features)

    # Apply median filter to the predictions
    filtered_predictions = apply_median_filter(predictions, kernel_size=3)

    return filtered_predictions


def process_directory(directory):
    """
    Read the files from the directory and extract all the audio files.
    """
    features = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_duration = librosa.get_duration(path=filepath)
        if file_duration < 4.0:
            continue
        audio, sample_rate = librosa.load(filepath, sr=None, offset=offset, duration=duration)
        # Extract Features
        melspectrogram = extract_features(audio, sample_rate, window_size, hop_size, mel_bands)
        # Convert to dB
        db_melspectrogram = db_conversion(melspectrogram)
        # Label each frame
        frame_labels = label_audio(db_melspectrogram, directory)
        features.append(db_melspectrogram.T)
        labels.append(frame_labels)
        # Plot spectrogram
        #plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title=f'Mel-Spectrogram of {filename}')

    return np.vstack(features), np.hstack(labels)



if __name__ == '__main__':
    # Load files
    foreground_features, foreground_labels = process_directory("../auxiliary2024/dataset/foreground")
    background_features, background_labels = process_directory("../auxiliary2024/dataset/background")

    # Combine features and labels to the same structures
    features = np.vstack((foreground_features, background_features))
    labels = np.hstack((foreground_labels, background_labels))

    # Split features and labels to train and test batches
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    svm_classifier = train_evaluate_svc(features_train, features_test, labels_train, labels_test)

    mlp_classifier = train_evaluate_mlp(features_train, features_test, labels_train, labels_test)



    new_audio_filepath = "../84-121123-0015.flac"

    # Predict using the SVM classifier
    svm_predictions = predict_audio_class(new_audio_filepath, svm_classifier, window_size, hop_size, mel_bands)
    print(f"SVM Predictions: {svm_predictions}")

    # Predict using the MLP classifier
    mlp_predictions = predict_audio_class(new_audio_filepath, mlp_classifier, window_size, hop_size, mel_bands)
    print(f"MLP Predictions: {mlp_predictions}")