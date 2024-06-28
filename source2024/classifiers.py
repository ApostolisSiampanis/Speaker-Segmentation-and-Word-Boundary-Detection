import os
import librosa
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf

from source2024.audio import load_and_preprocess_audio, plot_spectrogram, calculate_num_frames
from source2024.labeling import label_audio


def train_svm_classifier(train_features, train_labels):
    """
    Trains and evaluates the SVM classifier.
    """
    # Create the SVM Classifier
    svm_clf = LinearSVC(random_state=1)
    # Train the SVM Classifier
    svm_clf.fit(train_features, train_labels)
    # Save the SVM Classifier
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)
    path = "auxiliary2024/classifiers/svm_classifier.joblib"
    path = os.path.join(project_path, path).replace("\\", "/")
    joblib.dump(svm_clf, path)
    print(f'The SVM classifier has been trained and saved at {path}')


def load_svm_classifier(path):
    """
    Loads the trained SVM classifier.
    """
    return joblib.load(path)


def train_mlp_classifier(train_features, train_labels):
    """
    Trains the MLP Classifier.
    """
    # Create the MLP Classifier
    mlp_clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=100, random_state=1, early_stopping=True)
    # Train the MLP Classifier
    mlp_clf.fit(train_features, train_labels)
    # Save the MLP Classifier
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)
    path = "auxiliary2024/classifiers/mlp_classifier.joblib"
    path = os.path.join(project_path, path).replace("\\", "/")
    joblib.dump(mlp_clf, path)
    print(f'The MLP classifier has been trained and saved at {path}')


def load_mlp_classifier(path):
    """
    Loads the trained MLP classifier.
    """
    return joblib.load(path)


def train_least_squares_classifier(train_features, train_labels):
    """
    Trains the Least Squares Classifier.
    """
    # Add a bias term to the features
    train_features_b = np.c_[np.ones((train_features.shape[0], 1)), train_features]
    # Convert to TensorFlow tensors
    train_features_b = tf.constant(train_features_b, dtype=tf.float32)
    train_labels = tf.constant(train_labels.reshape(-1, 1), dtype=tf.float32)
    # Compute the least squares solution
    theta = tf.linalg.lstsq(train_features_b, train_labels)
    # Save the Least Square Theta
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)
    path = "auxiliary2024/classifiers/theta_least_squares_classifier.joblib"
    path = os.path.join(project_path, path).replace("\\", "/")
    joblib.dump(theta, path)
    print(f'The Least Squares classifier has been trained and saved at {path}')


def load_theta_least_squares_classifier(path):
    """
    Loads the Least Squares Theta.
    """
    return joblib.load(path)


def build_rnn_model(input_shape):
    """
    Builds the RNN model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.SimpleRNN(32, activation='sigmoid', return_sequences=True, seed=1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_rnn_classifier(train_features, train_labels):
    """
     Trains the RNN model.
    """
    # Define input shape based on the training data
    input_shape = (train_features.shape[1], train_features.shape[2])
    rnn_model = build_rnn_model(input_shape)
    # Train the model
    rnn_model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_split=0.2)
    # Save the RNN Classifier
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)
    path = "auxiliary2024/classifiers/rnn_classifier.keras"
    path = os.path.join(project_path, path).replace("\\", "/")
    rnn_model.save(path)
    print(f'The RNN classifier has been trained and saved at {path}')


def load_rnn_model(path):
    """
    Loads the RNN model.
    """
    return tf.keras.models.load_model(path)


def process_training_directory(directory, is_rnn=False):
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
        db_melspectrogram, sample_rate = load_and_preprocess_audio(filepath, offset=0.5, duration=duration)

        # Label each frame
        frame_labels = label_audio(db_melspectrogram, directory)
        features.append(db_melspectrogram.T)
        labels.append(frame_labels)
        # Plot spectrogram
        #plot_spectrogram(db_melspectrogram, sample_rate, title=f'Mel-Spectrogram of {filename}')

    if is_rnn:
        num_frames = calculate_num_frames(duration, sample_rate)
        features_reshaped = np.array(features).reshape((-1, num_frames, 96))  # Frames number, n_mels
        labels_reshaped = np.array(labels).reshape((-1, num_frames))
        return features_reshaped, labels_reshaped
    else:
        return np.vstack(features), np.hstack(labels)


def main():
    """
    Train the Classifiers.
    """
    # Load files for SVM, MLP and Least Square
    print("\nLoading train audio data...")
    print("\nExtracting features...")
    foreground_path = "../auxiliary2024/dataset/foreground"
    background_path = "../auxiliary2024/dataset/background"
    foreground_train_features, foreground_train_labels = process_training_directory(foreground_path, is_rnn=False)
    background_train_features, background_train_labels = process_training_directory(background_path, is_rnn=False)
    # Load files for RNN
    foreground_rnn_train_features, foreground_rnn_train_labels = process_training_directory(foreground_path, is_rnn=True)
    background_rnn_train_features, background_rnn_train_labels = process_training_directory(background_path, is_rnn=True)

    # Combine features and labels to the same structures
    # SVM, MLP and Least Square
    train_features = np.vstack((foreground_train_features, background_train_features))
    train_labels = np.hstack((foreground_train_labels, background_train_labels))
    # RNN
    train_rnn_features = np.vstack((foreground_rnn_train_features, background_rnn_train_features))
    train_rnn_labels = np.vstack((foreground_rnn_train_labels, background_rnn_train_labels))

    # Train classifiers
    print("\nClassifiers training started... This may take some time.")
    print("\nTraining SVM Classifier...")
    train_svm_classifier(train_features, train_labels)
    print("\nTraining MLP Classifier...")
    train_mlp_classifier(train_features, train_labels)
    print("\nTraining Least Squares Classifier...")
    train_least_squares_classifier(train_features, train_labels)
    print("\nTraining RNN Classifier...")
    train_rnn_classifier(train_rnn_features, train_rnn_labels)


if __name__ == '__main__':
    main()
