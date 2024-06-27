import os
from sklearn.metrics import accuracy_score

from source2024.audio_player import play_audio_boundaries
from source2024.labeling import label_test_audio, predict_audio_labels
from source2024.train_classifiers import load_svm_classifier, load_mlp_classifier, load_theta_least_squares_classifier, \
    load_rnn_model


def find_word_boundaries(predictions, sample_rate):
    """
    Find word boundaries.
    """
    in_word = False
    word_start = None
    word_boundaries = []

    i = 0
    while i < len(predictions):
        if predictions[i] == 1 and not in_word:
            in_word = True
            word_start = i
        elif in_word and (predictions[i] == 0 or i == len(predictions) - 1):
            word_start_time = word_start * 256 / sample_rate  # hop_size = 256
            word_end_time = i * 256 / sample_rate
            word_boundaries.append((word_start_time, word_end_time))
            in_word = False
        i += 1

    return word_boundaries


def main():
    while True:
        user_input = input("Please enter an audio file: ")
        test_audio_filepath = os.path.join("../", user_input)
        
        # Check if the file exists
        if os.path.exists(test_audio_filepath):
            break
        else:
            print(f'File "{user_input}" does not exist!')
    correct_labels = label_test_audio(test_audio_filepath)

    print("Waiting for SVM predictions...")

    # Predict using the SVM classifier
    path = "../auxiliary2024/classifiers/svm_classifier.joblib"
    svm_model = load_svm_classifier(path)
    svm_predictions, sample_rate = predict_audio_labels(test_audio_filepath, svm_model)
    # Find word Boundaries
    svm_boundaries = find_word_boundaries(svm_predictions, sample_rate)
    print(f'SVM classification accuracy is {accuracy_score(correct_labels, svm_predictions)}')
    print(f'SVM boundaries are {svm_boundaries}')

    # Predict using the MLP classifier
    path = "../auxiliary2024/classifiers/mlp_classifier.joblib"
    mlp_model = load_mlp_classifier(path)
    mlp_predictions, sample_rate = predict_audio_labels(test_audio_filepath, mlp_model)
    # Find word Boundaries
    mlp_boundaries = find_word_boundaries(mlp_predictions, sample_rate)
    print(f'MLP classification accuracy is {accuracy_score(correct_labels, mlp_predictions)}')
    print(f'MLP boundaries are {mlp_boundaries}')

    # Predict using the Least Squares classifier
    path = "../auxiliary2024/classifiers/theta_least_squares_classifier.joblib"
    least_squares_theta = load_theta_least_squares_classifier(path)
    least_squares_predictions, sample_rate = predict_audio_labels(test_audio_filepath, least_squares_theta)
    # Find word Boundaries
    least_squares_boundaries = find_word_boundaries(least_squares_predictions, sample_rate)
    print(f'Least Squares classification accuracy is {accuracy_score(correct_labels, least_squares_predictions)}')
    print(f'Least Squares boundaries are {least_squares_boundaries}')

    # Predict using the RNN classifier
    path = "../auxiliary2024/classifiers/rnn_classifier.keras"
    rnn_model = load_rnn_model(path)
    rnn_predictions, sample_rate = predict_audio_labels(test_audio_filepath, rnn_model)
    # Find word Boundaries
    rnn_boundaries = find_word_boundaries(rnn_predictions, sample_rate)
    print(f'RNN classification accuracy is {accuracy_score(correct_labels, rnn_predictions)}')
    print(f'RNN boundaries are {rnn_boundaries}')

    # Play the audio with the boundaries
    print("Playing audio with boundaries...")
    print("SVM boundaries:")
    play_audio_boundaries(test_audio_filepath, svm_boundaries)
    print("MLP boundaries:")
    play_audio_boundaries(test_audio_filepath, mlp_boundaries)
    print("Least Squares boundaries:")
    play_audio_boundaries(test_audio_filepath, least_squares_boundaries)
    print("RNN boundaries:")
    play_audio_boundaries(test_audio_filepath, rnn_boundaries)


if __name__ == '__main__':
    main()