import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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
        label = 1
    else:
        label = 0
    frames_num = melspectrogram.shape[1]
    labels = np.full(frames_num, label)
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

def process_directory(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_duration = librosa.get_duration(path=filepath)
        if file_duration < 4.0:
            continue
        audio, sample_rate = librosa.load(filepath, sr=None, offset=offset, duration=duration)
        print(sample_rate)
        # Extract Features
        melspectrogram = extract_features(audio, sample_rate, window_size, hop_size, mel_bands)
        # Convert to dB
        db_melspectrogram = db_conversion(melspectrogram)
        # Label each frame
        labels = label_audio(db_melspectrogram, directory)
        print(labels)
        # Plot spectrogram
        plot_spectrogram(db_melspectrogram, sample_rate, hop_size, title=f'Mel-Spectrogram of {filename}')

# Load files
process_directory("dataset/foreground")
process_directory("dataset/background")

#print(melspectrogram.shape)

# Transpose the result to have shape (frames, n_mfcc)
#melspectrogram = melspectrogram.T

#print(melspectrogram.shape)
# Print the first frame's MFCC feature vector
#for i in range(melspectrogram.shape[0]):
#    print(melspectrogram[i])
