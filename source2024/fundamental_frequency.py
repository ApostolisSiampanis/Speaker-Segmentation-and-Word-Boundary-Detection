import librosa
import numpy as np
import math

def mean_fundamental_frequency(audio_filepath, boundaries):
    """
    Calculate the mean fundamental frequency of the audio within the given boundaries.
    """
    audio, sample_rate = librosa.load(audio_filepath)

    mean_f0s = []
    for start, end in boundaries:
        # Extract the segment of the audio within the boundaries
        start_sample = librosa.time_to_samples(start, sr=sample_rate)
        end_sample = librosa.time_to_samples(end, sr=sample_rate)
        segment = audio[start_sample:end_sample]

        f0, voiced_flag, voiced_probs = librosa.pyin(segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
        # Filter out the silent frames
        f0 = f0[voiced_flag & (f0 > 105)]
        # Calculate the mean fundamental frequency of the segment
        mean_f0s.append(np.mean(f0))

    # Remove NaN values
    mean_f0s = [f0 for f0 in mean_f0s if not math.isnan(f0)]
    # Calculate the mean fundamental frequency
    mean_f0 = np.mean(mean_f0s)

    return mean_f0
