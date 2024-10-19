# Speech and Audio Processing (2024) - Speaker Segmentation and Word Boundary Detection

## Project Overview

This project was completed as an individual assignment for the "Speech and Audio Processing" course in the 8th semester of the 2024 academic year at the University of Piraeus, Department of Informatics. It involves the development of a Python-based program designed to segment speech recordings into individual words and classify them as either background or foreground sounds using machine learning techniques. The system processes audio input and detects word boundaries without any prior information about the number of words present in the recording.

## Course Information

- **Institution:** University of Piraeus
- **Department:** Department of Informatics
- **Course:** Speech and Audio Processing (2024)
- **Semester:** 8th

## Technologies Used

- **Python**
- **Libraries:**
    - `librosa`: For audio loading, feature extraction, and signal processing.
    - `NumPy`: For numerical computations and matrix operations.
    - `Matplotlib`: For data visualization and plotting.
    - `Pydub`: For audio manipulation and playback.
    - `Joblib`: For model serialization and loading.
    - `scikit-learn`: For machine learning models, including SVM and MLP classifiers.
    - `Tensorflow`: For building and training neural networks (RNN classifier).
    - `SciPy`: For applying signal processing techniques like median filtering.
    - `tempfile`: For temporary file creation during audio processing.
    - `subprocess`: For running external commands like `ffmpeg`.
 
## Classifiers Implemented

The system implements and compares the following classifiers for speech segmentation:

- **Support Vector Machine (SVM)**
- **Multilayer Perceptron (MLP)** (Three layers: 512, 256, 128 neurons)
- **Least Squares**
- **Recurrent Neural Networks (RNN)**

Each classifier is trained to identify whether a portion of the audio corresponds to background noise or speech using audio features extracted from the dataset.

## Dataset Details

The system utilizes two datasets for training and testing:

- **LibriSpeech-clean** (foreground speech dataset)
- **ESC-50** (background noise dataset)

Both datasets are processed to extract features using a sliding window technique, with a 50% overlap and Mel-spectrogram filters for feature extraction.

## Example Output


## Project Documenation

For detailed explanations of the code, classifiers, and algorithms used in this project can be found in the [Project-documentation.pdf](./docs/Project-documentation.pdf)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
