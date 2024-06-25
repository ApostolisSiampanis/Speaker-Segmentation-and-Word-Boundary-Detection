import scipy.signal

def apply_median_filter(predictions, kernel_size):
    """
    Apply median filter to the predictions.
    """
    return scipy.signal.medfilt(predictions, kernel_size=kernel_size)