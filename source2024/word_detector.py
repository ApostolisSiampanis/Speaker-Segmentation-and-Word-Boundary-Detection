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
    print("Hello")


if __name__ == '__main__':
    main()