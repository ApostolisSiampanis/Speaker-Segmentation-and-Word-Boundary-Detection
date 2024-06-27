import os
import subprocess
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

PLAYER = "ffplay"  # Path to ffplay executable

def _play_with_ffplay(seg):
    """
    Play the given segment using ffplay.
    """
    try:
        with NamedTemporaryFile("w+b", suffix=".wav", delete=False) as f:
            seg.export(f.name, "wav")
            subprocess.call([PLAYER, "-nodisp", "-autoexit", "-hide_banner", f.name])
    finally:
        os.remove(f.name)

def play_audio_boundaries(audio_filepath, boundaries):
    """
    Play the audio file with the given boundaries.
    """
    with open(audio_filepath, 'rb') as file:
        audio = AudioSegment.from_file(file)
        for start, end in boundaries:
            segment = audio[start * 1000:end * 1000]
            _play_with_ffplay(segment)
