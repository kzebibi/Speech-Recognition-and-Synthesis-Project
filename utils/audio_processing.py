import librosa
import numpy as np
from scipy import signal

def load_audio(file_path, sample_rate=16000):
    """Load audio file and resample if necessary."""
    audio, sr = librosa.load(file_path, sr=None)
    if sr != sample_rate:
        audio = librosa.resample(audio, sr, sample_rate)
    return audio

def preprocess_audio(audio, frame_length=512, hop_length=128):
    """Preprocess audio for ASR."""
    # Compute spectrogram
    spectrogram = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    # Convert to mel scale
    mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(spectrogram), sr=16000)
    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def synthesize_audio(mel_spectrogram, sample_rate=22050):
    """Convert mel spectrogram to audio."""
    # Inverse mel spectrogram
    S = librosa.feature.inverse.mel_to_stft(mel_spectrogram)
    # Griffin-Lim algorithm to reconstruct phase
    audio = librosa.griffinlim(S)
    return audio
