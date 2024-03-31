import soundfile as sf
import sounddevice as sd

from utils import normalize_compress_decompress

sample_length = 1323

audio, samplerate = sf.read('audio_02_2024a.wav')
normalize_compress_decompress(audio, samplerate, sample_length, [0.70, 0.90, 0.95])

