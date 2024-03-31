import soundfile as sf
from utils import normalize_compress_decompress

sample_length = 1323

for i in range(1, 4):
    audio, samplerate = sf.read(f'audio_0{i}_2024a.wav')
    normalize_compress_decompress(audio, samplerate, sample_length, [0.10, 0.20, 0.50, 0.70, 0.90], True)
