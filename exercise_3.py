import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

from utils import normalize_compress_decompress

sample_length = 1323


def graph_mse_cr(mse, CR, nombres):
    """Grafica MSE vs CR con los nombres de audio especificados."""
    for i in range(len(mse)):
        plt.plot(CR, mse[i, :], '-', label=nombres[i])
    plt.title('MSE vs CR[%]')
    plt.xlabel('CR[%]')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


sample_audios = ['audio_01_2024a.wav', 'audio_02_2024a.wav', 'audio_03_2024a.wav']
compression_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
mse = np.zeros((len(sample_audios), len(compression_rates)))
for i in range(len(sample_audios)):
    audio, samplerate = sf.read(sample_audios[i])
    mse[i] = normalize_compress_decompress(audio, samplerate, sample_length, compression_rates)
graph_mse_cr(mse, compression_rates, sample_audios)
