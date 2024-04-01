import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

from utils import normalize_compress_decompress

sample_length = 1323


def graph_signals_time(original_signal, reconstructed_signals, compress_rates):
    """Grafica señales (original y reconstruida) respecto del tiempo."""
    reconstructed_signals.insert(0, original_signal)
    for pos, signal in enumerate(reconstructed_signals):
        plt.figure(figsize=(10, 4))
        time = np.arange(0, len(signal)) / samplerate
        plt.plot(time, signal, color='limegreen')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Amplitud')
        title = "sin compresion"
        if pos > 0:
            title = "con compresion " + str(compress_rates[pos-1] * 100) + " %"
        plt.title('Señal de audio ' + title)
        plt.show()


audio, samplerate = sf.read('audio_02_2024a.wav')
normalize_compress_decompress(audio, samplerate, sample_length, [0.70, 0.90, 0.95], graph_signals_time)
