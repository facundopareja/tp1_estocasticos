import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from utils import covariance_matrix, segment_signal, normalize_signal


def graph_histogram_scatterplot(matrix, bins=50):
    plt.hist(x=matrix[0, :], bins=bins)
    plt.show()
    plt.hist(x=matrix[1, :], bins=bins)
    plt.show()
    plt.scatter(matrix[0, :], matrix[1, :])
    plt.show()


# 1.a
data, fs = sf.read('audio_01_2024a.wav')
signal_rms = np.linalg.norm(data)
signal = normalize_signal(data, signal_rms)
array_even_elements = data[np.arange(data.size - 1)]
segmented_signal = segment_signal(array_even_elements, 2)

# 1.b
graph_histogram_scatterplot(np.transpose(segmented_signal), 50)

# 1.c
C_x = covariance_matrix(segmented_signal)
eigenvalues, eigenvectors = np.linalg.eig(C_x)

# 1.d)
Y_m = np.dot(eigenvectors, np.transpose(segmented_signal))
graph_histogram_scatterplot(Y_m, 50)
