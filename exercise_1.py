import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from utils import covariance_matrix, segment_signal, normalize_signal


def graph_histogram_scatterplot(matrix):
    """ Grafica histograma para cada eje horizontal de la matriz que recibe
    Tambien grafica dispersion entre ambos componentes."""
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Gráfico en el centro
    ax_center = fig.add_subplot(gs[0, 0])
    ax_center.scatter(matrix[0, :], matrix[1, :], alpha=0.5)
    ax_center.set_title('Grafico de dispersión X0 vs X1')
    ax_center.set_xlabel('X0')
    ax_center.set_ylabel('X1')

    # Gráfico a la derecha
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.hist(matrix[1, :], bins=30, orientation='horizontal')
    ax_right.set_title('Histograma X1')
    ax_right.yaxis.tick_right()  # Mover los ticks del eje y a la derecha
    y_min, y_max = ax_center.get_ylim()
    ax_right.set_ylim(y_min, y_max)
    ax_right.set_xlabel('Cantidad de Muestras')
    ax_right.set_ylabel('X1')

    # Gráfico abajo
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.hist(matrix[0, :], bins=30)
    ax_bottom.set_title('Histograma X0')
    ax_bottom.xaxis.tick_bottom()  # Mover los ticks del eje x abajo
    x_min, x_max = ax_center.get_xlim()
    ax_bottom.set_xlim(x_min, x_max)
    ax_bottom.invert_yaxis()  # Invertir el eje y para que esté boca abajo
    ax_bottom.set_ylabel('Cantidad de Muestras')
    ax_bottom.set_xlabel('X0')

    plt.show()


# 1.a)
data, fs = sf.read('audio_01_2024a.wav')
signal_rms = np.linalg.norm(data)
signal = normalize_signal(data, signal_rms)
array_even_elements = signal[np.arange(data.size - 1)]
segmented_signal = segment_signal(array_even_elements, 2)

# 1.b)
graph_histogram_scatterplot(np.transpose(segmented_signal))

# 1.c)
C_x = covariance_matrix(segmented_signal)
eigenvalues, eigenvectors = np.linalg.eig(C_x)

# 1.d)
Y_m = np.dot(np.transpose(eigenvectors), np.transpose(segmented_signal))
graph_histogram_scatterplot(Y_m)
