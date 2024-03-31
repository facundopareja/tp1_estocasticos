import numpy as np
import sounddevice as sd


def segment_signal(signal, length):
    """Segmenta señal de acuerdo a la longitud dada."""
    array_list = []
    for i in range(0, len(signal), length):
        array_list.append(signal[i:i + length])
    return np.array(array_list)


def reconstruct_signal(signal, U, signal_rms):
    """Reconstruye la señal."""
    signal = np.dot(U, signal)
    return (signal.reshape(-1, order='F')) * signal_rms


def normalize_signal(signal, signal_rms):
    """Normaliza la señal de acuerdo a RMS."""
    return np.apply_along_axis(lambda x: x / signal_rms, 0, signal)


def calculate_mean(vector, number_samples):
    """Calcula esperanza del vector aleatorio."""
    number_columns = len(vector[1, :])
    value_list = []
    for i in range(number_columns):
        sum_i = np.sum(vector[:, i])
        value_list.append(sum_i)
    return (1 / number_samples) * np.array(value_list)


def covariance_matrix(vector):
    """Calcular matriz de covarianza del vector aleatorio."""
    array_size = vector.shape
    rows = array_size[0]
    columns = array_size[1]
    mean = calculate_mean(vector, rows)
    matrix = np.zeros((columns, columns))
    for i in range(rows):
        aux_array = vector[i, :] - mean
        aux_matrix = np.outer(aux_array, aux_array)
        matrix += aux_matrix
    return (1 / rows) * matrix


def pca_compression(X_m, compression_rate):
    """Aplica compresion PCA con cierta tasa a la matriz dada."""
    covariance_x = covariance_matrix(X_m)
    sample_length = X_m.shape[1]
    eigenvalues, eigenvectors = np.linalg.eig(covariance_x)
    k = int(np.ceil((1 - compression_rate) * sample_length))
    U = np.zeros((sample_length, k))
    for i in range(k):
        U[:, i] = eigenvectors[:, i]
    y_matrix = np.dot(np.transpose(U), np.transpose(X_m))
    return y_matrix, U


def calculate_mse(original_signal, reconstructed_signal):
    """Calcula MSE de señal reconstruida respecto de la original."""
    mse = 0
    length = len(reconstructed_signal)
    for i in range(length):
        mse += np.power((original_signal[i] - reconstructed_signal[i]), 2)
    return mse / length


def normalize_compress_decompress(audio, samplerate, sample_length, compress_rates, graph=False):
    """Aplica normalizacion, compresion PCA y decompresion, segun los parametros especificados."""
    print(f"Reproduciendo audio original")
    sd.play(audio, samplerate)
    sd.wait()
    signal_rms = np.linalg.norm(audio)
    audio = normalize_signal(audio, signal_rms)
    sample_number = int(len(audio) / sample_length)
    corrected_array = audio[np.arange(sample_length * sample_number)]
    X_m = segment_signal(corrected_array, sample_length)
    for compression_rate in compress_rates:
        Y_m, U = pca_compression(X_m, compression_rate)
        reconstructed_signal = reconstruct_signal(Y_m, U, signal_rms)
        print(f"Reproduciendo audio con compresion {compression_rate * 100} %")
        sd.play(reconstructed_signal, samplerate)
        sd.wait()
        if graph:
            print(calculate_mse(corrected_array, reconstructed_signal))
