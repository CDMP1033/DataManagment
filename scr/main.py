import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

data  = pd.read_csv("data/data_test.csv")
data = data.reindex([x for x in range(len(data))]).drop(["command"],axis=1)
data = data[0:20000]
print(data.describe())

total_len = len(data)
# Parámetros del filtro Butterworth
frecuencia_corte = 0.001  # Frecuencia de corte en Hz
frecuencia_muestreo = 0.0167  # Frecuencia de muestreo en Hz

# Generar señal aleatoria
t = data["timestamp"]
senal_aleatoria = data["temperature-1"]
print(senal_aleatoria.head)
# Crear filtro Butterworth
orden = 8  # Orden del filtro
frecuencia_normalizada = frecuencia_corte / (frecuencia_muestreo / 2)
b, a = butter(orden, frecuencia_normalizada, btype='low', analog=False, output='ba')


# Aplicar el filtro a la señal
senal_filtrada = filtfilt(b, a, senal_aleatoria)

# Cálculo de la Transformada de Fourier
fft_filtrada = np.abs(np.fft.fft(senal_aleatoria))
frecuencias = np.fft.fftfreq(len(senal_aleatoria), d=1 / frecuencia_muestreo)

# Graficar señal original y señal filtrada
plt.figure(2,figsize=(10, 6))
plt.plot(t, senal_aleatoria, label='Señal Original')
plt.plot(t, senal_filtrada, label='Señal Filtrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Filtro Butterworth')
plt.figure(1,figsize=(10, 6))
plt.plot(frecuencias, fft_filtrada, label='transformada')
plt.legend()
plt.grid(True)
plt.show()
