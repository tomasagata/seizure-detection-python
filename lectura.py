import pyedflib
import numpy as np
import matplotlib.pyplot as plt

def denoiseFourier(signal, perc=0.25):
    signal -= np.mean(signal)
    fhat = np.fft.fft(signal)
    psd = fhat * np.conj(fhat)
    th = perc * np.mean(abs(psd[round(len(psd)/2)]))
    indices = psd > th
    psd = psd * indices
    fhat = indices * fhat
    return np.fft.ifft(fhat)

def readSignals():
    lstColors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    f = pyedflib.EdfReader("./convulsiones_edf/chb01/chb01_03.edf.seizureFile.edf")
    n=f.signals_in_file
    denoiseFourier(f.readSignal(0))
    figure, axs = plt.subplots(n)
    signal_labels = f.getSignalLabels()
    for i in range(0, n):
        axs[i].set_title(signal_labels[i], x=-0.075, y=-0.1)
        axs[i].plot(denoiseFourier(f.readSignal(i), 0.1), lstColors[i % len(lstColors)])
    plt.show()

readSignals()
