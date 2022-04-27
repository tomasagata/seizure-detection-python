import pyedflib.highlevel as highlevel
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

def normalize(signal):
    return_array = np.array([])

    N = len(signal)

    sumOfSquares = 0
    for sample in signal:
        sumOfSquares += sample ** 2

    rms = np.sqrt( sumOfSquares / N )

    for i in range(len(signal)):
        return_array = np.append(return_array, signal[i] - rms)

    return return_array


if __name__ == '__main__':
    signals, signal_headers, header = highlevel.read_edf("chb01/blk01/chb01_03.edf.ceizure.edf")
    #plt.plot(signals[1])}
    #plt.show()
    # plt.plot(signals[1])
    # new_signal = normalize(signals[1])
    plt.plot(abs(normalize(signals[1])))
    plt.show()
    