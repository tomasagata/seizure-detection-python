import pyedflib.highlevel as highlevel
import matplotlib.pyplot as plt
from scipy.fft import fft

if __name__ == '__main__':
    signals, signal_headers, header = highlevel.read_edf('chb01/chb01_01.edf')
    #plt.plot(signals[1])}
    #plt.show()
    plt.plot(fft(signals[1]))
    plt.show()
    