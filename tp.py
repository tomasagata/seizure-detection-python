import pyedflib.highlevel as highlevel
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy.fft import fft


def searchExistingFilesAccordingTo(seizureFile):
    f = open(seizureFile, "r", encoding='utf-8')
    arr = json.loads(f.read())
    f.close()

    return_arr = []
    for arrObject in arr:
        p = Path(arrObject["subdirectory"] + arrObject["fileName"])
        if p.exists() and p.is_file():
            return_arr.append(arrObject)
            print("found file " + arrObject["fileName"])


    if len(arr) == len(return_arr):
        print("all files found!")
    elif len(return_arr) == 0:
        print("no files found. Exiting...")
        exit()
    else:
        print("found " + str(len(return_arr)) + " out of " + str(len(arr)) + " files.")

    return return_arr

def normalize(signal):
    return_array = np.array([])

    N = len(signal)

    sumOfSquares = 0
    for sample in signal:
        sumOfSquares += sample ** 2

    rms = np.sqrt( sumOfSquares / N )

    for i in range(N):
        if signal[i] > 0:
            t = (signal[i] - rms)
            if t > 0:
                return_array = np.append(return_array, t)
            else:
                return_array = np.append(return_array, 0)
        elif signal[i] < 0:
            t = (signal[i] + rms)
            if t < 0:
                return_array = np.append(return_array, t)
            else: 
                return_array = np.append(return_array, 0)
        # return_array = np.append(return_array, signal[i] - rms)

    return return_array


if __name__ == '__main__':

    seizureFilesArr = searchExistingFilesAccordingTo("cutSeizures.json")

    for seizureFile in seizureFilesArr:

        signals, signal_headers, header = highlevel.read_edf(seizureFile["subdirectory"] + seizureFile["fileName"])
        plt.plot(normalize(signals[1]))
        plt.show()
        exit()
    