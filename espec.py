from math import ceil
import pyedflib.highlevel as highlevel
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy.fft import fft
import numpy as np
from scipy import signal
from scipy.fft import fftshift
from matplotlib import mlab
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


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


def plot_specgram(data, title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    pxx,  freq, t, cax = plt.specgram(data, Fs=5000)
    fig.colorbar(cax).set_label('Intensity [dB]')




if __name__ == '__main__':

    seizureFilesArr = searchExistingFilesAccordingTo("cutSeizures.json")

    for seizureFile in seizureFilesArr:

        signals, signal_headers, header = highlevel.read_edf(seizureFile["subdirectory"] + seizureFile["fileName"])
        plot_specgram(signals[1],title='Spectrogram', x_label='time (in seconds)', y_label='frequency')
        plt.show()
        exit()
