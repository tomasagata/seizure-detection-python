import tp
from shutil import ReadError
from time import time
from xml import dom
import pyedflib
import numpy as np
import matplotlib.pyplot as plt

def denoiseFourier(signal, perc=0.25):
    perc = 0.25
    fhat = np.fft.fft(signal)
    psd = fhat * np.conj(fhat)
    th = perc * np.mean(abs(psd[round(len(psd)/2)]))
    indices = psd > th
    psd = psd * indices
    fhat = indices * fhat
    return np.fft.ifft(fhat)

def readSignals():
    lstColors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    f = pyedflib.EdfReader("./chb01/chb01_03.edf.seizureFile.edf")
    n=f.signals_in_file
    denoiseFourier(f.readSignal(0))
    figure, axs = plt.subplots(n)
    signal_labels = f.getSignalLabels()
    # return f.readSignal(i), signal_labels[i]
    for i in range(0, n):
        axs[i].set_title(signal_labels[i], x=-0.075, y=-0.1)
        axs[i].plot(f.readSignal(i), lstColors[i % len(lstColors)])
    plt.show()

# readSignals()

def convertDomainToTime(signal):
    testRate = 256
    signalSize = len(signal)
    signalDuration = signalSize/testRate
    period = 1/testRate

    timeDomain = [] ; i = 0
    while i < signalDuration:
        timeDomain.append(i)
        i+=period
    
    return timeDomain

def splitDomainIntoClasses(signal, beforeCrisisDuration = 120, crisisDuration = 40, afterCrisisDuration = 120):
    beforeCrisisClass = [] ; crisisClass = [] ; afterCrisisClass = [] 
    signalDuration = len(signal)
    i = 0
    while (i < (beforeCrisisDuration*256)) and (i < signalDuration):
        beforeCrisisClass.append(signal[i])
        i+=1
    while (i < ((beforeCrisisDuration + crisisDuration)*256)) and (i < signalDuration):
        crisisClass.append(signal[i])
        i+=1
    while (i < (beforeCrisisDuration + crisisDuration + afterCrisisDuration)*256) and (i < signalDuration):
        afterCrisisClass.append(signal[i])
        i+=1
    
    return beforeCrisisClass, crisisClass, afterCrisisClass

def showSignalByClasses():
    lstColors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    file = pyedflib.EdfReader("./chb01/chb01_03.edf.seizureFile.edf")
    channelAmount = file.signals_in_file
    figure, axis = plt.subplots(1,3)
    signal_labels = file.getSignalLabels() #Titulos de los canales
    timeDomain = convertDomainToTime(file.readSignal(0))
    amplitudeDomain = file.readSignal(0)
    # limit = tp.getHighestPointsOnSignal(amplitudeDomain)[0]
    limit = 800
    
    beforeClass , crisisClass, afterClass = splitDomainIntoClasses(timeDomain)
    
    beforeClassSignal = [] ; crisisClassSignal = [] ; afterClassSignal = [] ; i = 0
    for i in range(0, len(beforeClass)):
        beforeClassSignal.append(amplitudeDomain[i])
    for i in range(len(beforeClass), len(beforeClass) + len(crisisClass)):
        crisisClassSignal.append(amplitudeDomain[i])
    for i in range(len(beforeClass) + len(crisisClass), len(beforeClass) + len(crisisClass) + len(afterClass)):
        afterClassSignal.append(amplitudeDomain[i])
    
    # print(len(beforeClassSignal))
    # print(len(crisisClassSignal))
    # print(len(afterClassSignal))

    axis[0].plot(beforeClass, beforeClassSignal)
    axis[1].plot(crisisClass, crisisClassSignal)
    axis[2].plot(afterClass, afterClassSignal)
    
    axis[0].set_ylim([-limit, limit])
    axis[1].set_ylim([-limit, limit])
    axis[2].set_ylim([-limit, limit])
    
    plt.show()

# showSignalByClasses()

def showSignalsByClasses():
    lstColors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange']
    file = pyedflib.EdfReader("./chb01/chb01_03.edf.seizureFile.edf")
    channelAmount = file.signals_in_file
    figure, axis = plt.subplots(channelAmount,3)
    signal_labels = file.getSignalLabels() #Titulos de los canales
    timeDomain = convertDomainToTime(file.readSignal(0))
    amplitudeDomain = file.readSignal(0)
    # limit = tp.getHighestPointsOnSignal(amplitudeDomain1)[0]
    limit = 800
    
    beforeClass , crisisClass, afterClass = splitDomainIntoClasses(timeDomain)
    
    beforeClassSignal = [] ; crisisClassSignal = [] ; afterClassSignal = [] 

    for i in range(0, channelAmount):
        amplitudeDomain = file.readSignal(i)
        tempBeforeClassSignal = [] ; tempCrisisClassSignal = [] ; tempAfterClassSignal = [] ; j = 0
        for j in range(0, len(beforeClass)):
            tempBeforeClassSignal.append(amplitudeDomain[j])
        for j in range(len(beforeClass), len(beforeClass) + len(crisisClass)):
            tempCrisisClassSignal.append(amplitudeDomain[j])
        for j in range(len(beforeClass) + len(crisisClass), len(beforeClass) + len(crisisClass) + len(afterClass)):
            tempAfterClassSignal.append(amplitudeDomain[j])
        
        beforeClassSignal.append(tempBeforeClassSignal)
        crisisClassSignal.append(tempCrisisClassSignal)
        afterClassSignal.append(tempAfterClassSignal)

        axis[i,0].plot(beforeClass, beforeClassSignal[i], lstColors[i % len(lstColors)])
        axis[i,1].plot(crisisClass, crisisClassSignal[i], lstColors[i % len(lstColors)])
        axis[i,2].plot(afterClass, afterClassSignal[i], lstColors[i % len(lstColors)])

        axis[i,0].set_ylim([-limit, limit])
        axis[i,1].set_ylim([-limit, limit])
        axis[i,2].set_ylim([-limit, limit])

        axis[i,0].set_title(signal_labels[i], x=-0.3, y=-0.1)
    axis[channelAmount-1,0].set(xlabel = 'Before Crisis')
    axis[channelAmount-1,1].set(xlabel = 'During Crisis')
    axis[channelAmount-1,2].set(xlabel = 'After Crisis')
    plt.show()

showSignalsByClasses()