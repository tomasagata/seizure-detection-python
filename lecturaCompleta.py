import pyedflib
from numpy import mean, var, std, cov
import matplotlib.pyplot as plt
from math import floor, ceil

# def denoiseFourier(signal, perc=0.25):
#     perc = 0.25
#     fhat = np.fft.fft(signal)
#     psd = fhat * np.conj(fhat)
#     th = perc * np.mean(abs(psd[round(len(psd)/2)]))
#     indices = psd > th
#     psd = psd * indices
#     fhat = indices * fhat
#     return np.fft.ifft(fhat)

# def readSignals():
#     lstColors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#     f = pyedflib.EdfReader("./chb01/chb01_03.edf.seizureFile.edf")
#     n=f.signals_in_file
#     denoiseFourier(f.readSignal(0))
#     figure, axs = plt.subplots(n)
#     signal_labels = f.getSignalLabels()
#     # return f.readSignal(i), signal_labels[i]
#     for i in range(0, n):
#         axs[i].set_title(signal_labels[i], x=-0.075, y=-0.1)
#         axs[i].plot(f.readSignal(i), lstColors[i % len(lstColors)])
#     plt.show()

# # readSignals()

# def convertDomainToTime(signal):
#     testRate = 256
#     signalSize = len(signal)
#     signalDuration = signalSize/testRate
#     period = 1/testRate

#     timeDomain = [] ; i = 0
#     while i < signalDuration:
#         timeDomain.append(i)
#         i+=period
    
#     return timeDomain

# def calculateSegmentMeanWithWindow(signal, window):
#     windowSize = len(window)
#     signalSize = len(signal[0])
#     numberOfSegments = floor(signalSize/windowSize)

#     segmentMeans = []
#     for channel in signal:
#         for segment in range(numberOfSegments-1):
#             segmentMeans.append(round( np.mean(channel[ segment*windowSize : (segment + 1)*windowSize ]), 2))

#     return segmentMeans

# def calculateSegmentVarianceWithWindow(signal, window):
#     windowSize = len(window)
#     signalSize = len(signal[0])
#     numberOfSegments = floor(signalSize/windowSize)

#     segmentVariance = []
#     for channel in signal:
#         for segment in range(numberOfSegments-1):
#             segmentVariance.append(round( np.var(channel[ segment*windowSize : (segment + 1)*windowSize ]), 2))

#     return segmentVariance

# def calculateSegmentStandardDeviationWithWindow(signal, window):
#     windowSize = len(window)
#     signalSize = len(signal[0])
#     numberOfSegments = floor(signalSize/windowSize)

#     segmentVariance = []
#     for channel in signal:
#         for segment in range(numberOfSegments-1):
#             segmentVariance.append(round( np.var(channel[ segment*windowSize : (segment + 1)*windowSize ]), 2))

#     return segmentVariance

def getNumberOfSegmentsForSamplesWithWindowAndOverlap(samples, window, overlap):
    N = len(samples)
    W = len(window)
    return round(1 + ((N-1) - (N%W) - W + 1)/round(W * (1-overlap)))

def calculateFunctionOfSegmentWithWindow(signal, functionToExecute, window, overlap = 0):
    windowSize = len(window)
    signalSize = len(signal[0])-1
    distanceBetweenSegments = round(windowSize*(1-overlap))
    numberOfSegments = round(1 + (signalSize - ((signalSize+1)%windowSize) - windowSize + 1)/round(windowSize * (1-overlap)))

    results = []
    for channel in signal:
        for segment in range(numberOfSegments):
            results.append(round( functionToExecute(channel[ floor(segment*distanceBetweenSegments) : floor(segment*distanceBetweenSegments)+windowSize ]), 2))
    return results

# def absoluteVariationAverage(samples):
#     sampleSize = len(samples)

#     return sum(abs(samples))/sampleSize

# def calculateSelfCovariance(samples, window):
#     sampleSize = len(samples)
#     windowSize = len(window)
#     numberOfSegments = floor(sampleSize/windowSize)

#     results = []
#     for segment in range(numberOfSegments):
#         results.append(round( cov(samples[segment*windowSize : (segment + 1)*windowSize], samples[(segment+1)*windowSize : (segment+2)*windowSize]) ), 2)




example_signal = [1,2,3,4,5,6,7,8,9]
example_window = [1,1,1,1]
example_overlap = 0.5


print(calculateFunctionOfSegmentWithWindow([example_signal], mean, example_window, example_overlap))
print(getNumberOfSegmentsForSamplesWithWindowAndOverlap(example_signal, example_window, example_overlap))



