import pyedflib
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from math import floor, ceil

def denoiseFourier(signal, perc=0.25):
    perc = 0.25
    fhat = np.fft.fft(signal)
    psd = fhat * np.conj(fhat)
    th = perc * np.mean(abs(psd[round(len(psd)/2)]))
    indices = psd > th
    psd = psd * indices
    fhat = indices * fhat
    return np.fft.ifft(fhat)

def getNumberOfSegmentsForSamplesWithWindowAndOverlap(N, W, overlap):
    return round(1 + ((N-1) - (N%W) - W + 1)/round(W * (1-overlap)))

def calculateFunctionOfSegmentWithWindow(signal, functionToExecute, window, overlap = 0):
    windowSize = len(window)
    distanceBetweenSegments = round(windowSize*(1-overlap))
    numberOfSegments = getNumberOfSegmentsForSamplesWithWindowAndOverlap(len(signal), len(window), overlap)

    results = []

    for segment in range(numberOfSegments):
        displacement = floor(segment*distanceBetweenSegments)
        results.append(functionToExecute(signal[ displacement : displacement+windowSize ]*window).round(2))
    return results

def rectangularWindow(size):
    return ([1]*size)

def prom_var_abs(signal):
    n = len(signal)
    return sum(abs(signal))/n

# def splitAndCalculateFunctionPerSegments(data, functionToExecute, endOfSegment1, endOfSegment2):
#     results = []
#     results.append(functionToExecute(data[0:endOfSegment1]))
#     results.append(functionToExecute(data[endOfSegment1:endOfSegment2]))
#     results.append(functionToExecute(data[endOfSegment2:len(data)]))
#     return results

def calculateFunctionWithADisplacedVersionOfItself(data, functionToExecute, window, overlap):
    windowSize = len(window)
    distanceBetweenSegments = round(windowSize*(1-overlap))
    numberOfSegments = getNumberOfSegmentsForSamplesWithWindowAndOverlap(len(data), len(window), overlap)

    results = []
    for segment in range(numberOfSegments-1):
        displacement1 = floor(segment*distanceBetweenSegments)
        displacement2 = floor((segment+1)*distanceBetweenSegments)
        results.append(functionToExecute(data[displacement1 : displacement1+windowSize]*window, data[displacement2 : displacement2+windowSize]*window).round(2))
    return results

def periodogram(signal, window, overlap):
    windowSize = len(window)
    distanceBetweenSegments = round(windowSize*(1-overlap))
    numberOfSegments = getNumberOfSegmentsForSamplesWithWindowAndOverlap(len(signal), len(window), overlap)

    results = []

    for segment in range(numberOfSegments):
        displacement = floor(segment*distanceBetweenSegments)
        f, r = sig.periodogram(signal[ displacement : displacement+windowSize ]*window, 256, return_onesided=False)
        results.append(r.round(2))
    return results

def readCalculateAndWriteStatisticalDataOfSignalToFile(inputSignalFilePath, outputFilePath):
    f = pyedflib.EdfReader(inputSignalFilePath)
    n=f.signals_in_file

    statisticalDataPerChannel = []
    for i in range(n):
        channelData = {}
        denoised_signal = denoiseFourier(f.readSignal(i))
        channelData["varianza_rectangular_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, rectangularWindow(256), 0)
        channelData["varianza_rectangular_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, rectangularWindow(256), 0.5)
        channelData["varianza_hamming_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, sig.windows.hamming(256), 0)
        channelData["varianza_hamming_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, sig.windows.hamming(256), 0.5)
        channelData["varianza_bh_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, sig.windows.blackmanharris(256), 0)
        channelData["varianza_bh_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.var, sig.windows.blackmanharris(256), 0.5)
        channelData["desviacion_estandar_rectangular_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, rectangularWindow(256), 0)
        channelData["desviacion_estandar_rectangular_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, rectangularWindow(256), 0.5)
        channelData["desviacion_estandar_hamming_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, sig.windows.hamming(256), 0)
        channelData["desviacion_estandar_hamming_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, sig.windows.hamming(256), 0.5)
        channelData["desviacion_estandar_bh_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, sig.windows.blackmanharris(256), 0)
        channelData["desviacion_estandar_bh_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, np.std, sig.windows.blackmanharris(256), 0.5)
        channelData["prom_var_abs_rectangular_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, rectangularWindow(256), 0)
        channelData["prom_var_abs_rectangular_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, rectangularWindow(256), 0.5)
        channelData["prom_var_abs_hamming_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, sig.windows.hamming(256), 0)
        channelData["prom_var_abs_hamming_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, sig.windows.hamming(256), 0.5)
        channelData["prom_var_abs_bh_sin_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, sig.windows.blackmanharris(256), 0)
        channelData["prom_var_abs_bh_con_overlap"] = calculateFunctionOfSegmentWithWindow(denoised_signal, prom_var_abs, sig.windows.blackmanharris(256), 0.5)
        channelData["covarianza_rectangular_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, rectangularWindow(256), 0)
        channelData["covarianza_rectangular_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, rectangularWindow(256), 0.5)
        channelData["covarianza_hamming_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, sig.windows.hamming(256), 0)
        channelData["covarianza_hamming_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, sig.windows.hamming(256), 0.5)
        channelData["covarianza_bh_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, sig.windows.blackmanharris(256), 0)
        channelData["covarianza_bh_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.cov, sig.windows.blackmanharris(256), 0.5)
        channelData["pearson_corr_squared_rectangular_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, rectangularWindow(256), 0)
        channelData["pearson_corr_squared_rectangular_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, rectangularWindow(256), 0.5)
        channelData["pearson_corr_squared_hamming_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.hamming(256), 0)
        channelData["pearson_corr_squared_hamming_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.hamming(256), 0.5)
        channelData["pearson_corr_squared_bh_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.blackmanharris(256), 0)
        channelData["pearson_corr_squared_bh_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.blackmanharris(256), 0.5)
        channelData["correlacion_rectangular_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, rectangularWindow(256), 0)
        channelData["correlacion_rectangular_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, rectangularWindow(256), 0.5)
        channelData["correlacion_hamming_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.hamming(256), 0)
        channelData["correlacion_hamming_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.hamming(256), 0.5)
        channelData["correlacion_bh_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.blackmanharris(256), 0)
        channelData["correlacion_bh_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.blackmanharris(256), 0.5)
        channelData["periodograma_rectangular_sin_overlap"] = periodogram(denoised_signal, rectangularWindow(256), 0)
        channelData["periodograma_rectangular_con_overlap"] = periodogram(denoised_signal, rectangularWindow(256), 0.5)
        channelData["periodograma_hamming_sin_overlap"] = periodogram(denoised_signal, sig.windows.hamming(256), 0)
        channelData["periodograma_hamming_con_overlap"] = periodogram(denoised_signal, sig.windows.hamming(256), 0.5)
        channelData["periodograma_bh_sin_overlap"] = periodogram(denoised_signal, sig.windows.blackmanharris(256), 0)
        channelData["periodograma_bh_con_overlap"] = periodogram(denoised_signal, sig.windows.blackmanharris(256), 0.5)
        statisticalDataPerChannel.append(channelData)
    
    #with open(outputFilePath, "f+") as fout:
        

    f.close()

readCalculateAndWriteStatisticalDataOfSignalToFile("chb01/chb01_03.edf.seizureFile.edf", "")
