from curses import window
import pyedflib
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from math import floor, ceil
import xlsxwriter
from xlsxwriter.exceptions import DuplicateWorksheetName as DuplicateWorksheetNameException

def denoiseFourier(signal, perc=0.25):
    perc = 0.25
    fhat = np.fft.fft(signal)
    psd = fhat * np.conj(fhat)
    th = perc * np.mean(abs(psd[round(len(psd)/2)]))
    indices = psd > th
    psd = psd * indices
    fhat = indices * fhat
    return np.fft.ifft(fhat)

def getNumberOfSegmentsForSamplesWithWindowAndOverlap(N: int, W: int, overlap: float) -> int:
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
        results.append(functionToExecute(data[displacement1 : displacement1+windowSize]*window, data[displacement2 : displacement2+windowSize]*window).round(2)[0][1])
    return results

def periodogram(signal, window):
    f, r = sig.periodogram(signal, 256, return_onesided=False, window=window)
    return [f,r]

def timeDomainWithOverlap(signalSegments, overlap):
    windowSize = 256
    averageTimeOfWindow = round(windowSize/2)

    timeDomain = [] ; i = 0
    while i < signalSegments:
        t = (averageTimeOfWindow + i*windowSize*(1-overlap))/float(256)
        timeDomain.append(t)
        i+=1
    
    return timeDomain

def readCalculateAndWriteStatisticalDataOfSignalToFile(inputSignalFilePath, outputDirectory = 'book.xlsx'):
    f = pyedflib.EdfReader(inputSignalFilePath)
    n=f.signals_in_file
    signal_labels = f.getSignalLabels()

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
        channelData["pearson_corr_squared_rectangular_sin_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, rectangularWindow(256), 0)]
        channelData["pearson_corr_squared_rectangular_con_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, rectangularWindow(256), 0.5)]
        channelData["pearson_corr_squared_hamming_sin_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.hamming(256), 0)]
        channelData["pearson_corr_squared_hamming_con_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.hamming(256), 0.5)]
        channelData["pearson_corr_squared_bh_sin_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.blackmanharris(256), 0)]
        channelData["pearson_corr_squared_bh_con_overlap"] = [ abs(number) for number in calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.corrcoef, sig.windows.blackmanharris(256), 0.5)]
        # channelData["correlacion_rectangular_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, rectangularWindow(256), 0)
        # channelData["correlacion_rectangular_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, rectangularWindow(256), 0.5)
        # channelData["correlacion_hamming_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.hamming(256), 0)
        # channelData["correlacion_hamming_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.hamming(256), 0.5)
        # channelData["correlacion_bh_sin_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.blackmanharris(256), 0)
        # channelData["correlacion_bh_con_overlap"] = calculateFunctionWithADisplacedVersionOfItself(denoised_signal, np.correlate, sig.windows.blackmanharris(256), 0.5)
        channelData["periodograma_rectangular"] = periodogram(denoised_signal, "boxcar")
        channelData["periodograma_hamming"] = periodogram(denoised_signal, "hamming")
        channelData["periodograma_bh"] = periodogram(denoised_signal, "blackmanharris")
        statisticalDataPerChannel.append(channelData)
    

    workbook = xlsxwriter.Workbook(outputDirectory)
    keys = list(statisticalDataPerChannel[0].keys())

    for channel in range(len(statisticalDataPerChannel)):
        try:
            worksheet = workbook.add_worksheet(signal_labels[channel])
        except DuplicateWorksheetNameException:
            worksheet = workbook.add_worksheet(signal_labels[channel] + "(2)")

        for arrayNumber in range(len(statisticalDataPerChannel[channel])):
            key = keys[arrayNumber]
            startingColumn = arrayNumber * 4
            worksheet.write(0, startingColumn, key)
            if arrayNumber < len(statisticalDataPerChannel[channel])-3:
                worksheet.write(1, startingColumn, "T(segs)")
                worksheet.write(1, startingColumn + 1, "value")
                if arrayNumber % 2 == 0:
                    t = timeDomainWithOverlap(len(statisticalDataPerChannel[channel][key]), 0)
                    for valueIndex in range(len(statisticalDataPerChannel[channel][key])):
                        worksheet.write(2 + valueIndex, startingColumn, t[valueIndex])
                        worksheet.write(2 + valueIndex, startingColumn + 1, statisticalDataPerChannel[channel][key][valueIndex])
                else:
                    t = timeDomainWithOverlap(len(statisticalDataPerChannel[channel][key]), 0.5)
                    for valueIndex in range(len(statisticalDataPerChannel[channel][key])):
                        worksheet.write(2 + valueIndex, startingColumn, t[valueIndex])
                        worksheet.write(2 + valueIndex, startingColumn + 1, statisticalDataPerChannel[channel][key][valueIndex])
            else:
                worksheet.write(1, startingColumn, "f(hertz)")
                worksheet.write(1, startingColumn + 1, "value")
                for valueIndex in range(len(statisticalDataPerChannel[channel][key][1])):
                    worksheet.write(2 + valueIndex, startingColumn, statisticalDataPerChannel[channel][key][0][valueIndex])
                    worksheet.write(2 + valueIndex, startingColumn + 1, statisticalDataPerChannel[channel][key][1][valueIndex])



    workbook.close()
    f.close()


readCalculateAndWriteStatisticalDataOfSignalToFile("chb01/chb01_03.edf.seizureFile.edf")
