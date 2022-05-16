import tp
from shutil import ReadError
from time import time
from xml import dom
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from fitter import Fitter, get_common_distributions, get_distributions
import pandas as pd
from scipy.stats import cauchy, norm

def probDistributions(beforeClassSignal, crisisClassSignal,afterClassSignal):
    seizureFilesArr = tp.searchExistingFilesAccordingTo("cutSeizures.json")

    for seizureFile in seizureFilesArr:
        signals, signal_headers, header = tp.highlevel.read_edf(seizureFile["subdirectory"] + seizureFile["fileName"])
        data=signals[3]
        df = pd.DataFrame(data, columns =['Datos'])
        #sns.histplot(data=df, x="Datos", common_norm=True, kde=True)
        f = Fitter(df,
           distributions=['gamma',
                          'logistic',
                          "cauchy",
                          "norm",
                          "rayleigh"])
        f.fit()
        f.summary()
        plt.show()
        
        
        mu1, sigma1 = cauchy.fit(beforeClassSignal[3])
        mu2, sigma2 = cauchy.fit(crisisClassSignal[3])
        mu3, sigma3 = cauchy.fit(afterClassSignal[3])
        print("Before crisis: mu: "+ str(mu1) +"sigma: "+str(sigma1)+ "\n" + "During crisis: mu: "+ str(mu2) +"sigma: "+str(sigma2)+ "\n" + "After crisis: mu: "+ str(mu3) +"sigma: "+str(sigma3)+ "\n")
        '''
        Para hacer el scatter plot de solo estos 3 puntos, descomentar esto:
        plt.scatter([mu,mu1,mu2,mu3],[sigma,sigma1,sigma2,sigma3])
        plt.title("Plot Scatter")
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.show()
        '''
         
        #Para hacer box plot de los 3 bloques, descomentar esto:
        boxPlotThreeTypes(beforeClassSignal,crisisClassSignal,afterClassSignal)
        
        #Para hacer un box plot de los 3 tipos, descomentar acá.
        #boxPlotOfMusAndSigmas(beforeClassSignal,crisisClassSignal,afterClassSignal, seizureFilesArr)
        
        #Para hacer el Scatter Plot de los mu and sigma de todas las muestras con crisis según los bloques, descomentar acá.
        scatterDistOfAllMuAndSigma(beforeClassSignal, crisisClassSignal,afterClassSignal)
        
        exit()
        
        
def boxPlotOfMusAndSigmas(beforeClassSignal,crisisClassSignal,afterClassSignal, seizureFilesArr):
    listMuCrisis=[]
    listSigmaCrisis=[]
    listMuAfter=[]
    listSigmaAfter=[]
    listMuBefore=[]
    listSigmaBefore=[]
    
    for seizureFile in seizureFilesArr:
        signals, signal_headers, header = tp.highlevel.read_edf(seizureFile["subdirectory"] + seizureFile["fileName"])
        for i in range(len(signals)-1):
            data=signals[i]
            mu1, sigma1 = cauchy.fit(beforeClassSignal[i])
            listMuBefore.append(mu1)
            listSigmaBefore.append(sigma1)
            
            mu2, sigma2 = cauchy.fit(crisisClassSignal[i])
            listMuCrisis.append(mu2)
            listSigmaCrisis.append(sigma2)
            mu3, sigma3 = cauchy.fit(afterClassSignal[i])
            listMuAfter.append(mu3)
            listSigmaAfter.append(sigma3)
            
    fig = plt.figure()
    fig.add_subplot(131)
    plt.title('Before')
    plt.xlabel("Tiempo [S]")
    plt.ylabel("Amplitud [μV]")
    plt.boxplot(listMuBefore)
    fig.add_subplot(132)
    plt.title('Crisis')
    plt.boxplot(listMuCrisis)
    fig.add_subplot(133)
    plt.title('After')
    plt.boxplot(listMuAfter)
    plt.show()
        
    fig = plt.figure()
    fig.add_subplot(131)
    plt.title('Before')
    plt.xlabel("Tiempo [S]")
    plt.ylabel("Amplitud [μV]")
    plt.boxplot(listMuAfter)
    fig.add_subplot(132)
    plt.title('Crisis')
    plt.boxplot(listMuCrisis)
    fig.add_subplot(133)
    plt.title('After')
    plt.boxplot(listMuAfter)
    plt.show()
    
        
def boxPlotThreeTypes(beforeClassSignal,crisisClassSignal,afterClassSignal):
        fig = plt.figure()
        fig.add_subplot(131)
        plt.title('Before')
        plt.xlabel("Tiempo [S]")
        plt.ylabel("Amplitud [μV]")
        plt.boxplot(beforeClassSignal)
        fig.add_subplot(132)
        plt.title('Crisis')
        plt.boxplot(crisisClassSignal)
        fig.add_subplot(133)
        plt.title('After')
        plt.boxplot(afterClassSignal)
        plt.show()
        
def scatterDistOfAllMuAndSigma(beforeClassSignal, crisisClassSignal,afterClassSignal):
    seizureFilesArr = tp.searchExistingFilesAccordingTo("cutSeizures.json")
    listMu=[]
    listSigma=[]

    for seizureFile in seizureFilesArr:
        signals, signal_headers, header = tp.highlevel.read_edf(seizureFile["subdirectory"] + seizureFile["fileName"])
        for i in range(len(signals)):
            data=signals[i]
            mu, sigma = cauchy.fit(data)
        

            mu1, sigma1 = cauchy.fit(beforeClassSignal[i])
            listMu.append(mu1)
            listSigma.append(sigma1)
            
            mu2, sigma2 = cauchy.fit(crisisClassSignal[i])
            listMu.append(mu2)
            listSigma.append(sigma2)
            mu3, sigma3 = cauchy.fit(afterClassSignal[i])
            listMu.append(mu3)
            listSigma.append(sigma3)
            #print("Entire Block: mu: "+ str(mu) +" sigma: "+ str(sigma)+ "\n" + "Before crisis: mu: "+ str(mu1) +" sigma: "+str(sigma1)+ "\n" + "During crisis: mu: "+ str(mu2) +" sigma: "+str(sigma2)+ "\n" + "After crisis: mu: "+ str(mu3) +"sigma: "+str(sigma3)+ "\n")
    
        #Para hacer el scatter plot, descomentar esto:

        plt.scatter(listMu,listSigma)
        plt.title("Plot Scatter")
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.show()
        exit()
        
        #Para hacer box plot, descomentar esto
        plt.boxplot(data)
        plt.show()

        
        exit()
        

def denoiseFourier(signal, perc=0.25):
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
        axs[i].plot(denoiseFourier(f.readSignal(i)), lstColors[i % len(lstColors)])
    plt.show()
    
def calculateSignalToNoiseRatio(signal, axis=0, ddof=0):
    signalNpArray = np.asanyarray(signal)
    meanOfSignal = signalNpArray.mean(axis)
    standardDeviation = signalNpArray.std(axis=axis, ddof=ddof)
    return np.where(standardDeviation == 0, 0, meanOfSignal/standardDeviation)

        
def plot_specgram(before, during, after, title='', x_label='', y_label='', fig_size=None):
    #Para que se aprecie bien, no voy a poner que imprima todos los canales de una.
    #Si queres imprimir todos los canales:
    #var = len(before)
    #Caso contrario muestro los primeros 2 usando var = 2
    var = 2
    fs = 250
    
    for i in range (var):
        
        fig = plt.figure()
        if fig_size != None:
            fig.set_size_inches(fig_size[0], fig_size[1])
        fig.add_subplot(131)
        plt.title('Before')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        f, t, Zxx = signal.stft(before[i], fs, window='flattop', nperseg=50, noverlap=10)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax = 64, shading='gouraud')
        plt.colorbar()
        fig.add_subplot(132)
        plt.title('Crisis')
        f, t, Zxx = signal.stft(during[i], fs, window='flattop',nperseg=50, noverlap=10)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax = 64,shading='gouraud')
        plt.colorbar()
        fig.add_subplot(133)
        plt.title('After')
        f, t, Zxx = signal.stft(after[i], fs, window='flattop',nperseg=50, noverlap=10)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax = 64, shading='gouraud')
        plt.colorbar().set_label('Intensidad')
        plt.show()
        
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
    
    return denoiseFourier(beforeCrisisClass), denoiseFourier(crisisClass), denoiseFourier(afterCrisisClass)

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
    calculateMean(beforeClassSignal, crisisClassSignal, afterClassSignal)
    calculateVariance(beforeClassSignal, crisisClassSignal, afterClassSignal)
    calculateStandardDeviation(beforeClassSignal, crisisClassSignal, afterClassSignal)
    
    plot_specgram(beforeClassSignal, crisisClassSignal,afterClassSignal, title='Espectrograma', x_label='Tiempo [S]', y_label='Frecuencia [Hz]')
    #Punto 3 de bloques enteros:
    probDistributions(beforeClassSignal, crisisClassSignal,afterClassSignal)
    exit()

def calculateMean(beforeClassSignal, crisisClassSignal, afterClassSignal):
    outputFile = open('output.txt', 'w')
    outputFile.write('Mean\n')
    outputFile.write('Before - Crisis - After\n')
    for i in range (0, len(beforeClassSignal)):
        outputFile.write(str(round(np.mean(beforeClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.mean(crisisClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.mean(afterClassSignal[i]),2)))
        outputFile.write('\n')
    outputFile.close()

def calculateVariance(beforeClassSignal, crisisClassSignal, afterClassSignal):
    outputFile = open('output.txt', 'a')
    outputFile.write('\n\nVariance\n')
    outputFile.write('Before - Crisis - After\n')
    for i in range (0, len(beforeClassSignal)):
        outputFile.write(str(round(np.var(beforeClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.var(crisisClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.var(afterClassSignal[i]),2)))
        outputFile.write('\n')
    outputFile.close()

def calculateStandardDeviation(beforeClassSignal, crisisClassSignal, afterClassSignal):
    outputFile = open('output.txt', 'a')
    outputFile.write('\n\nStandard Deviation\n')
    outputFile.write('Before - Crisis - After\n')
    for i in range (0, len(beforeClassSignal)):
        outputFile.write(str(round(np.std(beforeClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.std(crisisClassSignal[i]),2)) + ' - ')
        outputFile.write(str(round(np.std(afterClassSignal[i]),2)))
        outputFile.write('\n')
    outputFile.close()

showSignalsByClasses()