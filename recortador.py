import pyedflib.highlevel as highlevel
import numpy as np
import json
from pathlib import Path

def searchExistingFilesAccordingTo(seizureFile):
    f = open(seizureFile, "r", encoding='utf-8')
    arr = json.loads(f.read())
    f.close()

    return_arr = []
    for arrObject in arr:
        p = Path(arrObject["subdirectory"] + arrObject["fileName"])
        if p.exists() and p.is_file():
            return_arr.append(arrObject)
        else:
            print("file " + str(p) + " not found. Omitting..." )

    if len(arr) == len(return_arr):
        print("all files found!")
    elif len(return_arr) == 0:
        print("no files found. Exiting...")
        exit()
    else:
        print("found " + str(len(return_arr)) + " out of " + str(len(arr)) + " files.")

    return return_arr


if __name__ == '__main__':

    seizures = searchExistingFilesAccordingTo("seizures.json")

    lastFileWritten = ""
    counter = 1

    for seizureFile in seizures:
        arch = seizureFile["subdirectory"] + seizureFile["fileName"]
        ti = seizureFile["startSeizureTime"] * 256
        tf = seizureFile["endSeizureTime"] * 256

        if ti - 30720 < 0:
            ti = 0
        else:
            ti -= 30720

        signals, signal_headers, header = highlevel.read_edf(arch)

        new_signal_headers = []
        for i in range( len(signal_headers)):
            new_signal_headers.append(signal_headers[i])
            new_signal_headers[i]["physical_max"] = 3000.0
            new_signal_headers[i]["physical_min"] = -3000.0 # modulo 3 veces mas grande que el original!

        numOfSignals = len(signals)
        N = len(signals[0])

        if tf + 30720 > N:
            tf = N
        else:
            tf += 30720
        
        if lastFileWritten == arch:
            writeFileName = arch + ".seizureFile" + str(counter) + ".edf"
            counter+=1
        else:
            writeFileName = arch + ".seizureFile.edf"
            counter = 1
            
        lastFileWritten = arch

        highlevel.write_edf(writeFileName, signals[:numOfSignals, ti:tf], new_signal_headers, header, False, 1, 1)

        print("successfully cut and written " + seizureFile["fileName"] + " at: \""+ writeFileName + "\"")