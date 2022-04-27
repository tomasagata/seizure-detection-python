import pyedflib.highlevel as highlevel
import numpy as np

if __name__ == '__main__':

    arch = input("dubdirectorio del archivo edf a recortar: ")
    ti = int(input("tiempo inicial de ceizure: ")) * 256
    tf = int(input("tiempo final de ceizure: ")) * 256

    if ti - 30720 < 0:
        ti = 0
    
    signals, signal_headers, header = highlevel.read_edf(arch)

    numOfSignals = len(signals)
    N = len(signals[0])

    if tf + 30720 > N:
        tf = N

    print(tf-ti )

    highlevel.write_edf_quick(arch + ".ceizure.edf", signals[:numOfSignals,ti:tf], 256)