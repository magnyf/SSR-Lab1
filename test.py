import numpy as np
import matplotlib.pyplot as pl
import scipy as sp
from scipy.fftpack import fft
from scipy.signal import lfilter

example = np.load('lab1_example.npz')['example'].item()

## Initial signal
# pl.plot(example['samples'])
# pl.show()

# Enframe part
samples = example['samples']
winlen = 20
winshift = 10

## sampling rate= 20kHz -> 400 for 20 ms
samplingRate = 20 # in kHz as the rest is in milliseconds

winlenPoints = winlen*samplingRate
winshiftPoints = winshift*samplingRate

# The shift is computed the Younes way (shifting from left to right)
frames = [samples[x:x+winlenPoints] for x in range(0, len(samples)-winlenPoints, winshiftPoints)]

# preemph
coef = 0.97
N = len(frames[0])
b = np.array([1.0 - coef for i in range(N)])
b[0] = 1.
a = np.array([1. for i in range(N)])
preemp = [ lfilter(b,a, frames[i]) for i in range(len(frames))]

#diff = [[preemp[i][j] - example['preemph'][i][j] for j in range(len(preemp[0]))] for i in range(len(preemp))]
#print(diff[0])


# Fast Fourrier Transform
#powerSpectrum(input, nfft)
entree= example['windowed']
nfft  = 512
N = len(entree)
output = [ sp.fftpack.fft(entree[i], nfft) for i in range(N)]
output = [ abs(output[i])**2 for i in range(N)]

#value = True
#for i in range(N):
#    if (value == False):
#        break;
#    for j in range(nfft):
#        if (value == False):
#            break;
#        value = value and (example['spec'][i][j] == output[i][j])

#print(value)



#pl.pcolormesh(preemp)
#pl.show


#pl.pcolormesh(frames)
#pl.show()

#pl.pcolormesh(example['frames'])
#pl.show()

#pl.pcolormesh(example['preemph'])
#pl.show()

# pl.pcolormesh(example['windowed'])
# pl.show()

#pl.pcolormesh(example['spec'])
#pl.show()

# pl.pcolormesh(example['mspec'])
# pl.show()

# pl.pcolormesh(example['mfcc'])
# pl.show()

# pl.pcolormesh(example['lmfcc'])
# pl.show()
