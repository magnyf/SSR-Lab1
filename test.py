import numpy as np
import matplotlib.pyplot as pl
import scipy as sp
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter
import scipy.signal
import math


example = np.load('lab1_example.npz')['example'].item()

### Additionnal functions
def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.

    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters

    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank

def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.

       mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
       lifter: lifering coefficient

    Returns:
       NxM array with lifeterd coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))

### MAIN

## Initial signal
# pl.plot(example['samples'])
# pl.show()

# Enframe part
samples = example['samples']
winlen = 20
winshift = 10

## sampling rate= 20kHz -> 400 for 20 ms
samplingRate = example['samplingrate'] # in kHz as the rest is in milliseconds

winlenPoints = winlen*samplingRate
winshiftPoints = winshift*samplingRate

# The shift is computed the Younes way (shifting from left to right)
frames = [samples[x:x+winlenPoints] for x in range(0, len(samples)-winlenPoints, winshiftPoints)]


# preemph
coef = 0.97
N = winlenPoints 
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

# cepstrum
filterbank = example['mspec']

print(example['mfcc'][0])

res =  [dct(filterbank[i],type=2, norm='ortho', axis= -1)[ :13] for i in range(len(filterbank))]  
print(res[0])
print(example['mfcc'] == res)




#pl.pcolormesh(preemp)
#pl.show


#pl.pcolormesh(frames)
#pl.show()

#pl.pcolormesh(example['frames'])
#pl.show()

#pl.pcolormesh(example['preemph'])
#pl.show()

preemph = np.array(example['preemph'])

# Generating the hamming window
hammingWindow = np.array(scipy.signal.hamming(len(preemph[0]), sym=False))

# Applying the hamming window filter
windowed =  [np.multiply(x, hammingWindow) for x in preemph]

##print(windowed == example['windowed'])

##pl.pcolormesh(windowed)
##pl.show()
##
##pl.pcolormesh(example['windowed'])
##pl.show()

#pl.pcolormesh(example['spec'])
#pl.show()

spec = example['spec']


nfft = len(spec[0])
mspecFilters = trfbank(samplingRate, nfft)


##pl.plot(mspecFilters)
##pl.show()

# Applying the filter to the fourier transform
mspec = np.matmul(spec, np.transpose(mspecFilters))

# Taking the log of the result
for i in range(len(mspec)):
    for j in range(len(mspec[i])):
        mspec[i][j] = math.log(mspec[i][j])

# Comparaison between the objective and our result (some elements are not equal, numerical instability?)
##print(mspec == example['mspec'])

##pl.pcolormesh(example['mspec'])
##pl.show()

##pl.pcolormesh(mspec)
##pl.show()

##pl.pcolormesh(example['mfcc'])
##pl.show()

mfcc = example['mfcc']

lmfcc = lifter(mfcc)

# Comparaison between the objective and our result
#print(lmfcc == example['lmfcc'])

#pl.pcolormesh(example['lmfcc'])
#pl.show()

#pl.pcolormesh(lmfcc)
#pl.show()


# pl.pcolormesh(example['lmfcc'])
# pl.show()

