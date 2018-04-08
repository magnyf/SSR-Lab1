from proto import *
import numpy as np
import matplotlib.pyplot as pl


## Part 4
example = np.load('lab1_example.npz')['example'].item()

mfccOutput = mfcc(example['samples'])

# pl.pcolormesh(example['lmfcc'])
# pl.show()

# pl.pcolormesh(mfccOutput)
# pl.show()

## Part 5
## Feature correlation for MFCC
data = np.load('lab1_data.npz')['data']

dataMfcc = []
for element in data:
	for frame in mfcc(element['samples']):
		dataMfcc += [frame]

coreCoeffs = np.corrcoef(np.transpose(dataMfcc))
pl.pcolormesh(coreCoeffs)
pl.colorbar()
pl.show()

## Feature correlation for mspec
def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

dataMspec = []
for element in data:
	for frame in mspec(element['samples']):
		dataMspec += [frame]

coreCoeffsMspec = np.corrcoef(np.transpose(dataMspec))
pl.pcolormesh(coreCoeffsMspec)
pl.colorbar()
pl.show()

# Computing distance between utterances

