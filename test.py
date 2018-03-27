import numpy as np
import matplotlib.pyplot as pl

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
	
pl.pcolormesh(frames)
pl.show()

pl.pcolormesh(example['frames'])
pl.show()

# pl.pcolormesh(example['preemph'])
# pl.show()

# pl.pcolormesh(example['windowed'])
# pl.show()

# pl.pcolormesh(example['spec'])
# pl.show()

# pl.pcolormesh(example['mspec'])
# pl.show()

# pl.pcolormesh(example['mfcc'])
# pl.show()

# pl.pcolormesh(example['lmfcc'])
# pl.show()