from proto import *
import numpy as np
import matplotlib.pyplot as pl

example = np.load('lab1_example.npz')['example'].item()

mfccOutput = mfcc(example['samples'])


pl.pcolormesh(example['lmfcc'])
pl.show()

pl.pcolormesh(mfccOutput)
pl.show()
