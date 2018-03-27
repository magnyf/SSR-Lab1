import numpy as np
import matplotlib.pyplot as pl

example = np.load('lab1_example.npz')['example'].item()

pl.plot(example['samples'])
pl.show()

pl.pcolormesh(example['frames'])
pl.show()

pl.pcolormesh(example['preemph'])
pl.show()

pl.pcolormesh(example['windowed'])
pl.show()

pl.pcolormesh(example['spec'])
pl.show()

pl.pcolormesh(example['mspec'])
pl.show()

pl.pcolormesh(example['mfcc'])
pl.show()

pl.pcolormesh(example['lmfcc'])
pl.show()