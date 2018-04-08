from proto import *
import numpy as np


example = np.load('lab1_example.npz')['example'].item()

mfcc(example)
