from proto import *
import numpy as np
import matplotlib.pyplot as pl
import scipy.cluster.hierarchy
from sklearn import mixture

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

# coreCoeffs = np.corrcoef(np.transpose(dataMfcc))
# pl.pcolormesh(coreCoeffs)
# pl.colorbar()
# pl.show()

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

# coreCoeffsMspec = np.corrcoef(np.transpose(dataMspec))
# pl.pcolormesh(coreCoeffsMspec)
# pl.colorbar()
# pl.show()


# Analysis of the data
# print([ element['digit'] for element in data])
# print([ element['speaker'] for element in data])

## Part 6
# Computing distance between utterances
# distanceUtterances = [np.zeros(len(data)) for x in range(len(data))]
# for i in range(len(data)):
# 	for j in range(i+1, len(data)):
# 		distanceUtterances[i][j] += dtw(mfcc(data[i]['samples']), mfcc(data[j]['samples']), dist)[0]
# 		distanceUtterances[j][i] = distanceUtterances[i][j]
# 	print("Line nb "+str(i)+' computed {:.2%}.'.format(i/len(data)))

# np.savetxt('distanceUtterances.txt', distanceUtterances, fmt='%f')
distanceUtterances = np.loadtxt('distanceUtterances.txt', dtype=float)

# pl.pcolormesh(distanceUtterances)
# pl.colorbar()
# pl.show()


link = scipy.cluster.hierarchy.linkage(distanceUtterances, method='complete')
# scipy.cluster.hierarchy.dendrogram(link, labels=tidigit2labels(data))
# pl.show()

# to plot with the same coefficient
#for n in [32]:
#	g = mixture.GaussianMixture(n_components=n)
#	g.fit(dataMfcc)
   #     pl.figure(1)
  #      pl.title("Word \"seven\" with 32 component")
 #       j = 1
#	for i in [16,17,24,38,39]:
#		result = []
  #              if (i!= 24):
 #                       pl.subplot(3,2,j)
#		        pl.pcolormesh(g.predict_proba(mfcc(data[i]['samples'])))
#                        pl.title("\"seven\" nb "+ str( i))
#		        pl.colorbar()
#                        j = j + 1
#                else:
#                        pl.subplot(3,2,5)
#                        pl.pcolormesh(g.predict_proba(mfcc(data[i]['samples'])))
#                        pl.title("\"zero\" nb "+str(i))
#        pl.show()



# to plot with the same word (diff coefficient)
pl.figure(1)
pl.title("Word \"seven\" nb 16")
j = 1
for n in [4,8,16,32]:
	g = mixture.GaussianMixture(n_components=n)
	g.fit(dataMfcc)
	result = []
        pl.subplot(2,2,j)
	pl.pcolormesh(g.predict_proba(mfcc(data[16]['samples'])))
        pl.title("\"seven\" with  "+ str( n ) + " coefficients")
	pl.colorbar()
        j = j + 1
pl.show()
