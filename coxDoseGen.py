import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm, lognorm
import nibabel as nib


## Ensure repeatability during testing
np.random.seed(12345)


## Constants
euler = 0.5772 




"""
The idea here is to create 128x128 images with a dose sensitive region where doses in that region are drawn from two distributions
with different means. We then construct a cox model to sample srvival times. The model will depend heavily on another covariate
such that we can remove the dependence on the dose sensitive region or make it stronger.
"""

def generateNoise(shape, blocksize=1, mean=0.0, scale=1.0):
    """
    Generates images with noise. The blocksize is the overall size of the 'pixels' filled with noise.
    For simplicity, blocksize should be a multiple of the image size. It will work even if not, but might look weird
    """
    if blocksize == 1:
        noise = np.random.normal(loc=mean, scale=scale, size=shape)
    else:
        noise = np.zeros(shape) ## has to be the right shape
        for ix,iy in product(range(0, int(shape[0]), blocksize), range(0, int(shape[1]), blocksize)):
            noise[ix:ix+blocksize, iy:iy+blocksize] = np.random.normal(loc=mean, scale=scale)

    # We probably want to do a gaussian blur on the noise now
    noise = gaussian_filter(noise, blocksize)

    return noise


def genSDImage(loc, scale, amp):
    """
    Generates a standard deviation image where the sd is gaussian distributed in space. loc is the y-coordinate where the maximum
    standard deviation is wanted
    """
    sd = np.zeros(imageSize)
    gaus = lambda x,mu,s : np.exp(-(x - mu)**2/(2.0*s**2))
    for i in range(0,imageSize[0]):
        sd[i,:] = gaus(i, loc, scale)

    sd *= amp
    sd[np.where(sd < 1.0)] = 1.0

    return sd

def generateDose(shape, region, bgdose=50.0, std=10.0, enhancement=2.0):
    """
    Generate a dose distribution of the correct shape, with background dose and an enhanced dose in a specified region.
    shape is the tuple image shape, eg (128,128). region is a list of indices where we want the enhancement.
    """
    mean = np.ones(shape) * bgdose
    stdImg = genSDImage(np.min(region[0]), (np.max(region[0]) - np.min(region[0]))/2, std)

    # plt.imshow(stdImg)
    # plt.show()

    mean[region] += enhancement

    bg = np.random.normal(loc=0.0, scale=stdImg)
    bg += mean
    ## smooth the dose a bit by convolving with a gaussian
    bg = gaussian_filter(bg, 10)

    return bg

def T(alpha, lam, N, betas, covSamples):
    """
    Return sampled survival times for a given set of covariates.

    """
    recipAlpha = 1.0 / alpha


    topline = alpha * np.log(np.random.uniform(size=N))
    # print(covSamples[0] ) 
    bottomline = lam * np.exp(betas[0]*covSamples[0] + betas[1]*covSamples[1])

    samples = recipAlpha * np.log(1.0 - (topline/bottomline))

    return samples

## Definitions for the images
imageSize = (128,128)
regionCentre = (64,64) ## These two match the distributions in the Chen paper
regionSize = (50,50)


## Define indices for the region in the centre
region = np.ogrid[int(regionCentre[0] - regionSize[0]/2):int(regionCentre[0] + regionSize[0]/2), int(regionCentre[1] - regionSize[1]/2):int(regionCentre[1] + regionSize[1]/2)]

def ChenPaperPermutationTest(doses, labels):
    pass

def example():
    ## Example calculation.
    noise = generateNoise(imageSize, blocksize=4, mean=0.0, scale=1.0)

    dose = generateDose(imageSize, region, bgdose=60.0, enhancement=-2.0)
    dose += noise



    plt.imshow(dose) 
    plt.colorbar()
    plt.show()

    noise = generateNoise(imageSize, blocksize=4, mean=0.0, scale=1.0)
    eventDose = generateDose(imageSize, region, bgdose=10.0, enhancement=-2.0)
    eventDose += noise




    print(np.mean(dose[region]), np.mean(eventDose[region]))



    # deadImage = nib.Nifti1Image(deadArray, np.eye(4))
    # nib.save(deadImage, sampleImagesDirectory + "generated{0}.nii".format(n))

if __name__ == "__main__":

    # example()

    ## constants for the Cox regression
    stdDev = 12.5
    mean = 15.6 ## std and mean survival time

    alpha = np.pi / (np.sqrt(6) * stdDev)
    lam = alpha * np.exp(-euler -(alpha * mean))

    uniformDose = 2.0 ## This makes it a bit more like the heart dose from Alan's paper
    ## Change uniformDose to 60 to replicate Chen paper


    ## Generate some doses with and without enhancement
    NNoE = 500
    NwE  = 500 ## two groups with 50 images each

    ## Create some arrays to hold the data
    noEnhancementGroup  = np.zeros((NNoE, *imageSize))
    withEnhancementGroup = np.zeros((NwE, *imageSize))


    for i in range(NNoE):
        noEnhancementGroup[i,:,:] = np.abs(generateDose(imageSize, region, bgdose=uniformDose, enhancement=0.0))# + generateNoise(imageSize, blocksize=4, mean=0.0, scale=1.0)

    for i in range(NwE):
        withEnhancementGroup[i,:,:] = generateDose(imageSize, region, bgdose=uniformDose, enhancement=2.0)# + generateNoise(imageSize, blocksize=4, mean=0.0, scale=1.0)

    combinedDose = np.vstack((noEnhancementGroup, withEnhancementGroup))
    N = combinedDose.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(noEnhancementGroup[0,:,:])
    ax2.imshow(withEnhancementGroup[0,:,:])
    plt.show()


    ## Now generate some survival times. Assume a covariate which is something like tumour size - a truncated normal with mean about 20. The other covariate is the per-voxel dose.abs
    ## Assume equal betas to begin with. beta = 5E-3

    covariate1 = truncnorm.rvs(-1, np.inf, loc=20, scale=20, size=combinedDose.shape[0])

    # plt.hist(covariate1, bins=100)
    # plt.show()

    covariate2 = np.zeros(combinedDose.shape[0])
    for i in range(covariate2.shape[0]):
        thisDose = combinedDose[i, :,:]
        covariate2[i] = np.mean(thisDose[region])

    # plt.hist(covariate2)
    # plt.show()

    betaCov1 = 1.0E-3
    betaCov2 = 5E-2


    betas = [betaCov1, betaCov2]

    sampledTimes = T(alpha, lam, N, betas, [covariate1 ,covariate2])
    plt.hist(sampledTimes)
    plt.show()

    ## Specify a time to select dead/alive
    timePoint = 12.0
    labels = sampledTimes < 12.0
    labels = labels.astype(np.int32)
    # print(labels.astype(np.int32))

    originalLabels = np.hstack((np.zeros(NNoE), np.ones(NwE)))

    with open("coxData.txt", 'w') as coxData:
        coxData.write("time,status,t,d,hiLo\n")
        for i in range(N):
            coxData.write("{0},{1},{2:.2f},{3:.2f},{4}\n".format(int(sampledTimes[i]), labels[i], covariate1[i], covariate2[i], originalLabels[i]) )
    matrix = np.eye(4)
    with open("simulationCovariates.txt", 'w') as cv, open("simulationIds.txt", 'w') as pid, open("studentStatus.txt", 'w') as sts:
        for i,(time,status,tsize) in enumerate(zip(sampledTimes, originalLabels, covariate1)):
            cv.write("{0}\t{1}\t{2:.2f}\n".format(int(time),status,tsize))
            sts.write("{0}\n".format(status))
            pid.write("{0}.nii\n".format(i))
            thisImage = nib.Nifti1Image(np.rot90(combinedDose[i,:,:], k=1), matrix)
            nib.save(thisImage, "generatedImages\\{0}.nii".format(i))
