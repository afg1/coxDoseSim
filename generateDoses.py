import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from joblib import Parallel


imageSize = (128,128)

regionCentre = (64,64)
regionSize = (16,16)

NEvent = 50
NNoEvent = 50

Nperm = 1000

def genSDImage(loc, scale, amp):
    sd = np.zeros(imageSize)
    gaus = lambda x,mu,s : np.exp(-(x - mu)**2/(2.0*s**2))
    for i in range(0,imageSize[0]):
        sd[i,:] = gaus(i, loc, scale)

    return amp * sd


if __name__ == "__main__":


    sd = genSDImage(32, 32, 10)
    meanDoseE = np.zeros(imageSize)
    meanDoseE[:,:] = 60.0
    meanDoseN = np.zeros(imageSize)
    meanDoseN[:,:] = 60.0

    meanDoseE[regionCentre[0]-regionSize[0]:regionCentre[0]+regionSize[0], regionCentre[1]-regionSize[1]:regionCentre[1]+regionSize[1] ] = 58.0
    meanDoseE = random_noise(meanDoseE, clip=False, var=1.0)

    meanDoseN[regionCentre[0]-regionSize[0]:regionCentre[0]+regionSize[0], regionCentre[1]-regionSize[1]:regionCentre[1]+regionSize[1] ] = 60.0
    meanDoseN = random_noise(meanDoseN, clip=False, var=1.0)




    simDoseE = np.zeros((NEvent, imageSize[0], imageSize[1])) 
    simDoseN = np.zeros((NNoEvent, imageSize[0], imageSize[1]))

    print(simDoseE.shape)
    for i in range(0, NEvent):
        simDoseE[i, :,:] = gaussian_filter(np.random.normal(loc=meanDoseE, scale=sd, size=(imageSize[0], imageSize[1])), 5)
    for i in range(0, NNoEvent):
        simDoseN[i, :,:] = gaussian_filter(np.random.normal(loc=meanDoseN, scale=sd, size=(imageSize[0], imageSize[1])), 5)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # ax1.imshow(meanDoseN)
    # ax2.imshow(meanDoseE)

    ax1.imshow(simDoseE[0,:,:])
    ax1.set_title("Events")
    ax2.imshow(simDoseN[0,:,:])
    ax2.set_title("No events")
    plt.show()


    t_true, p_analytical = ttest_ind(simDoseE, simDoseN, axis=0)
    print(t_true.shape, p_analytical.shape)

    fig2 = plt.figure(2)
    ax2_1 = fig2.add_subplot(121)
    ax2_2 = fig2.add_subplot(122)
    ax2_1.imshow(t_true)
    # ax2_2.imshow(simDoseE[:,:,0])
    ax2_2.contour(p_analytical, levels=[0.05])

    plt.show()

    stackedDoses = np.vstack((simDoseE, simDoseN))

    print(stackedDoses.shape)

    permutedT = np.zeros((Nperm, *imageSize))
    perm_p = np.ones(imageSize)

    for n in range(0, Nperm):
        permuted = np.random.permutation(stackedDoses)
        permutedT[n,:,:], _ = ttest_ind(permuted[:NEvent], permuted[NNoEvent:], axis=0)
        perm_p[np.where(np.abs(permutedT[n,:,:]) > np.abs(t_true))] += 1

    perm_p /= Nperm

    fig3 = plt.figure(3)
    ax3_1 = fig3.add_subplot(121)
    ax3_2 = fig3.add_subplot(122)
    ax3_1.imshow(t_true)
    ax3_2.contour(perm_p, levels=[0.05], colors='r')
    ax3_2.contour(p_analytical, levels=[0.05], colors='b')
    plt.show()

