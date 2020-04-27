# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:46:26 2020

@author: Ninha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model 
from skimage import morphology, measure

def load(filename = r'C:/Users/Ninha/Documents/Imperial/Y3 - Labs/A1 - '
         'Astronomical Image Processing/A1_mosaic.fits'):
    """
    Load the image 
    
    Parameters
    ----------
    filename : file path, file object, or file like object, optional
        File to get data from.  If opened, mode must be one of the
        following rb, rb+, or ab+. The default is r'C:/Users/Ninha/Documents/
        Imperial/Y3 - Labs/A1 - Astronomical Image Processing/A1_mosaic.fits'.

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    """
    data = fits.getdata(filename)   #Load the data as a numpy array
    return data

def header(filename =  r'C:/Users/Ninha/Documents/Imperial/Y3 - Labs/A1 - '
         'Astronomical Image Processing/A1_mosaic.fits'):
    """
    Obtain the header information from the image
    
    Parameters
    ----------
    filename : file path, file object, or file like object, optional
        File to get header from.  If an opened file object, its mode
        must be one of the following rb, rb+, or ab+). The default is r'C:/
        Users/Ninha/Documents/Imperial/Y3 - Labs/A1 - Astronomical Image 
        Processing/A1_mosaic.fits'.

    Returns
    -------
    header : TYPE
        DESCRIPTION.

    """
    header = fits.getheader(filename, 0)    #Load the header of the FITS image
    return header

def histogram(filename = r'C:/Users/Ninha/Documents/Imperial/Y3 - Labs/A1 - '
         'Astronomical Image Processing/A1_mosaic.fits'):
    """
    Plot a histogram of the pixel values within the image
    
    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is load().

    Returns
    -------
    None.

    """
    data = load(filename)
    head = header(filename)
    pixel_values = np.linspace(0,2**head['BITPIX']-head['SATURATE'],
                               2**head['BITPIX']-head['SATURATE'])
    occurences = np.bincount(data.flatten())[:2**head['BITPIX']-
                                             head['SATURATE']]
    plt.figure()
    #plt.bar is slow for large datasets
    plt.plot(pixel_values, occurences, 'k+', label = 'Data')
    plt.legend()
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of occurences')
    plt.title('Occurences against pixel value for the whole image')
    
def background(a = 344770, m = 3418, s = 12, width = 300,data = load()
               , header = header(), plot = False):
    """
    Determine the mean and standard deviation of the background pixel value 
    in the image
    
    Parameters
    ----------
    a : float, optional
        Initial guess for the amplitude of the gaussian. The default is 344770.
    m : float, optional
        Initial guess for the mean of the gaussian. The default is 3418.
    s : float, optional
        Initial guess for the standard deviation of the gaussian. The default 
        is 12.
    width : float, optional
        Number of considered pixels. The default is 300.
    data : TYPE, optional
        DESCRIPTION. The default is load().
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
  
    #Create the x and y lists for the histogram
    pixel_values = np.linspace(0,2**header['BITPIX']-1,2**header['BITPIX'])
    occurences = np.bincount(data.flatten())
    #Delete the count corresponding to the corners of the image
    pixel_values = np.delete(pixel_values, 3421)
    occurences = np.delete(occurences, 3421)
    #Only consider data within a given range of the guess mean for the background
    pixel_values = pixel_values[m-width//2:m+width//2]
    occurences = occurences[m-width//2:m+width//2]
    #Define a gaussian function to fit to the peak
    def gaussian(x, amplitude, mean, stand_dev):
        return amplitude*np.exp(-((x-mean)**2)/(2*stand_dev**2))
    #Least squares fit to data
    result = Model(gaussian).fit(occurences, x = pixel_values, 
                                 amplitude = a, mean = m, 
                                 stand_dev = s)
    #If plot is true, plot the data and resulting fit
    if plot == True:
        plt.figure()
        plt.bar(pixel_values, occurences, width = 1, color = 'black', 
                alpha = 0.3, label = 'Data')
        plt.plot(pixel_values, result.best_fit, color = 'black', 
                 label = ' Gaussian Fit')
        plt.legend()
        plt.xlabel('Pixel value')
        plt.ylabel('Occurences')
        plt.title('Number of occurences against pixel value near the peak of '
                  'the background')
    return result.values    #Return results as a dictionary
    
def remove_background(data = load(), background = background(),
                      save_fits= False):
    """
    Remove the background from the image by subtracting the mean background 
    count from the image and setting all of the resulting negative pixels to 0.

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is load().
    background : TYPE, optional
        DESCRIPTION. The default is background().
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    

    """
    
    
    #Subtract the background count and set negative values to zero
    data = data.astype('float64')
    data -= background['mean']
    data[data < 0] = 0
    #Save as a new image if required
    if save_fits == True:
        hdu = fits.PrimaryHDU(data)
        hdu.writeto('A1_mosaic_no_background.fits')
    return data

def binary_sources(std_thresh = 3.5, hole_area_thresh = 5, 
                   blob_size_thresh = 15, data = remove_background(), 
                   background = background(), plot = False):
    """
    Create a binary image of the sources present in the original image

    Parameters
    ----------
    n : float, optional
        The number of standard deviations above the mean in order for a pixel 
        value to result from a source. The default is 3.5.
    data : TYPE, optional
        DESCRIPTION. The default is remove_background().
    background : TYPE, optional
        DESCRIPTION. The default is background().

    Returns
    -------
    None.

    """
    
    #Remove n standard deviations and convert to a binary image
    data = data.astype('float64')
    data -= std_thresh*background['stand_dev']
    data[data < 0] = 0
    data[data > 0] = 1
    data = data.astype(np.int8)
    
# =============================================================================
#     labels = measure.label(data)
#     filled = morphology.remove_small_holes(data, area_threshold = 
#                                            hole_area_thresh)
#    
#     no_small_blobs = morphology.remove_small_objects(filled, min_size = 
#                                                      blob_size_thresh)
#     labels = measure.label(filled)
#     if plot == True:
#        fig, axes = plt.subplots(1,3, figsize = (9,3), sharex = True, 
#                                 sharey = True)
#        ax = axes.ravel()
#        ax[0].set_title('Data')
#        ax[0].imshow(data, cmap = 'gray')
#        ax[1].set_title('Filled')
#        ax[1].imshow(filled, cmap = 'gray')
#        ax[2].set_title('No small objects')
#        ax[2].imshow(no_small_blobs, cmap = 'gray')
#     return filled, labels
# =============================================================================
       
    #Fill in the holes within sources
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.max()
    mask = data

# =============================================================================
#     #filled = sp.ndimage.morphology.binary_fill_holes(data).astype(int)
# =============================================================================
    
    filled = morphology.reconstruction(seed, mask, method = 'erosion')
    #Remove objects less than or equal to 2 pixels across in any direction
    selem = morphology.disk(2)
    res = morphology.white_tophat(filled, selem)
    
    #label the blobs in the image
    
    labels = measure.label(filled-res)
    
    if plot == True:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, 
                                 sharey=True)
        ax = axes.ravel()
        ax[0].set_title('Filled')
        ax[0].imshow(filled, cmap = 'gray')
        ax[1].set_title('Filled-res')
        ax[1].imshow(filled-res, cmap = 'gray')
        ax[2].set_title('Res')
        ax[2].imshow(res, cmap = 'gray')
   
        
    return filled - res, labels
       
def sources(image = remove_background(),sources = binary_sources()[0],
            labels = binary_sources()[1],data2 = load(),header = header(), 
            plot = False):
    """
    

    Parameters
    ----------
    image : TYPE, optional
        DESCRIPTION. The default is remove_background().
    sources : TYPE, optional
        DESCRIPTION. The default is binary_sources()[0].
    labels : TYPE, optional
        DESCRIPTION. The default is binary_sources()[1].
    data2 : TYPE, optional
        DESCRIPTION. The default is load().
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    positions : TYPE
        DESCRIPTION.
    radii : TYPE
        DESCRIPTION.
    intensity : TYPE
        DESCRIPTION.

    """
    data = image*sources
    measures = measure.regionprops(labels, data)
# =============================================================================
#     for m in measures:
#         if m.max_intensity >= header['SATURATE']:
#             measures.remove(m)
# =============================================================================
    intensity = []
    positions = []
    radii = []
    #
    eccentricity = []
    peak_intensity = []
    for m in measures:
        intensity.append(m.area*m.mean_intensity)
        positions.append(m.centroid)
        radii.append(m.equivalent_diameter/2)
        #eccentricity.append(m.eccentricity)
        peak_intensity.append(m.max_intensity)
    peak_intensity += background()['mean']
    if plot == True:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, 
                                 sharey=True)
        ax = axes.ravel()
        im1 = ax[0].imshow(np.log2(data))
        ax[0].set_title('Sources Found')
        fig.colorbar(im1)
        im2 = ax[1].imshow(np.log2(data2))
        fig.colorbar(im2)
        ax[1].set_title('Original Image')
        for i in range(labels.max()):
            if peak_intensity[i] >= header['SATURATE']:
                c = plt.Circle([positions[i][1],positions[i][0]], radii[i], 
                               color = 'blue', linewidth = 2, fill = True)
            elif peak_intensity[i] < header['SATURATE']:
                 c = plt.Circle([positions[i][1],positions[i][0]], radii[i], 
                                color = 'red', linewidth = 2, fill = False) 
            ax[0].add_patch(c)
    return positions, radii, intensity, peak_intensity
   
def calibrate_fluxes(counts = sources()[2], header = header()):
    """
    

    Parameters
    ----------
    counts : TYPE, optional
        DESCRIPTION. The default is sources().
    standard_star : TYPE, optional
        DESCRIPTION. The default is header()['MAGZPT'].
    standard_star_error : TYPE, optional
        DESCRIPTION. The default is header()['MAGZRR'].

    Returns
    -------
    apparent_magnitude : TYPE
        DESCRIPTION.

    """
    standard_star = header['MAGZPT']
    standard_star_error = header['MAGZRR']
    apparent_magnitude = standard_star - 2.5*np.log10(counts)
    return apparent_magnitude

def catalogue(data = sources()):
    """
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is sources().

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    apparent_magnitude = calibrate_fluxes(counts = data[2])
    data = {'Position': data[0], 'Radius': data[1], 'Total pixel count': 
            data[2], 'Apparent Magnitude': apparent_magnitude}
    df = pd.DataFrame(data, columns = ['Position', 'Radius', 
                                       'Total pixel count', 
                                       'Apparent Magnitude'])
    return df

def number_count(data = catalogue()):
    """
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is catalogue().

    Returns
    -------
    None.

    """
    apparent_magnitude = data['Apparent Magnitude'].to_numpy()
    m = np.linspace(6, max(apparent_magnitude), 50)
    n_less_m = []
    for n in m:
        n_less_m.append(sum(i <= n for i in apparent_magnitude))
    x_lobf = np.linspace(6, max(m), 100)
    y, cov = np.polyfit(m[:42], np.log(n_less_m[:42]), deg = 1, full = False, 
                        cov = True)
    y_lobf = y[0]*x_lobf + y[1]
    plt.figure()
    plt.plot(m, np.log(n_less_m),'r+')
    plt.plot(x_lobf, y_lobf,'k-')
    return y[0], cov[0][0]

# =============================================================================
# n = np.arange(100)
# hdu = fits.PrimaryHDU(n)
# hdu.writeto('new2.fits')
# =============================================================================
