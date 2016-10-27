import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    m, n = kernel.shape
    m2, n2 = m // 2, n // 2
    mn = m * n
    kernel = np.ravel(kernel)

    newImg = np.empty(img.shape)

    if img.ndim == 3:
        h, w, chans = img.shape
    else:
        chans = 1
        h, w = img.shape
        # Make img always end up with 3 dimensions
        img = img[:, :, np.newaxis]

    paddedWorkspace = np.zeros((h + m - 1, w + n - 1, chans), dtype=img.dtype)
    # Pad the original image in our reusable workspace
    paddedWorkspace[m2:m2+h, n2:n2+w] = img

    for x in xrange(w):
        for y in xrange(h):
            # Extract the area from the workspace we are cross-correlating
            # and compute dot product
            sliced = np.reshape(paddedWorkspace[y:y+m, x:x+n], (mn, chans))
            newImg[y, x] = np.dot(kernel, sliced)

    return newImg

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, np.flipud(np.fliplr(kernel)))

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # Radius of Gaussian
    rx, ry = (1 - width) * 0.5, (1 - height) * 0.5

    # We assume each element of the kernel takes its value from the
    # distance of its midpoint to the center of the Gaussian 
    kernelX = np.arange(rx, rx + width, 1.0) ** 2
    kernelY = np.arange(ry, ry + height, 1.0) ** 2

    # Formula for Gaussian is exp(-(x^2+y^2)/(2sigma^2))/(2pi*sigma^2)
    # However we will compute the kernel from the outer product of two vectors
    twoSig2 = 2 * sigma * sigma

    kernelX = np.exp(- kernelX / twoSig2)
    kernelY = np.exp(- kernelY / twoSig2) / (twoSig2 * np.pi)

    # Kernel is outer product
    kernel = np.outer(kernelX, kernelY)

    # We need to normalize the kernel
    s = np.sum(kernelY) * np.sum(kernelX)

    return kernel / s
    
def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
