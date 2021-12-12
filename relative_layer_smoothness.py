import numpy as np
import cv2
from scipy.signal import convolve2d


"""
Zero padding the image
"""


def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


"""
Converting Point Spread function to optical transfer function
"""


def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')

    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    otf = np.fft.fft2(psf)

    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


"""
Normalizing the image
"""


def im2double(im):
    try:
        N, M, D = im.shape
    except:
        N, M = im.shape
        D = 1
    for i in range(D):
        min_pixel = np.min(im[:, :, i])
        max_pixel = np.max(im[:, :, i])
        im[:, :, i] = (im[:, :, i] - min_pixel)/(max_pixel - min_pixel)
    return im


"""
Applying filters over the image in circular manner
"""


def imfilter(I, f):
    try:
        m, n = f.shape
        M, N = I.shape
        if m == 1:
            order = list(range(1, N)) + [0]
            Y = -convolve2d(I, f, mode='same', boundary='wrap')
            Y = Y[:, order]
        elif n == 1:
            order = list(range(1, M)) + [0]
            Y = -convolve2d(I, f, mode='same', boundary='wrap')
            Y = Y[order, :]
        else:
            Y = convolve2d(I, f, mode='same', boundary='wrap')
        return Y
    except Exception as e:
        print(str(e))


"""
Channel wise FFT2 computation
"""


def fft2(I):
    ff = np.zeros(I.shape, dtype=complex)
    N, M, D = I.shape
    for channel in range(D):
        ff[:, :, channel] = np.fft.fft2(I[:, :, channel])
    return ff


"""
Channel wise IFFT2 computation
"""


def ifft2(ff):
    I = np.zeros(ff.shape)
    N, M, D = ff.shape
    for channel in range(D):
        I[:, :, channel] = np.fft.ifft2(ff[:, :, channel])
    return I


"""
Main Algorithm for reflection Removal
"""


def reflection_removal(I, lamda, lb, hb, L1_0=None):
    N, M, D = I.shape

    # filters
    f1 = np.array([[1, -1]])
    f2 = np.array([[1], [-1]])
    f3 = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

    sizeI2D = (N, M)
    otfFx = psf2otf(f1, sizeI2D)
    otfFy = psf2otf(f2, sizeI2D)
    otfL = psf2otf(f3, sizeI2D)

    Normin1 = np.multiply(np.repeat(abs(otfL)[:, :, np.newaxis], 3, axis=2)**2, fft2(I))
    Denormin1 = abs(otfL)**2
    Denormin2 = abs(otfFx)**2 + abs(otfFy)**2

    if D > 1:
        Denormin1 = np.repeat(Denormin1[:, :, np.newaxis], D, axis=2)
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)

    eps = 1e-16
    L1 = L1_0

    thr = 0.05

    for i in range(1, 4):

        beta = 2**(i-1)/thr
        Denormin = lamda*Denormin1 + beta*Denormin2  # This is correct

        # update g
        gFx = np.zeros(L1.shape)
        gFy = np.zeros(L1.shape)

        for channel in range(D):
            gFx[:, :, channel] = -1.0*imfilter(L1[:, :, channel], f1)
            gFy[:, :, channel] = -1.0*imfilter(L1[:, :, channel], f2)

        t = np.repeat((np.sum(abs(gFx), axis=2) < 1.0/beta)[:, :, np.newaxis], D, axis=2)
        gFx[t] = 0

        t = np.repeat((np.sum(abs(gFy), axis=2) < 1.0/beta)[:, :, np.newaxis], D, axis=2)
        gFy[t] = 0

        Normin2 = np.concatenate(((gFx[:, -1, :] - gFx[:, 0, :]).reshape(-1, 1, D), -np.diff(gFx, 1, 1)), axis=1)
        Normin2 = Normin2 + np.concatenate(((gFy[-1, :, :] - gFy[0, :, :]).reshape(1, -1, D), -np.diff(gFy, 1, 0)), axis=0)

        FL1 = np.divide((lamda*Normin1 + beta*fft2(Normin2)), (Denormin+eps))
        L1 = np.real(ifft2(FL1))

        # normalize L1
        for c in range(0, D):
            L1t = L1[:, :, c]
            for k in range(0, 500):
                dt = (sum(L1t[L1t < lb[:, :, c]]) + sum(L1t[L1t > hb[:, :, c]]))*2/len(L1t.ravel())
                L1t = L1t-dt
                if abs(dt) < 1/len(L1t.ravel()):
                    break
            L1[:, :, c] = L1t

        t = L1 < lb
        L1[t] = lb[t]
        t = L1 > hb
        L1[t] = hb[t]

        # L2
        L2 = I-L1

    return L1, L2


"""
I : Image
Lambda : Regularization parameter
lb: lower bound of the Layer 1,need to be same dimention with input I 
hb: upper bound of the Layer 1,need to be same dimention with input I
I0: initialization of Layer 1, default as the input I
"""
if __name__ == "__main__":
    file_path = "<file path>"

    # Reading and normalizing the image
    img = cv2.imread(file_path)
    out = im2double(img.astype('float'))

    # Running the core algorithm
    L1, L2 = reflection_removal(out, 10, np.zeros(out.shape), out, out)

    # Showing images - Additional multiplicative constant to improve the brightness
    cv2.imshow("Original Image", img)
    cv2.imshow("Image without Reflection", L1*1.7)
    cv2.imshow("Reflection of Image", L2*2)
