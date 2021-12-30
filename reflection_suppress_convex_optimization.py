import numpy as np
import scipy
import cv2
from copy import deepcopy

"""
Normalizing the image 
"""


def im2double(im):
    try:
        N, M, D = im.shape
    except:
        N, M = im.shape
        D = 1
    y = np.zeros(im.shape, dtype=float)
    for i in range(D):
        min_pixel = np.min(im[:, :, i])
        max_pixel = np.max(im[:, :, i])
        y[:, :, i] = (im[:, :, i] - min_pixel)*1.0/(max_pixel - min_pixel)
    return y


"""
Defining the threshold function
"""


def hard_threshold(array, threshold):
    cpy_array = deepcopy(array)
    indices = array <= threshold
    cpy_array[indices] = 0.0
    return cpy_array


"""
Suppression Algorithm
"""


def reflectSuppress(Im, h, epsilon):

    Y = im2double(Im)
    m, n, r = Y.shape
    T = np.zeros(Y.shape)
    Y_Laplacian_2 = np.zeros(Y.shape)

    for dim in range(0, r):

        GRAD = grad(Y[:, :, dim])
        GRAD_x = GRAD[:, :, 0]
        GRAD_y = GRAD[:, :, 1]
        GRAD_norm = np.sqrt(GRAD_x**2 + GRAD_y**2)

        GRAD_norm_thresh = hard_threshold(GRAD_norm, h)
        ind = (GRAD_norm_thresh == 0)

        GRAD_x[ind] = 0
        GRAD_y[ind] = 0

        GRAD_thresh = np.zeros((GRAD.shape[0], GRAD.shape[1], 2))

        GRAD_thresh[:, :, 0] = GRAD_x
        GRAD_thresh[:, :, 1] = GRAD_y
        Y_Laplacian_2[:, :, dim] = div(grad(div(GRAD_thresh)))

    rhs = Y_Laplacian_2 + epsilon * Y

    for dim in range(0, r):
        T[:, :, dim] = PoissonDCT_variant(rhs[:, :, dim], 1, 0, epsilon)

    return T


"""
DCT implementation using Numpy
"""


def dct(y):
    N = len(y)
    y2 = np.empty(2*N, float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = np.rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])


"""
IDCT implementation using Numpy
"""


def idct(y):
    N = len(y)
    y2 = np.empty(2*N, float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = np.rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])


"""
DCT2 implementation for layerwise computation
"""


def dct2(y):
    M, N = y.shape
    a = np.zeros(y.shape)
    b = np.zeros(y.shape)

    for i in range(M):
        a[i, :] = scipy.fft.dct(y[i, :])
    for j in range(N):
        b[:, j] = scipy.fft.dct(a[:, j])

    return b


"""
IDCT2 implementation for layerwise calculation
"""


def idct2(y):
    M, N = y.shape
    a = np.zeros(y.shape)
    b = np.zeros(y.shape)

    for j in range(N):
        b[:, j] = scipy.fft.idct(y[:, j])

    for i in range(M):
        a[i, :] = scipy.fft.idct(b[i, :])

    return a


"""
solve the equation  (mu*L^2 - lambda*L + epsilon)*u = rhs via DCT 
where L means Laplacian operator
"""


def PoissonDCT_variant(rhs, mu, lamda, epsilon):

    M, N = rhs.shape

    k = np.array([range(1, M+1)])
    l = np.array([range(1, N+1)])
    k = k.T
    eN = np.ones((1, N))
    eM = np.ones((M, 1))
    k = np.cos(np.pi/M*(k-1))
    l = np.cos(np.pi/N*(l-1))

    k = scipy.sparse.kron(k, eN)
    l = scipy.sparse.kron(eM, l)

    kappa = 2*(k.toarray()+l.toarray()-2)
    const = mu * kappa**2 - lamda * kappa + epsilon
    u = dct2(rhs)
    u = u/const
    u = idct2(u)

    return u


"""
 compute the gradient of a 2D image array
"""


def grad(A):

    m, n = A.shape
    B = np.zeros((m, n, 2))

    Ar = np.zeros((m, n))
    Ar[:, 0:n-1] = A[:, 1:n]
    Ar[:, n-1] = A[:, n-1]

    Au = np.zeros((m, n))
    Au[0:m-1, :] = A[1:m, :]
    Au[m-1, :] = A[m-1, :]

    B[:, :, 0] = Ar-A
    B[:, :, 1] = Au-A

    return B


""" 
compute the divergence of gradient
Input A is a matrix of size m*n*2
A[:,:,0] is the derivative along the x direction
A[:,:,1] is the derivative along the y direction
"""


def div(A):

    m, n, _ = A.shape
    B = np.zeros((m, n))

    T = A[:, :, 0]
    T1 = np.zeros((m, n))
    T1[:, 1:n] = T[:, 0:n-1]

    B = B+T-T1

    T = A[:, :, 1]
    T1 = np.zeros((m, n))
    T1[1:m, :] = T[0:m-1, :]

    B = B+T-T1

    return B


if __name__ == "__main__":

    file_path = '<path to file>'

    # Threshold below which gradients will be clipped of
    h = 0.043

    # epsilon term added to cost function to make it convex. Should not be high to impact the data fidelity cost function
    e = 1e-10

    img = cv2.imread(file_path)  # Read image here
    out = im2double(img.astype('float'))  # Convert to normalized floating poin

    # Running the main algorithm
    L1 = reflectSuppress(out, h, e)
    L2 = out - L1

    cv2.imshow("Original Image", img)
    cv2.imshow("Image without Reflection", L1)
    cv2.imshow("Reflection of Image", L2)
