# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 01:15:02 2021

@author: ratan
"""


"""
I : Image
Lambda : Regularization parameter
lb: lower bound of the Layer 1,need to be same dimention with input I 
hb: upper bound of the Layer 1,need to be same dimention with input I
I0: initialization of Layer 1, default as the input I
"""

import numpy as np
import cv2
from scipy.signal import convolve2d


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




def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf



def im2double(im):  
    try:
        N, M, D = im.shape
    except:
        N,M = im.shape
        D = 1
    print(N,M,D)
    for i in range(D):
      print(i)
      min_pixel = np.min(im[:,:,i])
      max_pixel = np.max(im[:,:,i])
      im[:,:,i] = (im[:,:,i] - min_pixel)/(max_pixel - min_pixel)
    return im




def imfilter(I, f):
  #-p[[1,2,3,4,0],:] - f2 (Reorder rows) (2,1)
  #-p[:,[1,2,3,4,0]] -f1 (Reorder columns) (1,2)
  # p - f3

  try:
    m,n = f.shape
    print("m - {}, n - {}".format(m,n))
    M, N = I.shape
    if m == 1:
      order = list(range(1,N)) + [0]
      Y = -convolve2d(I, f, mode='same', boundary='wrap')
      print(Y.shape)
      Y = Y[:, order]
    elif n == 1:
      order = list(range(1,M)) + [0]
      Y = -convolve2d(I, f, mode='same', boundary='wrap')
      print(Y.shape)
      Y = Y[order, :]
    else:
      Y = convolve2d(I, f, mode='same', boundary='wrap')
    print("Y Shape - {}".format(Y.shape))    
    return Y
  except Exception as e:
    print(str(e))



def fft2(I):
    ff = np.zeros(I.shape, dtype=complex) 
    N,M,D = I.shape
    for channel in range(D):
        ff[:,:,channel] = np.fft.fft2(I[:,:,channel])
    return ff

def ifft2(ff):
    I = np.zeros(ff.shape) 
    N,M,D = ff.shape
    for channel in range(D):
        I[:,:,channel] = np.fft.ifft2(ff[:,:,channel])
    return I



def reflection_removal(I, lamda, lb, hb, L1_0 = None):
    N, M, D = I.shape;
    
    # filters
    f1 = np.array([[1, -1]])
    f2 = np.array([[1], [-1]])
    f3 = np.array([[0, -1, 0], 
          [-1, 4, -1],
          [0, -1, 0]])
    
    sizeI2D = (N, M)
    otfFx = psf2otf(f1,sizeI2D)
    otfFy = psf2otf(f2,sizeI2D)
    otfL = psf2otf(f3,sizeI2D)
    
    Normin1 = np.multiply(np.repeat(abs(otfL)[:, :, np.newaxis], 3, axis = 2)**2, fft2(I))
    Denormin1 = abs(otfL)**2 
    Denormin2 = abs(otfFx)**2 + abs(otfFy )**2
    
    print("-*"*10)
    print(np.sum(fft2(I)))
    print(np.sum(np.repeat(abs(otfL)[:, :, np.newaxis], 3, axis = 2)))
    print(np.sum(Denormin1))
    print(np.sum(Denormin2))
    print("-*"*10)

    
    if D>1:
        Denormin1 = np.repeat(Denormin1[:,:, np.newaxis], D, axis=2)
        Denormin2 = np.repeat(Denormin2[:,:, np.newaxis], D, axis=2)
    
    
    eps = 1e-16
    L1 = L1_0
        
    thr = 0.05
    

  
    for i in range(1,4):
        
        beta = 2**(i-1)/thr
        Denormin   = lamda*Denormin1 + beta*Denormin2  # This is correct
        
        print(" Printing the BETA - {}".format(beta))
        print("Printing Sum of Denormin - {}".format(np.sum(Denormin)))
        

        # update g
        gFx = np.zeros(L1.shape)
        gFy = np.zeros(L1.shape)

        for channel in range(D):
            print("-"*20)
            print("Processing Channel - {}".format(channel))
            print("Printing Shape of L1[:,:,channel] - {}".format(L1[:,:,channel].shape))
            print("Printing the shape of filter - {}".format(f1.shape))

            gFx[:,:, channel] = -1.0*imfilter(L1[:,:,channel],f1);
            gFy[:,:, channel] = -1.0*imfilter(L1[:,:,channel],f2);

        print("sum of gFx {}".format(np.sum(gFx)))
        print("sum of gFy {}".format(np.sum(gFy)))
        
        #gL = imfilter(L1,f3,'circular');
        print(gFx.shape)
        print(gFx[0,:,0])
        print((np.sum(abs(gFx),axis = 2).shape))

        t = np.repeat((np.sum(abs(gFx),axis = 2)<1.0/beta)[:, :, np.newaxis], D, axis = 2)

        gFx[t] = 0;

        print("Sum of the t - data - {}".format(np.sum(t)))

        t = np.repeat((np.sum(abs(gFy),axis = 2)<1.0/beta)[:,:, np.newaxis], D, axis = 2)

        gFy[t] = 0;

        print("Sum of the t - data - {}".format(np.sum(t)))
        
        print("sum of gFx {}".format(np.sum(gFx)))
        print("sum of gFy {}".format(np.sum(gFy)))

    
        #Correct till here

        
    #     t = repmat(sum(abs(gL),3)<1/beta,[1 1 D]);
    #     gL(t) = 0;
        
        # compute L1
    
        #     ((1,28,3),(27,28,3) ,axis = 0)
        print(np.sum(gFx[:,-1,:] - gFx[:, 0,:]))
        rms = -np.diff(gFx,1,1)
        print(rms.shape)
        print(np.sum(rms))
        Normin2 = np.concatenate(((gFx[:,-1,:] - gFx[:, 0,:]).reshape(-1,1,D), -np.diff(gFx,1,1)), axis = 1)
                
        print("Sum of normin 2 - {}".format(np.sum(Normin2)))
        
        print(Normin2.shape)
        #Normin2 = Normin2 + [[gFy[-1,:,:] - gFy[0, :,:]], -np.diff(gFy,1,0)]
        
        #((28,1,3),(28,27,3), axis = 1)
        print(np.sum(gFy[-1,:,:] - gFy[0, :,:]))
        rms = -np.diff(gFy,1,0)
        print(rms.shape)
        print(np.sum(rms))
        
        Normin2 = Normin2 + np.concatenate(((gFy[-1,:,:] - gFy[0,:,:]).reshape(1,-1,D), -np.diff(gFy,1,0)), axis = 0)
        print("Sum of normin 2 - {}".format(np.sum(Normin2)))
        print(Normin2.shape)
        
        #Correct till here
        
        print(eps)
        print(Normin1.shape)
        print(np.sum(Normin1))
        print(np.sum(lamda*Normin1))
        
        # Correct till here
        
        print(np.sum(beta*fft2(Normin2)))
        print(np.sum(Denormin+eps))

        FL1 = np.divide((lamda*Normin1 + beta*fft2(Normin2)),(Denormin+eps))
        print("Sum of FL1 {}".format(np.sum(FL1)))

        L1 = np.real(ifft2(FL1))
        
        print("Sum of L1 {}".format(np.sum(L1)))

        # normalize L1
        for c in range(0,D):
            L1t = L1[:,:,c]
            for k in range(0,500):
                dt = (sum(L1t[L1t<lb[:,:,c]] )+ sum(L1t[L1t>hb[:,:,c]] ))*2/len(L1t.ravel())
                L1t = L1t-dt
                if abs(dt)<1/len(L1t.ravel()):
                    break 
            L1[:,:,c] = L1t
        
        t = L1<lb
        L1[t] = lb[t]
        t = L1>hb
        L1[t] = hb[t]
        
        # L2
        L2 = I-L1;

    return L1, L2

#%%

if __name__ == "__main__":
    file_path = "<path to file>"
    img = cv2.imread(file_path) # Read image here
    out =im2double(img.astype('float')) # Convert to normalized floating poin
    L1, L2 = reflection_removal(out, 10, np.zeros(out.shape), out, out)   
    cv2.imshow("Original Image", img)
    cv2.imshow("Image without Reflection", L1)
    cv2.imshow("Reflection of Image", L2)