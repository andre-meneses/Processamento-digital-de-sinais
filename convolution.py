from scipy.fft import rfft, rfftfreq,irfft,fft,ifft
import numpy as np

def convolution(x,h):
    y = []

    n_points = len(x) + len(h) - 1
    
    for n in range(n_points):
        s = 0

        lim_inf = max(0,n - len(x) + 1)
        lim_sup = min(len(h) - 1, n)

        for k in range(lim_inf,lim_sup + 1):
            s += h[k]*x[n - k]
        
        y.append(s) 

    return y

def zero_padding(x,final_size):
    padded = []

    for n in range(final_size):
        if n < len(x):
            padded.append(x[n])
        else:
            padded.append(0)

    return padded

def overlap_add(x,h,l):

    convolution_size = len(x) + len(h) - 1
    dft_size = l + len(h) - 1

    n_blocks = int(len(x)/l) 

    y = np.zeros(convolution_size)

    H = fft(zero_padding(h,dft_size))

    for r in range(n_blocks):
        x_r = x[l*r:l*(r + 1)]

        X = fft(zero_padding(x_r,dft_size))

        Y = [X[n]*H[n] for n in range(len(X))]

        y_temp = ifft(Y)

        for n in range (len(Y)):
            y[n + l*r] += y_temp[n]

    return y

        

    
