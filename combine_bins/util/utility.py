import autograd.numpy as np

def trapz( y, x):
    d = np.diff(x)
    return np.sum((y[0:-1] + y[1:]) * d / 2)

def get_F_c(A, bins):
    # calculate the Fi and ci for a binning scheme
    # binning can be disjoint
    # bins should only be defined in [0, 1], then bins for negative s
    # are defined as beeing the opposite
    # bins = [[interval1, interval2], [interval3], ...]
    # interval = (a, b) tuple
    # all intervals should be disjoint and the union should be (0, 1) (not checked ...)
    Fs = []
    cs = []
    
    for bs in reversed(bins):
        F = 0
        for b in bs:          
            s_in_b = np.linspace(-b[1], -b[0], 100)
            F += trapz(A.F(s_in_b), s_in_b)
        Fs.append(F)
    
    for bs in bins:
        F = 0
        for b in bs:
            s_in_b = np.linspace(b[0], b[1], 100)
            F += trapz(A.F(s_in_b), s_in_b)            
        Fs.append(F)

    Fs = np.array(Fs)
#     Fs = Fs/np.sum(Fs)
    
    mid = int(len(Fs)/2)
    for i, bs in enumerate(bins):
        c = 0
        for b in bs:
            s_in_b = np.linspace(b[0], b[1], 100)
            c += trapz(np.sqrt(A.F(s_in_b)*A.F(-s_in_b))*A.c(s_in_b), s_in_b)            

        cs.append(c / np.sqrt(Fs[mid-1-i]*Fs[mid+i]))
    cs = np.array(cs[::-1] + cs)
    return Fs, cs 

import time, sys
from IPython.display import clear_output
def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)