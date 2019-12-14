import autograd.numpy as np
def test_F_1(s):
    return 0.3+0.5*np.sin(-np.pi*s) + 3*np.exp(-(s-0.3)**2/0.1)

def test_c_1(s):
    return np.cos(-0.7*np.cos(np.pi*s) + 2.4*np.cos(2*np.pi*s)  )