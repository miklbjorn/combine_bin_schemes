from scipy.optimize import minimize
from scipy import stats
from autograd import hessian
import autograd.numpy as np


class BinnedFitter(object):
    def __init__(self, bins, Fs, cs):
        self.sort_bins(bins)
        self.Fs = Fs
        self.F_inv = np.flip(self.Fs)
        self.cs = cs
        
    def sort_bins(self, bins):
        n_intervals = sum([len(bs) for bs in bins])
        bin_changes = []
        for i, bs in enumerate(bins):
            for b in bs:
                bin_changes.append((b[1], i+1))
                bin_changes.append((-b[0], -(i+1)))
        self.bin_changes = sorted(bin_changes)
        
    def predict_yields(self, x, N=1000):
        Ys = self.Fs + x**2*self.F_inv+np.sqrt(self.Fs*self.F_inv)*self.cs*x
        Ys = Ys/np.sum(Ys)
        return N*Ys

    def predict_dN_dx(self, x, N=1000):
        Ys = self.Fs + x**2*self.F_inv+np.sqrt(self.Fs*self.F_inv)*self.cs*x
        dYs = 2*x*self.F_inv+np.sqrt(self.Fs*self.F_inv)*self.cs
        dYs = dYs/np.sum(Ys)
        return N*dYs

    def fit(self, data):
        Ns = self.get_bin_yields(data)
        res = minimize(self.chi_sq,
                      0,
                      (Ns,),
                      tol=1e-4)#,
#                       bounds=[(-1, 1)])
        
        H_func = hessian(self.chi_sq)
        res.H = H_func(np.array(res.x), Ns) 
        res.cov_mat = np.linalg.inv(res.H)
        res.x_unc = np.sqrt(np.diag(res.cov_mat))
        return res

    def chi_sq(self, x, data):
        ''' The function to me minimized when fitting x, y, is the chi-square
            xy: a list with elements [xm, ym, xp, yp]
            data: The yields to which the fit is being made
        '''
        N = np.sum(data)
        predictions = self.predict_yields(x, N)
        uncertainty_squared = data
        for i, u in enumerate(uncertainty_squared):
            if u==0: uncertainty_squared[i]=1 # make sure there are no divisions by zero
        minLL = np.sum(0.5*(data - predictions)**2/uncertainty_squared) # least squares fit
        return minLL
    
    def get_bin_yields(self, data):
        N = np.zeros(2*max([b[1] for b in self.bin_changes]))
        mid = int(len(N)/2)
        for d in data:
            i = 0
            while self.bin_changes[i][0] < d and i < len(self.bin_changes): 
                i += 1
            i = self.bin_changes[i][1]
            if i < 0:
                index = mid + i
            else:
                index = mid -1  + i
            N[index] += 1
        return N 