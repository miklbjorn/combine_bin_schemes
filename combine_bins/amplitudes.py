from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import stats
from autograd import hessian
import autograd.numpy as np

from combine_bins.util import trapz

class Amp(stats.rv_continuous):
    def __init__(self, *args, **kwargs):
        args = list(args)
        self.F = args.pop(0)
        self.c = args.pop(0)
        self.x = args.pop(0)
        self.norm = 1
        self.ss = np.linspace(-1, 1, 100)
        self.norms = {}
        kwargs['a'] = -1
        kwargs['b'] = 1
        super(Amp, self).__init__(*args, **kwargs)
        
        self.ppf_int = interp1d(np.linspace(0, 1, 100), self.ppf(np.linspace(0, 1, 100)), kind='cubic')
        
    def _x_norm(self, x):
        key = np.sum(x)
        if key in self.norms:
            return self.norms[key]
#         print ('key', key, 'x', x, 'N', np.trapz(self._get_pdf(self.ss, x, False), self.ss))
        self.norms[key] = trapz(self._get_pdf(self.ss, x, False), self.ss)
        return self.norms[key]
        
    def _pdf(self, s):
        return self._get_pdf(s, self.x)
    
    def generate(self, n):
        uniform = np.random.uniform(size=n)
        return self.ppf_int(uniform)
    
    def chi_sq(self, x, data):
#         print ('chi', x, data)
        return - np.sum(np.log(self._get_pdf(data, x)))
    
    def fit(self, data):
        res = minimize(self.chi_sq,
                      self.x*0.9,
                      (data,),
                      tol=1e-4)#,
#                       bounds=[(-1, 1)])
        
        H_func = hessian(self.chi_sq)
        res.H = H_func(np.array(res.x), data) 
        res.cov_mat = np.linalg.inv(res.H)
        res.x_unc = np.sqrt(np.diag(res.cov_mat))
        return res

    def sensitivity(self, x=0):
        deriv_x = self._get_dpdf_dx(self.ss, x)
        pdf = self._get_pdf(self.ss, x)
        Q = trapz( (deriv_x / np.sqrt(pdf))**2, self.ss)
        return Q

        
class Amp_p(Amp):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_pdf(self, s, x, normalise=True):
        A = self.F(s)+x**2*self.F(-s)+np.sqrt(self.F(s)*self.F(-s))*self.c(s)*x
        if normalise: A = A/self._x_norm(x)
        return A

    def _get_dpdf_dx(self, s, x):
        A = 2*x*self.F(-s)+np.sqrt(self.F(s)*self.F(-s))*self.c(s)
        A = A/self._x_norm(x)
        return A

class Amp_m(Amp):
    def __init__(self, F, c, x, a=-1, b=1):
        super().__init__(F, c, x)
        
    def _get_pdf(self, s, x, normalise=True):
        A = self.F(-s)+self.x**2*self.F(s)+np.sqrt(self.F(s)*self.F(-s))*self.c(s)*x
        if normalise: A = A/self._x_norm(x)
        return A
    
class Amp_c(Amp):
    def __init__(self, F, c, a=-1, b=1):
        super().__init__(F, c, 1)
        
    def _get_pdf(self, s, x=1, normalise=True):
        A = self.F(-s)+self.F(s)+np.sqrt(self.F(s)*self.F(-s))*self.c(s)
        if normalise: A = A/self._x_norm(x)
        return A
