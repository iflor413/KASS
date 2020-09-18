import numpy as np

def factors(num, dims=2):
    '''
        Calculates all possible factors
        of size 'dims' whose product equals
        'num'. Negative numbers are limited
        to having one negative value.
        
        Returns a numpy.ndarray with shape:
        (factor, dims), if dims >= 2
    '''
    
    def _factors(num):
        factors = np.arange(abs(num))+1
        factors = np.take(factors, np.argwhere(np.mod(num, factors) == 0)).flatten()
        
        if factors.shape[0]%2 == 1: #if factor is root
            factors = np.insert(factors, int(factors.shape[0]/2), factors[int(factors.shape[0]/2)])
        
        half_a = factors[:int(factors.shape[0]/2)]
        half_b = np.flip(factors[int(factors.shape[0]/2):])
        
        if num < 0: #if num is negative
            half_a *=-1
        
        return np.stack((half_a, half_b), axis=-1)
    
    def add_dim(pri_factors):
        new_factors = np.empty((0, pri_factors.shape[1]+1), dtype=pri_factors.dtype)
        for factor in pri_factors:
            for m in range(factor.shape[0]):
                m_factors = _factors(factor[m])[1:] #exclude [1, multiple]
                
                if m_factors.shape[0] > 0:
                    mod_factors = np.delete(factor, m)
                    mod_factors = np.repeat(mod_factors[np.newaxis, :], m_factors.shape[0], axis=0)
                    mod_factors = np.append(mod_factors, m_factors, axis=-1)
                    
                    new_factors = np.concatenate((new_factors, mod_factors))
        
        return new_factors
    
    if dims == 1:
        return np.asarray(num)
    elif dims == 2:
        return _factors(num)
    elif dims > 2:
        pri_factors = _factors(num)
        
        for n in range(dims-2):
            pri_factors = add_dim(pri_factors)
        
        return pri_factors
    else:
        raise AttributeError("Can only find factors in 1 or more dims.")
