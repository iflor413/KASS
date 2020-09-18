'''
    This module contains all functions used for
    the randomization of keras object attributes.
'''

import numpy as np

def format_input_shape(input_shape, data_format):
    '''Returns a formated input_shape according to data_format'''
    
    if data_format == 'channels_first':
        return input_shape[2:]
    elif data_format is None or data_format == 'channels_last':
        return input_shape[1:-1]
    else:
        raise AttributeError("'{}' is not a valid data_format.".format(data_format))

def axes_(input_shape):
    axes = np.random.permutation(np.arange(np.array(input_shape[1:]).shape[0]))
    return np.sort(axes[:(1 if axes.shape[0] == 0 else np.random.randint(1, axes.shape[0]+1))]).tolist()

def noise_shape_(input_shape):
    noise_shape = list(input_shape[1:]) if input_shape[0] is None else list(input_shape)
    noise_shape[0 if len(noise_shape) == 1 else np.random.choice(np.arange(len(noise_shape)-1)+1)] = 1
    
    return tuple(noise_shape)

def kernel_size_(input_shape, data_format, strides, dilation_rate=None):
    shape = np.array(format_input_shape(input_shape, data_format))
    strides = np.array(strides)
    
    def from_dilation_rate(dilation_rate):
        assert dilation_rate.shape[0] == shape.shape[0] and not np.any(dilation_rate==0) and np.all(dilation_rate > 0) and np.all(dilation_rate <= shape)
        
        max_kernel_size = np.floor((shape-1+dilation_rate)/dilation_rate)
        
        return tuple([1 if max_ks < 1 else np.random.randint(1, max_ks+1) for max_ks in max_kernel_size])
    
    def from_strides(strides):
        assert strides.shape[0] == shape.shape[0] and np.all(strides > 0) and np.all(strides < shape)
        
        max_kernel_size = np.where(strides==1, shape, shape-strides)
        
        return tuple([1 if max_ks < 1 else np.random.randint(1, max_ks+1) for max_ks in max_kernel_size])
    
    if not dilation_rate is None and not np.all(dilation_rate==1):
        #'strides' is default dims*(1,), so 'kernel_size' is limited to 'dilation_rate'.
        dilation_rate = np.array(dilation_rate)
        
        return from_dilation_rate(dilation_rate)
    else:
        return from_strides(strides)
    #If both 'dilation_rate' and 'strides' are default dims*(1,), either are valid in determining 'kernel_size'.

def strides_(input_shape, data_format, kernel_size, dilation_rate=(1,), null=False):
    shape = np.array(format_input_shape(input_shape, data_format))
    kernel_size = np.array(kernel_size)
    dilation_rate = np.array(dilation_rate)
    
    assert kernel_size.shape[0] == shape.shape[0]# and np.all(kernel_size > 0) and np.all(kernel_size <= shape)
    
    kernel_size = np.clip(kernel_size, 1, shape)
    
    if np.all(dilation_rate == 1):
        max_strides = np.where(np.logical_or(kernel_size==1, kernel_size==shape), 1, shape-kernel_size)
        
        return tuple([1 if max_s < 1 else np.random.randint(1, max_s+1) for max_s in max_strides])
    else:
        return tuple([1 for _ in shape])

def dilation_rate_(input_shape, data_format, kernel_size, strides, null=False):
    shape = np.array(format_input_shape(input_shape, data_format))
    kernel_size = np.array(kernel_size)
    strides = np.array(strides)
    
    assert kernel_size.shape[0] == shape.shape[0] and strides.shape[0] == shape.shape[0] and np.all(strides > 0) and np.all(strides <= shape)
    
    if np.all(strides==1):
        assert not np.any(kernel_size==0)
        
        #with np.errstate(divide='ignore'):
        #   max_dilation_rates = np.where(kernel_size!=1, np.floor((shape-1)/(kernel_size-1)), 1) 
        
        denominator = np.where(kernel_size!=1, kernel_size-1, 1) 
        numerator = np.where(kernel_size!=1, shape-1, 1)
        
        max_dilation_rates = numerator/denominator
        
        return tuple([1 if max_dr < 1 else np.random.randint(1, max_dr+1) for max_dr in max_dilation_rates])
    else:
        return tuple([1 for _ in shape])

def output_padding_(strides):
    return tuple([np.random.randint(0, s) for s in strides])

def cropping_(input_shape, data_format):
    cropping_dims = format_input_shape(input_shape, data_format)
    
    cropping = []
    for cd in cropping_dims:
        kernel_size = np.random.randint(1, cd+1)
        off_set = np.random.randint(0, cd-kernel_size+1)
        
        cropping.append((off_set, cd-(kernel_size+off_set)))
    
    return tuple(cropping)

def pool_size_(input_shape, data_format, strides):
    shape = np.array(format_input_shape(input_shape, data_format))
    strides = np.array(strides)
    
    assert shape.shape[0] == strides.shape[0] and np.all(strides > 0) and np.all(strides < shape)
    
    max_pool_sizes = np.where(strides==1, shape, shape-strides)
    
    return tuple([np.random.randint(1, max_ps+1) for max_ps in max_pool_sizes])

def strides_pooling_(input_shape, data_format, pool_size):
    shape = np.array(format_input_shape(input_shape, data_format))
    pool_size = np.array(pool_size)
    
    assert shape.shape[0] == pool_size.shape[0] and np.all(pool_size > 0) and np.all(pool_size <= shape)
    
    max_strides = np.where(np.logical_or(pool_size==1, pool_size==shape), 1, shape-pool_size)
    
    return tuple([np.random.randint(1, max_s+1) for max_s in max_strides])

def axis_(input_shape):
    return np.random.randint(1, np.array(input_shape).shape[0])

def shared_axes_(input_shape):
    axes = np.random.permutation(np.arange(len(input_shape)-1)+1)
    
    return tuple(axes[:(2 if axes.shape[0]+1 == 2 else np.random.randint(2, axes.shape[0]+1))])

