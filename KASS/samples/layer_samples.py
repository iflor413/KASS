from KASD.initializers import initializers, serialize as _serialize_initializer, get as _get_initializer
from KASD.regularizers import regularizers, serialize as _serialize_regularizer, get as _get_regularizer
from KASD.constraints import constraints, serialize as _serialize_constraint, get as _get_constraint
from KASD.activations import activations

from ..initializers import randomize as _randomize_initializer
from ..regularizers import randomize as _randomize_regularizer
from ..constraints import randomize as  _randomize_constraint
from ..utils.math import factors
from ..utils.rand_funcs import*
from .. import Process, run

from collections import deque
from copy import deepcopy
import numpy as np

data_formats = ['channels_last', 'channels_first']
paddings = ['valid', 'causal', 'same']
interpolations = ['nearest', 'bilinear']
implementations = [1, 2]
merge_modes = ['sum', 'mul', 'concat', 'ave']

def regularizer_(reg):
    if reg is None: return None
    
    reg = _get_regularizer(reg)
    reg = _serialize_regularizer(reg)
    
    _randomize_regularizer(reg)
    
    return reg

def initializer_(init):
    if init is None: return None

    init = _get_initializer(init)
    init = _serialize_initializer(init)
    
    _randomize_initializer(init)
    
    return init

def constraint_(const, input_shape):
    if const is None: return None
    
    const = _get_constraint(const)
    const = _serialize_constraint(const)
    
    _randomize_constraint(const, input_shape)
    
    return const

default_ranges = {
'Dense/units': [1, 128],
'Dropout/seed': [1, 1024],
'RepeatVector/n': [1, 64],
'ActivityRegularization/l1': [-1.0, 1.0],
'ActivityRegularization/l2': [-1.0, 1.0],
'SpatialDropout1D/seed': [1, 1024],
'SpatialDropout2D/seed': [1, 1024],
'SpatialDropout3D/seed': [1, 1024],
'Conv1D/filters': [1, 128],
'Conv2D/filters': [1, 128],
'SeparableConv1D/filters': [1, 128],
'SeparableConv1D/depth_multiplier': [1, 32],
'SeparableConv2D/filters': [1, 128],
'SeparableConv2D/depth_multiplier': [1, 32],
'DepthwiseConv2D/filters': [1, 128],
'DepthwiseConv2D/depth_multiplier': [1, 32],
'Conv2DTranspose/filters': [1, 128],
'Conv3D/filters': [1, 128],
'Conv3DTranspose/filters': [1, 128],
'UpSampling1D/size': [2, 32],
'UpSampling2D/size': ([2, 32], [2, 32]),
'UpSampling3D/size': ([2, 32], [2, 32], [2, 32]),
'ZeroPadding1D/padding': (([0, 32], [0, 32]),),
'ZeroPadding2D/padding': (([0, 32], [0, 32]), ([0, 32], [0, 32])),
'ZeroPadding3D/padding': (([0, 32], [0, 32]), ([0, 32], [0, 32]), ([0, 32], [0, 32])),
'SimpleRNN/units': [1, 128],
'GRU/units': [1, 128],
'LSTM/units': [1, 128],
'SimpleRNNCell/units': [1, 128],
'GRUCell/units': [1, 128],
'LSTMCell/units': [1, 128],
'CuDNNGRU/units': [1, 128],
'CuDNNLSTM/units': [1, 128],
'BatchNormalization/momentum': [-10, 10],
'BatchNormalization/epsilon': [1e-5, 1e-2],
'GaussianNoise/stddev': [1e-3, 10],
'AlphaDropout/seed': [1, 1024],
'LeakyReLU/alpha': [0, 16],
'ELU/alpha': [0, 16],
'ThresholdedReLU/theta': [0, 10],
'ReLU/threshold/max_value': [0, 16],
'ReLU/negative_slope': [0, 16],
'ConvLSTM2D/filters': [1, 128],
'ConvLSTM2DCell/filters': [1, 128]}

###Layer Samples###

def _sample_null(serial, attributes=[], ranges=default_ranges): pass

InputLayer_sample=Add_sample=Subtract_sample=Multiply_sample=Average_sample=Maximum_sample=Minimum_sample=Concatenate_sample=Lambda_sample=_sample_null

def Dot_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def normalize():
    	return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())
    
def Dense_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['Dense/units'][0], ranges['Dense/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Activation_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def activation():
        return activations.choice()

    run(queue, attributes, locals())

def Dropout_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()
    @Process(serial, queue)
    def noise_shape():
        return noise_shape_(input_shape)
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['Dropout/seed'][0], ranges['Dropout/seed'][1]+1)
    
    run(queue, attributes, locals())

def Flatten_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats+[None])

    run(queue, attributes, locals())

def Reshape_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def target_shape():
        _factors = factors(np.prod(input_shape[1:]), dims=np.array(output_shape[1:]).shape[0])
        
        if not isinstance(_factors, (list, np.ndarray)):
            _factors = np.array([[_factors]])
        
        _factors = np.concatenate((_factors, np.flip(_factors, axis=-1)))
        
        return tuple(_factors[np.random.randint(0, _factors.shape[0])].tolist())
    
    run(queue, attributes, locals())

def Permute_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def dims():
        return tuple(np.random.permutation(np.arange(np.array(input_shape[1:]).shape[0])+1).tolist())

    run(queue, attributes, locals())

def RepeatVector_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def n():
        return np.random.randint(ranges['RepeatVector/n'][0], ranges['RepeatVector/n'][1]+1)

    run(queue, attributes, locals())

def ActivityRegularization_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def l1():
        return np.random.uniform(ranges['ActivityRegularization/l1'][0], ranges['ActivityRegularization/l1'][1])
    @Process(serial, queue)
    def l2():
        return np.random.uniform(ranges['ActivityRegularization/l2'][0], ranges['ActivityRegularization/l2'][1])
    
    run(queue, attributes, locals())

def Masking_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def mask_value():
        return np.random.sample()

    run(queue, attributes, locals())

def SpatialDropout1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()
    @Process(serial, queue)
    def noise_shape():
        return noise_shape_(input_shape)
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['SpatialDropout1D/seed'][0], ranges['SpatialDropout1D/seed'][1]+1)
    
    run(queue, attributes, locals())

def SpatialDropout2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()
    @Process(serial, queue)
    def noise_shape():
        return noise_shape_(input_shape)
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['SpatialDropout2D/seed'][0], ranges['SpatialDropout2D/seed'][1]+1)
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def SpatialDropout3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()
    @Process(serial, queue)
    def noise_shape():
        return noise_shape_(input_shape)
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['SpatialDropout3D/seed'][0], ranges['SpatialDropout3D/seed'][1]+1)
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def Conv1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv1D/filters'][0], ranges['Conv1D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats) if padding() != 'causal' else 'channels_last'
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Conv2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv2D/filters'][0], ranges['Conv2D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def SeparableConv1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['SeparableConv1D/filters'][0], ranges['SeparableConv1D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def depth_multiplier():
        return np.random.randint(ranges['SeparableConv1D/depth_multiplier'][0], ranges['SeparableConv1D/depth_multiplier'][1]+1)
    @Process(serial, queue)
    def depthwise_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def pointwise_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def depthwise_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def pointwise_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def depthwise_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def pointwise_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def SeparableConv2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['SeparableConv2D/filters'][0], ranges['SeparableConv2D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def depth_multiplier():
        return np.random.randint(ranges['SeparableConv2D/depth_multiplier'][0], ranges['SeparableConv2D/depth_multiplier'][1]+1)
    @Process(serial, queue)
    def depthwise_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def pointwise_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def depthwise_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def pointwise_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def depthwise_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def pointwise_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def DepthwiseConv2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def depth_multiplier():
        return np.random.randint(ranges['DepthwiseConv2D/depth_multiplier'][0], ranges['DepthwiseConv2D/depth_multiplier'][1]+1)
    @Process(serial, queue)
    def depthwise_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def depthwise_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def depthwise_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Conv2DTranspose_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv2DTranspose/filters'][0], ranges['Conv2DTranspose/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return (min(dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))),)*2 #assert dilation_rate[0] == dilation_rate[1]
    @Process(serial, queue)
    def output_padding():
        return output_padding_(strides())
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Conv3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv3D/filters'][0], ranges['Conv3D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Conv3DTranspose_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv3DTranspose/filters'][0], ranges['Conv3DTranspose/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def output_padding():
        return output_padding_(strides())
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Cropping1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def cropping():
        return cropping_(input_shape, 'channels_last')[0]

    run(queue, attributes, locals())

def Cropping2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def cropping():
        return cropping_(input_shape, data_format())

    run(queue, attributes, locals())

def Cropping3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def cropping():
        return cropping_(input_shape, data_format())

    run(queue, attributes, locals())

def UpSampling1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def size():
        return np.random.randint(ranges['UpSampling1D/size'][0], ranges['UpSampling1D/size'][1]+1)

    run(queue, attributes, locals())

def UpSampling2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def size():
        return tuple([np.random.randint(size[0], size[1]+1) for size in ranges['UpSampling2D/size']])
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def interpolation():
        return np.random.choice(interpolations)

    run(queue, attributes, locals())

def UpSampling3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def size():
        return tuple([np.random.randint(size[0], size[1]+1) for size in ranges['UpSampling3D/size']])
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def ZeroPadding1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def padding():
        return tuple([(np.random.randint(padding[0][0], padding[0][1]+1), np.random.randint(padding[1][0], padding[1][1]+1)) for padding in ranges['ZeroPadding1D/padding']])[0]

    run(queue, attributes, locals())

def ZeroPadding2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def padding():
        return tuple([(np.random.randint(padding[0][0], padding[0][1]+1), np.random.randint(padding[1][0], padding[1][1]+1)) for padding in ranges['ZeroPadding2D/padding']])
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def ZeroPadding3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def padding():
        return tuple([(np.random.randint(padding[0][0], padding[0][1]+1), np.random.randint(padding[1][0], padding[1][1]+1)) for padding in ranges['ZeroPadding3D/padding']])
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def MaxPooling1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def MaxPooling2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def MaxPooling3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def AveragePooling1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def AveragePooling2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def AveragePooling3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def pool_size():
        return pool_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_pooling_(input_shape, data_format(), pool_size())
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'

    run(queue, attributes, locals())

def GlobalMaxPooling1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def GlobalMaxPooling2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def GlobalMaxPooling3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def GlobalAveragePooling2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def GlobalAveragePooling1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def GlobalAveragePooling3D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)

    run(queue, attributes, locals())

def LocallyConnected1D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv1D/filters'][0], ranges['Conv1D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return 'valid'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def LocallyConnected2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['Conv1D/filters'][0], ranges['Conv1D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return 'valid'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice(include=[None])
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def RNN_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def cell():
        def rand_cell(cell):
            cell['input'] = input_
            cell['input_shape'] = input_shape
            cell['output_shape'] = output_shape
            
            sample_functions[cell['class_name']](cell, attributes=attributes, ranges=ranges)
            del cell['input'], cell['input_shape'], cell['output_shape']
        
        cells = deepcopy(serial['config']['cell'])
        
        if isinstance(cell, list):
            [rand_cell(cells) for cell in cells]
        else:
            rand_cell(cells)
        
        return cells
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def unroll():
        return bool(np.random.randint(0, 2))
        
    run(queue, attributes, locals())

def SimpleRNN_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['SimpleRNN/units'][0], ranges['SimpleRNN/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def unroll():
        return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())

def GRU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['GRU/units'][0], ranges['GRU/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    @Process(serial, queue)
    def implementation():
        return np.random.choice(implementations)
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def reset_after():
        return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())

def LSTM_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['LSTM/units'][0], ranges['LSTM/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def unit_forget_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    @Process(serial, queue)
    def implementation():
        return np.random.choice(implementations)
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def unroll():
        return bool(np.random.randint(0, 2))
    
    run(queue, attributes, locals())

def SimpleRNNCell_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['SimpleRNNCell/units'][0], ranges['SimpleRNNCell/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
    	 return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
    	return np.random.sample()

    run(queue, attributes, locals())

def GRUCell_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['GRUCell/units'][0], ranges['GRUCell/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    @Process(serial, queue)
    def implementation():
        return np.random.choice(implementations)
    @Process(serial, queue)
    def reset_after():
        return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())

def LSTMCell_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['LSTMCell/units'][0], ranges['LSTMCell/units'][1]+1)
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def unit_forget_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    @Process(serial, queue)
    def implementation():
        return np.random.choice(implementations)

    run(queue, attributes, locals())

def StackedRNNCells_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def cells():
        cells = deepcopy(serial['config']['cells'])
        
        for i in range(len(cells)):
            cell = cells[i]
            cell['input'] = input_
            cell['input_shape'] = input_shape
            cell['output_shape'] = output_shape
            
            sample_functions[cell['class_name']](cell, attributes=attributes, ranges=ranges)
            del cell['input'], cell['input_shape'], cell['output_shape']
        
        return cells
    
    run(queue, attributes, locals())

def CuDNNGRU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])
 
    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['CuDNNGRU/units'][0], ranges['CuDNNGRU/units'][1]+1)
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())

def CuDNNLSTM_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])
           
    @Process(serial, queue)
    def units():
        return np.random.randint(ranges['CuDNNLSTM/units'][0], ranges['CuDNNLSTM/units'][1]+1)
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def unit_forget_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))

    run(queue, attributes, locals())

def BatchNormalization_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def axis():
        return axis_(input_shape)
    @Process(serial, queue)
    def momentum():
        return np.random.uniform(ranges['BatchNormalization/momentum'][0], ranges['BatchNormalization/momentum'][1])
    @Process(serial, queue)
    def epsilon():
        return np.random.uniform(ranges['BatchNormalization/epsilon'][0], ranges['BatchNormalization/epsilon'][1])
    @Process(serial, queue)
    def center():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def scale():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def beta_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def gamma_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def moving_mean_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def moving_variance_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def beta_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def gamma_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def beta_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def gamma_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)

    run(queue, attributes, locals())

def Embedding_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    #input_dim
    #output_dim
    @Process(serial, queue)
    def embeddings_initializer():
        return initializer_(initializers.choice())
    @Process(serial, queue)
    def embeddings_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def embeddings_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def mask_zero():
        return bool(np.random.randint(0, 2))
    
    #input_length=None,

    run(queue, attributes, locals())

def GaussianNoise_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def stddev():
        return np.random.uniform(ranges['GaussianNoise/stddev'][0], ranges['GaussianNoise/stddev'][1])

    run(queue, attributes, locals())

def GaussianDropout_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()

    run(queue, attributes, locals())

def AlphaDropout_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def rate():
        return np.random.sample()
    @Process(serial, queue)
    def noise_shape():
        return noise_shape_(input_shape) #ignore batch_size
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['AlphaDropout/seed'][0], ranges['AlphaDropout/seed'][1]+1)
    
    run(queue, attributes, locals())

def LeakyReLU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def alpha():
        return np.random.randint(ranges['LeakyReLU/alpha'][0], ranges['LeakyReLU/alpha'][1]+1)

    run(queue, attributes, locals())

def PReLU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def alpha_initializer():
        return initializer_(initializers.choice()) if len(input_shape) == 3 else initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix']) if len(input_shape) < 3 else initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity and Orthogonal, conditionally
    @Process(serial, queue)
    def alpha_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def alpha_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def shared_axes():
        return shared_axes_(input_shape)

    run(queue, attributes, locals())

def ELU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def alpha():
        return np.random.uniform(ranges['ELU/alpha'][0], ranges['ELU/alpha'][1])

    run(queue, attributes, locals())

def ThresholdedReLU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def theta():
        return np.random.uniform(ranges['ThresholdedReLU/theta'][0], ranges['ThresholdedReLU/theta'][1])

    run(queue, attributes, locals())

def Softmax_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def axis():
        return axis_(input_shape)

    run(queue, attributes, locals())

def ReLU_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def threshold():
        return np.random.uniform(ranges['ReLU/threshold/max_value'][0], ranges['ReLU/threshold/max_value'][1])
    @Process(serial, queue)
    def max_value():
        return np.random.uniform(threshold(), ranges['ReLU/threshold/max_value'][1])
    @Process(serial, queue)
    def negative_slope():
        return np.random.uniform(ranges['ReLU/negative_slope'][0], ranges['ReLU/negative_slope'][1])

    run(queue, attributes, locals())

def Bidirectional_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def layer():
        assert len(input_shape) == 5 or len(input_shape) == 3
        
        layer = deepcopy(serial['config']['layer'])
        layer['input'] = input_
        layer['input_shape'] = input_shape
        layer['output_shape'] = output_shape
        
        sample_functions[layer['class_name']](layer, ranges=ranges)
        #(layer)
        
        del layer['input'], layer['input_shape'], layer['output_shape']
        
        return layer
    @Process(serial, queue)
    def merge_mode():
         return np.random.choice(merge_modes+[None])
    
    run(queue, attributes, locals())

def TimeDistributed_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def layer():
        assert (len(input_) == 1 and len(input_shape) >= 2) or len(input_) > 1
        
        layer = deepcopy(serial['config']['layer'])
        layer['input'] = input_
        layer['input_shape'] = np.delete(input_shape, 1)
        layer['output_shape'] = output_shape
        
        sample_functions[layer['class_name']](layer, ranges=ranges)
        #randomize(layer)
        
        del layer['input'], layer['input_shape'], layer['output_shape']
        
        return layer
    
    run(queue, attributes, locals())

def ConvRNN2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])
    
    @Process(serial, queue)
    def cell():
        def rand_cell(cell):
            cell['input'] = input_
            if cell['class_name'] == 'ConvLSTM2DCell':
                cell['input_shape'] = np.delete(input_shape, 1)
            else:
                cell['input_shape'] = input_shape
            cell['output_shape'] = output_shape
            
            sample_functions[cell['class_name']](cell, attributes=attributes, ranges=ranges)
            
            del cell['input'], cell['input_shape'], cell['output_shape']
        
        cells = deepcopy(serial['config']['cell'])
        
        if isinstance(cell, list):
            [rand_cell(cells) for cell in cells]
        else:
            rand_cell(cells)
        
        return cells
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def return_state():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def unroll():
        return False
    
    run(queue, attributes, locals())

def ConvLSTM2D_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['ConvLSTM2D/filters'][0], ranges['ConvLSTM2D/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(np.delete(input_shape, 1), data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(np.delete(input_shape, 1), data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(np.delete(input_shape, 1), data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def unit_forget_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def activity_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def return_sequences():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def go_backwards():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def stateful():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()

    run(queue, attributes, locals())

def ConvLSTM2DCell_sample(serial, attributes=[], ranges=default_ranges):
    input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])
      
    @Process(serial, queue)
    def filters():
        return np.random.randint(ranges['ConvLSTM2DCell/filters'][0], ranges['ConvLSTM2DCell/filters'][1]+1)
    @Process(serial, queue)
    def padding():
        return np.random.choice(np.delete(paddings, [1])) #removes 'causal'
    @Process(serial, queue)
    def data_format():
        return np.random.choice(data_formats)
    @Process(serial, queue)
    def kernel_size():
        return kernel_size_(input_shape, data_format(), strides(), dilation_rate=dilation_rate())
    @Process(serial, queue)
    def strides():
        return strides_(input_shape, data_format(), kernel_size(), dilation_rate=dilation_rate(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def dilation_rate():
        return dilation_rate_(input_shape, data_format(), kernel_size(), strides(), null=np.random.randint(0, 2))
    @Process(serial, queue)
    def activation():
        return activations.choice()
    @Process(serial, queue)
    def recurrent_activation():
        return activations.choice()
    @Process(serial, queue)
    def use_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def recurrent_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['2DMatrix'])) #removes Identity
    @Process(serial, queue)
    def bias_initializer():
        return initializer_(initializers.choice(exclude=initializers.labels['>=2DMatrix'])) #removes Identity and Orthogonal
    @Process(serial, queue)
    def unit_forget_bias():
        return bool(np.random.randint(0, 2))
    @Process(serial, queue)
    def kernel_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def recurrent_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def bias_regularizer():
        return regularizer_(regularizers.choice(include=[None]))
    @Process(serial, queue)
    def kernel_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def recurrent_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def bias_constraint():
        return constraint_(constraints.choice(include=[None]), input_shape)
    @Process(serial, queue)
    def dropout():
        return np.random.sample()
    @Process(serial, queue)
    def recurrent_dropout():
        return np.random.sample()
    
    run(queue, attributes, locals())

###Layer Samples###

_globals = globals()

sample_functions = {}
for key in list(_globals.keys()):
    _split = key.split('_')
    
    if len(_split) == 2 and _split[1] == 'sample':
        sample_functions[_split[0]] = _globals[key]

