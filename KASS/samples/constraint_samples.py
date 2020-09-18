from ..utils.rand_funcs import axes_
from .. import Process, run

from collections import deque
import numpy as np

default_ranges = {
'MaxNorm/max_value': [0, 10],
'MinMaxNorm/min_value/max_value': [0, 10]}

###Constraint Samples###

def _sample_null(serial, input_shape, attributes=[], ranges=default_ranges): pass

Constraint_sample=NonNeg_sample=_sample_null

def MaxNorm_sample(serial, input_shape, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def max_value():
        return np.random.uniform(ranges['MaxNorm/max_value'][0], ranges['MaxNorm/max_value'][1])
        
    @Process(serial, queue)
    def axis():
        return axes_(input_shape)
    
    run(queue, attributes, locals())

def UnitNorm_sample(serial, input_shape, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def axis():
        return axes_(input_shape)
    
    run(queue, attributes, locals())

def MinMaxNorm_sample(serial, input_shape, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def min_value():
        return np.random.randint(ranges['MinMaxNorm/min_value/max_value'][0], ranges['MinMaxNorm/min_value/max_value'][1])
    
    @Process(serial, queue)
    def max_value():
        return np.random.randint(min_value(), ranges['MinMaxNorm/min_value/max_value'][1])
    
    @Process(serial, queue)
    def rate():
        return np.random.sample()
    
    @Process(serial, queue)
    def axis():
        return axes_(input_shape)
    
    run(queue, attributes, locals())

###Constraint Samples###

_globals = globals()

sample_functions = {}
for key in list(_globals.keys()):
    _split = key.split('_')
    
    if len(_split) == 2 and _split[1] == 'sample':
        sample_functions[_split[0]] = _globals[key]

