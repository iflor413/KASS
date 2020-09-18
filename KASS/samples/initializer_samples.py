from collections import deque
from .. import Process, run

import numpy as np

modes = ["fan_in", "fan_out", "fan_avg"]
distributions = ["normal", "uniform"]

default_ranges = {
'Constant/value': [-10, 10],
'RandomNormal/mean': [-10, 10],
'RandomNormal/stddev': [0, 5],
'RandomNormal/seed': [1, 1024],
'RandomUniform/minval/maxval': [-10, 10],
'RandomUniform/seed': [1, 1024],
'TruncatedNormal/mean': [-10, 10],
'TruncatedNormal/stddev': [0, 5],
'TruncatedNormal/seed': [1, 1024],
'VarianceScaling/scale': [1e-3, 10],
'VarianceScaling/seed': [1, 1024],
'Orthogonal/gain': [-10, 10],
'Orthogonal/seed': [1, 1024],
'Identity/gain': [-10, 10]}

###Initializer Samples###

def _sample_null(serial, attributes=[], ranges=default_ranges):
    return {}

Zeros_sample=Ones_sample=_sample_null

def Constant_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def value():
    	return np.random.uniform(ranges['Constant/value'][0], ranges['Constant/value'][1])
    
    run(queue, attributes, locals())

def RandomNormal_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def mean():
    	return np.random.uniform(ranges['RandomNormal/mean'][0], ranges['RandomNormal/mean'][1])
    
    @Process(serial, queue)
    def stddev():
    	return np.random.uniform(ranges['RandomNormal/stddev'][0], ranges['RandomNormal/stddev'][1])
    
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['RandomNormal/seed'][0], ranges['RandomNormal/seed'][1]+1)
    
    run(queue, attributes, locals())

def RandomUniform_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def minval():
    	return np.random.uniform(ranges['RandomUniform/minval/maxval'][0], maxval())
    
    @Process(serial, queue)
    def maxval():
    	return np.random.uniform(minval(), ranges['RandomUniform/minval/maxval'][1])
    
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['RandomUniform/seed'][0], ranges['RandomUniform/seed'][1]+1)
    
    run(queue, attributes, locals())

def TruncatedNormal_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def mean():
    	return np.random.uniform(ranges['TruncatedNormal/mean'][0], ranges['TruncatedNormal/mean'][1])
    
    @Process(serial, queue)
    def stddev():
    	return np.random.uniform(ranges['TruncatedNormal/stddev'][0], ranges['TruncatedNormal/stddev'][1])
        
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['TruncatedNormal/seed'][0], ranges['TruncatedNormal/seed'][1]+1)
    
    run(queue, attributes, locals())

def VarianceScaling_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def scale():
    	return np.random.uniform(ranges['VarianceScaling/scale'][0], ranges['VarianceScaling/scale'][1])
    
    @Process(serial, queue)
    def mode():
    	return np.random.choice(modes)
    
    @Process(serial, queue)
    def distribution():
    	return np.random.choice(distributions)
        
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['VarianceScaling/seed'][0], ranges['VarianceScaling/seed'][1]+1)
    
    run(queue, attributes, locals())

def Orthogonal_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def gain():
    	return np.random.uniform(ranges['Orthogonal/gain'][0], ranges['Orthogonal/gain'][1])
        
    @Process(serial, queue)
    def seed():
        return np.random.randint(ranges['Orthogonal/seed'][0], ranges['Orthogonal/seed'][1]+1)
    
    run(queue, attributes, locals())

def Identity_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def gain():
    	return np.random.uniform(ranges['Identity/gain'][0], ranges['Identity/gain'][1])
    
    run(queue, attributes, locals())

###Initializer Samples###

_globals = globals()

sample_functions = {}
for key in list(_globals.keys()):
    _split = key.split('_')
    
    if len(_split) == 2 and _split[1] == 'sample':
        sample_functions[_split[0]] = _globals[key]

