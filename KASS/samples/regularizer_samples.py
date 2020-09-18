from collections import deque
from .. import Process, run

import numpy as np

default_ranges = {
'L1L2/l1': [-1.0, 1.0],
'L1L2/l2': [-1.0, 1.0]}

###Regularizer Samples###

def L1L2_sample(serial, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def l1():
    	return np.random.uniform(ranges['L1L2/l1'][0], ranges['L1L2/l1'][1])
    
    @Process(serial, queue)
    def l2():
    	return np.random.uniform(ranges['L1L2/l2'][0], ranges['L1L2/l2'][1])
    
    run(queue, attributes, locals())

###Regularizer Samples###

_globals = globals()

sample_functions = {}
for key in list(_globals.keys()):
    _split = key.split('_')
    
    if len(_split) == 2 and _split[1] == 'sample':
        sample_functions[_split[0]] = _globals[key]

