'''
Description:
    Contains a randomization function that allows for
    the randomization of attributes in serials from custom 
    and native initializers.

Customization:
    Use the 'custom' wrapper to allow for randomization
    support of custom initializers.
    
    Ex:
    from KASS import Process, run
    from KASS.initializers import custom
    from collections import deque
    
    @custom('{custom_initializer_class_name}')
    def sample(serial, attributes=[], ranges=default_ranges):
        queue = deque([])
        
        @Process(serial, queue)
        def attr1():
            pass
        
        @Process(serial, queue)
        def attr2():
            pass
        
        ...
        
        run(queue, attributes, locals())
        
Functionality:
    *randomize  : (func) Used to randomize a serial.
    
    *custom     : (@func) Used to register a sample function for randomization support.
'''

from .samples.initializer_samples import sample_functions, default_ranges

def randomize(serial, attributes=[], ranges=default_ranges):
    '''
        Description:
            This function is used to randomize initializer
            attributes in an advanced serial format using
            pre-defined parameters.
        
        Arguments:
            *serial         : An initializer serial.
            
            *attributes     :If an empty list, then all attributes
                            are randomized, otherwise only
                            specified attributes are randomized.
                            
            *ranges         : Is a dict of ranges for
                            for quantitative attributes.
                            (see ..._samples.default_ranges)
        
        Non-Randomizable:
            Zeros       Ones
        
        returns None
    '''
    if not 'class_name' in serial or not 'config' in serial:
        raise AttributeError("Serial must contain keys: class_name and config.")
    
    class_name = serial['class_name']
    
    if not class_name in sample_functions:
        raise AttributeError("'{}' is not supported for randomization. Ensure custom keras object is registered and a sample function is provided.".format(class_name))
    
    sample_functions[class_name](serial, attributes=attributes, ranges=ranges)

def custom(class_name, override=False):
    '''
        Description:
            This is a wrapper used to include custom sample
            functions to allow for randomization of custom
            initializers.
        
        Arguments:
            *class_name     : Class_name of keras object found
                            in serial['class_name'] or name 
                            of class/func.
            
            *override       : When True, will override previously
                            supported sample function.
            
    '''
    assert isinstance(class_name, str)
    
    if class_name in sample_functions and not override:
        raise AttributeError("Cannot override '{}'. Must be a unique initializer.".format(class_name))
    
    def wrapper(func):
        sample_functions.update({class_name: func})
        
        return func
    return wrapper

