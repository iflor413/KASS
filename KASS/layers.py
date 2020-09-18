'''
Description:
    Contains a randomization function that allows for
    the randomization of attributes in advanced serials/series
    from custom and native layers.

Customization:
    Use the 'custom' wrapper to allow for randomization
    support of custom layers.
    
    Ex:
    from KASS import Process, run
    from KASS.layers import custom
    from collections import deque
    
    @custom('{custom_layer_class_name}')
    def sample(serial, attributes=[], ranges=default_ranges):
        input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])

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

from KASD.layers import is_advanced_serial, update, is_advanced_series
from .samples.layer_samples import sample_functions, default_ranges

def randomize(serial, attributes=[], ranges=default_ranges, only_cell=False):
    '''
        Description:
            This function is used to randomize layer
            attributes in an advanced serial format using
            pre-defined parameters.
        
        Arguments:
            *serial         : An advanced serial/series.
            
            *attributes     :
                            Randomizing an Advanced Serial:
                                If an empty list, then all attributes
                                are randomized, otherwise only
                                specified attributes are randomized.
                            
                            Randomizing an Advanced Series:
                                If a dict with a key as the name of
                                a serial and list of attributes as 
                                the value, then only those attributes
                                are randomized. Otherwise, each serial's
                                attributes are randomized.
                            
            *ranges         : Is a dict of ranges for
                            for quantitative attributes.
                            (see ..._samples.default_ranges)
                            
            *only_cell      : When True, is only applicable
                            to layers with a 'cell' attribute,
                            will convert serial to a randomized
                            serial['config']['cell'].
        
        Non-Randomizable:
            InputLayer      Add             Subtract
            Multiply        Average         Maximum
            Minimum         Concatenate     Lambda
        
        returns None
    '''
    if is_advanced_serial(serial):
        class_name = serial['class_name']
        
        if not class_name in sample_functions:
            raise AttributeError("'{}' is not supported for randomization. Ensure custom keras object is registered and a sample function is provided.".format(class_name))
        
        sample_functions[class_name](serial, attributes=attributes, ranges=ranges)
        update(serial)
        
        if only_cell and 'cell' in serial['config']:
            cell = serial['config']['cell']
            serial.clear()
            serial.update(cell)
    elif is_advanced_series(serial):
        for key, value in serial.items():
            randomize(value, attributes=attributes[key] if isinstance(attributes, dict) and key in attributes else [], ranges=default_ranges)
    else:
        raise AttributeError("'serial' must be an advanced serial/series.")

def custom(class_name, override=False):
    '''
        Description:
            This is a wrapper used to include custom sample
            functions to allow for randomization of custom
            layers.
        
        Arguments:
            *class_name     : Class_name of keras object found
                            in serial['class_name'] or name 
                            of class/func.
            
            *override       : When True, will override previously
                            supported sample function.
            
    '''
    assert isinstance(class_name, str)
    
    if class_name in sample_functions and not override:
        raise AttributeError("Cannot override '{}'. Must be a unique layer.".format(class_name))
    
    def wrapper(func):
        sample_functions.update({class_name: func})
        
        return func
    return wrapper

