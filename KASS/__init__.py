from types import FunctionType as FunctionType
from functools import wraps as wraps

def Process(serial, queue):
    '''
        Description:
            This wrapper processes the queue, where
            if an attribute is in the queue, then
            the attribute on the serial is randomized
            and removed from queue, otherwise the
            value of the attribute in the serial is
            returned (could be modified or original).
            
            In the event an attribute dependent on
            another which in itself is dependent on
            the formal is encountered, then the latter
            attribute will use the formal's original
            attribute from the serial in order to
            randomize, which will then allow for the
            formal attribute to randomize based on the
            newly modified latter attribute. This will
            result in the removal of the latter
            attribute from the queue aswell.
        
        Arguments:
            *serial     : Is a collection of attribute values
                        that can get randomized.
            
            *queue      : Is a local empty deque instance.
        
        returns serial['config'][{attribute}]
    '''
    
    def wrapper(func):
        attribute = func.__name__
        
        @wraps(func)
        def call():
            attributes = [method.__name__ for method in queue]
            
            if attribute in attributes:
                index = attributes.index(attribute)
                queue.rotate(-index)
                queue.popleft()
                queue.rotate(index)
                
                serial['config'][attribute] = func()
            
            return serial['config'][attribute]
        call._decorator_name_ = attribute #set call.__name__ to func.__name__
        return call
    return wrapper

def run(queue, attributes, locals_):
    '''
        Description:
            Is a function to initiate the randomization
            process of all attributes in a sample function
            one at a time to prevent deadlocks from self
            dependent pairs.
        
        Arguments:
            *queue          : Is a local empty deque instance.
            
            *attributes     : Is a list of attributes to be
                            randomized.
                            
            *locals_        : Is the result of locals() from
                            sample functions.
        
        returns None
    '''
    
    #establish queue
    [queue.append(method) for method in list(locals_.values()) if isinstance(method, FunctionType) and (len(attributes) == 0 or method.__name__ in attributes)]
    #process queue
    while len(queue) != 0:
        queue[0]()

from . import layers
from . import initializers
from . import regularizers
from . import constraints

from . import utils
from . import samples

