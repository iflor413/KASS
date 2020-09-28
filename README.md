This library is a tool extension for the keras machine
learning API, offering a sampling tool that is capable
of randomizing advanced serials which includes layers,
constraints, regularizers and initializers. Functionality
also extends to custom keras objects and attributes.
However sampling is user defined (see below for more
information).  

Randomization is performed by a modulated function, which
allows for the dynamic randomization of independent and
multi-dependent attributes of a serialized keras object.  

## Requirements:
The KASD library is required which can be found here:  
https://github.com/iflor413/KASD

## Sampling Example
```
>>> from keras.layers import Input, Dense
>>> from KASD.layers import serialize
>>> from KASS.layers import randomize
>>> 
>>> tensor = Dense(10)(Input(shape=(20,)))
>>> serial = serialize(tensor)
>>> 
>>> print(serial)
>>> randomize(serial)
>>> print(serial)
```

## Note:
Sample functions found in the samples file are based on
the 2.3.1 keras version and may not be compatible with
previous or future versions. However the core functionality
is fully functional across multiple keras versions. For
sampler support of custom keras objects/attributes or
modification see below.

**Custom Layer:**
```
>>> from KASS import Process, run
>>> from KASS.layers import custom
>>> from collections import deque
>>> 
>>> @custom('{custom_layer_class_name}', override=False) #override=True, to replace existing sampler function
>>> def sample(serial, attributes=[], ranges=default_ranges):
>>>     input_, input_shape, output_shape, queue = serial['input'], serial['input_shape'], serial['output_shape'], deque([])
>>> 
>>>     @Process(serial, queue)
>>>     def attr1():
>>>         return #value
>>>     
>>>     @Process(serial, queue)
>>>     def attr2():
>>>         return #value
>>>     
>>>     #...
>>>     
>>>     run(queue, attributes, locals())
```

**Custom Constraint:**
```
>>> from KASS import Process, run
>>> from KASS.constraints import custom
>>> from collections import deque
>>> 
>>> @custom('{custom_constraint_class_name}', override=False) #override=True, to replace existing sampler function
>>> def sample(serial, input_shape, attributes=[], ranges=default_ranges):
>>>     queue = deque([])
>>>     
>>>     @Process(serial, queue)
>>>     def attr1():
>>>         return #value
>>>     
>>>     @Process(serial, queue)
>>>     def attr2():
>>>         return #value
>>>     
>>>     #...
>>>     
>>>     run(queue, attributes, locals())
```

**Custom Initializer:**
```
>>> from KASS import Process, run
>>> from KASS.initializers import custom
>>> from collections import deque
>>> 
>>> @custom('{custom_initializer_class_name}', override=False) #override=True, to replace existing sampler function
>>> def sample(serial, attributes=[], ranges=default_ranges):
>>>     queue = deque([])
>>>     
>>>     @Process(serial, queue)
>>>     def attr1():
>>>         return #value
>>>     
>>>     @Process(serial, queue)
>>>     def attr2():
>>>         return #value
>>>     
>>>     #...
>>>     
>>>     run(queue, attributes, locals())
```

**Custom Regularizer:**
```
>>> from KASS import Process, run
>>> from KASS.regularizers import custom
>>> from collections import deque
>>> 
>>> @custom('{custom_regularizer_class_name}', override=False) #override=True, to replace existing sampler function
>>> def sample(serial, attributes=[], ranges=default_ranges):
>>>     queue = deque([])
>>>     
>>>     @Process(serial, queue)
>>>     def attr1():
>>>         return #value
>>>     
>>>     @Process(serial, queue)
>>>     def attr2():
>>>         return #value
>>>     
>>>     #...
>>>     
 >>>    run(queue, attributes, locals())
```

## Repository:
https://github.com/iflor413/KASS

## Compatibility:
**Python:** >= 2.7  
**Keras:** 2.0.8, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.2, 2.2.4, 2.3.1 (see Note section)

