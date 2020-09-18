This library offers a randomization tool for advanced
serial sampling of native and custom keras object
attributes. Randomization is performed by a modulated
function, which allows for the dynamic randomization
of interdependent attributes of a serialized keras
object.

## Note:
Sample functions found in the sample directory
are based on the 2.3.1 keras version and may not
be compatible with previous versions. However the
core functionality is fully functional across
multiple keras versions. For sampler support of
custom keras objects or modification see below.

**Custom Layer:**
```
from KASS import Process, run
from KASS.layers import custom
from collections import deque

@custom('{custom_layer_class_name}') #override=True, to replace existing sampler function
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
```

**Custom Constraint:**
```
from KASS import Process, run
from KASS.constraints import custom
from collections import deque

@custom('{custom_constraint_class_name}') #override=True, to replace existing sampler function
def sample(serial, input_shape, attributes=[], ranges=default_ranges):
    queue = deque([])
    
    @Process(serial, queue)
    def attr1():
        pass
    
    @Process(serial, queue)
    def attr2():
        pass
    
    ...
    
    run(queue, attributes, locals())
```

**Custom Initializer:**
```
from KASS import Process, run
from KASS.initializers import custom
from collections import deque

@custom('{custom_initializer_class_name}') #override=True, to replace existing sampler function
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
```

**Custom Regularizer:**
```
from KASS import Process, run
from KASS.regularizers import custom
from collections import deque

@custom('{custom_regularizer_class_name}') #override=True, to replace existing sampler function
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
```

## Repository:
https://github.com/iflor413/KASS

## Compatibility:
**Python:** >= 2.7
**Keras:** 2.0.8, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.2, 2.2.4, 2.3.1 (see Note section)

