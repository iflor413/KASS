from KASD.layers import get, serialize, deserialize
from KASS.layers import randomize
from keras.layers import Input

import traceback

def checkPerformance(print_results=False):
    def get_sub_cls(sub_class_name, sub_arg, sub_kwargs):
        sub_cls = get(sub_class_name)
        
        if sub_cls is None:
            raise NotImplementedError(sub_class_name)
        
        if 'layer_cell' in sub_kwargs:
            layer_cell = sub_kwargs['layer_cell']
            
            if isinstance(layer_cell, list):
                sub_sub_cls = ([get_sub_cls(*LS) for LS in layer_cell],)
            else:
                sub_sub_cls = (get_sub_cls(layer_cell),)
            
            sub_arg = sub_sub_cls+sub_arg
            del sub_kwargs['layer_cell']
        return sub_cls(*sub_arg, **sub_kwargs)
        
    def checkFunctionality(input_batch_shape, class_name, *arg, **kwargs):
        _input = Input(batch_shape=input_batch_shape) if isinstance(input_batch_shape, tuple) else [Input(batch_shape=batch_shape) for batch_shape in input_batch_shape]
        
        print('='*(40+60*print_results))
        
        check_list = {"Build": False, "Serialization": False, "Randomization": False, "Deserialization": False}
        print('{} Test Results:'.format(class_name))
        
        #######Build Test#######
        
        try:
            cls = get(class_name)
            
            if cls is None:
                raise NotImplementedError(class_name)
            
            if 'layer_cell' in kwargs:
                sub_class_name, sub_arg, sub_kwargs = kwargs['layer_cell']
                arg = (get_sub_cls(sub_class_name, sub_arg, sub_kwargs),)+arg
                del kwargs['layer_cell']
            
            tensor = cls(*arg, **kwargs)(_input)
            tensors = (_input if isinstance(_input, list) else [_input])+[tensor]
            
            check_list['Build'] = True
        except:
            print('Build: Failed')
            traceback.print_exc()
        
        #######Serialization Test#######
        
        if check_list['Build']:
            try:
                serial = serialize(tensors)
                serial = serial[list(serial.keys())[-1]]
                
                if print_results:
                    print('Advanced Serialization Result:\n', serial, '\n')
                check_list['Serialization'] = True
            except:
                print('Serialization: Failed')
                traceback.print_exc()
        
        #######Randomization Test#######
        
        if check_list['Serialization']:
            try:
                randomize(serial)
                
                if print_results:
                    print('Randomization Result:\n', serial, '\n')
                check_list['Randomization'] = True
            except:
                print('Randomization: Failed')
                traceback.print_exc()
        
        #######Deserialization Test#######
        
        if check_list['Build'] and check_list['Serialization']:
            try:
                tensor = deserialize(serial)
                
                if print_results:
                    print('Advanced Deserialization Result:\n', tensor, '\n')
                check_list['Deserialization'] = True
            except:
                print('Deserialization: Failed')
                print('With serial:\n', serial)
                traceback.print_exc()
        
        if not False in check_list.values():
            print('Fully Functional!')
        print()
    
    a = (None, 10)
    b = (None, 10, 10)
    bb = (10, 10, 10)
    c = (None, 10, 10, 10)
    cc = (10, 10, 10, 10)
    d = (None, 10, 10, 10, 10)
    dd = (10, 10, 10, 10, 10)
    
    #advanced_activations
    checkFunctionality(a, 'LeakyReLU')
    checkFunctionality(a, 'PReLU')
    checkFunctionality(a, 'ThresholdedReLU')
    checkFunctionality(a, 'Softmax')
    checkFunctionality(a, 'ReLU')
    
    #convolutional
    checkFunctionality(b, 'Conv1D', *(10, 1))
    checkFunctionality(c, 'Conv2D', *(10, 1))
    checkFunctionality(d, 'Conv3D', *(10, 1))
    checkFunctionality(c, 'Conv2DTranspose', *(10, 1))
    checkFunctionality(d, 'Conv3DTranspose', *(10, 1))
    checkFunctionality(b, 'SeparableConv1D', *(10, 1))
    checkFunctionality(c, 'SeparableConv2D', *(10, 1))
    checkFunctionality(c, 'DepthwiseConv2D', *(1,))
    checkFunctionality(b, 'UpSampling1D')
    checkFunctionality(c, 'UpSampling2D')
    checkFunctionality(d, 'UpSampling3D')
    checkFunctionality(b, 'ZeroPadding1D')
    checkFunctionality(c, 'ZeroPadding2D')
    checkFunctionality(d, 'ZeroPadding3D')
    checkFunctionality(b, 'Cropping1D')
    checkFunctionality(c, 'Cropping2D')
    checkFunctionality(d, 'Cropping3D')
    
    #convolutional_recurrent
    print('(ConvLSTM2DCell)')
    checkFunctionality(dd, 'ConvRNN2D', layer_cell=('ConvLSTM2DCell', (9, 1), {}))
    checkFunctionality(dd, 'ConvLSTM2D', *(10, 1))
    
    #core
    checkFunctionality(a, 'Masking')
    checkFunctionality(a, 'Dropout', *(1.0,))
    checkFunctionality(b, 'SpatialDropout1D', *(1.0,))
    checkFunctionality(c, 'SpatialDropout2D', *(1.0,))
    checkFunctionality(d, 'SpatialDropout3D', *(1.0,))
    checkFunctionality(a, 'Activation', *('relu',))
    checkFunctionality(b, 'Reshape', target_shape=(5, 20))
    checkFunctionality(b, 'Permute', *((2, 1),))
    checkFunctionality(b, 'Flatten')
    checkFunctionality(a, 'RepeatVector', *(3,))
    checkFunctionality(a, 'Lambda', *((lambda x: x ** 2), ))
    checkFunctionality(a, 'Dense', *(10,))
    checkFunctionality(a, 'ActivityRegularization')
    
    #cudnn_recurrent
    checkFunctionality(b, 'CuDNNGRU', *(10,)) #need cuda support to test
    checkFunctionality(b, 'CuDNNLSTM', *(10,)) #need cuda support to test
    
    #embeddings
    checkFunctionality(a, 'Embedding', *(1000, 64))
    
    #local
    checkFunctionality(b, 'LocallyConnected1D', *(10, 1))
    checkFunctionality(c, 'LocallyConnected2D', *(10, 1))
    
    #merge
    checkFunctionality([a, a, a], 'Add')
    checkFunctionality([a, a], 'Subtract')
    checkFunctionality([a, a, a], 'Multiply')
    checkFunctionality([a, a, a], 'Average')
    checkFunctionality([a, a, a], 'Maximum')
    checkFunctionality([a, a, a], 'Minimum')
    checkFunctionality([a, a, a], 'Concatenate')
    checkFunctionality([a, a], 'Dot', axes=-1)
    
    #noise
    checkFunctionality(a, 'GaussianNoise', *(1.0,))
    checkFunctionality(a, 'GaussianDropout', *(1.0,))
    checkFunctionality(a, 'AlphaDropout', *(1.0,))
    
    #normalization
    checkFunctionality(a, 'BatchNormalization')
    
    #pooling
    checkFunctionality(b, 'MaxPooling1D')
    checkFunctionality(b, 'AveragePooling1D')
    checkFunctionality(c, 'MaxPooling2D')
    checkFunctionality(c, 'AveragePooling2D')
    checkFunctionality(d, 'MaxPooling3D')
    checkFunctionality(d, 'AveragePooling3D')
    checkFunctionality(b, 'GlobalAveragePooling1D')
    checkFunctionality(b, 'GlobalMaxPooling1D')
    checkFunctionality(c, 'GlobalAveragePooling2D')
    checkFunctionality(c, 'GlobalMaxPooling2D')
    checkFunctionality(d, 'GlobalAveragePooling3D')
    checkFunctionality(d, 'GlobalMaxPooling3D')
    
    #recurrent
    print('(StackedRNNCells(GRUCell, LSTMCell, SimpleRNNCell))')
    checkFunctionality(bb, 'RNN', layer_cell=('StackedRNNCells', (), {'layer_cell': [('GRUCell', (10,), {}), ('LSTMCell', (10,), {}), ('SimpleRNNCell', (10,), {})]}))
    print('(SimpleRNNCell)')
    checkFunctionality(bb, 'RNN', layer_cell=('SimpleRNNCell', (10,), {}))
    checkFunctionality(bb, 'SimpleRNN', *(10,))
    print('(GRUCell)')
    checkFunctionality(bb, 'RNN', layer_cell=('GRUCell', (10,), {}))
    checkFunctionality(bb, 'GRU', *(10,))
    print('(LSTMCell)')
    checkFunctionality(bb, 'RNN', layer_cell=('LSTMCell', (10,), {}))
    checkFunctionality(bb, 'LSTM', *(10,))
    
    #wrappers
    checkFunctionality(b, 'TimeDistributed', layer_cell=('Dense', (10,), {}))
    checkFunctionality(bb, 'Bidirectional', layer_cell=('LSTM', (10,), {}))
    
checkPerformance()


