import numpy as np


# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
def OOB(activation, layer_name):

    try:
        print('TEST', np.shape(activation[layer_name]), activation[layer_name])
    except:
        print('layer "{}" not found.'.format(layer_name))
    
    return True