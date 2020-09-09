

class ModelBase:
    '''
    This is the base model class that all other architectures will need to derive from
    '''
    def __init__(self, n_channels, n_classes, base_filters, final_convolution_layer):
        '''
        This defines all defaults that the model base uses
        '''
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.final_convolution_layer = final_convolution_layer