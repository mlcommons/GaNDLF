
from .modelBase import ModelBase
import   monai.networks.nets.dynunet as dynunet 


class dynunet_wrapper(ModelBase):
    """
    More info: https://docs.monai.io/en/stable/networks.html#dynunet

    Args:
        spatial_dims (int): number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (Sequence[Union[Sequence[int], int]]): convolution kernel size.
        strides (Sequence[Union[Sequence[int], int]]): convolution strides for each blocks.
        upsample_kernel_size (Sequence[Union[Sequence[int], int]]): convolution kernel size for transposed convolution layers. The values should equal to strides[1:].
        filters (Optional[Sequence[int]]): number of output channels for each blocks.  Defaults to None.
        dropout (Union[Tuple, str, float, None]): dropout ratio. Defaults to no dropout.
        norm_name (Union[Tuple, str]): feature normalization type and arguments. Defaults to INSTANCE. 
        act_name (Union[Tuple, str]): activation layer type and arguments. Defaults to leakyrelu.
        deep_supervision (bool): whether to add deep supervision head before output. Defaults to False.
        deep_supr_num (int): number of feature maps that will output during deep supervision head. The value should be larger than 0 and less than the number of up sample layers. Defaults to 1.
        res_block (bool): whether to use residual connection based convolution blocks during the network. Defaults to False.
        trans_bias (bool): whether to set the bias parameter in transposed convolution layers. Defaults to False.
    """
    
    def __init__(self,parameters:dict ):
        super(dynunet_wrapper, self).__init__(parameters)

        #checking for validation
        assert ("kernel_size" in parameters["model"]) == True, "\033[0;31m`kernel_size` key missing in parameters"
        assert ("strides" in parameters["model"]) == True, "\033[0;31m`strides` key missing in parameters"   

        # defining some defaults
        
        if not ("upsample_kernel_size" in parameters["model"]):
            parameters["model"]["upsample_kernel_size"] = parameters["model"]["strides"][1:]

        if not ("filters" in parameters["model"]):
            parameters["model"]["filters"] = None

        if not ("act_name" in parameters["model"]):
            parameters["model"]["act_name"] = ('leakyrelu', {'inplace': True, 'negative_slope': 0.01})
        
        if not ("deep_supervision" in parameters["model"]):
            parameters["model"]["deep_supervision"] = False
        
        if not ("deep_supr_num" in parameters["model"]):
            parameters["model"]["deep_supr_num"] = 1 
        
        if not ("res_block" in parameters["model"]):
            parameters["model"]["res_block"] = False
        
        if not ("trans_bias" in parameters["model"]):
            parameters["model"]["trans_bias"] = False
        
        # if not ("norm_type" in parameters["model"]):
        #     self.norm_type = "INSTANCE"
        
        if not ("dropout" in parameters):
            parameters["model"]["dropout"] = None
        
        
       
        
        self.model = dynunet.DynUNet(spatial_dims= self.n_dimensions,
                                    in_channels=self.n_channels,
                                    out_channels=self.base_filters, #? Is it correct? 
                                    kernel_size= parameters["model"]["kernel_size"],
                                    strides=parameters["model"]["strides"],
                                    upsample_kernel_size=parameters["model"]["upsample_kernel_size"],  #The values should equal to strides[1:]
                                    filters=parameters["model"]["filters"],#? self.base_filter??? , number of output channels for each blocks
                                    dropout=parameters["model"]["dropout"],#dropout ratio. Defaults to no dropout
                                    norm_name=self.norm_type, #? Is it correct??
                                    act_name=parameters["model"]["act_name"],
                                    deep_supervision=parameters["model"]["deep_supervision"],
                                    deep_supr_num=parameters["model"]["deep_supr_num"], # number of feature maps that will output during deep supervision head.
                                    res_block=parameters["model"]["res_block"],
                                    trans_bias=parameters["model"]["trans_bias"])
        
        
    def forward(self, x):
        return self.model.forward(x)