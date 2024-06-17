import torch
from mamba_ssm import Mamba
from config import config
class MambaTab(torch.nn.Module):
    """
    This class defines the MambaTab model
    """
    def __init__(self,input_features,n_class,intermediate_representation=config['REPRESENTATION_LAYER']):
        super(MambaTab, self).__init__()
        self.linear_layer=torch.nn.Linear(input_features,intermediate_representation)
        self.relu=torch.nn.ReLU()
        self.layer_norm=torch.nn.LayerNorm(intermediate_representation)

        self.mamba=Mamba(d_model=intermediate_representation,d_state=32,d_conv=4,expand=2) # Please use different parameters settings for different configurations
        self.output_layer=torch.nn.Linear(intermediate_representation,n_class)
    
    def forward(self, x):
         x=self.linear_layer(x)
         x=self.layer_norm(x)
         x=self.relu(x)
         x=self.mamba(x)
         x=self.output_layer(x)
         return x