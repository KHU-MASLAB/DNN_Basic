import torch
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self,
            in_N: int,
            depth: int,
            width: int,
            out_N: int,
            activation: str = 'gelu',
            param_init: str = 'he',
            batchnorm: bool = False,
            dropout: float = 0):
        
        super(MLP, self).__init__()
        self.depth = depth
        self.batchnorm = batchnorm
        self.dropout = dropout
        
        # Linear layers
        self.Linear = torch.nn.ModuleList()
        self.Linear.append(torch.nn.Linear(in_N, width, bias=True))
        for i in range(depth - 1):
            self.Linear.append(torch.nn.Linear(width, width, bias=True))
        self.Linear.append(torch.nn.Linear(width, out_N, bias=True))
        
        # BatchNorm layers
        if batchnorm:
            self.BatchNorm = torch.nn.ModuleList()
            for j in range(depth):
                self.BatchNorm.append(torch.nn.BatchNorm1d(num_features=width))
        
        # Activation function
        if activation.lower() == 'relu':
            self.Activation = torch.nn.ReLU()
        if activation.lower() == 'gelu':
            self.Activation = torch.nn.GELU()
        if activation.lower() == 'silu':
            self.Activation = torch.nn.SiLU()
        if activation.lower() == 'mish':
            self.Activation = torch.nn.Mish()
        if activation.lower() == 'elu':
            self.Activation = torch.nn.ELU()
        if activation.lower() == 'tanh':
            self.Activation = torch.nn.Tanh()
        if activation.lower() == 'none':
            self.Activation = None
        
        # Dropout layer
        if dropout:
            self.Dropout = torch.nn.Dropout(dropout)
        
        # Initialization
        if param_init == 'default':
            self.apply(self.__initializer_default)
        elif param_init == 'he':
            self.apply(self.__initializer_He)
    
    def __initializer_default(self, layer):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if isinstance(layer, (torch.nn.Linear)):
            # torch.nn.init.kaiming_normal_(layer.weight, a=0, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5), nonlinearity='leaky_relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def __initializer_He(self, layer):
        if isinstance(layer, (torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(layer.weight, a=0, nonlinearity='relu')
            # torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5), nonlinearity='leaky_relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Input layer
        y = self.Linear[0](x)
        y = self.BatchNorm[0](y) if self.batchnorm else y
        y = self.Activation(y) if self.Activation else y
        y = self.Dropout(y) if self.dropout else y
        # Hidden layers
        for i in range(1, self.depth):
            y = self.Linear[i](y)
            y = self.BatchNorm[i](y) if self.batchnorm else y
            y = self.Activation(y) if self.Activation else y
            y = self.Dropout(y) if self.dropout else y
        # Output layer
        y = self.Linear[-1](y)
        
        return y