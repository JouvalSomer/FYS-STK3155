import torch

""" The FFNN """
class Net(torch.nn.Module):
    def __init__(self, num_hidden_units, num_hidden_layers, inputs, outputs=1, inputnormalization=None):
        
        super(Net, self).__init__()        
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.inputnormalization = inputnormalization

        # Dimensions of input/output
        self.inputs =  inputs
        self.outputs = outputs
        
        # Create the inputlayer 
        self.input_layer = torch.nn.Linear(self.inputs, self.num_hidden_units)
        torch.nn.init.xavier_uniform_(self.input_layer.weight)


        # Create the hidden layers
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(
            self.num_hidden_units, self.num_hidden_units)
            for i in range(self.num_hidden_layers - 1)])

        # Create the output layer
        self.output_layer = torch.nn.Linear(self.num_hidden_units, self.outputs)
        
        # Use hyperbolic tangent as activation:
        """ Want to try: Tanh, ReLU, LeakyReLU and Sigmoid """
        self.activation = torch.nn.Tanh()
        
        self.Initialize_weights()

    def forward(self, x):
        """[Compute NN output]

        Args:
            x ([torch.Tensor]): input tensor
        Returns:
            [torch.Tensor]: [The NN output]
        """
        # # Transform the shape of the Tensor to match what is expected by torch.nn.Linear
        # if self.inputnormalization is not None:
        #     x = InputNormalization(x)
        
        """ (n,) -> (n,1)  """
        x = torch.unsqueeze(x, 1) 

        # x[..., -1] = x[..., -1] / tmax
    
        out = self.input_layer(x)
        
        # The first hidden layer:
        out = self.activation(out)

        # The other hidden layers:
        for i, linearLayer in enumerate(self.linear_layers):
            out = linearLayer(out)
            out = self.activation(out)

        # No activation in the output layer:
        out = self.output_layer(out)

        """ (n,1) -> (n,)  """
        out = torch.squeeze(out, 1)

        return out
    
    def Initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)