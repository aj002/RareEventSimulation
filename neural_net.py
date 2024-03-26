import torch
import torch.nn as nn
import torch.nn.init as init

# class TwoLayerNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(TwoLayerNet, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.tanh = nn.Tanh()
#         self.layer2 = nn.Linear(hidden_size, output_size)

#     def forward(self, X_tensor, t_tensor):
#         combined_input = torch.cat((X_tensor, t_tensor), dim=1)
#         x = self.layer1(combined_input)
#         x = self.tanh(x)
#         x = self.layer2(x)
#         return x



class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights and biases to zero
        # self._init_weights_to_zero()

    def forward(self, X_tensor, t_tensor):
        combined_input = torch.cat((X_tensor, t_tensor), dim=1)
        x = self.layer1(combined_input)
        x = self.tanh(x)
        x = self.layer2(x)
        return x
    
    def _init_weights_to_zero(self):
        with torch.no_grad():
            # Initialize weights to zero
            init.zeros_(self.layer1.weight)
            init.zeros_(self.layer2.weight)
            # Initialize biases to zero
            init.zeros_(self.layer1.bias)
            init.zeros_(self.layer2.bias)





class TwoLayerNet_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet_Linear, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Identity()  # Replace nn.Tanh() with nn.Identity()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, X_tensor, t_tensor):
        combined_input = torch.cat((X_tensor, t_tensor), dim=1)
        x = self.layer1(combined_input)
        x = self.activation(x)
        x = self.layer2(x)
        return x
