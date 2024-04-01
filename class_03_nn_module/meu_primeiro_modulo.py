import torch

class RedeDeSoma(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.camada_linear = torch.nn.Linear(2,1)

    def forward(self, x):
        y_hat = self.camada_linear(x)
        return y_hat
    

