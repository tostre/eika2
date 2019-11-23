import torch.nn as nn

class Lin_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(Lin_Net, self).__init__()
        self.act_function = nn.ReLU()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(0, num_hidden_layers)])
        self.lin4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act_function(self.lin1(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.act_function(hidden_layer(x))
        x = self.lin4(x)
        return x
