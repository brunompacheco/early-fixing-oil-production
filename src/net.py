import torch
import torch.nn as nn


class ObjSurrogate(nn.Module):
    def __init__(self, n_in=17572, layers=[100, 10, 1]) -> None:
        super().__init__()
        hidden_layers = list()

        n_hidden_prior = n_in
        for n_hidden in layers:
            hidden_layers += [
                nn.Linear(n_hidden_prior, n_hidden),
                nn.ReLU(),
            ]
            n_hidden_prior = n_hidden

        self.net = nn.Sequential(*hidden_layers[:-1])  # ignore last activation

    def forward(self, A, b, z_c, z_gl):
        x = torch.hstack([A.flatten(1), b, z_c, z_gl])

        return self.net(x)
