import torch
import torch.nn as nn


class ObjSurrogate(nn.Module):
    def __init__(self, n_in=13, layers=[40, 20, 10, 10, 10], add_dropout=False) -> None:
        super().__init__()
        hidden_layers = list()

        n_hidden_prior = n_in
        for n_hidden in layers:
            hidden_layers += [
                nn.Linear(n_hidden_prior, n_hidden),
                nn.Dropout(p=0.2),
                nn.ReLU(),
            ]
            n_hidden_prior = n_hidden

        # output layer
        hidden_layers += [
            nn.Linear(n_hidden_prior, 1),
        ]

        self.net = nn.Sequential(*hidden_layers)

        if add_dropout == True:
            self.add_dropout()

    def add_dropout(self, p=0.05):
        self.net = torch.nn.Sequential(
            self.net[:1] +
            torch.nn.Sequential(*[torch.nn.Dropout(p=p),]) +
            self.net[1:3] +
            torch.nn.Sequential(*[torch.nn.Dropout(p=p),]) +
            self.net[3:]
        )

    def forward(self, bsw, gor, z_c, z_gl, q_gl_max):
        x = torch.hstack([z_c, z_gl, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e3, (q_gl_max.unsqueeze(1) - 1e5) / 2e5])

        return self.net(x) * 2e3

class Fixer(nn.Module):
    def __init__(self, n_in=3, layers=[10, 10, 10]) -> None:
        super().__init__()
        hidden_layers = list()

        n_hidden_prior = n_in
        for n_hidden in layers:
            hidden_layers += [
                nn.Linear(n_hidden_prior, n_hidden),
                # nn.Dropout(p=0.2),
                nn.ReLU(),
            ]
            n_hidden_prior = n_hidden

        # output layer
        hidden_layers += [
            nn.Linear(n_hidden_prior, 10),
        ]

        self.net = nn.Sequential(*hidden_layers)

    def forward(self, bsw, gor, q_gl_max):
        x = torch.hstack([bsw.unsqueeze(1), gor.unsqueeze(1) / 1e3, (q_gl_max.unsqueeze(1) - 1e5) / 2e5])

        logits = self.net(x)

        return logits.unflatten(1, (2, 5))
