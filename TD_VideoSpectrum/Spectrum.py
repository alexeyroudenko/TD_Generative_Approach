import torch.nn as nn


class SpectrumEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(64, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 96)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(96, 4)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x