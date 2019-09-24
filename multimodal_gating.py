import torch
from torch import nn


class Multimodal_Gated_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, mode=0):
        super().__init__()
        self.mode = mode

        self.fc_h1 = nn.Linear(input_dimension, output_dimension)
        self.fc_h2 = nn.Linear(input_dimension, output_dimension)
        self.fc_h3 = nn.Linear(input_dimension, output_dimension)

        if self.mode == 0:
            self.fc_z1 = nn.Linear(3 * input_dimension, output_dimension)
            self.fc_z2 = nn.Linear(3 * input_dimension, output_dimension)
            self.fc_z3 = nn.Linear(3 * input_dimension, output_dimension)
        elif self.mode == 1:
            self.fc_z1 = nn.Linear(2 * input_dimension, output_dimension)
            self.fc_z2 = nn.Linear(2 * input_dimension, output_dimension)
            self.fc_z3 = nn.Linear(2 * input_dimension, output_dimension)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()

        self.sigm1 = nn.Sigmoid()
        self.sigm2 = nn.Sigmoid()
        self.sigm3 = nn.Sigmoid()

    def forward(self, inputs):
        # hi computation
        h1 = self.tanh1(self.fc_h1(inputs[0]))
        h2 = self.tanh2(self.fc_h2(inputs[1]))
        h3 = self.tanh3(self.fc_h3(inputs[2]))

        # zi computation
        if self.mode == 0:
            concatenated = torch.cat(inputs, dim=1)
            z1 = self.sigm1(self.fc_z1(concatenated))
            z2 = self.sigm2(self.fc_z2(concatenated))
            z3 = self.sigm3(self.fc_z3(concatenated))
        elif self.mode == 1:
            term1 = torch.cat([inputs[0], inputs[1] + inputs[2]], dim=1)
            term2 = torch.cat([inputs[1], inputs[0] + inputs[2]], dim=1)
            term3 = torch.cat([inputs[2], inputs[0] + inputs[1]], dim=1)
            z1 = self.sigm1(self.fc_z1(term1))
            z2 = self.sigm2(self.fc_z2(term2))
            z3 = self.sigm3(self.fc_z3(term3))

        # h computation
        if self.mode == 0:
            h = torch.mul(z1, h1)\
                + torch.mul(z2, h2)\
                + torch.mul(z3, h3)
        elif self.mode == 1:
            h = torch.mul(z1, h1) + torch.mul((1 - z1), h2 + h3)\
                + torch.mul(z2, h2) + torch.mul((1 - z2), h1 + h3)\
                + torch.mul(z3, h3) + torch.mul((1 - z3), h1 + h2)

        return h
