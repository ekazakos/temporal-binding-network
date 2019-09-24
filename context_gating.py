import torch
from torch import nn
import torch.nn.functional as F


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):

        x = self.fc(x)  #FC layer 
        x = self.cg(x)  #Context Gating Unit
        x = F.normalize(x)  #normalise

        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1)

        x = torch.cat((x, x1), 1)

        return F.glu(x, 1)
