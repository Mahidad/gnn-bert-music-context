"""
Task 2 models: the GraphSAGE encoder, plus the CNN baseline it must be
compared against (baseline B2 in the brief).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEClassifier(nn.Module):
    """
    Implements the brief's GraphSAGE update:
        h_i^(l+1) = sigma( W^(l) . CONCAT[ h_i^(l), MEAN_{j in N(i)} h_j^(l) ] )

    PyG's SAGEConv does exactly that concat-of-self-and-neighbour-mean
    internally, so each SAGEConv layer is one round of "every segment
    gossips with its neighbours and updates its own opinion".

    Two layers is the sensible default: after layer 1 a node knows about
    its direct neighbours, after layer 2 it knows about neighbours of
    neighbours. Going much deeper causes over-smoothing -- every node
    converges to the same averaged representation and the model loses
    the ability to tell segments apart.
    """

    def __init__(self, in_dim, hidden_dim=128, num_classes=10,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, x, edge_index, batch):
        """Returns the graph-level embedding g (used again in Task 3)."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Mean-pool all node vectors into one vector for the whole track,
        # matching the brief's readout g = (1/|V|) * sum_i h_i.
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch):
        g = self.encode(x, edge_index, batch)
        return self.classifier(g)


class MelCNNBaseline(nn.Module):
    """
    Baseline B2: a conventional 2D CNN over the log-mel spectrogram, with
    no graph and no text. This is the comparison that makes the Task 2
    result meaningful -- it answers "does modelling the song as a graph
    of related segments actually buy us anything over treating the
    spectrogram as an image?"
    """

    def __init__(self, num_classes=10, n_mels=128):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        return self.classifier(self.pool(self.features(x)))
