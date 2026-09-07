"""
Task 4: contrastive dual-encoder for cross-modal retrieval.

Tasks 1-3 asked audio and text to predict the SAME labels, and text won
every time -- captions describe the tags almost directly, so the graph had
little left to add. Retrieval changes the question. Here the model must
match a caption to the correct audio clip out of many candidates, so the
audio encoder cannot free-ride on the text: it has to place each clip
somewhere meaningful in a shared space on its own.

Architecture (following the brief):
    g_i = normalize(proj_g(GNN(G_i)))
    t_i = normalize(proj_t(BERT_CLS(caption_i)))
    S_ij = g_i . t_j / tau
    L = symmetric InfoNCE over S

Symmetric means the loss is computed in both directions -- caption finds
audio, and audio finds caption -- which is what CLIP does and what the
retrieval metrics measure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel


class ProjectionHead(nn.Module):
    """Maps an encoder output into the shared embedding space."""

    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphTower(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, edge_index))),
                          p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class ContrastiveGNNBert(nn.Module):
    def __init__(self, node_dim, embed_dim=256,
                 bert_name="distilbert-base-uncased", gnn_hidden=128,
                 gnn_layers=2, gnn_dropout=0.3, dropout=0.1,
                 init_temperature=0.07):
        super().__init__()
        self.graph_tower = GraphTower(node_dim, gnn_hidden, gnn_layers, gnn_dropout)
        self.bert = AutoModel.from_pretrained(bert_name)

        self.graph_proj = ProjectionHead(self.graph_tower.out_dim, embed_dim,
                                         dropout=dropout)
        self.text_proj = ProjectionHead(self.bert.config.hidden_size, embed_dim,
                                        dropout=dropout)

        # Temperature is learned in log space so it stays positive without
        # a constraint, which is how CLIP parameterises it. A fixed
        # temperature forces you to guess how sharp the similarity
        # distribution should be; learning it lets the model decide.
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature)))

    def encode_graph(self, batch):
        g = self.graph_tower(batch.x, batch.edge_index, batch.batch)
        return F.normalize(self.graph_proj(g), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.text_proj(out.last_hidden_state[:, 0, :]), dim=-1)

    def forward(self, batch):
        g = self.encode_graph(batch)
        t = self.encode_text(batch.input_ids, batch.attention_mask)
        # clamp keeps the scale from exploding early in training, which
        # otherwise makes the softmax saturate and gradients vanish
        scale = self.log_temperature.exp().clamp(max=100.0)
        return g, t, scale

    def param_groups(self, bert_lr, head_lr, weight_decay):
        bert_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (bert_params if name.startswith("bert.") else other_params).append(param)
        groups = [{"params": other_params, "lr": head_lr, "weight_decay": weight_decay}]
        if bert_params:
            groups.append({"params": bert_params, "lr": bert_lr,
                           "weight_decay": weight_decay})
        return groups


def info_nce_loss(g, t, scale):
    """
    Symmetric InfoNCE.

    Every clip in the batch acts as a negative for every other clip, so a
    larger batch means harder negatives and a sharper training signal.
    That is why contrastive training benefits from big batches far more
    than the supervised tasks did.
    """
    logits = scale * g @ t.T                      # (B, B)
    targets = torch.arange(g.size(0), device=g.device)
    loss_audio_to_text = F.cross_entropy(logits, targets)
    loss_text_to_audio = F.cross_entropy(logits.T, targets)
    return (loss_audio_to_text + loss_text_to_audio) / 2


@torch.no_grad()
def retrieval_metrics(g, t, ks=(1, 5, 10)):
    """
    Recall@K in both directions over the full evaluation set.

    Note this is retrieval against every clip in the split, not within a
    batch -- reporting batch-level recall would make the task look far
    easier than it is, since a 16-way choice is much simpler than a
    1030-way one.
    """
    sim = g @ t.T
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    results = {}
    for name, matrix in (("audio_to_text", sim), ("text_to_audio", sim.T)):
        ranks = (matrix.argsort(dim=1, descending=True) == targets.view(-1, 1))
        rank_positions = ranks.float().argmax(dim=1)
        for k in ks:
            results[f"{name}_R@{k}"] = float((rank_positions < k).float().mean())
        results[f"{name}_median_rank"] = float(rank_positions.median().item()) + 1
    results["n_candidates"] = n
    return results
