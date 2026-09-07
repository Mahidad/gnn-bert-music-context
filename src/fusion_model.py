"""
Task 3: GNN-BERT fusion for multi-label music context understanding.

One class, four fusion variants, selected by a string. Keeping them in a
single model means the ablation compares fusion strategies and nothing
else -- same encoders, same head, same training loop. If each variant were
a separate class, differences in initialisation or head size would
contaminate the comparison.

Variants:
  bert_only        y = W t                    (text alone -- Task 1 control)
  gnn_only         y = W g                    (structure alone -- Task 2 control)
  concat           y = W [g ; t]              (stapled together, no interaction)
  cross_attention  y = W [g ; Attn(g, H)]     (structure queries the text)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel

VARIANTS = ["bert_only", "gnn_only", "concat", "cross_attention"]


class GraphEncoder(nn.Module):
    """GraphSAGE encoder producing one vector g per graph (same as Task 2)."""

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
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class CrossAttention(nn.Module):
    """
    The graph asks the text questions.

    A single query is built from the graph vector g, and it attends over
    every token embedding in H_text. The output is a text summary weighted
    by what the structure found relevant -- "given that this track repeats
    heavily, which words in the caption matter?"

    Padding tokens are masked out. Without that, attention would spread
    weight onto [PAD] positions and the summary would be diluted by
    whatever those embeddings happen to contain.
    """

    def __init__(self, graph_dim, text_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must divide by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(graph_dim, hidden_dim)
        self.k_proj = nn.Linear(text_dim, hidden_dim)
        self.v_proj = nn.Linear(text_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, g, h_text, attention_mask, return_weights=False):
        B, L, _ = h_text.shape

        q = self.q_proj(g).view(B, self.num_heads, 1, self.head_dim)
        k = self.k_proj(h_text).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_text).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale        # (B, heads, 1, L)

        mask = attention_mask.view(B, 1, 1, L).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = (self.dropout(weights) @ v)                 # (B, heads, 1, head_dim)
        attended = attended.transpose(1, 2).reshape(B, -1)
        attended = self.out_proj(attended)

        if return_weights:
            # averaged over heads -> (B, L), one weight per caption token.
            # This is what powers the interpretability case studies.
            return attended, weights.mean(dim=1).squeeze(1)
        return attended


class GNNBertFusion(nn.Module):
    def __init__(self, node_dim, num_tags, variant="cross_attention",
                 bert_name="distilbert-base-uncased", gnn_hidden=128,
                 gnn_layers=2, gnn_dropout=0.3, fusion_dim=256,
                 attn_heads=4, dropout=0.3):
        super().__init__()
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.variant = variant

        self.needs_text = variant != "gnn_only"
        self.needs_graph = variant != "bert_only"

        if self.needs_text:
            self.bert = AutoModel.from_pretrained(bert_name)
            text_dim = self.bert.config.hidden_size
        if self.needs_graph:
            self.gnn = GraphEncoder(node_dim, gnn_hidden, gnn_layers, gnn_dropout)
            graph_dim = self.gnn.out_dim

        if variant == "bert_only":
            z_dim = text_dim
        elif variant == "gnn_only":
            z_dim = graph_dim
        elif variant == "concat":
            z_dim = graph_dim + text_dim
        else:
            self.cross_attn = CrossAttention(graph_dim, text_dim, fusion_dim,
                                             attn_heads, dropout)
            z_dim = graph_dim + self.cross_attn.out_dim

        self.head = nn.Sequential(
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_tags),
        )
        self.z_dim = z_dim

    def encode(self, batch, return_attention=False, shuffle_graph=False):
        """
        Returns the fused representation z (and optionally attention weights).

        shuffle_graph randomly permutes the graph vectors across the batch,
        so each caption is paired with a DIFFERENT track's audio. If test
        performance is unchanged by this, the model was ignoring the audio
        branch -- a direct test of whether fusion uses the graph at all,
        costing one inference pass instead of a retrain.
        """
        g = t = h_text = None

        if self.needs_graph:
            g = self.gnn(batch.x, batch.edge_index, batch.batch)
            if shuffle_graph:
                g = g[torch.randperm(g.size(0), device=g.device)]

        if self.needs_text:
            out = self.bert(input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask)
            h_text = out.last_hidden_state
            t = h_text[:, 0, :]                      # CLS token

        weights = None
        if self.variant == "bert_only":
            z = t
        elif self.variant == "gnn_only":
            z = g
        elif self.variant == "concat":
            z = torch.cat([g, t], dim=-1)
        else:
            if return_attention:
                attended, weights = self.cross_attn(
                    g, h_text, batch.attention_mask, return_weights=True)
            else:
                attended = self.cross_attn(g, h_text, batch.attention_mask)
            z = torch.cat([g, attended], dim=-1)

        return (z, weights) if return_attention else z

    def forward(self, batch, return_attention=False, shuffle_graph=False):
        if return_attention:
            z, weights = self.encode(batch, return_attention=True,
                                     shuffle_graph=shuffle_graph)
            return self.head(z), weights
        return self.head(self.encode(batch, shuffle_graph=shuffle_graph))

    def param_groups(self, bert_lr, head_lr, weight_decay):
        """
        BERT gets a much smaller learning rate than everything else.

        BERT arrives pretrained and only needs nudging; the GNN, the
        attention block and the head start from random initialisation and
        need to move fast. One shared learning rate either destroys BERT's
        pretrained weights or leaves the new modules barely trained.
        """
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
