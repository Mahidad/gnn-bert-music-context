"""
Task 1: BERT-based multi-label tag classifier.

Model: pretrained BERT/DistilBERT -> CLS embedding -> linear head ->
sigmoid per tag. Trained with binary cross-entropy because each tag is
an independent yes/no decision -- a track can be "mellow" AND "piano"
AND "pop" all at once, which is what makes this multi-*label* rather
than multi-*class* (where only one answer could be right).
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score


class BertTagClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_tags=50, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler_output, so we grab the CLS token
        # manually -- this matches the spec's t = BERT_CLS(X_text) step.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)  # raw logits (sigmoid applied later)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss": total_loss / len(dataloader.dataset),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
