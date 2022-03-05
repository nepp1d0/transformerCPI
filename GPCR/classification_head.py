import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, batch_size=8, hidden_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.layer = nn.Linear(hidden_size * batch_size, batch_size)
        self.head = nn.Sigmoid()

    def forward(self, x):
        # x = Tensor([batch_size, max_input_len, hidden_size)]
        cls_embeddings = x[:, 1, :]
        cls_ = cls_embeddings.squeeze()
        out = self.layer(torch.flatten(cls_))
        return self.head(out)

