import torch
from torch import nn


class SiameseDamageNet(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)
        return self.embedding(feat)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_emb = self.encode(pre)
        post_emb = self.encode(post)
        delta = torch.abs(post_emb - pre_emb)
        stacked = torch.cat([pre_emb, post_emb, delta], dim=1)
        logits = self.classifier(stacked)
        return logits.squeeze(1)

