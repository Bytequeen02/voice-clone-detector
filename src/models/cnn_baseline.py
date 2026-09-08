"""
Baseline model: a small CNN operating on log-mel spectrograms.

WHY THIS AS THE BASELINE (not a pretrained transformer):
  - Trains from scratch in a reasonable time on a laptop GPU or even
    CPU for a demo-sized dataset - no big download, no license issues.
  - Log-mel + CNN is a well-established, well-understood approach for
    audio spoof/anti-spoofing style classification, so it is a safe,
    explainable starting point for a hackathon timeline.
  - It gives us a working end-to-end pipeline FAST, which we then use
    to validate every other module (risk scoring, explainability,
    API) before spending time on anything fancier.

This is intentionally a small architecture (4 conv blocks). If you
have more time/GPU budget, see src/models/ssl_advanced.py notes in
the README for the "advanced" option (wav2vec2 / WavLM feature
extractor + classifier head).
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # makes it robust to variable time-frame counts
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mels, time_frames)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)   # raw logits - softmax applied outside (see inference)
