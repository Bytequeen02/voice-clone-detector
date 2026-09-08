"""
PyTorch Dataset wrapping the manifests produced by prepare_dataset.py.
Applies the SAME preprocessing functions used at inference time.
"""
import json
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_preprocessing import preprocess_file

LABEL_TO_IDX = {"real": 0, "cloned": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class VoiceCloneDataset(Dataset):
    def __init__(self, manifest_path: str, cfg: dict):
        with open(manifest_path, "r") as f:
            self.items = json.load(f)
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        features = preprocess_file(item["path"], self.cfg)   # (n_mels, time)
        tensor = torch.from_numpy(features).unsqueeze(0)      # (1, n_mels, time) -> CNN channel dim
        label = LABEL_TO_IDX[item["label"]]
        return tensor, label
