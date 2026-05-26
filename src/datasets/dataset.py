import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):
    """
    Dataset para detección de cambios en imágenes satelitales.
    Combina imágenes PRE y POST y retorna máscara de cambio.
    """

    def __init__(self, tiles_dirs, labels_dirs):
        self.samples = []

        for tiles_dir, labels_dir in zip(tiles_dirs, labels_dirs):

            ids = sorted([
                f.split("_")[1]
                for f in os.listdir(tiles_dir)
                if "pre" in f
            ])

            for tile_id in ids:
                self.samples.append((tiles_dir, labels_dir, tile_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        tiles_dir, labels_dir, tile_id = self.samples[idx]

        pre_path = os.path.join(tiles_dir, f"tile_{tile_id}_pre.npy")
        post_path = os.path.join(tiles_dir, f"tile_{tile_id}_post.npy")
        label_path = os.path.join(labels_dir, f"tile_{tile_id}_label.npy")

        pre = np.load(pre_path)
        post = np.load(post_path)

        # Concatenación de bandas (PRE + POST)
        x = np.concatenate([pre, post], axis=0)

        label = np.load(label_path)
        label = np.expand_dims(label, axis=0)

        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return x, label