import torch
import numpy as np

import config


from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


image_paths = sorted(Path(config.IMAGE_DIR).glob('*.tif'))
mask_paths = sorted(Path(config.MASK_DIR).glob('*.tif'))

train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths,
    mask_paths,
    test_size=0.15,
    random_state=42
)


class ParticleDataset(Dataset):

    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = np.array(
            Image.open(self.image_paths[idx])
        )

        mask = np.array(
            Image.open(self.mask_paths[idx])
        )

        image = image.astype(np.float32)/255.0
        mask = (mask > 0).astype(np.float32)

        image = image[None, :, :]
        mask = mask[None, :, :]

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        return image, mask

train_dataset = ParticleDataset(
    train_images,
    train_masks
)
val_dataset = ParticleDataset(
    val_images,
    val_masks
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)
