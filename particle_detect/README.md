# SEM Particle Segmentation using U-Net

A PyTorch implementation of a U-Net segmentation model for detecting TiO₂ particles in Scanning Electron Microscopy (SEM) images.

The project demonstrates semantic segmentation of particle regions from grayscale SEM images and provides a foundation for downstream particle size measurements and oversized particle detection.

---

## Dataset

The model was trained on the NanoSEM-464 dataset.

Dataset characteristics:

- Grayscale SEM images
- Resolution: 768 × 1024 pixels
- Binary particle masks
- TiO₂ particle samples
- Pixel-wise segmentation task

---

## Model

### Architecture

- U-Net
- ResNet34 encoder
- ImageNet pretrained encoder weights
- Single-channel input (grayscale)
- Single-channel binary mask output

### Training

- Framework: PyTorch
- Optimizer: AdamW
- Learning Rate: 1e-4
- Batch Size: 16
- Epochs: 100

### Loss Function

The model was trained using a combination of Dice Loss and Binary Cross Entropy:

```python
loss = DiceLoss + BCEWithLogitsLoss
```

This combines region overlap optimization with stable pixel-wise classification.

---

## Results

### Validation Metrics

| Metric | Score |
|----------|----------|
| Dice Score | 0.96 |
| IoU Score | 0.92 |

## Training History

The model converged steadily over 100 epochs with decreasing training and validation loss.

![Training Curve](output/loss_history.pdf)

---

## Example Prediction

Example segmentation result on a validation image.

![Prediction Example](output/Results_preview.pdf)

From left to right:

1. SEM image
2. Ground truth mask
3. Predicted mask

