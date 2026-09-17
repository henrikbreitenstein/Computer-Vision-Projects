import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import config
import data.data as data
import models.unet as unet

from segmentation_models_pytorch.losses import DiceLoss


dice_loss = DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

model = unet.get_model()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4
)

device = torch.device('cuda')
model.to(device)

train_losses = []
val_losses = []

def eval():

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for images,masks in data.val_loader:

            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)

            loss = loss_fn(pred, masks)
            running_loss += loss.item()

    return running_loss/ len(data.val_loader)

def train():

    for epoch in range(config.EPOCHS):

        model.train()

        running_loss = 0

        for images, masks in data.train_loader:
            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)

            loss = loss_fn(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss)
        val_losses.append(eval())
        print(epoch, running_loss)

def save():

    torch.save(
        model.state_dict(),
        'output/unet_particles.pth'
    )

    df = pd.DataFrame({
        'train_losses' : train_losses,
        'val_losses' : val_losses
    })
    df.to_csv('output/loss_history.csv', index=False)

def load():
    model.load_state_dict(
        torch.load('output/unet_particles.pth')
    )

if __name__ == "__main__":
    train()
    save()
    load()
    print(eval())

