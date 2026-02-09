#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:57:22 2026
@author: armandoiachini
"""

import torch
from model import MLP_Class
from torch import nn
from torch.utils.data import DataLoader
from dataset_musdb import Dataset_Musdb_Frames_Pairs
import matplotlib.pyplot as plt
from torch.optim import Adam

dataset = Dataset_Musdb_Frames_Pairs()
batch_size = 8

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 40
loss_curve = []

model = MLP_Class(1025)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()

        y_hat = model(x_batch)

        error = loss_fn(y_hat, y_batch)
        error.backward()
        optimizer.step()

        total_loss += error.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print("Total Loss Average:", avg_loss)
    loss_curve.append(avg_loss)

    torch.save(model.state_dict(), "mlp_drum_mask_latest_musdb.pth")

plt.plot(loss_curve)
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("Training Loss Curve")
plt.show()

torch.save(model.state_dict(), "mlp_drum_mask_50frames.pth")
print("Saved model weights to mlp_drum_mask_final.pth")