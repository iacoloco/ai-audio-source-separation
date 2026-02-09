#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:39:46 2026
@author: armandoiachini
"""

import musdb
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd

from model import MLP_Class
from dsp_functions import stft_Complex, magnitude, istft_Complex, IRM

ROOT = "/Users/armandoiachini/Documents/Uni/AIforAudio/assesstment2/Dataset"

# ------------------------------------------------------
# Load test track 
# ------------------------------------------------------
db_test = musdb.DB(root=ROOT, subsets="test")
test = db_test.tracks[0]

print("Example test track:", test.name)
print("Target:", list(test.targets.keys()))
print("t.audio dtype:", test.audio.dtype)
print("t.audio shape:", test.audio.shape)

mix = test
drum_audio = test.targets["drums"].audio
print("Track Name:", mix.name)

# ------------------------------------------------------
# Get audio (numpy) -> float32 -> mono
# ------------------------------------------------------
mix_audio = mix.audio.astype(np.float32)
drum_audio = drum_audio.astype(np.float32)

mix_audio = mix_audio[:, 0]
drum_audio = drum_audio[:, 0]

# ------------------------------------------------------
# Normalise using MIX scale (same factor for mix and drum)
# ------------------------------------------------------
scale = np.max(np.abs(mix_audio)) + 1e-8
mix_audio = mix_audio / scale
drum_audio = drum_audio / scale

print("Mix after convertion:", mix_audio.dtype, mix_audio.shape)

# ------------------------------------------------------
# Convert to torch tensors
# ------------------------------------------------------
mix_tensor = torch.from_numpy(mix_audio)
drum_tensor = torch.from_numpy(drum_audio)

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
model = MLP_Class(1025)
#model.load_state_dict(torch.load("mlp_drum_mask_50frames.pth", map_location="cpu"))
model.load_state_dict(torch.load("mlp_drum_mask_50frames.pth", map_location="cpu"))
model.eval()

# ----------------------------------------------------------
# STFT + magnitude
# ----------------------------------------------------------
X_mix = stft_Complex(mix_tensor)
Mix_mag = magnitude(X_mix)

# ----------------------------------------------------------
# Predict mask_hat (same preprocessing as training)
# ----------------------------------------------------------
with torch.no_grad():
    mask_hat = model(torch.log1p(Mix_mag.T))
mask_hat = mask_hat.T

print("mask_hat min/max/mean:",
      mask_hat.min().item(),
      mask_hat.max().item(),
      mask_hat.mean().item())
# ----------------------------------------------------------
# Apply mask + iSTFT
# ----------------------------------------------------------

#S_hat_drums = mask_hat * X_mix
Mix_ang = torch.angle(X_mix)
Drums_mag_hat = mask_hat * Mix_mag
S_hat_drums = Drums_mag_hat * torch.exp(1j * Mix_ang)

S_drums = istft_Complex(S_hat_drums, length=mix_tensor.shape[0])

# ----------------------------------------------------------
# Save + play
# ----------------------------------------------------------
sf.write("test_drums_separation.wav", S_drums.numpy(), 44100)
print("Saved: test_drums_separation.wav")

#sd.play(S_drums.numpy(), 44100)
#sd.wait()

#sd.play(mix_audio, 44100)
#sd.wait()
sf.write("Mix_original.wav", mix_audio, 44100)
print("Saved: Mix_original.wav")

print("S_drums:", S_drums.dtype, S_drums.shape)
print("S_hat_drums:", S_hat_drums.dtype, S_hat_drums.shape)


