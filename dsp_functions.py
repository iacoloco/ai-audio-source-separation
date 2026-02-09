#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:14:59 2026

@author: armandoiachini
"""
"dsp_function.py"
"TRAINING"

import torch

#----------------------------------------------------------
# Functions
#----------------------------------------------------------

def stft_Complex(y, n_fft=2048, hop=512, center=True):
    # window on same device + dtype as input y
    window = torch.hann_window(n_fft, device=y.device, dtype=y.dtype)
    X = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=center
    )
    return X

def magnitude(X):
    return torch.abs(X)

def IRM(D_mag, M_mag, eps=1e-8):
    mask = D_mag / (M_mag + eps)
    return torch.clamp(mask, 0.0, 1.0)

def istft_Complex(X, n_fft=2048, hop=512, length=None, center=True):
    # X is complex; output is real float tensor
    window = torch.hann_window(n_fft, device=X.device, dtype=torch.float32)
    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        length=length,
        center=center
    )
    return y


