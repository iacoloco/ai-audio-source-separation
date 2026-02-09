#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 18:17:31 2026
"dataset_musdb.py"
@author: armandoiachini
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import musdb
import random

from dsp_functions import stft_Complex, magnitude, IRM


class Dataset_Musdb_Frames_Pairs(Dataset):

    def __init__(self):
        super().__init__()

        self.root = "/Users/armandoiachini/Documents/Uni/AIforAudio/assesstment2/Dataset"
        self.subset = "train"
        self.target = "drums"

        self.db = musdb.DB(root=self.root, subsets=self.subset)
        self.tracks = list(self.db.tracks)
        self.frames_per_track = 20

    def __len__(self):
        return len(self.tracks) * self.frames_per_track

    def __getitem__(self, idx):

        track_idx = idx // self.frames_per_track
        track_idx = min(track_idx, len(self.tracks) - 1)
        track = self.tracks[track_idx]

        # Get mix and drum
        mix = track.audio
        drum = track.targets[self.target].audio

        # Convert to float32
        mix = mix.astype(np.float32)
        drum = drum.astype(np.float32)

        # To mono (Get left channel)
        mix = mix[:, 0]
        drum = drum[:, 0]
        
        # --- pick random chunk (FASTER than full-track STFT) ---
        sr = 44100
        chunk_seconds = 6
        chunk_len = chunk_seconds * sr

        if len(mix) > chunk_len:
            start = random.randint(0, len(mix) - chunk_len)
            mix = mix[start:start + chunk_len]
            drum = drum[start:start + chunk_len]


        # Normalise using MIX scale 
        scale = np.max(np.abs(mix)) + 1e-8
        mix = mix / scale
        drum = drum / scale

        # To tensor
        mix = torch.from_numpy(mix)
        drum = torch.from_numpy(drum)

        # STFT
        mix_stft = stft_Complex(mix)
        drum_stft = stft_Complex(drum)

        # Magnitude
        mix_mag = magnitude(mix_stft)
        drum_mag = magnitude(drum_stft)

        # IRM (Target)
        irm_mask = IRM(drum_mag, mix_mag)

        # choose a random frame every time (better variety)
        frame_t = mix_mag.shape[1]
        random_frame = random.randint(0, frame_t - 1)

        # INPUT: log-magnitude (much easier to learn)
        x = mix_mag[:, random_frame])
        y = irm_mask[:, random_frame]

        return x, y