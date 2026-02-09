#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:31:41 2026

@author: armandoiachini
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 16:57:50 2026

@author: armandoiachini
"""

"Analisis MUSDB"

import musdb
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd

from dsp_functions import stft_Complex, magnitude, IRM , istft_Complex

ROOT = "/Users/armandoiachini/Documents/Uni/AIforAudio/assesstment2/Dataset"

db_train = musdb.DB(root=ROOT, subsets="train")
db_test = musdb.DB(root=ROOT, subsets="test")

print("Training tracks: " , len(db_train.tracks))
print("Test tracks", len(db_test.tracks))

test=db_test.tracks[0]

print("Example test track: " , test.name)
print("Mix shape:  " , test.audio.shape)
print("Target:", list(test.targets.keys()))
print("type(t.audio):", type(test.audio))
print("t.audio dtype:", test.audio.dtype)
print("t.audio shape:", test.audio.shape)

#------------------------------------------------------
#Source Separation test s_hat = IRM * MIX

mix = test
drum_audio = test.targets["drums"].audio

print("Track Name: " , mix.name)

#get audio
mix_audio = mix.audio

#Converto to float 32
mix_audio = mix_audio.astype(np.float32)
drum_audio = drum_audio.astype(np.float32)
#Converto to Mono
mix_audio = mix_audio[:,0]
drum_audio = drum_audio[:,0]

print("Mix type after convertion to Float32 and Mono: ", mix_audio.dtype, mix_audio.shape)

#Convert from numpy top Tensors (Uses GPU)
mix_tensor = torch.from_numpy(mix_audio)
drum_tensor = torch.from_numpy(drum_audio)

#stft
X_mix = stft_Complex(mix_tensor)
X_drum = stft_Complex(drum_tensor)

#Take the magnitude
Mix_mag = magnitude(X_mix)
Drum_mag = magnitude(X_drum)
print("Mix mag shape", Mix_mag.shape)

#get the IRM from the model 

#IRM
irm = IRM(Drum_mag, Mix_mag)

#Apply mask
S_hat_drums = irm * X_mix

#ISTFT ->
S_drums = istft_Complex(S_hat_drums , length=mix_tensor.shape[0])

#Save

#sf.write("test_drums_separation.wav", S_drums.numpy(), 44100)
sd.play(S_drums, 44100)
sd.wait()

print("S_Drums ", S_drums.dtype, S_drums.shape)
print("S_hat_Drums ", S_hat_drums.dtype,S_hat_drums.shape)





