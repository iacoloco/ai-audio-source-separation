# Music Source Separation (Drums) — IRM + MLP (PyTorch)

Course project for **Audio Software Engineering** (School of Computing and Engineering, University of West London), January 2026.

## Abstract
The goal of this project was to train a neural network **Multi-Layer Perceptron (MLP)** model using the **Ideal Ratio Mask (IRM)** as a target, in order to estimate a time–frequency mask and extract the **drum source** from a mixture track.

The **MUSDB18** dataset was used (not included in this repository). In this project, 144 tracks of 7 seconds each from different genres were used, with isolated stems for drums, bass, vocals, and other instruments. MUSDB18 is officially split into 100 training and 50 test full-length tracks.

The final implementation did not produce the expected separation results, even though the analysis pipeline (**STFT → IRM → masking → iSTFT**) was computed correctly. The loss curve showed a learning trend during training, however the model could not predict a mask that produces a good drum separation. Different numbers of epochs were tested without success.

## Method (pipeline)
STFT → IRM target → MLP predicts mask → apply mask → iSTFT reconstruction

## Files
- `train.py` — training loop
- `test.py` — inference / evaluation
- `model.py` — MLP model
- `dsp_functions.py` — STFT/iSTFT + masking utilities
- `dataset_musdb.py` — dataset loading / batching
- `analysis.py` — plots / analysis utilities

## Dataset
This repo does **not** include MUSDB18 audio.
You need to download MUSDB18 separately and set the dataset path inside the scripts.

## Run
```bash
pip install -r requirements.txt
python train.py
python test.py
