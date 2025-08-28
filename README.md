
**This repository contains the implementation and resources for the paper:**

**“MFCCResNet: A Deep Squeeze Excitation Residual Network with MFCC Features for MP3 Steganography Detection”**

**Overview**
Steganography in MP3 audio is a popular method for covert communication, often exploited in cybercrime and stego-malware. Detecting such hidden content is a critical forensic audio analysis task.

We propose MFCCResNet, a deep residual network enhanced with squeeze-and-excitation (SE) blocks trained on Mel-Frequency Cepstral Coefficients (MFCCs) to detect steganographic artifacts in MP3 audio.

**Key Features:**

1. MFCC-based compact, perceptually relevant representation.

2. Residual connections for robust deep learning without vanishing gradients.

3. Attention mechanism for temporal feature reweighting.

4. Robust under re-encoding and compression artifacts.

5. Achieves 98.38% detection accuracy at 128 kbps with 5% embedding rate.

**Dataset**
1. We used a balanced dataset of 20,800 MP3 audio files (10,400 cover + 10,400 stego) -> Sample data are uploaded here

2. HMMSec2024 dataset (benchmark stego MP3 audio). **Link:** https://cloud.ovgu.de/s/pYGzb8EzeM2ZMWr

3. In-house dataset (converted from Zenodo acoustic scene dataset (https://github.com/stella-ap/MFCCResNet) + stego embedded with MP3Stego(https://github.com/stella-ap/MFCCResNet)).

**Guidelines for dataset preparation are included in the repository.**

**Methodology**
_1. Feature Extraction (MFCC)_
Pre-emphasis filtering

Windowing (256, 512, 1024 sample windows)

FFT + Mel filter banks

Log compression

Discrete Cosine Transform → MFCC coefficients

_2. Deep Learning Model (MFCCResNet)_
Input: 2D MFCC feature maps

Initial convolution + pooling

Residual blocks with SE mechanism

Attention mechanism

Global average pooling + dense layers

Output: Binary classification (Cover vs Stego)

** Authors**
1. Stella J
2. Karthikeyan P
