# Using the Captcha Model

There are two primary ways to consume the CAPTCHA model depending on your use-case: the **Production Rust API** (using the ONNX export) and the **PyTorch Pipeline** (for training/evaluation scripts).

## 1. Production Rust API (Recommended)

The `api/` directory contains a high-performance Rust server designed for production deployment. It serves the exported `.onnx` model (`api/model/captcha_model.onnx`).

### Usage
- The Rust server handles image preprocessing natively and performs inference via the ONNX runtime.
- For serving the production API, build and run the Rust project in `api/`. (The crowdsourcing backend in `crowdsource/` was used separately for data collection).

---

## 2. Python/PyTorch Pipeline (For Training & Dev)

If you need to use the trained PyTorch model for inference or evaluation in your own Python script, follow these steps:

### 2.1 Define the Architecture
You must use the exact `CaptchaModel` class defined in `scripts/train.py`. The weights are tied to this specific layer structure.

### 2.2 Implementation Guide

```python
import torch
import cv2
import numpy as np

# 1. Load the architecture
model = CaptchaModel(vocab_size=37) # 36 chars + 1 blank

# 2. Load the weights (CPU or GPU)
device = torch.device("cpu")
model.load_state_dict(torch.load("models/captcha_model.pth", map_location=device))
model.eval()

# 3. Preprocess your image
# Image must be (40, 200) grayscale
image = cv2.imread("path_to_captcha.jpg", cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

# IMPORTANT: Use the same normalization used in training
image = (image / 255.0 - 0.5) / 0.5

# Add batch and channel dimensions: [1, 1, 40, 200]
input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)

# 4. Predict
with torch.no_grad():
    outputs = model(input_tensor) # Shape: [T, B, C]
    
# 5. Decode CTC Output
# Use greedy decoding:
# - Get argmax of dimensions
# - Remove repeated characters
# - Remove 'blank' tokens (index 0)
# - Map indices back to your character string
prediction = my_decode_function(outputs)
print(prediction.upper())
```

### 2.3 Requirements
- `torch`
- `numpy`
- `opencv-python`

---

## Model Specifications

Based on the implementation in `scripts/train.py`, here are the detailed technical specifications of the trained CAPTCHA model:

### 1. Overall Architecture
The model is a **CRNN (Convolutional Recurrent Neural Network)** built using PyTorch. It uses a CNN backbone for feature extraction from the image and a Bidirectional GRU (RNN) for sequence prediction, optimized using **CTC (Connectionist Temporal Classification)** loss.

### 2. Input Specifications
*   **Format:** Grayscale images (1 channel)
*   **Dimensions:** 200 pixels (width) × 40 pixels (height)
*   **Preprocessing:** Each image is normalized from 0-255 down to -1 to 1 using the linear transformation `(image / 255.0 - 0.5) / 0.5`.

### 3. CNN Feature Extractor (Vision)
The CNN reduces the image down to a sequence of feature maps. It consists of 4 main convolutional blocks:
*   **Block 1:** `Conv2d` (1 -> 64 channels, 3x3 kernel) -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2)`. (Image becomes 20x100).
*   **Block 2:** `Conv2d` (64 -> 128 channels, 3x3 kernel) -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2)`. (Image becomes 10x50).
*   **Block 3:** Two `Conv2d` layers (128 -> 256 -> 256 channels, 3x3 kernel) each followed by `BatchNorm2d` and `ReLU`. Then `MaxPool2d((2, 1))`. (Image becomes 5x50. *Notice it only downsamples height here*).
*   **Block 4:** `Conv2d` (256 -> 512 channels, 3x3 kernel) -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d((2, 1))`. (Image roughly becomes 2x50).

The output of the CNN is a feature map of shape `[Batch, 512 channels, 2 height, 50 sequence length]`. This is flattened along the height/channels into `[Batch, 50, 1024]` to feed into the RNN.

### 4. Sequence Modeling (RNN)
*   **Type:** `Bidirectional GRU`
*   **Input Size:** 1024
*   **Hidden Size:** 256 (Outputs 512 because it's bidirectional)
*   **Layers:** 2 layers
*   **Dropout:** 0.3
*   **Time Steps (Sequence Length):** 50 (The model makes 50 character predictions per image).

### 5. Output Layer & Vocabulary
*   **Fully Connected Layer:** `Linear(512, 37)` maps the GRU outputs to character probabilities.
*   **Vocabulary Size:** 37 tokens.
    *   36 alphanumeric characters: `0-9` and `a-z` (all characters are lowercased during training).
    *   1 CTC "Blank" token (at index 0).

### 6. Training Hyperparameters
*   **Loss Function:** `nn.CTCLoss(blank=0, zero_infinity=True)`
*   **Optimizer:** `Adam`
*   **Learning Rate (LR):** `0.001` (1e-3)
*   **Batch Size:** 16
*   **Epochs:** 50
*   **Validation Split:** 20% of the dataset (`test_size=0.2` with a fixed random seed of 42).
