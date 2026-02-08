# Using the Captcha Model (Pseudo-code)

To use the trained model for inference in your own Python script, follow these steps:

### 1. Define the Architecture
You must use the exact `CaptchaModel` class defined in `scripts/train_model.py`. The weights are tied to this specific layer structure.

### 2. Implementation Guide

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
image = (image - np.mean(image)) / (np.std(image) + 1e-5)

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

### 3. Requirements
- `torch`
- `numpy`
- `opencv-python`
