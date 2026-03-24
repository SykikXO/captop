#!/usr/bin/env python3
"""Export PyTorch captcha model to ONNX format."""

import torch
import torch.nn as nn

# Character set (must match training)
CHARS = sorted(list("0123456789abcdefghijklmnopqrstuvwxyz"))
VOCAB_SIZE = len(CHARS) + 1  # +1 for CTC blank token


class CaptchaModel(nn.Module):
    """CNN + BiGRU model for captcha recognition."""
    
    def __init__(self, vocab_size):
        super(CaptchaModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.GRU(1024, 256, bidirectional=True, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)


def main():
    import os
    model_path = "models/captcha_model.pth"
    onnx_path = "api/model/captcha_model.onnx"
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"Loading PyTorch model from {model_path}...")
    model = CaptchaModel(VOCAB_SIZE)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Fix RNN buffer copy warnings
    model.rnn.flatten_parameters()
    
    # Create dummy input: [batch, channels, height, width] = [1, 1, 40, 200]
    dummy_input = torch.randn(1, 1, 40, 200)
    
    print(f"Exporting to ONNX format (opset 20): {onnx_path}")
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["image"],
            output_names=["output"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "output": {1: "batch_size"}
            },
            opset_version=18,
            do_constant_folding=True,
        )
    
    print(f"✓ Successfully exported to {onnx_path}")
    
    # Verify the export
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")


if __name__ == "__main__":
    main()
