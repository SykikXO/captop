import torch
import torch.nn as nn
import os
import sys

# Add the project root to sys.path so 'scripts' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import CaptchaModel, VOCAB_SIZE

def quantize_to_int8(model_path, save_path):
    print(f"Loading model from {model_path}...")
    model = CaptchaModel(VOCAB_SIZE)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Step 1: Prep for quantization
    # Note: Dynamic quantization is easiest for GRU/RNN based models
    print("Applying dynamic quantization (int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.GRU}, 
        dtype=torch.qint8
    )

    # Step 2: Save
    torch.save(quantized_model.state_dict(), save_path)
    
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Reduction: {(1 - quant_size/orig_size)*100:.1f}%")
    print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    MODEL_IN = "models/captcha_model.pth"
    MODEL_OUT = "models/captcha_model_int8.pth"
    
    if os.path.exists(MODEL_IN):
        quantize_to_int8(MODEL_IN, MODEL_OUT)
    else:
        print(f"Error: Could not find {MODEL_IN}")
