import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

# --- Configuration ---
MODEL_PATH = "models/captcha_model.pth"
TEST_DIR = "data/test"
IMG_WIDTH = 200
IMG_HEIGHT = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHARS = sorted(list("0123456789abcdefghijklmnopqrstuvwxyz"))
VOCAB_SIZE = len(CHARS) + 1
num_to_char = {i + 1: char for i, char in enumerate(CHARS)}

# --- Model Architecture (Must match training script) ---
class CaptchaModel(nn.Module):
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

def decode_predictions(outputs):
    # Greedy decoding for CTC
    outputs = outputs.permute(1, 0, 2) # [B, T, C]
    outputs = torch.softmax(outputs, dim=2)
    predictions = torch.argmax(outputs, dim=2)
    
    decoded_preds = []
    for i in range(predictions.size(0)):
        pred = predictions[i]
        char_list = []
        for j in range(len(pred)):
            char_idx = pred[j].item()
            if char_idx != 0: # blank
                if j > 0 and char_idx == pred[j-1].item():
                    continue
                char_list.append(num_to_char[char_idx])
        decoded_preds.append("".join(char_list))
    return decoded_preds

def predict():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = CaptchaModel(VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    print(f"Found {len(test_images)} test images. Decoding...")

    results = []
    with torch.no_grad():
        for img_path in test_images:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32)
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)
            image = np.expand_dims(image, axis=0) # [1, H, W]
            image = np.expand_dims(image, axis=0) # [1, 1, H, W]
            
            image_tensor = torch.tensor(image).to(DEVICE)
            outputs = model(image_tensor)
            prediction = decode_predictions(outputs)[0]
            
            basename = os.path.basename(img_path)
            results.append(f"{basename}: {prediction.upper()}")
            print(f"{basename}: {prediction}")

    with open("data/test_results.txt", "w") as f:
        f.write("\n".join(results))
    print(f"Results saved to data/test_results.txt")

if __name__ == "__main__":
    predict()
