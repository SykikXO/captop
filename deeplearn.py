import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = "captchas"
IMG_WIDTH = 200
IMG_HEIGHT = 40
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_TO_INT = {char: i for i, char in enumerate(CHAR_SET)}
INT_TO_CHAR = {i: char for i, char in enumerate(CHAR_SET)}
NUM_CHARS = 6
VOCAB_SIZE = len(CHAR_SET)

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        # Process label
        label_chars = self.labels[idx]
        label_ints = [CHAR_TO_INT[c] for c in label_chars]
        
        return torch.from_numpy(img), torch.tensor(label_ints, dtype=torch.long)

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        # Conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 20x100
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 10x50
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 5x25
        )
        
        # Heads for each character
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 5 * 25, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 5 output layers for 5 characters
        self.heads = nn.ModuleList([
            nn.Linear(256, VOCAB_SIZE) for _ in range(NUM_CHARS)
        ])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Get output from each head
        outputs = [head(x) for head in self.heads]
        return outputs

def train_model():
    # Load data
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    image_paths = [os.path.join(DATA_DIR, f) for f in image_files]
    labels = [f.split(".")[0] for f in image_files] # Assumes filename is [label].jpg

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    train_dataset = CaptchaDataset(X_train, y_train)
    val_dataset = CaptchaDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Sum loss for all 5 characters
            loss = 0
            for i in range(NUM_CHARS):
                loss += criterion(outputs[i], targets[:, i])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total_chars = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                
                loss = 0
                for i in range(NUM_CHARS):
                    loss += criterion(outputs[i], targets[:, i])
                    
                    # Accuracy per character
                    _, predicted = torch.max(outputs[i], 1)
                    correct += (predicted == targets[:, i]).sum().item()
                    total_chars += targets.size(0)
                
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        acc = (correct / total_chars) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, Val Acc: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

def predict_captcha(img_path, model_path="model.pth"):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Image not found"
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0) # (1, 1, H, W)
    img_tensor = torch.from_numpy(img).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        prediction = ""
        for i in range(NUM_CHARS):
            _, predicted = torch.max(outputs[i], 1)
            prediction += INT_TO_CHAR[predicted.item()]
    
    return prediction

if __name__ == "__main__":
    train_model()
