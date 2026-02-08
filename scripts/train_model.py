import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# --- Configuration ---
DATA_DIR = "data/captchas"
BATCH_SIZE = 16
IMG_WIDTH = 200
IMG_HEIGHT = 40
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character set (based on research findings)
# Training had ABCDEFGHIJKLMNPQRSTUVWXYZ and 123456789
# Test had 0123456789 and abcdefghijklmnopqrstuvwxyz
# We normalize everything to lowercase for consistency.
CHARS = sorted(list("0123456789abcdefghijklmnopqrstuvwxyz"))
VOCAB_SIZE = len(CHARS) + 1  # +1 for CTC blank token

char_to_num = {char: i + 1 for i, char in enumerate(CHARS)}
num_to_char = {i + 1: char for i, char in enumerate(CHARS)}

def encode_label(label):
    return [char_to_num[c.lower()] for c in label if c.lower() in char_to_num]

# --- Dataset ---
class CaptchaDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        
        # [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        
        # Extract label from filename (stripping _1 etc for duplicates)
        label_str = os.path.basename(img_path).split(".")[0].split("_")[0]
        label = torch.tensor(encode_label(label_str), dtype=torch.long)
        label_len = torch.tensor(len(label), dtype=torch.long)
        
        return torch.tensor(image), label, label_len

def collate_fn(batch):
    images, labels, label_lens = zip(*batch)
    images = torch.stack(images)
    
    # Pad labels for CTC
    max_label_len = max(label_lens)
    padded_labels = torch.zeros((len(labels), max_label_len), dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
        
    return images, padded_labels, torch.stack(label_lens)

# --- Model Architecture ---
class CaptchaModel(nn.Module):
    def __init__(self, vocab_size):
        super(CaptchaModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 40x200 -> 20x100
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 20x100 -> 10x50
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # 10x50 -> 5x50
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # 5x50 -> 2x50 (approx, depending on exact padding/stride)
        )
        
        # Calculate RNN input size: 512 * H_final (which is 2)
        self.rnn = nn.GRU(1024, 256, bidirectional=True, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        # x: [B, 1, 40, 200]
        x = self.cnn(x)
        # B, 512, 2, 50
        x = x.permute(0, 3, 1, 2) # [B, 50, 512, 2]
        x = x.reshape(x.size(0), x.size(1), -1) # [B, 50, 1024]
        
        x, _ = self.rnn(x) # [B, 50, 512]
        x = self.fc(x) # [B, 50, vocab_size]
        
        return x.permute(1, 0, 2) # [50, B, vocab_size]

# --- Training / Validation ---
def train():
    image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    
    train_ds = CaptchaDataset(train_paths)
    val_ds = CaptchaDataset(val_paths)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = CaptchaModel(VOCAB_SIZE).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter('runs/captcha_experiment')
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lens in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # log_probs: [T, B, C]
            # targets: [B, S]
            # input_lengths: [B]
            # target_lengths: [B]
            input_lengths = torch.full(size=(images.size(0),), fill_value=50, dtype=torch.long).to(DEVICE)
            
            # CTC loss expects targets to be concatenated if they have different lengths
            # but we padded them and passed lengths.
            loss = criterion(outputs.log_softmax(2), labels.to(DEVICE), input_lengths, label_lens.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels, label_lens in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                input_lengths = torch.full(size=(images.size(0),), fill_value=50, dtype=torch.long).to(DEVICE)
                loss = criterion(outputs.log_softmax(2), labels.to(DEVICE), input_lengths, label_lens.to(DEVICE))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/captcha_model.pth")
            print("Model saved.")
            
    writer.close()

if __name__ == "__main__":
    train()
