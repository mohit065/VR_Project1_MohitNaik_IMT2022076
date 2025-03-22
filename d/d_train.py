import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 16
EPOCHS = 20
IMAGE_DIR = "../datasets/dataset2/1/face_crop"
MASK_DIR = "../datasets/dataset2/1/face_crop_segmentation"
MODEL_PATH = "..d/d_model.pth"

# Load images and masks
def load_data(image_dir, mask_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))  # Ensure correct order
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files[:5000], mask_files[:5000]):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # Convert to channel-first

        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=0) / 255.0  # Add channel dimension and normalize

        images.append(img)
        masks.append(mask)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# Load dataset
X, Y = load_data(IMAGE_DIR, MASK_DIR)
print(f"Dataset loaded: {X.shape}, {Y.shape}")

# Split dataset (80% Train, 20% Validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.masks[idx], dtype=torch.float32)

# Create DataLoaders
train_dataset = SegmentationDataset(X_train, Y_train)
val_dataset = SegmentationDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.ModuleList([
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        ])
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.upconv)):
            x = self.upconv[i](x)
            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.decoder[i](x)
        
        return torch.sigmoid(self.final(x))

# Initialize model, loss function, and optimizer
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    iou_scores, dice_scores = [], []
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        y_true = (masks.cpu().numpy().flatten() > 0.5).astype(int)
        y_pred = (outputs.cpu().detach().numpy().flatten() > 0.5).astype(int)

        iou_scores.append(jaccard_score(y_true, y_pred))
        dice_scores.append(f1_score(y_true, y_pred))

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}, Mean IoU: {np.mean(iou_scores):.4f}, Mean Dice: {np.mean(dice_scores):.4f}")

# Evaluate on validation set after training
model.eval()
val_iou_scores, val_dice_scores = [], []

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        y_true = (masks.cpu().numpy().flatten() > 0.5).astype(int)
        y_pred = (outputs.cpu().numpy().flatten() > 0.5).astype(int)

        val_iou_scores.append(jaccard_score(y_true, y_pred))
        val_dice_scores.append(f1_score(y_true, y_pred))

print(f"Final Validation IoU: {np.mean(val_iou_scores):.4f}, Final Validation Dice: {np.mean(val_dice_scores):.4f}")

# Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")