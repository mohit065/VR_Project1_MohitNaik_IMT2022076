import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

IMAGE_DIR = "../datasets/dataset2/1/face_crop"
MASK_DIR = "../datasets/dataset2/1/face_crop_segmentation"
MODEL_PATH = "./model.pth"
RESULTS_DIR = "./results"
IMG_HEIGHT, IMG_WIDTH = 128, 128
NUM_IMAGES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

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

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

def load_and_preprocess(image_path, mask_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size) / 255.0
    img = np.transpose(img, (2, 0, 1))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size)
    mask = np.expand_dims(mask, axis=0) / 255.0

    return img, mask

image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))
random_indices = np.random.choice(len(image_files), NUM_IMAGES, replace=False)
selected_images = [image_files[i] for i in random_indices]
selected_masks = [mask_files[i] for i in random_indices]
iou_scores, dice_scores = [], []

for img_file, mask_file in zip(selected_images, selected_masks):
    img_path = os.path.join(IMAGE_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, mask_file)
    img, mask = load_and_preprocess(img_path, mask_path)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = model(img_tensor).squeeze().cpu().numpy()
        
    pred_mask = (pred_mask > 0.5).astype(int)
    true_mask = (mask.squeeze() > 0.5).astype(int)
    iou_scores.append(jaccard_score(true_mask.flatten(), pred_mask.flatten()))
    dice_scores.append(f1_score(true_mask.flatten(), pred_mask.flatten()))

    if len(iou_scores) <= 5:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title("Input Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        result_path = os.path.join(RESULTS_DIR, f"result_{len(iou_scores)}.png")
        plt.savefig(result_path)
        plt.close()

print(f"Average IoU: {np.mean(iou_scores):.4f}")
print(f"Average Dice: {np.mean(dice_scores):.4f}")
print(f"Visualization results saved to {RESULTS_DIR}")