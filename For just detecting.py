import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set device to CPU for inference
device = torch.device("cpu")

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# BiFPN with 7 layers for real-time inference
class BiFPNLayer(nn.Module):
    def __init__(self, input_channels, num_channels):
        super(BiFPNLayer, self).__init__()
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0)
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, P3, P4, P5):
        P3_in = self.conv3(P3)
        P4_in = self.conv4(P4)
        P5_in = self.conv5(P5)

        batch_size, channels, height, width = P5_in.shape
        P5_in_flattened = P5_in.view(batch_size, channels, -1).permute(2, 0, 1)
        P5_attn, _ = self.attention(P5_in_flattened, P5_in_flattened, P5_in_flattened)
        P5_attn = P5_attn.permute(1, 2, 0).view(batch_size, channels, height, width)

        P3_down = nn.functional.interpolate(P3_in, size=P4_in.shape[2:], mode='bilinear', align_corners=False)
        P5_up = nn.functional.interpolate(P5_attn, size=P4_in.shape[2:], mode='bilinear', align_corners=False)

        P4_td = P4_in + P5_up
        P4_td = self.dropout(P4_td)
        P4_up = nn.functional.interpolate(P4_td, size=P3_down.shape[2:], mode='bilinear', align_corners=False)

        P3_td = P3_down + P4_up
        P3_td = self.dropout(P3_td)

        return P3_td, P4_td, P5_attn

# Stacked BiFPN
class StackedBiFPN(nn.Module):
    def __init__(self, input_channels, num_channels, num_layers=7):
        super(StackedBiFPN, self).__init__()
        self.layers = nn.ModuleList([BiFPNLayer(input_channels, num_channels) for _ in range(num_layers)])

    def forward(self, P3, P4, P5):
        for layer in self.layers:
            P3, P4, P5 = layer(P3, P4, P5)
        return P3, P4, P5

# DetectionModel class
class DetectionModel(nn.Module):
    def __init__(self, num_classes, max_bboxes):
        super(DetectionModel, self).__init__()
        backbone = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.stage0 = nn.Sequential(*backbone.features[:2])
        self.stage1 = nn.Sequential(*backbone.features[2:4])
        self.stage2 = nn.Sequential(*backbone.features[4:6])
        self.stage3 = nn.Sequential(*backbone.features[6:8])

        self.adjust_channels3 = nn.Conv2d(32, 256, kernel_size=1, padding=0)
        self.adjust_channels4 = nn.Conv2d(80, 256, kernel_size=1, padding=0)
        self.adjust_channels5 = nn.Conv2d(224, 256, kernel_size=1, padding=0)

        # Use StackedBiFPN
        self.bifpn = StackedBiFPN(input_channels=256, num_channels=256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_head = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            Swish(),
            nn.Dropout(0.5)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(256, num_classes * max_bboxes),
            nn.Sigmoid()
        )

        self.regression_head = nn.Sequential(
            nn.Linear(256, max_bboxes * 4)
        )

    def forward(self, x):
        P3 = self.stage0(x)
        P4 = self.stage1(P3)
        P5 = self.stage2(P4)

        P3 = self.adjust_channels3(P3)
        P4 = self.adjust_channels4(P4)
        P5 = self.adjust_channels5(P5)

        P3, P4, P5 = self.bifpn(P3, P4, P5)

        features = self.pool(P3).view(x.size(0), -1)
        features = self.shared_head(features)
        classification = self.classification_head(features).view(-1, 20, 1)
        bbox_regression = self.regression_head(features).view(-1, 20, 4)
        return classification, bbox_regression

# Load the trained model
model = DetectionModel(num_classes=1, max_bboxes=20).to(device)
model.load_state_dict(torch.load(
    r"C:\Users\3s\Desktop\Custom Models For Object Detection Using Mobilenetv2 & Efficientnetb7/detection_model.pth",
    map_location=device,
    weights_only=True  # Added for secure loading
))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Real-time inference function
def real_time_inference(image_path, model):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)  # Convert PIL image to numpy array for plotting

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_cls, pred_reg = model(image)

    pred_cls = pred_cls.cpu().numpy()[0]
    pred_reg = pred_reg.cpu().numpy()[0]

    # Draw bounding boxes only for detected objects
    plt.imshow(original_image)
    ax = plt.gca()
    object_detected = False

    for i in range(len(pred_cls)):
        class_prob = pred_cls[i][0]
        if class_prob > 0.000001:  # Detection threshold
            object_detected = True
            x, y, w, h = pred_reg[i]
            x_center = int(x * original_image.shape[1])
            y_center = int(y * original_image.shape[0])
            box_width = int(w * original_image.shape[1])
            box_height = int(h * original_image.shape[0])

            x1 = max(0, x_center - box_width // 2)
            y1 = max(0, y_center - box_height // 2)
            x2 = min(original_image.shape[1], x_center + box_width // 2)
            y2 = min(original_image.shape[0], y_center + box_height // 2)

            # Draw the bounding box
            rect = plt.Rectangle((x1, y1), box_width, box_height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"Prob: {class_prob:.2f}", color='red', fontsize=8)

    if object_detected:
        plt.axis('off')
        plt.show()
    else:
        print("No objects detected with confidence above the threshold.")

# Example usage
image_path = r"C:\Users\3s\Desktop\WWW\2.jpg"
real_time_inference(image_path, model)
