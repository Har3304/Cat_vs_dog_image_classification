import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import transforms
from PIL import Image
import os


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            flatten_dim = self._forward_features(dummy).view(1, -1).size(1)
        print(f"Auto-detected flatten dimension: {flatten_dim}")

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



dataset = datasets.ImageFolder("/content/drive/MyDrive/dataset/cat_dog_dataset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_classes_for_model_loading = 2
model = CNNModel(num_classes_for_model_loading)
model.load_state_dict(torch.load(
    "/content/drive/MyDrive/dataset/cnn_model.pth",
    map_location=DEVICE
))
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def classify_image(image_path, class_names):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return class_names[pred.item()], conf.item()
test_folder = "/content/drive/MyDrive/dataset/testing_images"
class_names = dataset.classes

for file in os.listdir(test_folder):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(test_folder, file)
    label, confidence = classify_image(path, class_names)
    print(f"{file} → {label} ({confidence:.4f})")
