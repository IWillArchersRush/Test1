import os
import zipfile
import urllib.request
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(conv_1x1_bn(inp, hidden_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, c, stride, t))
                input_channel = c
        features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

base_dir = "E:/T3"
zip_path = os.path.join(base_dir, "tiny-imagenet-200.zip")
extract_path = os.path.join(base_dir, "tiny-imagenet-200")
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

if not os.path.exists(extract_path):
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dir = os.path.join(extract_path, "train")
val_dir = os.path.join(extract_path, "val")
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

def parse_val_folder(val_dir, class_to_idx):
    images, labels = [], []
    with open(os.path.join(val_dir, "val_annotations.txt")) as f:
        for line in f:
            filename, class_id = line.split("\t")[:2]
            if class_id in class_to_idx:
                label = class_to_idx[class_id]
                images.append(os.path.join(val_dir, "images", filename))
                labels.append(label)
    return images, labels

val_images, val_labels = parse_val_folder(val_dir, train_dataset.class_to_idx)

class TinyValDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = default_loader(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

val_dataset = TinyValDataset(val_images, val_labels, transform_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

model = MobileNetV2().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

checkpoint_path = os.path.join(base_dir, "model_checkpoint.pth")
best_model_path = os.path.join(base_dir, "best_model.pth")
log_path = os.path.join(base_dir, "training_log.txt")
start_epoch = 0
best_acc = 0.0
epochs = 250

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"Resuming from epoch {start_epoch} | Best acc: {best_acc:.2f}%")

if start_epoch == 0:
    with open(log_path, "w") as f:
        f.write("Epoch\tTrain Loss\tVal Accuracy (%)\tTime (s)\n")

for epoch in range(start_epoch, epochs):
    start_time = time.time()
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    avg_loss = train_loss / len(train_loader)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = 100 * correct / total
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% - Time: {epoch_time:.1f}s")

    with open(log_path, "a") as f:
        f.write(f"{epoch+1}\t{avg_loss:.4f}\t{acc:.2f}\t{epoch_time:.1f}\n")

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc
    }, checkpoint_path)

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), best_model_path)
        print(f" New best model saved at epoch {epoch+1} with acc {acc:.2f}%")
