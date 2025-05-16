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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

base_dir = "E:/Test2"
dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = os.path.join(base_dir, "tiny-imagenet-200.zip")
extract_path = os.path.join(base_dir, "tiny-imagenet-200")
log_path = os.path.join(base_dir, "training_log.txt")
checkpoint_path = os.path.join(base_dir, "model_checkpoint.pth")

if not os.path.exists(extract_path):
    print("Downloading Tiny ImageNet dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    print("Done.")
else:
    print("Dataset already exists.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = os.path.join(extract_path, 'train')
val_dir = os.path.join(extract_path, 'val')

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

def parse_val_folder(val_dir):
    images = []
    labels = []
    label_map = {}
    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            img_file, label = parts[0], parts[1]
            label_map.setdefault(label, len(label_map))
            img_path = os.path.join(val_dir, 'images', img_file)
            if os.path.isfile(img_path):
                images.append(img_path)
                labels.append(label_map[label])
    return images, labels, label_map

val_images, val_labels, label_map = parse_val_folder(val_dir)

class TinyImageNetValDataset(Dataset):
    def __init__(self, images, labels, transform=None):
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

val_dataset = TinyImageNetValDataset(val_images, val_labels, transform_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, 200)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

start_epoch = 0
epochs = 250

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

if start_epoch == 0:
    with open(log_path, "w") as f:
        f.write("Epoch\tTrain Loss\tVal Accuracy (%)\tTime (s)\n")

for epoch in range(start_epoch, epochs):
    start_time = time.time()
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    epoch_time = time.time() - start_time
    print(f"Validation Accuracy: {val_acc:.2f}% - Epoch Time: {epoch_time:.2f} seconds")

    with open(log_path, "a") as log_file:
        log_file.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{val_acc:.2f}\t{epoch_time:.2f}\n")

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, checkpoint_path)
