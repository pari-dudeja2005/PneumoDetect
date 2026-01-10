import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import ResNet18_Weights

BATCH_SIZE = 16
IMAGE_SIZE = 224
MODEL_PATH = "models/resnet18_pneumonia.pth"
DATA_DIR = "data/val"
CLASSES = ["Normal", "Pneumonia"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

