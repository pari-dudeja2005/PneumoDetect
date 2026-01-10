from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet18_pneumonia.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ["Normal", "Pneumonia"]

@app.get("/")
def root():
    return {"message": "Explainable Medical AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

