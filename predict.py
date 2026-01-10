import torch
from torchvision import transforms
from PIL import Image
from train_model import resnet18, ResNet18_Weights, GradCAM, ClassifierOutputTarget, show_cam_on_image
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("models/resnet18_pneumonia.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def predict_and_gradcam(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    print("Prediction:", "PNEUMONIA" if pred==1 else "NORMAL")

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    img_np = input_tensor.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    predict_and_gradcam("data/test/PNEUMONIA/person1_virus_1.jpeg")
