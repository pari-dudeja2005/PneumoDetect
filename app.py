import io
import torch
import streamlit as st
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet18_pneumonia.pth"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="PneumoDetect",
    page_icon="P",
    layout="centered"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION
# ============================================================

def predict(image, model, device):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    classes = [
        "Normal",
        "Pneumonia"
    ]

    prediction = classes[predicted.item()]

    confidence = confidence.item() * 100

    return prediction, confidence


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, device = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# MAIN UI
# ============================================================

st.title("PneumoDetect")

st.write(
    "AI-based chest X-ray classification for "
    "Normal and Pneumonia detection."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        io.BytesIO(
            uploaded_file.read()
        )
    )

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width="stretch"
    )

    st.divider()

    if st.button(
        "Analyze X-ray",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Analyzing X-ray..."
        ):

            prediction, confidence = predict(
                image,
                model,
                device
            )

        st.subheader("Result")

        if prediction == "Pneumonia":

            st.error(
                f"Pneumonia detected"
            )

        else:

            st.success(
                f"Normal"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "This application is intended for educational "
            "and research purposes and is not a medical diagnosis."
        )

else:

    st.info(
        "Upload a chest X-ray image to begin."
    )

