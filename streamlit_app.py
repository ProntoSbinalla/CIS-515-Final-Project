import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(page_title="Aging vs Procedure Classifier", layout="centered")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_bottom(img, crop_frac=0.15):
    width, height = img.size
    crop_h = int(height * (1 - crop_frac))
    return img.crop((0, 0, width, crop_h))


class DifferenceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, before, after):
        f1 = self.resnet(before)
        f2 = self.resnet(after)
        diff = f2 - f1
        return self.classifier(diff)


@st.cache_resource
def load_model():
    checkpoint = torch.load("./Model/aging_vs_procedure_model.pth", map_location=device)

    model = DifferenceModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", ["Aging", "Procedure"])
    crop_frac = checkpoint.get("crop_frac", 0.15)

    return model, class_names, crop_frac


def preprocess_image(img, crop_frac=0.15):
    img = img.convert("RGB")
    img = crop_bottom(img, crop_frac)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(img)


def predict_pair(model, before_img, after_img, crop_frac=0.15):
    before_tensor = preprocess_image(before_img, crop_frac).unsqueeze(0).to(device)
    after_tensor = preprocess_image(after_img, crop_frac).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(before_tensor, after_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    return pred_idx, probs


st.title("Aging vs Procedure Classifier")
st.write("Upload a before image and an after image to get a prediction.")

model, class_names, crop_frac = load_model()

col1, col2 = st.columns(2)

with col1:
    before_file = st.file_uploader("Upload Before Image", type=["png", "jpg", "jpeg"])
    
    if before_file is not None:
        before_img = Image.open(before_file)
        st.image(before_img, caption="Before Image", use_container_width=True)

with col2:
    after_file = st.file_uploader("Upload After Image", type=["png", "jpg", "jpeg"])
    
    if after_file is not None:
        after_img = Image.open(after_file)
        st.image(after_img, caption="After Image", use_container_width=True)

if st.button("Predict"):
    if before_file is None or after_file is None:
        st.error("Please upload both images first.")
    else:
        before_img = Image.open(before_file)
        after_img = Image.open(after_file)

        pred_idx, probs = predict_pair(model, before_img, after_img, crop_frac)

        st.subheader("Result")
        st.write(f"Predicted class: **{class_names[pred_idx]}**")

        st.bar_chart(
            {
                "Class": class_names,
                "Probability": [float(probs[0]), float(probs[1])]
            },
            x="Class",
            y="Probability"
        )

        st.write(f"Aging probability: {probs[0]:.3f}")
        st.write(f"Procedure probability: {probs[1]:.3f}")