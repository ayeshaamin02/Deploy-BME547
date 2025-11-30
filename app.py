import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import joblib

# ================================
# SETTINGS
# ================================
IMAGE_SIZE_ML = (90, 90)        # SVM/KNN image size
IMAGE_SIZE_DL = (224, 224)      # DL models image size
CLASS_NAMES_ML = ["Ascaris_lumbricoides", "Hookworm_egg"]
CLASS_NAMES_DL = ["Ascaris lumbricoides", "Hookworm"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SVM_MODEL_PATH = "svm_raw_pipeline.pkl"
KNN_MODEL_PATH = "knn_raw_pipeline.pkl"

MODEL_PATHS_DL = {
    "ResNet-18": "resnet18_best.pth",
    "EfficientNet-B0": "efficientnetb0_best.pth",
    "MobileNetV2": "mobilenetv2_best.pth",
    #"VGG16": "vgg16_best.pth",
    "Custom CNN": "customcnn_reduced_best.pth"
}

YOLO_MODEL_PATH = "best.pt"

# ================================
# UTILITIES
# ================================
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

def preprocess_ml_image(img):
    img = img.convert("L")                         # grayscale
    img = img.resize(IMAGE_SIZE_ML)
    img_arr = np.array(img, dtype=np.float32)/255.0
    img_arr = img_arr.flatten().reshape(1, -1)
    return img_arr

preprocess_dl = transforms.Compose([
    transforms.Resize(IMAGE_SIZE_DL),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
class CustomCNNReduced(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNReduced, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self._to_linear = 128 * (IMAGE_SIZE_DL[0] // 16) * (IMAGE_SIZE_DL[1] // 16)


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dl_model(model_name):
    if model_name == "YOLOv8":
        return YOLO(YOLO_MODEL_PATH)

    num_classes = len(CLASS_NAMES_DL)
    if model_name == "ResNet-18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    #elif model_name == "VGG16":
        #model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        #model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "Custom CNN":
        model = CustomCNNReduced(num_classes)
    else:
        raise ValueError("Unknown model.")

    model.load_state_dict(torch.load(MODEL_PATHS_DL[model_name], map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_dl_class(model, image):
    img_tensor = preprocess_dl(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(prob, 1)
    return CLASS_NAMES_DL[pred.item()], conf.item()

# ================================
# STREAMLIT UI
# ================================
st.title("🔬 Parasite Egg Detection – Multi-Model")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        st.write("### 🔍 SVM & KNN Predictions")

        # --- Load ML models ---
        svm_model = load_pipeline(SVM_MODEL_PATH)
        knn_model = load_pipeline(KNN_MODEL_PATH)
        X_ml = preprocess_ml_image(image)

        # SVM prediction
        svm_idx = svm_model.predict(X_ml)[0]
        svm_label = CLASS_NAMES_ML[svm_idx].replace("_", " ")
        try:
            svm_conf = svm_model.predict_proba(X_ml)[0][svm_idx]
        except:
            svm_conf = None
        st.write(f"SVM Prediction: {svm_label}" + (f" — Confidence: {svm_conf*100:.2f}%" if svm_conf else ""))

        # KNN prediction
        knn_idx = knn_model.predict(X_ml)[0]
        knn_label = CLASS_NAMES_ML[knn_idx].replace("_", " ")
        try:
            knn_conf = knn_model.predict_proba(X_ml)[0][knn_idx]
        except:
            knn_conf = None
        st.write(f"KNN Prediction: {knn_label}" + (f" — Confidence: {knn_conf*100:.2f}%" if knn_conf else ""))

        # --- DL models ---
        st.write("### 🔍 Deep Learning Predictions")
        dl_predictions = []
        for mdl_name in ["Custom CNN", "ResNet-18", "EfficientNet-B0", "MobileNetV2"]:
            mdl = load_dl_model(mdl_name)
            lbl, conf = predict_dl_class(mdl, image)
            st.write(f"{mdl_name}: {lbl} — {conf*100:.2f}%")
            dl_predictions.append(lbl)

        # --- YOLOv8 ---
        st.write("### 🔍 YOLOv8 Detection")
        yolo_model = load_dl_model("YOLOv8")
        yolo_results = yolo_model.predict(
            source=image,      # image or batch
            conf=0.35,         # minimum confidence threshold
            iou=0.45,          # NMS IoU threshold
            max_det=50         # maximum detections per image
        )

        st.image(yolo_results[0].plot(), caption="YOLOv8 Prediction", use_container_width=True)

        # --- Majority Vote ---
        all_preds = [svm_label, knn_label] + dl_predictions
        final_pred = max(set(all_preds), key=all_preds.count)
        st.write(f"### ✅ Final Prediction (Majority Vote): {final_pred}")

