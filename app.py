import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import json

# ============================================================
# GAIA TOMATO DISEASE AI
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato Doctor",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "GAIA_TOMATO_VIT_BEST.pt"
CONFIG_PATH = "GAIA_TOMATO_CONFIG.json"

IMAGE_SIZE = 224

DEFAULT_CLASSES = [
    "Late_blight",
    "healthy",
    "Early_blight",
    "Septoria_leaf_spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Bacterial_spot",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Leaf_Mold",
    "Spider_mites_Two_spotted_spider_mite",
    "Powdery_Mildew"
]

CLASS_NAMES_DISPLAY = {
    "Late_blight": "Late Blight",
    "healthy": "Healthy",
    "Early_blight": "Early Blight",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Bacterial_spot": "Bacterial Spot",
    "Target_Spot": "Target Spot",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Leaf_Mold": "Leaf Mold",
    "Spider_mites_Two_spotted_spider_mite": "Spider Mites",
    "Powdery_Mildew": "Powdery Mildew"
}

# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A serious fungal-like disease that can rapidly damage tomato leaves, stems and fruits.",
        "action":
            "Remove heavily infected material, improve airflow and avoid prolonged leaf wetness. Consider an appropriate locally approved fungicide.",
        "severity": "High"
    },

    "Early_blight": {
        "description":
            "A fungal disease commonly producing dark lesions and concentric rings on older leaves.",
        "action":
            "Remove infected leaves, improve field sanitation and avoid overhead irrigation.",
        "severity": "Moderate"
    },

    "Septoria_leaf_spot": {
        "description":
            "A fungal disease producing numerous small spots, often beginning on lower leaves.",
        "action":
            "Remove infected foliage, reduce leaf wetness and maintain good field sanitation.",
        "severity": "Moderate"
    },

    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description":
            "A viral disease commonly associated with leaf curling, yellowing and stunted growth.",
        "action":
            "Control whiteflies, remove severely affected plants and use resistant varieties where available.",
        "severity": "High"
    },

    "Bacterial_spot": {
        "description":
            "A bacterial disease that can cause small dark lesions on leaves and fruit.",
        "action":
            "Remove infected material, avoid working with wet plants and maintain good sanitation.",
        "severity": "Moderate"
    },

    "Target_Spot": {
        "description":
            "A fungal disease characterized by circular target-like lesions on leaves and sometimes fruit.",
        "action":
            "Improve airflow, remove infected leaves and use appropriate disease-management practices.",
        "severity": "Moderate"
    },

    "Tomato_mosaic_virus": {
        "description":
            "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",
        "action":
            "Remove severely infected plants and disinfect tools. Avoid spreading plant sap between plants.",
        "severity": "High"
    },

    "Leaf_Mold": {
        "description":
            "A fungal disease that commonly develops under humid conditions.",
        "action":
            "Improve ventilation, reduce humidity and avoid prolonged moisture on leaves.",
        "severity": "Moderate"
    },

    "Spider_mites_Two_spotted_spider_mite": {
        "description":
            "Spider mites feed on leaves and can cause stippling, yellowing and general plant stress.",
        "action":
            "Inspect the underside of leaves and use an appropriate locally approved mite-management strategy.",
        "severity": "Moderate"
    },

    "Powdery_Mildew": {
        "description":
            "A fungal disease characterized by white powdery growth on plant surfaces.",
        "action":
            "Improve airflow, remove severely affected leaves and apply an appropriate locally approved treatment.",
        "severity": "Moderate"
    },

    "healthy": {
        "description":
            "The model did not detect one of the 10 target tomato diseases.",
        "action":
            "Continue monitoring the plant and maintain good irrigation, nutrition and field hygiene.",
        "severity": "Low"
    }
}

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        linear-gradient(
            rgba(0,0,0,0.58),
            rgba(0,0,0,0.72)
        ),
        url("tomato_farmer_africa.jpg");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.gaia-header {
    text-align: center;
    padding: 25px;
    border-radius: 22px;
    background: rgba(0,0,0,0.42);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

.gaia-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 0;
}

.gaia-subtitle {
    color: #eeeeee;
    font-size: 18px;
}

/* Cards */
.gaia-card {
    background: rgba(255,255,255,0.94);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
}

/* Prediction */
.prediction {
    font-size: 30px;
    font-weight: 800;
    color: #222;
}

.confidence {
    font-size: 21px;
    color: #444;
}

/* Small labels */
.label {
    font-size: 14px;
    font-weight: 700;
    color: #666;
    text-transform: uppercase;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    opacity: 0.8;
    margin-top: 30px;
}

/* Mobile */
@media (max-width: 768px) {

    .gaia-title {
        font-size: 34px;
    }

    .gaia-subtitle {
        font-size: 15px;
    }

}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="gaia-header">

<div class="gaia-title">
🍅 GAIA Tomato Doctor
</div>

<div class="gaia-subtitle">
AI-powered tomato disease screening for farmers
</div>

</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_resource
def load_config():

    classes = DEFAULT_CLASSES

    if os.path.exists(CONFIG_PATH):

        try:

            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)

            if "classes" in config:
                classes = config["classes"]

        except Exception:
            pass

    return classes


CLASS_NAMES = load_config()

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# MODEL
# ============================================================

class GaiaTomatoModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GaiaTomatoModel(NUM_CLASSES)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Handle several possible saving formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove common prefixes
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("model."):
            new_key = new_key[6:]

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    model.to(device)
    model.eval()

    return model, device


try:

    model, device = load_model()

except Exception as e:

    st.error("Could not load the GAIA tomato model.")

    st.code(str(e))

    st.stop()


# ============================================================
# TRANSFORM
# ============================================================

transform = T.Compose([

    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ============================================================
# UNCERTAINTY
# ============================================================

def calculate_uncertainty(probabilities):

    probabilities = np.asarray(probabilities)

    probabilities = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    max_entropy = np.log(len(probabilities))

    normalized_entropy = entropy / max_entropy

    confidence = np.max(probabilities)

    uncertainty = normalized_entropy

    return entropy, uncertainty, confidence


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    image_rgb = image.convert("RGB")

    tensor = transform(image_rgb)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    probabilities = probabilities.cpu().numpy()

    prediction_index = int(
        np.argmax(probabilities)
    )

    prediction = CLASS_NAMES[prediction_index]

    confidence = float(
        probabilities[prediction_index]
    )

    entropy, uncertainty, _ = calculate_uncertainty(
        probabilities
    )

    ranking = np.argsort(
        probabilities
    )[::-1]

    top_predictions = []

    for idx in ranking[:3]:

        top_predictions.append(
            (
                CLASS_NAMES[idx],
                float(probabilities[idx])
            )
        )

    return (
        prediction,
        confidence,
        entropy,
        uncertainty,
        top_predictions,
        probabilities
    )


# ============================================================
# UPLOAD AREA
# ============================================================

st.markdown(
    """
<div class="gaia-card">

<h2>📷 Upload a tomato leaf</h2>

<p>
Take a clear photograph of the affected tomato leaf and upload it.
For best results, use good lighting and keep the leaf clearly visible.
</p>

</div>
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    with left:

        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_container_width=True
        )

    with right:

        with st.spinner(
            "GAIA is analyzing the leaf..."
        ):

            (
                prediction,
                confidence,
                entropy,
                uncertainty,
                top_predictions,
                probabilities
            ) = predict(image)

        display_name = CLASS_NAMES_DISPLAY.get(
            prediction,
            prediction
        )

        # ----------------------------------------------------
        # Prediction card
        # ----------------------------------------------------

        st.markdown(
            f"""
<div class="gaia-card">

<div class="label">
GAIA SCREENING RESULT
</div>

<div class="prediction">
{display_name}
</div>

<div class="confidence">
Confidence: {confidence * 100:.2f}%
</div>

</div>
""",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        if uncertainty >= 0.60 or confidence < 0.60:

            st.warning(
                f"""
⚠️ **Low-confidence prediction**

GAIA is not sufficiently certain about this result.

Confidence: **{confidence * 100:.2f}%**

Uncertainty: **{uncertainty * 100:.2f}%**

Please take another clear photograph or have the plant examined by an agricultural professional.
"""
            )

            uncertainty_status = "HIGH UNCERTAINTY"

        elif uncertainty >= 0.40 or confidence < 0.75:

            st.info(
                f"""
ℹ️ **Moderate uncertainty**

Confidence: **{confidence * 100:.2f}%**

Consider taking another image from a different angle.
"""
            )

            uncertainty_status = "MODERATE UNCERTAINTY"

        else:

            st.success(
                f"""
✅ **Prediction appears stable**

Confidence: **{confidence * 100:.2f}%**
"""
            )

            uncertainty_status = "LOW UNCERTAINTY"


    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            display_name
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%"
        )

    with col3:

        st.metric(
            "Uncertainty",
            f"{uncertainty * 100:.1f}%"
        )


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    info = DISEASE_INFO.get(
        prediction,
        {}
    )

    if info:

        st.markdown(
            f"""
<div class="gaia-card">

<h2>🌱 What GAIA detected</h2>

<p>
{info.get("description", "")}
</p>

<h3>Recommended next step</h3>

<p>
{info.get("action", "")}
</p>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # TOP 3
    # ========================================================

    st.markdown(
        """
<div class="gaia-card">

<h2>🔎 Alternative possibilities</h2>

"""
        ,
        unsafe_allow_html=True
    )

    for disease, probability in top_predictions:

        name = CLASS_NAMES_DISPLAY.get(
            disease,
            disease
        )

        st.write(
            f"**{name}** — {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "Advanced model information"
    ):

        st.write(
            f"**Model:** GAIA ViT-Small Patch16 224"
        )

        st.write(
            f"**Number of classes:** {NUM_CLASSES}"
        )

        st.write(
            f"**Device:** {device}"
        )

        st.write(
            f"**Entropy:** {entropy:.4f}"
        )

        st.write(
            f"**Normalized uncertainty:** "
            f"{uncertainty:.4f}"
        )

        st.write(
            f"**Status:** {uncertainty_status}"
        )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.markdown(
        """
<div class="gaia-card">

<h2>🌿 How to get the best result</h2>

<ul>
<li>Use a clear image of the tomato leaf.</li>
<li>Avoid very dark or blurry photographs.</li>
<li>Make sure the affected area is visible.</li>
<li>Take the photograph close enough to see symptoms.</li>
<li>If GAIA reports high uncertainty, take another image.</li>
</ul>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

🍅 <b>GAIA</b> — AI-assisted agricultural intelligence

<br><br>

For screening and decision support only. 
The result should not replace professional agricultural diagnosis.

</div>
""",
    unsafe_allow_html=True
)
