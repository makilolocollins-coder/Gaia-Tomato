import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
import json
import base64


# ============================================================
# GAIA TOMATO AI
# Production Streamlit Application
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "Makky07/gaiatomato07"

MODEL_FILENAME = "GAIA_TOMATO_VIT_BEST.pt"
CONFIG_FILENAME = "GAIA_TOMATO_CONFIG.json"

BACKGROUND_PATH = Path(
    "assets/tomato_farmer_africa.jpg"
)

DEFAULT_IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {

    "Late_blight":
        "Late Blight",

    "healthy":
        "Healthy",

    "Early_blight":
        "Early Blight",

    "Septoria_leaf_spot":
        "Septoria Leaf Spot",

    "Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",

    "Bacterial_spot":
        "Bacterial Spot",

    "Target_Spot":
        "Target Spot",

    "Tomato_mosaic_virus":
        "Tomato Mosaic Virus",

    "Leaf_Mold":
        "Leaf Mold",

    "Spider_mites_Two_spotted_spider_mite":
        "Two-Spotted Spider Mites",

    "Powdery_Mildew":
        "Powdery Mildew"
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A destructive tomato disease that can rapidly affect leaves, stems and fruit.",
        "action":
            "Remove severely infected material, improve airflow and avoid prolonged leaf wetness. Use only locally approved treatments.",
        "severity": "High"
    },

    "Early_blight": {
        "description":
            "A fungal disease commonly associated with dark lesions and concentric rings on older leaves.",
        "action":
            "Remove affected leaves, improve field sanitation and avoid overhead irrigation.",
        "severity": "Moderate"
    },

    "Septoria_leaf_spot": {
        "description":
            "A fungal disease producing numerous small spots, often beginning on lower leaves.",
        "action":
            "Remove infected foliage, improve sanitation and reduce prolonged leaf moisture.",
        "severity": "Moderate"
    },

    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description":
            "A viral disease commonly associated with leaf curling, yellowing and reduced plant growth.",
        "action":
            "Monitor and manage whiteflies, remove severely affected plants and consider resistant varieties.",
        "severity": "High"
    },

    "Bacterial_spot": {
        "description":
            "A bacterial disease that can produce dark spots on leaves, stems and fruit.",
        "action":
            "Maintain field sanitation, avoid handling wet plants and remove severely infected material.",
        "severity": "Moderate"
    },

    "Target_Spot": {
        "description":
            "A fungal disease characterized by circular target-like lesions.",
        "action":
            "Improve airflow, remove affected leaves and use appropriate locally approved disease management.",
        "severity": "Moderate"
    },

    "Tomato_mosaic_virus": {
        "description":
            "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",
        "action":
            "Remove severely affected plants and disinfect tools to reduce mechanical spread.",
        "severity": "High"
    },

    "Leaf_Mold": {
        "description":
            "A fungal disease associated with humid conditions and poor ventilation.",
        "action":
            "Improve ventilation, reduce humidity and minimize prolonged moisture on leaves.",
        "severity": "Moderate"
    },

    "Spider_mites_Two_spotted_spider_mite": {
        "description":
            "Spider mites feed on tomato leaves and may cause stippling, yellowing and plant stress.",
        "action":
            "Inspect the underside of leaves and apply an appropriate locally approved management strategy if infestation is confirmed.",
        "severity": "Moderate"
    },

    "Powdery_Mildew": {
        "description":
            "A fungal disease characterized by powdery white growth on plant surfaces.",
        "action":
            "Improve airflow, remove severely affected foliage and use an appropriate locally approved treatment.",
        "severity": "Moderate"
    },

    "healthy": {
        "description":
            "GAIA did not detect one of the target tomato diseases.",
        "action":
            "Continue monitoring the crop and maintain good irrigation, nutrition and field hygiene.",
        "severity": "Low"
    }
}


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def apply_background():

    if not BACKGROUND_PATH.exists():
        return

    try:

        encoded = base64.b64encode(
            BACKGROUND_PATH.read_bytes()
        ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(0,0,0,0.62),
                        rgba(0,0,0,0.72)
                    ),
                    url(
                        "data:image/jpeg;base64,{encoded}"
                    );

                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            .block-container {{
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}

            .hero {{
                background: rgba(0,0,0,0.48);
                backdrop-filter: blur(10px);
                border-radius: 24px;
                padding: 38px 30px;
                text-align: center;
                color: white;
                margin-bottom: 28px;
                border: 1px solid rgba(255,255,255,0.15);
            }}

            .hero-title {{
                font-size: 48px;
                font-weight: 800;
                margin-bottom: 8px;
            }}

            .hero-subtitle {{
                font-size: 18px;
                opacity: 0.92;
            }}

            .card {{
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                padding: 28px;
                margin-bottom: 22px;
                box-shadow:
                    0 10px 35px rgba(0,0,0,0.20);
            }}

            .result-card {{
                background: rgba(255,255,255,0.97);
                border-radius: 22px;
                padding: 30px;
                box-shadow:
                    0 10px 35px rgba(0,0,0,0.20);
            }}

            .prediction {{
                font-size: 30px;
                font-weight: 800;
                color: #171717;
                margin-top: 8px;
                margin-bottom: 8px;
            }}

            .confidence {{
                font-size: 20px;
                color: #444;
            }}

            .metric {{
                background: rgba(245,245,245,0.9);
                border-radius: 15px;
                padding: 15px;
                text-align: center;
                margin-bottom: 10px;
            }}

            .metric-value {{
                font-size: 25px;
                font-weight: 800;
            }}

            .metric-label {{
                font-size: 13px;
                color: #555;
            }}

            .footer {{
                text-align: center;
                color: white;
                margin-top: 35px;
                padding: 20px;
                opacity: 0.85;
            }}

            @media (max-width: 768px) {{

                .hero-title {{
                    font-size: 34px;
                }}

                .hero-subtitle {{
                    font-size: 15px;
                }}

                .prediction {{
                    font-size: 25px;
                }}

            }}

            </style>
            """,
            unsafe_allow_html=True
        )

    except Exception:
        pass


apply_background()


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_hf_config():

    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CONFIG_FILENAME
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    return config


try:

    CONFIG = load_hf_config()

except Exception as error:

    st.error(
        "GAIA configuration could not be loaded "
        "from Hugging Face."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# READ MODEL CONFIG
# ============================================================

MODEL_NAME = CONFIG.get(
    "model",
    "vit_small_patch16_224"
)

IMAGE_SIZE = int(
    CONFIG.get(
        "image_size",
        DEFAULT_IMAGE_SIZE
    )
)

CLASS_NAMES = CONFIG.get(
    "classes",
    []
)

NUM_CLASSES = int(
    CONFIG.get(
        "num_classes",
        len(CLASS_NAMES)
    )
)


if not CLASS_NAMES:

    st.error(
        "No class names were found in the "
        "Hugging Face model configuration."
    )

    st.stop()


if len(CLASS_NAMES) != NUM_CLASSES:

    st.error(
        "Model configuration error: "
        "number of classes does not match "
        "the class list."
    )

    st.stop()


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class GaiaTomatoModel(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.Linear(
                embed_dim,
                1024
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                1024,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading GAIA Tomato AI..."
)
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    model = GaiaTomatoModel(
        NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove common prefixes
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("model."):

            new_key = new_key[
                len("model.") :
            ]

        if new_key.startswith("module."):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


try:

    model = load_model()

except Exception as error:

    st.error(
        "GAIA Tomato model could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image = image.convert(
        "RGB"
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    confidence, prediction_index = torch.max(
        probabilities,
        dim=0
    )

    prediction_index = (
        prediction_index.item()
    )

    confidence = (
        confidence.item()
    )

    probabilities_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return (
        prediction_index,
        confidence,
        probabilities_np
    )


# ============================================================
# UNCERTAINTY CALCULATION
# ============================================================

def calculate_uncertainty(
    probabilities
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probabilities *
        np.log(probabilities)
    )

    max_entropy = np.log(
        len(probabilities)
    )

    if max_entropy > 0:

        normalized_entropy = (
            entropy /
            max_entropy
        )

    else:

        normalized_entropy = 0.0

    uncertainty_percent = (
        normalized_entropy * 100
    )

    return (
        entropy,
        uncertainty_percent
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🍅 GAIA Tomato AI
        </div>

        <div class="hero-subtitle">
            AI-powered tomato disease screening
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="card">

        <h2>🌱 Analyze Your Tomato Leaf</h2>

        <p>
        Upload a clear image of a tomato leaf and GAIA
        will analyze it using a Vision Transformer
        trained across 11 tomato health and disease classes.
        </p>

        <p>
        For best results, use a well-lit image where the
        leaf is clearly visible.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            "The uploaded image could not be read."
        )

        st.code(
            str(error)
        )

        st.stop()


    # --------------------------------------------------------
    # Image + prediction
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    with left:

        st.markdown(
            """
            <div class="card">
                <h3>Uploaded Image</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    with right:

        with st.spinner(
            "GAIA is analyzing the tomato leaf..."
        ):

            (
                prediction_index,
                confidence,
                probabilities
            ) = predict_image(
                image
            )


        predicted_class = (
            CLASS_NAMES[
                prediction_index
            ]
        )

        display_name = DISPLAY_NAMES.get(
            predicted_class,
            predicted_class
        )


        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        (
            entropy,
            uncertainty_percent
        ) = calculate_uncertainty(
            probabilities
        )


        confidence_percent = (
            confidence * 100
        )


        # ----------------------------------------------------
        # Decision threshold
        # ----------------------------------------------------
        #
        # This is a screening flag, NOT a clinical
        # or agricultural diagnosis.
        #

        uncertain = (

            confidence_percent < 60

            or

            uncertainty_percent > 60
        )


        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )


        st.markdown(
            "### GAIA Prediction"
        )


        st.markdown(
            f"""
            <div class="prediction">
                {display_name}
            </div>

            <div class="confidence">
                Confidence:
                <b>{confidence_percent:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metric_1, metric_2 = st.columns(
            2
        )


        with metric_1:

            st.markdown(
                f"""
                <div class="metric">

                    <div class="metric-value">
                        {confidence_percent:.2f}%
                    </div>

                    <div class="metric-label">
                        Confidence
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with metric_2:

            st.markdown(
                f"""
                <div class="metric">

                    <div class="metric-value">
                        {uncertainty_percent:.2f}%
                    </div>

                    <div class="metric-label">
                        Uncertainty
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # Prediction status
        # ----------------------------------------------------

        if uncertain:

            st.warning(
                "⚠️ GAIA is uncertain about this prediction. "
                "Try uploading a clearer image with the leaf "
                "occupying most of the frame."
            )

        else:

            st.success(
                "✓ Prediction appears stable."
            )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.markdown(
        """
        <div class="card">

            <h2>📊 Prediction Probabilities</h2>

        </div>
        """,
        unsafe_allow_html=True
    )


    probability_data = {

        DISPLAY_NAMES.get(
            CLASS_NAMES[i],
            CLASS_NAMES[i]
        ):
        float(
            probabilities[i]
        )

        for i in range(
            NUM_CLASSES
        )
    }


    st.bar_chart(
        probability_data
    )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.markdown(
        """
        <div class="card">

            <h2>🔎 Top 3 Predictions</h2>

        </div>
        """,
        unsafe_allow_html=True
    )


    top_k = min(
        3,
        NUM_CLASSES
    )


    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        class_name = CLASS_NAMES[
            index
        ]

        name = DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        probability = (
            probabilities[index] * 100
        )

        st.write(
            f"**{rank}. {name}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            float(
                probabilities[index]
            )
        )


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    info = DISEASE_INFO.get(
        predicted_class
    )


    if info is not None:

        st.markdown(
            f"""
            <div class="card">

                <h2>🌿 About {display_name}</h2>

                <p>
                    <b>Description:</b><br>
                    {info["description"]}
                </p>

                <p>
                    <b>Recommended action:</b><br>
                    {info["action"]}
                </p>

                <p>
                    <b>Severity:</b>
                    {info["severity"]}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # TECHNICAL INFORMATION
    # ========================================================

    with st.expander(
        "Technical analysis"
    ):

        st.write(
            f"**Model:** {MODEL_NAME}"
        )

        st.write(
            f"**Input size:** "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}"
        )

        st.write(
            f"**Device:** {DEVICE}"
        )

        st.write(
            f"**Entropy:** {entropy:.4f}"
        )

        st.write(
            f"**Normalized uncertainty:** "
            f"{uncertainty_percent:.2f}%"
        )

        st.write(
            f"**Number of classes:** "
            f"{NUM_CLASSES}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>GAIA Tomato AI</b><br><br>

        AI-assisted tomato disease screening.<br>

        This system is intended to support agricultural
        decision-making and does not replace assessment
        by a qualified agricultural or plant-health expert.

    </div>
    """,
    unsafe_allow_html=True
)
