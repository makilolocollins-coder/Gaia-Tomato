import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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
# APPLICATION CONFIGURATION
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
                url("data:image/jpeg;base64,{encoded}");

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
            box-shadow: 0 10px 35px rgba(0,0,0,0.20);
        }}

        .result-card {{
            background: rgba(255,255,255,0.97);
            border-radius: 22px;
            padding: 30px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.20);
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


apply_background()


# ============================================================
# LOAD MODEL CONFIGURATION FROM HUGGING FACE
# ============================================================

@st.cache_data(show_spinner=False)
def load_hf_config():

    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CONFIG_FILENAME
    )

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


try:

    CONFIG = load_hf_config()

except Exception as error:

    st.error(
        "GAIA configuration could not be loaded from Hugging Face."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# READ EXACT MODEL CONFIG
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


if len(CLASS_NAMES) != NUM_CLASSES:

    st.error(
        "Model configuration error: "
        "number of classes does not match class list."
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
# LOAD WEIGHTS
# ============================================================

@st.cache_resource(show_spinner="Loading GAIA Tomato AI...")

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
    # Handle possible checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):
