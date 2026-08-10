import json
import base64
from pathlib import Path

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

from huggingface_hub import hf_hub_download


# ============================================================
# GAIA TOMATO AI
# Production-ready Streamlit application
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "Makky07/gaiatomato07"

MODEL_FILENAME = "GAIA_TOMATO_VIT_BEST.pt"
CONFIG_FILENAME = "GAIA_TOMATO_CONFIG.json"

ROOT_DIR = Path(__file__).resolve().parent

BACKGROUND_FILES = [
    ROOT_DIR / "tomato_farmer_africa.jpg",
    ROOT_DIR / "assets" / "tomato_farmer_africa.jpg",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEFAULT CLASSES
# ============================================================

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
    "Powdery_Mildew",
]


DISPLAY_NAMES = {
    "Late_blight": "Late Blight",
    "healthy": "Healthy",
    "Early_blight": "Early Blight",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",
    "Bacterial_spot": "Bacterial Spot",
    "Target_Spot": "Target Spot",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Leaf_Mold": "Leaf Mold",
    "Spider_mites_Two_spotted_spider_mite":
        "Two-Spotted Spider Mites",
    "Powdery_Mildew": "Powdery Mildew",
}


# ============================================================
# DIAGNOSTIC KNOWLEDGE BASE
# ============================================================

DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A destructive disease that can rapidly damage tomato leaves, stems and fruit.",
        "action":
            "Remove severely affected material, improve airflow and avoid prolonged leaf wetness. Follow locally approved disease-management practices.",
        "severity": "High",
    },

    "Early_blight": {
        "description":
            "A fungal disease commonly associated with dark lesions and concentric rings on older leaves.",
        "action":
            "Remove affected leaves, improve field sanitation and avoid overhead irrigation where possible.",
        "severity": "Moderate",
    },

    "Septoria_leaf_spot": {
        "description":
            "A fungal disease that produces numerous small spots, often beginning on lower leaves.",
        "action":
            "Remove infected foliage, improve sanitation and reduce prolonged moisture on leaves.",
        "severity": "Moderate",
    },

    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description":
            "A viral disease commonly associated with leaf curling, yellowing and reduced plant growth.",
        "action":
            "Monitor and manage whiteflies, remove severely affected plants and consider resistant varieties.",
        "severity": "High",
    },

    "Bacterial_spot": {
        "description":
            "A bacterial disease that can produce dark spots on leaves, stems and fruit.",
        "action":
            "Maintain field sanitation, avoid handling wet plants and remove severely infected material.",
        "severity": "Moderate",
    },

    "Target_Spot": {
        "description":
            "A fungal disease characterized by circular target-like lesions.",
        "action":
            "Improve airflow, remove affected leaves and follow locally approved disease-management practices.",
        "severity": "Moderate",
    },

    "Tomato_mosaic_virus": {
        "description":
            "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",
        "action":
            "Remove severely affected plants and disinfect tools to reduce mechanical spread.",
        "severity": "High",
    },

    "Leaf_Mold": {
        "description":
            "A fungal disease associated with humid conditions and poor ventilation.",
        "action":
            "Improve ventilation, reduce humidity and minimize prolonged moisture on leaves.",
        "severity": "Moderate",
    },

    "Spider_mites_Two_spotted_spider_mite": {
        "description":
            "Spider mites feed on tomato leaves and can cause stippling, yellowing and plant stress.",
        "action":
            "Inspect the underside of leaves and use an appropriate locally approved management strategy if infestation is confirmed.",
        "severity": "Moderate",
    },

    "Powdery_Mildew": {
        "description":
            "A fungal disease characterized by powdery white growth on plant surfaces.",
        "action":
            "Improve airflow, remove severely affected foliage and follow locally approved treatment recommendations.",
        "severity": "Moderate",
    },

    "healthy": {
        "description":
            "GAIA did not detect one of the target tomato diseases in the uploaded image.",
        "action":
            "Continue crop monitoring and maintain good irrigation, nutrition and field hygiene.",
        "severity": "Low",
    },
}


# ============================================================
# BACKGROUND
# ============================================================

def find_background():

    for path in BACKGROUND_FILES:

        if path.exists():
            return path

    return None


def get_background_uri():

    path = find_background()

    if path is None:
        return None

    try:

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

        return (
            "data:image/jpeg;base64,"
            + encoded
        )

    except Exception:

        return None


BACKGROUND_URI = get_background_uri()


# ============================================================
# UI
# ============================================================

if BACKGROUND_URI:

    background = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(3,18,8,0.86),
                rgba(5,25,10,0.78)
            ),
            url("{BACKGROUND_URI}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """

else:

    background = """
    .stApp {
        background:
            linear-gradient(
                135deg,
                #06140a,
                #123b1d,
                #071b0c
            );
    }
    """


st.markdown(
    f"""
    <style>

    {background}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }}

    .brand {{
        color: white;
        font-size: 27px;
        font-weight: 900;
    }}

    .brand span {{
        color: #91e66d;
    }}

    .nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 5px 20px;
    }}

    .nav-right {{
        color: rgba(255,255,255,0.65);
        font-size: 12px;
        letter-spacing: 1.5px;
    }}

    .hero {{
        text-align: center;
        color: white;
        padding: 50px 15px 60px;
    }}

    .badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(145,230,109,0.15);
        border: 1px solid rgba(145,230,109,0.35);
        color: #a8ee89;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 20px;
    }}

    .hero h1 {{
        font-size: clamp(42px,7vw,75px);
        line-height: 0.98;
        letter-spacing: -3px;
        margin: 0;
        font-weight: 900;
    }}

    .hero h1 span {{
        color: #91e66d;
    }}

    .hero p {{
        max-width: 680px;
        margin: 24px auto 0;
        font-size: 18px;
        line-height: 1.6;
        color: rgba(255,255,255,0.82);
    }}

    .glass {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 25px;
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.22);
        color: white;
    }}

    .glass h2,
    .glass h3 {{
        color: white;
    }}

    .result {{
        background: rgba(255,255,255,0.97);
        border-radius: 25px;
        padding: 30px;
        color: #142519;
        box-shadow: 0 25px 70px rgba(0,0,0,0.28);
    }}

    .label {{
        color: #65806c;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    .diagnosis {{
        color: #142519;
        font-size: 35px;
        font-weight: 900;
        margin: 6px 0 15px;
    }}

    .confidence {{
        color: #18351d;
        font-size: 40px;
        font-weight: 900;
    }}

    .warning-box {{
        background: #fff3d6;
        border-left: 5px solid #d89b19;
        padding: 18px;
        border-radius: 12px;
        color: #654600;
        margin-top: 18px;
    }}

    .success-box {{
        background: #e7f8e2;
        border-left: 5px solid #4b9b3f;
        padding: 18px;
        border-radius: 12px;
        color: #245d20;
        margin-top: 18px;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 14px;
        border: none;
        background: #83d95f;
        color: #102411;
        font-weight: 900;
        padding: 14px 20px;
        font-size: 16px;
    }}

    .stButton > button:hover {{
        background: #a0ed7e;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.07);
        border: 2px dashed rgba(142,226,107,0.55);
        border-radius: 20px;
        padding: 10px;
    }}

    .footer {{
        text-align: center;
        color: rgba(255,255,255,0.60);
        padding: 45px 10px 15px;
        font-size: 13px;
    }}

    @media(max-width:768px) {{

        .nav-right {{
            display: none;
        }}

        .hero {{
            padding: 35px 10px 45px;
        }}

        .hero h1 {{
            letter-spacing: -2px;
        }}

        .result,
        .glass {{
            padding: 21px;
        }}

        .diagnosis {{
            font-size: 28px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="nav">

        <div class="brand">
            GAIA<span>🍅</span>
        </div>

        <div class="nav-right">
            TOMATO HEALTH INTELLIGENCE
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            ✦ AI-POWERED CROP HEALTH
        </div>

        <h1>
            Know your crop.<br>
            <span>Grow with confidence.</span>
        </h1>

        <p>
            Upload a tomato leaf image and GAIA will
            screen it for 11 tomato health and disease
            conditions using a Vision Transformer.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_config():

    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CONFIG_FILENAME,
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


try:

    CONFIG = load_config()

except Exception as error:

    st.error(
        "GAIA could not load the model configuration."
    )

    st.code(str(error))

    st.stop()


MODEL_NAME = CONFIG.get(
    "model",
    "vit_small_patch16_224",
)

IMAGE_SIZE = int(
    CONFIG.get(
        "image_size",
        224,
    )
)

CLASS_NAMES = CONFIG.get(
    "classes",
    DEFAULT_CLASSES,
)

NUM_CLASSES = int(
    CONFIG.get(
        "num_classes",
        len(CLASS_NAMES),
    )
)


if len(CLASS_NAMES) != NUM_CLASSES:

    st.error(
        "Model configuration error: "
        "class count does not match the class list."
    )

    st.stop()


# ============================================================
# MODEL
# ============================================================

class GaiaTomatoModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes,
    ):

        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )

        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.Linear(
                embed_dim,
                1024,
            ),

            nn.GELU(),

            nn.Dropout(
                0.30,
            ),

            nn.Linear(
                1024,
                512,
            ),

            nn.GELU(),

            nn.Dropout(
                0.20,
            ),

            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(
        checkpoint,
        nn.Module,
    ):

        return checkpoint.state_dict()

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Unsupported model checkpoint format."
        )

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ]

    for key in possible_keys:

        value = checkpoint.get(key)

        if isinstance(
            value,
            dict,
        ):

            return value

    return checkpoint


def clean_state_dict(
    state_dict,
):

    cleaned = {}

    prefixes = [
        "module.",
        "model.",
        "net.",
    ]

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(prefix):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned[new_key] = value

    return cleaned


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
    )

    model = GaiaTomatoModel(
        MODEL_NAME,
        NUM_CLASSES,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [
            0.485,
            0.456,
            0.406,
        ],
        [
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    image,
):

    tensor = transform(
        image.convert("RGB")
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0,
    )

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return (
        index.item(),
        float(confidence.item()),
        probabilities,
    )


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================

def calculate_diagnostics(
    probabilities,
    confidence,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )

    max_entropy = float(
        np.log(
            len(probabilities)
        )
    )

    uncertainty = (
        entropy / max_entropy * 100.0
        if max_entropy > 0
        else 0.0
    )
