import json
import base64
import csv
import uuid
from datetime import datetime, timezone
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
# Production Streamlit application
#
# Upload collection:
# Every analyzed image is automatically saved with:
#   - unique ID
#   - original image
#   - timestamp
#   - GAIA prediction
#   - confidence
#   - entropy
#   - uncertainty
#   - human-review status
#
# Storage:
#   gaia_data/
#       images/
#       gaia_predictions.csv
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
# GAIA DATA COLLECTION STORAGE
# ============================================================

GAIA_DATA_DIR = ROOT_DIR / "gaia_data"
GAIA_IMAGE_DIR = GAIA_DATA_DIR / "images"
GAIA_LOG_FILE = GAIA_DATA_DIR / "gaia_predictions.csv"

GAIA_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GAIA_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_COLUMNS = [
    "record_id",
    "timestamp_utc",
    "image_filename",
    "prediction",
    "display_prediction",
    "confidence",
    "entropy",
    "uncertainty_percent",
    "review_status",
]


def ensure_log_file():
    """
    Create the prediction log if it does not already exist.
    """

    if GAIA_LOG_FILE.exists():
        return

    with open(
        GAIA_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()


ensure_log_file()


def calculate_diagnostics(
    probabilities,
    confidence,
):
    """
    Calculate entropy and normalized uncertainty.
    """

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
            * np.log(probabilities)
        )
    )

    max_entropy = float(
        np.log(len(probabilities))
    )

    uncertainty = (
        entropy / max_entropy * 100.0
        if max_entropy > 0
        else 0.0
    )

    return {
        "entropy": entropy,
        "uncertainty_percent": float(
            uncertainty
        ),
        "confidence_percent": float(
            confidence * 100.0
        ),
    }


def save_upload_record(
    image,
    prediction,
    display_prediction,
    confidence,
    probabilities,
):
    """
    Save the farmer's uploaded image and its GAIA prediction.

    Every new prediction starts with:

        Needs Human Review

    This allows a human reviewer to verify the AI result
    before the image is eventually used for future training.
    """

    record_id = uuid.uuid4().hex

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    diagnostics = calculate_diagnostics(
        probabilities,
        confidence,
    )

    image_filename = (
        f"{record_id}.jpg"
    )

    image_path = (
        GAIA_IMAGE_DIR /
        image_filename
    )

    # Save a normalized JPEG copy.
    image_rgb = image.convert(
        "RGB"
    )

    image_rgb.save(
        image_path,
        format="JPEG",
        quality=95,
    )

    row = {
        "record_id": record_id,
        "timestamp_utc": timestamp,
        "image_filename": image_filename,
        "prediction": prediction,
        "display_prediction": display_prediction,
        "confidence": f"{confidence:.6f}",
        "entropy": f"{diagnostics['entropy']:.6f}",
        "uncertainty_percent": (
            f"{diagnostics['uncertainty_percent']:.4f}"
        ),
        "review_status": "Needs Human Review",
    }

    with open(
        GAIA_LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writerow(row)

    return {
        "record_id": record_id,
        "timestamp": timestamp,
        "image_path": str(image_path),
        "log_path": str(GAIA_LOG_FILE),
        "entropy": diagnostics["entropy"],
        "uncertainty_percent":
            diagnostics["uncertainty_percent"],
    }


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

    .review-box {{
        background: #fff3d6;
        border-left: 5px solid #d89b19;
        padding: 18px;
        border-radius: 12px;
        color: #654600;
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
# MAIN APPLICATION
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error(
        "GAIA could not load the model."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# UPLOAD AREA
# ============================================================

left, right = st.columns(
    [1.05, 0.95]
)


with left:

    st.markdown(
        """
        <div class="glass">

        <h2>
            🍃 Upload a tomato leaf
        </h2>

        <p>
            Upload a clear image of a tomato leaf.
            GAIA will analyze it using the trained
            Vision Transformer.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )


with right:

    st.markdown(
        """
        <div class="glass">

        <h3>
            🔬 GAIA diagnostic system
        </h3>

        <p>
            The system screens for 11 target
            tomato conditions.
        </p>

        <p>
            Every analyzed image is automatically
            retained for human review and potential
            future model improvement.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_container_width=True,
        )

        analyze = st.button(
            "🍅 Analyze with GAIA",
            type="primary",
        )

        if analyze:

            with st.spinner(
                "GAIA is analyzing the leaf..."
            ):

                (
                    predicted_index,
                    confidence,
                    probabilities,
                ) = predict(
                    model,
                    image,
                )

            prediction = CLASS_NAMES[
                predicted_index
            ]

            display_prediction = DISPLAY_NAMES.get(
                prediction,
                prediction.replace(
                    "_",
                    " ",
                ),
            )

            diagnostics = calculate_diagnostics(
                probabilities,
                confidence,
            )

            # ------------------------------------------------
            # SAVE IMAGE + AI RESULT
            # ------------------------------------------------

            saved_record = save_upload_record(
                image=image,
                prediction=prediction,
                display_prediction=display_prediction,
                confidence=confidence,
                probabilities=probabilities,
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                '<div class="result">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="label">GAIA Diagnosis</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="diagnosis">'
                f'{display_prediction}'
                f'</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="label">'
                    'Confidence'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'{confidence * 100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col2:

                st.markdown(
                    '<div class="label">'
                    'Uncertainty'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'{diagnostics["uncertainty_percent"]:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            disease_info = DISEASE_INFO.get(
                prediction,
                {
                    "description":
                        "GAIA identified this as one of its target classes.",
                    "action":
                        "Have the result reviewed before taking major crop-management action.",
                    "severity":
                        "Unknown",
                },
            )

            st.markdown(
                f"""
                <div class="warning-box">

                <strong>Description</strong><br>

                {disease_info["description"]}

                <br><br>

                <strong>Recommended action</strong><br>

                {disease_info["action"]}

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="review-box">

                <strong>🧑‍🌾 Needs Human Review</strong><br>

                This prediction has been stored for
                human verification. The reviewed image
                can later become part of GAIA's future
                training dataset.

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="success-box">

                <strong>✓ Image saved to GAIA dataset collection</strong><br><br>

                Record ID:
                <code>{saved_record["record_id"]}</code>

                <br><br>

                Timestamp:
                <code>{saved_record["timestamp"]}</code>

                <br><br>

                Status:
                <strong>Needs Human Review</strong>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # TOP PREDICTIONS
            # ------------------------------------------------

            st.markdown(
                "### Model probabilities"
            )

            ranked_indices = np.argsort(
                probabilities
            )[::-1]

            for index in ranked_indices[:5]:

                class_name = CLASS_NAMES[
                    int(index)
                ]

                class_display = DISPLAY_NAMES.get(
                    class_name,
                    class_name.replace(
                        "_",
                        " ",
                    ),
                )

                probability = (
                    float(
                        probabilities[index]
                    )
                    * 100
                )

                st.write(
                    f"**{class_display}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(
                        probability / 100,
                        1.0,
                    )
                )

            # ------------------------------------------------
            # STORAGE INFORMATION
            # ------------------------------------------------

            with st.expander(
                "📁 GAIA data collection"
            ):

                st.write(
                    "Collected image:"
                )

                st.code(
                    str(
                        saved_record[
                            "image_path"
                        ]
                    )
                )

                st.write(
                    "Prediction log:"
                )

                st.code(
                    str(
                        saved_record[
                            "log_path"
                        ]
                    )
                )

                st.write(
                    "The CSV contains the prediction, "
                    "confidence, uncertainty, timestamp, "
                    "image filename and human-review status."
                )

    except Exception as error:

        st.error(
            "GAIA could not process this image."
        )

        st.code(
            str(error)
        )


# ============================================================
# DATA COLLECTION STATUS
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    "### 🧠 GAIA Learning Dataset"
)

try:

    ensure_log_file()

    with open(
        GAIA_LOG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    total_records = len(rows)

    review_records = sum(
        1
        for row in rows
        if row.get(
            "review_status"
        ) == "Needs Human Review"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Images Collected",
            total_records,
        )

    with col2:

        st.metric(
            "Needs Human Review",
            review_records,
        )

    with col3:

        st.metric(
            "Dataset Status",
            "Collecting",
        )

except Exception:

    st.info(
        "No collection records are available yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        GAIA Tomato AI · AI-assisted crop health screening

        <br><br>

        GAIA predictions are screening results and
        should be reviewed by a qualified human before
        major crop-management decisions.

    </div>
    """,
    unsafe_allow_html=True,
)
