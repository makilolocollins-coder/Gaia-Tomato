import json
import base64
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
from supabase import create_client


# ============================================================
# GAIA TOMATO AI
# ============================================================
# Production Streamlit application
#
# Features:
#   - Hugging Face Vision Transformer
#   - Supabase database
#   - Supabase Storage
#   - Disease-organized image storage
#   - Backend human-review flag
#   - Farmer-facing diagnosis interface
#   - GitHub-hosted background image
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

BACKGROUND_IMAGE = (
    ROOT_DIR / "tomato_farmer_africa.jpg"
)

HF_REPO_ID = "Makky07/gaiatomato07"

MODEL_FILENAME = "GAIA_TOMATO_VIT_BEST.pt"

CONFIG_FILENAME = "GAIA_TOMATO_CONFIG.json"

SUPABASE_BUCKET = "gaia-images"

# IMPORTANT:
# This is your exact Supabase table name.
SUPABASE_TABLE = "GAIA Diagnosis Database"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SUPABASE SECRETS
# ============================================================
#
# Put these in Streamlit Cloud:
#
# Settings
#   -> Secrets
#
# Example:
#
# SUPABASE_URL = "https://pelxleyfheicfccmprjm.supabase.co"
# SUPABASE_KEY = "your_publishable_or_appropriate_key"
#
# DO NOT put the real key directly in GitHub.
# ============================================================

try:

    SUPABASE_URL = st.secrets[
        "SUPABASE_URL"
    ]

    SUPABASE_KEY = st.secrets[
        "SUPABASE_KEY"
    ]

except Exception:

    SUPABASE_URL = None
    SUPABASE_KEY = None


# ============================================================
# SUPABASE CLIENT
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_supabase_client():

    if not SUPABASE_URL:

        raise RuntimeError(
            "SUPABASE_URL is missing from "
            "Streamlit Secrets."
        )

    if not SUPABASE_KEY:

        raise RuntimeError(
            "SUPABASE_KEY is missing from "
            "Streamlit Secrets."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )


# ============================================================
# MODEL DEFAULTS
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
        "Powdery Mildew",
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Late_blight": {

        "description":
            "A destructive disease that can rapidly damage tomato leaves, stems and fruit.",

        "action":
            "Remove severely affected material, improve airflow and avoid prolonged leaf wetness. Follow locally approved disease-management practices.",

        "severity":
            "High",
    },


    "Early_blight": {

        "description":
            "A fungal disease commonly associated with dark lesions and concentric rings on older leaves.",

        "action":
            "Remove affected leaves, improve field sanitation and avoid prolonged leaf wetness.",

        "severity":
            "Moderate",
    },


    "Septoria_leaf_spot": {

        "description":
            "A fungal disease producing numerous small spots, often beginning on lower leaves.",

        "action":
            "Remove infected foliage, improve sanitation and reduce prolonged moisture on leaves.",

        "severity":
            "Moderate",
    },


    "Tomato_Yellow_Leaf_Curl_Virus": {

        "description":
            "A viral disease commonly associated with leaf curling, yellowing and reduced plant growth.",

        "action":
            "Monitor and manage whiteflies, remove severely affected plants and consider resistant varieties.",

        "severity":
            "High",
    },


    "Bacterial_spot": {

        "description":
            "A bacterial disease that can produce dark spots on leaves, stems and fruit.",

        "action":
            "Maintain field sanitation, avoid handling wet plants and remove severely infected material.",

        "severity":
            "Moderate",
    },


    "Target_Spot": {

        "description":
            "A fungal disease characterized by circular target-like lesions.",

        "action":
            "Improve airflow, remove affected leaves and follow locally approved disease-management practices.",

        "severity":
            "Moderate",
    },


    "Tomato_mosaic_virus": {

        "description":
            "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",

        "action":
            "Remove severely affected plants and disinfect tools to reduce mechanical spread.",

        "severity":
            "High",
    },


    "Leaf_Mold": {

        "description":
            "A fungal disease associated with humid conditions and poor ventilation.",

        "action":
            "Improve ventilation, reduce humidity and minimize prolonged moisture on leaves.",

        "severity":
            "Moderate",
    },


    "Spider_mites_Two_spotted_spider_mite": {

        "description":
            "Spider mites feed on tomato leaves and can cause stippling, yellowing and plant stress.",

        "action":
            "Inspect the underside of leaves and use an appropriate locally approved management strategy if infestation is confirmed.",

        "severity":
            "Moderate",
    },


    "Powdery_Mildew": {

        "description":
            "A fungal disease characterized by powdery white growth on plant surfaces.",

        "action":
            "Improve airflow, remove severely affected foliage and follow locally approved treatment recommendations.",

        "severity":
            "Moderate",
    },


    "healthy": {

        "description":
            "GAIA did not detect one of the target tomato diseases in the uploaded image.",

        "action":
            "Continue crop monitoring and maintain good irrigation, nutrition and field hygiene.",

        "severity":
            "Low",
    },
}


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def get_background_css():

    if not BACKGROUND_IMAGE.exists():

        return """
        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    #164b24 0%,
                    #07150b 45%,
                    #030a05 100%
                );
        }
        """

    try:

        image_bytes = (
            BACKGROUND_IMAGE.read_bytes()
        )

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        return f"""
        .stApp {{

            background-image:

                linear-gradient(
                    rgba(2, 16, 7, 0.78),
                    rgba(2, 18, 8, 0.90)
                ),

                url(
                    "data:image/jpeg;base64,{encoded}"
                );

            background-size: cover;

            background-position: center;

            background-attachment: fixed;

            min-height: 100vh;
        }}
        """

    except Exception:

        return """
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


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(

    f"""
    <style>

    {get_background_css()}

    /* -------------------------------------------------- */
    /* GLOBAL */
    /* -------------------------------------------------- */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    .block-container {{

        max-width: 1180px;

        padding-top: 1rem;

        padding-bottom: 4rem;
    }}


    /* -------------------------------------------------- */
    /* BRAND */
    /* -------------------------------------------------- */

    .brand-row {{

        display: flex;

        justify-content:
            space-between;

        align-items: center;

        padding:
            10px 2px 10px;
    }}

    .brand {{

        font-size: 30px;

        font-weight: 950;

        color: white;

        letter-spacing:
            -1.5px;
    }}

    .brand-tomato {{

        color:
            #91e66d;
    }}

    .brand-subtitle {{

        color:
            rgba(255,255,255,.58);

        font-size:
            11px;

        font-weight:
            800;

        letter-spacing:
            2px;
    }}


    /* -------------------------------------------------- */
    /* HERO */
    /* -------------------------------------------------- */

    .hero {{

        text-align:
            center;

        color:
            white;

        padding:
            62px 12px 48px;
    }}

    .badge {{

        display:
            inline-block;

        padding:
            8px 17px;

        border-radius:
            999px;

        background:
            rgba(145,230,109,.13);

        border:
            1px solid
            rgba(145,230,109,.35);

        color:
            #a8ee89;

        font-size:
            12px;

        font-weight:
            850;

        letter-spacing:
            1.4px;
    }}

    .hero h1 {{

        margin:
            22px 0 0;

        color:
            white;

        font-size:
            clamp(44px, 7vw, 78px);

        line-height:
            .98;

        letter-spacing:
            -4px;

        font-weight:
            950;
    }}

    .hero h1 span {{

        color:
            #91e66d;
    }}

    .hero p {{

        max-width:
            720px;

        margin:
            25px auto 0;

        color:
            rgba(255,255,255,.80);

        font-size:
            18px;

        line-height:
            1.65;
    }}


    /* -------------------------------------------------- */
    /* GLASS CARD */
    /* -------------------------------------------------- */

    .glass-card {{

        background:
            rgba(255,255,255,.095);

        border:
            1px solid
            rgba(255,255,255,.17);

        border-radius:
            25px;

        padding:
            28px;

        backdrop-filter:
            blur(18px);

        -webkit-backdrop-filter:
            blur(18px);

        box-shadow:
            0 25px 80px
            rgba(0,0,0,.24);

        color:
            white;
    }}

    .glass-card h2,
    .glass-card h3 {{

        color:
            white;
    }}

    .glass-card p {{

        color:
            rgba(255,255,255,.72);
    }}


    /* -------------------------------------------------- */
    /* RESULT CARD */
    /* -------------------------------------------------- */

    .result-card {{

        background:
            rgba(255,255,255,.97);

        border-radius:
            25px;

        padding:
            30px;

        color:
            #132519;

        box-shadow:
            0 25px 75px
            rgba(0,0,0,.30);
    }}

    .result-card h2,
    .result-card h3 {{

        color:
            #132519;
    }}

    .label {{

        color:
            #66806c;

        font-size:
            11px;

        font-weight:
            900;

        text-transform:
            uppercase;

        letter-spacing:
            1.5px;
    }}

    .diagnosis {{

        color:
            #142519;

        font-size:
            34px;

        font-weight:
            950;

        margin:
            7px 0 15px;
    }}

    .confidence {{

        color:
            #18351d;

        font-size:
            40px;

        font-weight:
            950;
    }}


    /* -------------------------------------------------- */
    /* STATUS BOXES */
    /* -------------------------------------------------- */

    .warning-box {{

        background:
            #fff3d6;

        border-left:
            5px solid
            #d89b19;

        padding:
            17px;

        border-radius:
            12px;

        color:
            #654600;

        margin-top:
            18px;
    }}

    .success-box {{

        background:
            #e7f8e2;

        border-left:
            5px solid
            #4b9b3f;

        padding:
            17px;

        border-radius:
            12px;

        color:
            #245d20;

        margin-top:
            18px;
    }}


    /* -------------------------------------------------- */
    /* UPLOADER */
    /* -------------------------------------------------- */

    [data-testid="stFileUploader"] {{

        background:
            rgba(255,255,255,.07);

        border:
            2px dashed
            rgba(145,230,109,.52);

        border-radius:
            20px;

        padding:
            10px;
    }}


    /* -------------------------------------------------- */
    /* BUTTON */
    /* -------------------------------------------------- */

    .stButton > button {{

        width:
            100%;

        min-height:
            52px;

        border-radius:
            15px;

        border:
            none;

        background:
            #83d95f;

        color:
            #102411;

        font-weight:
            950;

        font-size:
            16px;

        box-shadow:
            0 10px 30px
            rgba(131,217,95,.18);
    }}

    .stButton > button:hover {{

        background:
            #a0ed7e;

        color:
            #102411;
    }}


    /* -------------------------------------------------- */
    /* METRICS */
    /* -------------------------------------------------- */

    [data-testid="stMetric"] {{

        background:
            rgba(255,255,255,.08);

        border:
            1px solid
            rgba(255,255,255,.13);

        border-radius:
            16px;

        padding:
            15px;
    }}

    [data-testid="stMetricLabel"] {{

        color:
            rgba(255,255,255,.62) !important;
    }}

    [data-testid="stMetricValue"] {{

        color:
            white !important;
    }}


    /* -------------------------------------------------- */
    /* EXPANDER */
    /* -------------------------------------------------- */

    [data-testid="stExpander"] {{

        background:
            rgba(255,255,255,.075);

        border:
            1px solid
            rgba(255,255,255,.13);

        border-radius:
            15px;
    }}


    /* -------------------------------------------------- */
    /* FOOTER */
    /* -------------------------------------------------- */

    .footer {{

        text-align:
            center;

        color:
            rgba(255,255,255,.54);

        padding:
            50px 10px 15px;

        font-size:
            13px;

        line-height:
            1.7;
    }}


    /* -------------------------------------------------- */
    /* MOBILE */
    /* -------------------------------------------------- */

    @media(max-width:768px) {{

        .brand-subtitle {{
            display:
                none;
        }}

        .hero {{
            padding:
                40px 8px 38px;
        }}

        .hero h1 {{
            letter-spacing:
                -2.5px;
        }}

        .glass-card,
        .result-card {{
            padding:
                21px;
        }}

        .diagnosis {{
            font-size:
                28px;
        }}

        .confidence {{
            font-size:
                34px;
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
    <div class="brand-row">

        <div class="brand">
            GAIA<span class="brand-tomato">🍅</span>
        </div>

        <div class="brand-subtitle">
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
# MODEL CONFIGURATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
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
        "class count does not match class list."
    )

    st.stop()


# ============================================================
# MODEL
# ============================================================

class GaiaTomatoModel(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_classes,
    ):

        super().__init__()

        self.backbone = (
            timm.create_model(

                model_name,

                pretrained=False,

                num_classes=0,
            )
        )

        embed_dim = (
            self.backbone.num_features
        )

        self.head = nn.Sequential(

            nn.Linear(
                embed_dim,
                1024,
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                1024,
                512,
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                512,
                num_classes,
            ),
        )


    def forward(
        self,
        x,
    ):

        features = (
            self.backbone(x)
        )

        return self.head(
            features
        )


# ============================================================
# CHECKPOINT EXTRACTION
# ============================================================

def extract_state_dict(
    checkpoint
):

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


    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):

        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):

            return value


    return checkpoint


# ============================================================
# CLEAN STATE DICT
# ============================================================

def clean_state_dict(
    state_dict
):

    cleaned = {}


    for key, value in (
        state_dict.items()
    ):

        new_key = key

        changed = True


        while changed:

            changed = False


            for prefix in (
                "module.",
                "model.",
                "net.",
            ):

                if new_key.startswith(
                    prefix
                ):

                    new_key = (
                        new_key[
                            len(prefix):
                        ]
                    )

                    changed = True


        cleaned[new_key] = value


    return cleaned


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
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


    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )


    state_dict = (
        clean_state_dict(
            state_dict
        )
    )


    model.load_state_dict(

        state_dict,

        strict=True,
    )


    model.to(
        DEVICE
    )


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
    )


    tensor = tensor.to(
        DEVICE
    )


    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = (
            F.softmax(
                logits,
                dim=1,
            )[0]
        )


    confidence, index = (
        torch.max(
            probabilities,
            dim=0,
        )
    )


    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )


    return (

        index.item(),

        float(
            confidence.item()
        ),

        probabilities,
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnostics(
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

        entropy
        / max_entropy
        * 100

        if max_entropy > 0

        else 0.0
    )


    confidence_pct = (
        confidence * 100
    )


    uncertain = (

        confidence_pct < 60

        or uncertainty > 60
    )


    return (

        entropy,

        uncertainty,

        confidence_pct,

        uncertain,
    )


# ============================================================
# SAFE STORAGE FOLDER NAME
# ============================================================

def clean_storage_folder(
    prediction
):

    folder = str(
        prediction
    ).strip()


    if not folder:

        folder = "unknown"


    folder = (
        folder
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )


    return folder


# ============================================================
# SAVE DIAGNOSIS TO SUPABASE
# ============================================================

def save_diagnosis_to_supabase(

    uploaded_file,

    crop,

    prediction,

    confidence,

    needs_human_review,
):

    """
    Saves:

        1. Original image -> Supabase Storage

        2. Diagnosis metadata -> Supabase table

    Storage structure:

        gaia-images/
            diagnoses/
                Late_blight/
                    2026/
                        08/
                            12/
                                uuid.jpg

    The human-review fields are backend-only.
    They are not shown to the farmer.
    """

    try:

        supabase = (
            get_supabase_client()
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        now = datetime.now(
            timezone.utc
        )


        date_folder = (
            now.strftime(
                "%Y/%m/%d"
            )
        )


        # ----------------------------------------------------
        # UNIQUE FILE ID
        # ----------------------------------------------------

        unique_id = str(
            uuid.uuid4()
        )


        # ----------------------------------------------------
        # ORIGINAL FILE NAME
        # ----------------------------------------------------

        original_name = (

            uploaded_file.name

            or "uploaded_image.jpg"
        )


        extension = (
            Path(
                original_name
            )
            .suffix
            .lower()
        )


        if extension not in (

            ".jpg",
            ".jpeg",
            ".png",
            ".webp",

        ):

            extension = ".jpg"


        # ----------------------------------------------------
        # DISEASE FOLDER
        # ----------------------------------------------------

        disease_folder = (
            clean_storage_folder(
                prediction
            )
        )


        # ----------------------------------------------------
        # STORAGE PATH
        # ----------------------------------------------------
        #
        # Example:
        #
        # diagnoses/
        #     Late_blight/
        #         2026/08/12/
        #             uuid.jpg
        #
        # ----------------------------------------------------

        storage_path = (

            f"diagnoses/"
            f"{disease_folder}/"
            f"{date_folder}/"
            f"{unique_id}"
            f"{extension}"
        )


        # ----------------------------------------------------
        # IMAGE BYTES
        # ----------------------------------------------------

        image_bytes = (
            uploaded_file.getvalue()
        )


        if not image_bytes:

            raise RuntimeError(
                "Uploaded image contains no data."
            )


        # ----------------------------------------------------
        # MIME TYPE
        # ----------------------------------------------------

        content_type = (

            uploaded_file.type

            or "image/jpeg"
        )


        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        storage = (

            supabase
            .storage
            .from_(
                SUPABASE_BUCKET
            )
        )


        storage.upload(

            storage_path,

            image_bytes,

            {
                "content-type":
                    content_type,

                "cache-control":
                    "3600",

                "upsert":
                    False,
            },
        )


        # ----------------------------------------------------
        # DATABASE RECORD
        # ----------------------------------------------------

        record = {

            "image_path":
                storage_path,

            "crop":
                crop,

            "prediction":
                prediction,

            "confidence":
                float(
                    confidence
                ),

            "created_at":
                now.isoformat(),

            "needs_human_review":
                bool(
                    needs_human_review
                ),

            "human_diagnosis":
                None,

            "reviewed_at":
                None,

            "approved_for_training":
                False,
        }


        response = (

            supabase

            .table(
                SUPABASE_TABLE
            )

            .insert(
                record
            )

            .execute()
        )


        return {

            "success":
                True,

            "storage_path":
                storage_path,

            "record":
                response.data,
        }


    except Exception as error:

        return {

            "success":
                False,

            "error":
                str(error),

        }


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown(

    """
    <div class="glass-card">

        <h2>
            🔬 Analyze a tomato leaf
        </h2>

        <p>
            Upload a clear tomato leaf photograph.
            Good lighting, focus and a visible leaf
            generally provide better screening conditions.
        </p>

    </div>
    """,

    unsafe_allow_html=True,
)


uploaded = st.file_uploader(

    "Upload tomato leaf image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],

    label_visibility="collapsed",
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded is None:

    st.markdown(

        """
        <div class="glass-card"
             style="
                 text-align:center;
                 margin-top:25px;
             ">

            <div style="
                font-size:55px;
                margin-bottom:10px;
            ">
                🍃
            </div>

            <h2>
                Your crop health starts here
            </h2>

            <p>
                Upload a tomato leaf photograph above
                to begin AI-assisted disease screening.
            </p>

        </div>
        """,

        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "GAIA could not read this image."
        )

        st.stop()


    left, right = st.columns(

        [0.95, 1.05],

        gap="large",
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with left:

        st.markdown(

            """
            <div class="glass-card">

                <h3>
                    📷 Uploaded image
                </h3>

            </div>
            """,

            unsafe_allow_html=True,
        )


        st.image(

            image,

            use_container_width=True,
        )


        analyze = st.button(

            "🍅 ANALYZE WITH GAIA",

            use_container_width=True,
        )


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        if not analyze:

            st.markdown(

                """
                <div class="result-card">

                    <div class="label">
                        READY
                    </div>

                    <div class="diagnosis">
                        Ready to analyze
                    </div>

                    <p>
                        Click
                        <b>
                            Analyze with GAIA
                        </b>
                        to run the trained
                        Vision Transformer.
                    </p>

                </div>
                """,

                unsafe_allow_html=True,
            )


        else:

            try:

                # ====================================================
                # MODEL
                # ====================================================

                with st.spinner(
                    "GAIA is analyzing the leaf..."
                ):

                    model = (
                        load_model()
                    )


                    idx, conf, probs = (
                        predict(
                            model,
                            image,
                        )
                    )


                # ====================================================
                # PREDICTION
                # ====================================================

                if idx < 0 or idx >= len(
                    CLASS_NAMES
                ):

                    raise RuntimeError(
                        "The model returned an "
                        "invalid class index."
                    )


                disease = (
                    CLASS_NAMES[idx]
                )


                name = (
                    DISPLAY_NAMES.get(
                        disease,
                        disease,
                    )
                )


                # ====================================================
                # DIAGNOSTICS
                # ====================================================

                (
                    entropy,

                    uncertainty,

                    confidence_pct,

                    uncertain,

                ) = diagnostics(

                    probs,

                    conf,
                )


                # ====================================================
                # BACKEND REVIEW FLAG
                # ====================================================
                #
                # IMPORTANT:
                # This is saved in Supabase only.
                #
                # It is NOT shown to the farmer.
                #
                # ====================================================

                needs_human_review = (

                    confidence_pct < 60

                    or uncertainty > 60
                )


                # ====================================================
                # DISEASE INFORMATION
                # ====================================================

                info = (

                    DISEASE_INFO.get(

                        disease,

                        {

                            "description":
                                "GAIA detected a target condition.",

                            "action":
                                "Consult a qualified plant-health professional.",

                            "severity":
                                "Unknown",
                        },
                    )
                )


                # ====================================================
                # SAVE TO SUPABASE
                # ====================================================

                save_result = (

                    save_diagnosis_to_supabase(

                        uploaded_file=uploaded,

                        crop="tomato",

                        prediction=disease,

                        confidence=conf,

                        needs_human_review=
                            needs_human_review,
                    )
                )


                # ====================================================
                # RESULT STATUS
                # ====================================================

                if uncertain:

                    status_box = (

                        '<div class="warning-box">'

                        '⚠ <b>Lower-confidence result.</b><br>'

                        'Try a clearer close-up with good lighting '
                        'and the affected leaf occupying most '
                        'of the image.'

                        '</div>'
                    )

                else:

                    status_box = (

                        '<div class="success-box">'

                        '✓ <b>Prediction generated.</b><br>'

                        'GAIA produced a relatively confident '
                        'screening result.'

                        '</div>'
                    )


                # ====================================================
                # MAIN RESULT
                # ====================================================

                st.markdown(

                    f"""
                    <div class="result-card">

                        <div class="label">
                            GAIA DETECTION
                        </div>

                        <div class="diagnosis">
                            {name}
                        </div>

                        <div class="label">
                            CONFIDENCE
                        </div>

                        <div class="confidence">
                            {confidence_pct:.2f}%
                        </div>

                        {status_box}

                        <br>

                        <b>Severity:</b>
                        {info["severity"]}

                    </div>
                    """,

                    unsafe_allow_html=True,
                )


                # ====================================================
                # CONFIDENCE BAR
                # ====================================================

                st.progress(

                    min(
                        max(
                            conf,
                            0.0,
                        ),
                        1.0,
                    )
                )


                # ====================================================
                # METRICS
                # ====================================================

                c1, c2, c3 = (
                    st.columns(3)
                )


                c1.metric(

                    "Confidence",

                    f"{confidence_pct:.2f}%",
                )


                c2.metric(

                    "Uncertainty",

                    f"{uncertainty:.2f}%",
                )


                c3.metric(

                    "Entropy",

                    f"{entropy:.4f}",
                )


                # ====================================================
                # TOP PREDICTIONS
                # ====================================================

                st.markdown(
                    "### Top predictions"
                )


                top_indices = (

                    np.argsort(
                        probs
                    )[::-1][:3]
                )


                for rank, i in enumerate(

                    top_indices,

                    1,
                ):

                    prediction_class = (

                        CLASS_NAMES[i]
                    )


                    prediction_name = (

                        DISPLAY_NAMES.get(

                            prediction_class,

                            prediction_class,
                        )
                    )


                    prediction_pct = (

                        float(
                            probs[i]
                        )
                        * 100
                    )


                    st.write(

                        f"**{rank}. "
                        f"{prediction_name} — "
                        f"{prediction_pct:.2f}%**"
                    )


                    st.progress(

                        float(
                            probs[i]
                        )
                    )


                # ====================================================
                # DIAGNOSTIC GUIDANCE
                # ====================================================

                st.markdown(
                    "### 🌱 Diagnostic guidance"
                )


                st.markdown(

                    f"""
                    <div class="result-card">

                        <div class="label">
                            WHAT GAIA DETECTED
                        </div>

                        <h2>
                            {name}
                        </h2>

                        <p>
                            {info["description"]}
                        </p>

                        <hr>

                        <div class="label">
                            RECOMMENDED NEXT STEP
                        </div>

                        <p>
                            {info["action"]}
                        </p>

                    </div>
                    """,

                    unsafe_allow_html=True,
                )


                # ====================================================
                # HEALTHY
                # ====================================================

                if disease == "healthy":

                    st.success(

                        "🌱 GAIA did not detect one of "
                        "its target tomato diseases in "
                        "this image."
                    )


                # ====================================================
                # LOW CONFIDENCE
                # ====================================================

                if uncertain:

                    st.warning(

                        "For a stronger screening result, "
                        "try a clearer close-up with good "
                        "lighting and the affected leaf "
                        "occupying most of the image."
                    )


                # ====================================================
                # DATABASE SAVE ERROR
                # ====================================================
                #
                # Do not expose database implementation
                # details to the farmer.
                #
                # ====================================================

                if not save_result["success"]:

                    st.warning(

                        "GAIA completed the analysis, "
                        "but the result could not be saved "
                        "to the diagnostic database."
                    )


                # ====================================================
                # ADVANCED INFORMATION
                # ====================================================

                with st.expander(
                    "⚙ Advanced AI information"
                ):

                    st.write(
                        f"Model: `{MODEL_NAME}`"
                    )

                    st.write(
                        f"Input size: "
                        f"`{IMAGE_SIZE} × {IMAGE_SIZE}`"
                    )

                    st.write(
                        f"Device: `{DEVICE}`"
                    )

                    st.write(
                        f"Classes: `{NUM_CLASSES}`"
                    )

                    st.write(
                        f"Entropy: "
                        f"`{entropy:.6f}`"
                    )

                    st.write(
                        f"Normalized uncertainty: "
                        f"`{uncertainty:.2f}%`"
                    )


            except Exception:

                st.error(

                    "GAIA could not complete the analysis. "
                    "Please try another image."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    """
    <div class="footer">

        <strong>
            GAIA Tomato AI 🍅
        </strong>

        <br>
        AI-assisted tomato crop health screening.

        <br><br>

        Results are intended to support
        agricultural decision-making and
        should not replace assessment by a
        qualified plant-health professional.

    </div>
    """,

    unsafe_allow_html=True,
)
