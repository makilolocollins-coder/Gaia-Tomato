import base64
import json
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
# Production Streamlit Application
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

BACKGROUND_PATH = ROOT_DIR / "tomato_farmer_africa.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_MODEL_NAME = "vit_small_patch16_224"
DEFAULT_IMAGE_SIZE = 224

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
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A destructive tomato disease that can rapidly affect leaves, stems and fruit.",
        "action":
            "Remove severely infected material, improve airflow and avoid prolonged leaf wetness. Use locally approved disease-management practices.",
        "severity": "High",
    },

    "Early_blight": {
        "description":
            "A fungal disease commonly associated with dark lesions and concentric rings on older leaves.",
        "action":
            "Remove affected leaves, improve field sanitation and avoid prolonged leaf wetness.",
        "severity": "Moderate",
    },

    "Septoria_leaf_spot": {
        "description":
            "A fungal disease producing numerous small spots, often beginning on lower leaves.",
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
            "Improve airflow, remove affected leaves and use locally approved disease-management practices.",
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
            "Spider mites feed on tomato leaves and may cause stippling, yellowing and plant stress.",
        "action":
            "Inspect the underside of leaves and use an appropriate locally approved management strategy if infestation is confirmed.",
        "severity": "Moderate",
    },

    "Powdery_Mildew": {
        "description":
            "A fungal disease characterized by powdery white growth on plant surfaces.",
        "action":
            "Improve airflow, remove severely affected foliage and use an appropriate locally approved treatment.",
        "severity": "Moderate",
    },

    "healthy": {
        "description":
            "GAIA did not detect one of the target tomato diseases.",
        "action":
            "Continue monitoring the crop and maintain good irrigation, nutrition and field hygiene.",
        "severity": "Low",
    },
}


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def get_background_uri():

    if not BACKGROUND_PATH.exists():
        return None

    try:
        encoded = base64.b64encode(
            BACKGROUND_PATH.read_bytes()
        ).decode("utf-8")

        return (
            "data:image/jpeg;base64,"
            + encoded
        )

    except Exception:
        return None


background_uri = get_background_uri()


# ============================================================
# PREMIUM UI
# ============================================================

if background_uri:

    background_css = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(4, 18, 8, 0.86),
                rgba(5, 24, 10, 0.78)
            ),
            url("{background_uri}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """

else:

    background_css = """
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

    {background_css}

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

    .nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 5px 20px;
        color: white;
    }}

    .brand {{
        font-size: 27px;
        font-weight: 900;
    }}

    .brand span {{
        color: #91e66d;
    }}

    .nav-right {{
        font-size: 12px;
        color: rgba(255,255,255,0.72);
        letter-spacing: 1px;
    }}

    .hero {{
        text-align: center;
        padding: 45px 15px 45px;
        color: white;
    }}

    .hero-badge {{
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
        font-size: clamp(40px, 7vw, 76px);
        line-height: 1;
        letter-spacing: -3px;
        margin: 0;
        font-weight: 900;
    }}

    .hero h1 span {{
        color: #91e66d;
    }}

    .hero p {{
        max-width: 700px;
        margin: 22px auto 0;
        font-size: 18px;
        line-height: 1.6;
        color: rgba(255,255,255,0.84);
    }}

    .glass {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 24px;
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.22);
        color: white;
    }}

    .glass h2,
    .glass h3 {{
        color: white;
    }}

    .upload-title {{
        font-size: 24px;
        font-weight: 800;
    }}

    .upload-subtitle {{
        margin-top: 7px;
        color: rgba(255,255,255,0.70);
    }}

    .result {{
        background: rgba(255,255,255,0.97);
        border-radius: 24px;
        padding: 30px;
        color: #142519;
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }}

    .result-label {{
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #65806c;
    }}

    .result-name {{
        font-size: 36px;
        font-weight: 900;
        margin: 7px 0 15px;
    }}

    .big-number {{
        font-size: 40px;
        font-weight: 900;
        color: #18351d;
    }}

    .metric-label {{
        color: #718078;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 800;
    }}

    .stable {{
        display: inline-block;
        padding: 7px 13px;
        border-radius: 20px;
        background: #e8f8e4;
        color: #26701e;
        font-size: 13px;
        font-weight: 800;
    }}

    .uncertain {{
        display: inline-block;
        padding: 7px 13px;
        border-radius: 20px;
        background: #fff2d8;
        color: #8b5b00;
        font-size: 13px;
        font-weight: 800;
    }}

    .severity-high {{
        color: #a82d2d;
        font-weight: 900;
    }}

    .severity-moderate {{
        color: #9b6500;
        font-weight: 900;
    }}

    .severity-low {{
        color: #28702d;
        font-weight: 900;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.07);
        border: 2px dashed rgba(145,230,109,0.55);
        border-radius: 18px;
        padding: 10px;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 14px;
        border: none;
        background: #83d95f;
        color: #102411;
        font-weight: 900;
        padding: 14px;
        font-size: 16px;
    }}

    .stButton > button:hover {{
        background: #a0ed7e;
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
            padding: 30px 5px 35px;
        }}

        .hero h1 {{
            letter-spacing: -2px;
        }}

        .hero p {{
            font-size: 16px;
        }}

        .glass,
        .result {{
            padding: 21px;
            border-radius: 20px;
        }}

        .result-name {{
            font-size: 29px;
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

        <div class="hero-badge">
            ✦ AI-POWERED CROP HEALTH
        </div>

        <h1>
            Know your crop.<br>
            <span>Grow with confidence.</span>
        </h1>

        <p>
            Upload a tomato leaf image and GAIA will
            analyze it using a Vision Transformer trained
            to recognize 11 tomato health and disease classes.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_config():

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CONFIG_FILENAME,
    )

    with open(
        path,
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
    DEFAULT_MODEL_NAME,
)

IMAGE_SIZE = int(
    CONFIG.get(
        "image_size",
        DEFAULT_IMAGE_SIZE,
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
        "class count does not match num_classes."
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
# CHECKPOINT PROCESSING
# ============================================================

def get_state_dict(checkpoint):

    if isinstance(checkpoint, nn.Module):

        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):

        value = checkpoint.get(key)

        if isinstance(value, dict):

            return value

    return checkpoint


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if not torch.is_tensor(value):
            continue

        new_key = key

        for prefix in (
            "module.",
            "model.",
            "net.",
        ):

            if new_key.startswith(prefix):

                new_key = new_key[
                    len(prefix):
                ]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# LOAD MODEL ONLY WHEN REQUIRED
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

    state_dict = get_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    if not state_dict:

        raise RuntimeError(
            "No tensor weights were found "
            "inside the checkpoint."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
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
# UNCERTAINTY
# ============================================================

def calculate_uncertainty(
    probabilities,
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

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    maximum_entropy = np.log(
        len(probabilities)
    )

    normalized = (
        entropy / maximum_entropy
        if maximum_entropy > 0
        else 0.0
    )

    return (
        float(entropy),
        float(normalized * 100.0),
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    """
    <div class="glass">

        <div class="upload-title">
            🔬 Analyze a tomato leaf
        </div>

        <div class="upload-subtitle">
            Upload a clear photograph of a tomato leaf.
            Good lighting and a focused leaf give better results.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    label_visibility="collapsed",
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()


    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )


    image_col, result_col = st.columns(
        [0.95, 1.05],
        gap="large",
    )


    # ========================================================
    # IMAGE PANEL
    # ========================================================

    with image_col:

        st.markdown(
            """
            <div class="glass">

                <h3>Uploaded leaf</h3>

                <p style="
                    color:rgba(255,255,255,0.70);
                ">
                    Review your image before analysis.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        analyze = st.button(
            "🍅  ANALYZE WITH GAIA",
            use_container_width=True,
        )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyze:

        try:

            with st.spinner(
                "GAIA is loading its AI model..."
            ):

                model = load_model()


            with st.spinner(
                "GAIA is analyzing the tomato leaf..."
            ):

                (
                    index,
                    confidence,
                    probabilities,
                ) = predict(
                    model,
                    image,
                )


            predicted_class = CLASS_NAMES[
                index
            ]

            prediction_name = DISPLAY_NAMES.get(
                predicted_class,
                predicted_class,
            )

            entropy, uncertainty = (
                calculate_uncertainty(
                    probabilities
                )
            )

            confidence_pct = (
                confidence * 100.0
            )


            # Conservative screening rule
            is_uncertain = (
                confidence_pct < 60.0
                or uncertainty > 60.0
            )


            info = DISEASE_INFO.get(
                predicted_class,
                {},
            )

            severity = info.get(
                "severity",
                "Unknown",
            )


            # Save result across Streamlit reruns
            st.session_state["prediction"] = {
                "predicted_class":
                    predicted_class,

                "prediction_name":
                    prediction_name,

                "confidence":
                    confidence_pct,

                "probabilities":
                    probabilities,

                "entropy":
                    entropy,

                "uncertainty":
                    uncertainty,

                "severity":
                    severity,

                "info":
                    info,

                "model":
                    MODEL_NAME,

                "image_size":
                    IMAGE_SIZE,
            }


        except Exception as error:

            st.error(
                "GAIA could not complete the analysis."
            )

            st.code(
                str(error)
            )

            st.stop()


    # ========================================================
    # SHOW SAVED RESULT
    # ========================================================

    result = st.session_state.get(
        "prediction"
    )


    with result_col:

        if result is None:

            st.markdown(
                """
                <div class="result">

                    <div class="result-label">
                        READY
                    </div>

                    <div class="result-name">
                        Ready to analyze
                    </div>

                    <p>
                        Click
                        <b>Analyze with GAIA</b>
                        to run the trained Vision Transformer.
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            confidence_pct = result[
                "confidence"
            ]

            uncertainty = result[
                "uncertainty"
            ]

            severity = result[
                "severity"
            ]

            prediction_name = result[
                "prediction_name"
            ]

            if (
                confidence_pct < 60
                or uncertainty > 60
            ):

                status_html = (
                    '<span class="uncertain">'
                    '⚠ REVIEW IMAGE'
                    '</span>'
                )

            else:

                status_html = (
                    '<span class="stable">'
                    '✓ PREDICTION STABLE'
                    '</span>'
                )


            st.markdown(
                f"""
                <div class="result">

                    <div class="result-label">
                        GAIA DETECTION
                    </div>

                    <div class="result-name">
                        {prediction_name}
                    </div>

                    {status_html}

                    <br><br>

                    <div class="result-label">
                        CONFIDENCE
                    </div>

                    <div class="big-number">
                        {confidence_pct:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            st.progress(
                min(
                    max(
                        confidence_pct / 100.0,
                        0.0,
                    ),
                    1.0,
                )
            )


            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )


            m1, m2 = st.columns(2)


            with m1:

                st.markdown(
                    f"""
                    <div class="glass">

                        <div class="metric-label">
                            Uncertainty
                        </div>

                        <div style="
                            font-size:32px;
                            font-weight:900;
                            color:white;
                        ">
                            {uncertainty:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            with m2:

                severity_class = (
                    "severity-high"
                    if severity == "High"
                    else
                    "severity-moderate"
                    if severity == "Moderate"
                    else
                    "severity-low"
                )

                st.markdown(
                    f"""
                    <div class="glass">

                        <div class="metric-label">
                            Severity
                        </div>

                        <div class="{severity_class}"
                             style="font-size:28px;">
                            {severity}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # ========================================================
    # DETAILED RESULTS
    # ========================================================

    result = st.session_state.get(
        "prediction"
    )


    if result is not None:

        probabilities = result[
            "probabilities"
        ]

        predicted_class = result[
            "predicted_class"
        ]

        prediction_name = result[
            "prediction_name"
        ]

        info = result[
            "info"
        ]

        uncertainty = result[
            "uncertainty"
        ]

        entropy = result[
            "entropy"
        ]


        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )


        # ====================================================
        # TOP 3
        # ====================================================

        st.markdown(
            """
            <div class="glass">

                <h2>
                    Prediction breakdown
                </h2>

                <p style="
                    color:rgba(255,255,255,0.70);
                ">
                    GAIA's three strongest predictions.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        for rank, idx in enumerate(
            top_indices,
            start=1,
        ):

            name = DISPLAY_NAMES.get(
                CLASS_NAMES[idx],
                CLASS_NAMES[idx],
            )

            probability = (
                probabilities[idx] * 100.0
            )


            c1, c2 = st.columns(
                [3, 1]
            )


            with c1:

                st.markdown(
                    f"""
                    <div style="
                        color:white;
                        font-weight:700;
                        font-size:16px;
                        padding-top:8px;
                    ">
                        {rank}. {name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            with c2:

                st.markdown(
                    f"""
                    <div style="
                        color:#9bea79;
                        font-weight:900;
                        text-align:right;
                        padding-top:8px;
                    ">
                        {probability:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            st.progress(
                float(
                    probabilities[idx]
                )
            )


        # ====================================================
        # GUIDANCE
        # ====================================================

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )


        st.markdown(
            f"""
            <div class="result">

                <div class="result-label">
                    AI-ASSISTED CROP GUIDANCE
                </div>

                <h2>
                    {prediction_name}
                </h2>

                <p>
                    {info.get(
                        "description",
                        "GAIA detected a target condition."
                    )}
                </p>

                <hr>

                <p>
                    <b>Recommended action</b>
                </p>

                <p>
                    {info.get(
                        "action",
                        "Consult a qualified plant-health professional."
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


        if predicted_class == "healthy":

            st.success(
                "🌱 GAIA did not detect one of its "
                "target tomato diseases in this image."
            )


        if (
            result["confidence"] < 60
            or uncertainty > 60
        ):

            st.warning(
                "GAIA is uncertain about this image. "
                "Try a clearer, well-lit photograph with "
                "the leaf filling most of the frame."
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "⚙ Advanced AI analysis"
        ):

            st.write(
                f"Model: `{result['model']}`"
            )

            st.write(
                f"Input resolution: "
                f"`{result['image_size']} × "
                f"{result['image_size']}`"
            )

            st.write(
                f"Device: `{DEVICE}`"
            )

            st.write(
                f"Entropy: `{entropy:.4f}`"
            )

            st.write(
                f"Normalized uncertainty: "
                f"`{uncertainty:.2f}%`"
            )

            st.write(
                f"Number of classes: "
                f"`{NUM_CLASSES}`"
            )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <br><br>

        <div class="glass"
             style="
                text-align:center;
                padding:55px 25px;
             ">

            <div style="font-size:52px;">
                🍃
            </div>

            <h2>
                Your crop health starts here
            </h2>

            <p style="
                color:rgba(255,255,255,0.70);
                max-width:550px;
                margin:auto;
            ">
                Upload a tomato leaf image above.
                GAIA will analyze the image and provide
                an AI-assisted disease screening result.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>GAIA Tomato AI</strong>

        <br><br>

        AI-assisted tomato crop health screening.

        <br><br>

        Results are intended to support agricultural
        decision-making and should not replace assessment
        by a qualified plant-health professional.

    </div>
    """,
    unsafe_allow_html=True,
)
