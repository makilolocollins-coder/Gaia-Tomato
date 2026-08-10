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
# PAGE CONFIG
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

BACKGROUND_PATH = Path("assets/tomato_farmer_africa.jpg")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS DISPLAY NAMES
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
    "Powdery_Mildew": "Powdery Mildew"
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A destructive disease that can rapidly affect tomato leaves, stems and fruit.",
        "action":
            "Remove severely infected material, improve airflow and avoid prolonged leaf wetness.",
        "severity": "High"
    },

    "Early_blight": {
        "description":
            "A fungal disease commonly associated with dark lesions and concentric rings.",
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
            "A viral disease commonly associated with leaf curling, yellowing and reduced growth.",
        "action":
            "Monitor whiteflies, remove severely affected plants and consider resistant varieties.",
        "severity": "High"
    },

    "Bacterial_spot": {
        "description":
            "A bacterial disease that can produce dark spots on leaves, stems and fruit.",
        "action":
            "Maintain field sanitation and avoid handling wet plants.",
        "severity": "Moderate"
    },

    "Target_Spot": {
        "description":
            "A fungal disease characterized by circular target-like lesions.",
        "action":
            "Improve airflow, remove affected leaves and use locally approved management.",
        "severity": "Moderate"
    },

    "Tomato_mosaic_virus": {
        "description":
            "A viral disease that can cause mosaic patterns, leaf distortion and reduced growth.",
        "action":
            "Remove severely affected plants and disinfect tools to reduce mechanical spread.",
        "severity": "High"
    },

    "Leaf_Mold": {
        "description":
            "A fungal disease associated with humid conditions and poor ventilation.",
        "action":
            "Improve ventilation, reduce humidity and minimize prolonged leaf moisture.",
        "severity": "Moderate"
    },

    "Spider_mites_Two_spotted_spider_mite": {
        "description":
            "Spider mites feed on tomato leaves and may cause stippling, yellowing and plant stress.",
        "action":
            "Inspect the underside of leaves and apply an appropriate locally approved management strategy.",
        "severity": "Moderate"
    },

    "Powdery_Mildew": {
        "description":
            "A fungal disease characterized by powdery white growth on plant surfaces.",
        "action":
            "Improve airflow, remove severely affected foliage and use appropriate locally approved treatment.",
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
# BACKGROUND
# ============================================================

def get_background():

    if not BACKGROUND_PATH.exists():
        return None

    try:
        encoded = base64.b64encode(
            BACKGROUND_PATH.read_bytes()
        ).decode()

        return (
            f"data:image/jpeg;base64,{encoded}"
        )

    except Exception:
        return None


background = get_background()


# ============================================================
# PREMIUM UI
# ============================================================

if background:

    background_css = f"""
        .stApp {{
            background:
                linear-gradient(
                    120deg,
                    rgba(7, 20, 11, 0.88),
                    rgba(10, 35, 17, 0.70)
                ),
                url("{background}");

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

    /* Navigation */

    .nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 5px 25px 5px;
        color: white;
    }}

    .brand {{
        font-size: 25px;
        font-weight: 900;
        letter-spacing: -0.5px;
    }}

    .brand span {{
        color: #8ee26b;
    }}

    .nav-right {{
        font-size: 13px;
        color: rgba(255,255,255,0.75);
    }}

    /* Hero */

    .hero {{
        padding: 55px 30px 60px 30px;
        text-align: center;
        color: white;
    }}

    .hero-badge {{
        display: inline-block;
        padding: 8px 15px;
        border-radius: 30px;
        background: rgba(142,226,107,0.15);
        border: 1px solid rgba(142,226,107,0.35);
        color: #a8ee89;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 20px;
    }}

    .hero h1 {{
        font-size: clamp(42px, 7vw, 76px);
        line-height: 0.98;
        letter-spacing: -3px;
        margin: 0;
        font-weight: 900;
    }}

    .hero h1 span {{
        color: #91e66d;
    }}

    .hero p {{
        max-width: 650px;
        margin: 22px auto 0 auto;
        font-size: 18px;
        line-height: 1.6;
        color: rgba(255,255,255,0.83);
    }}

    /* Glass cards */

    .glass {{
        background: rgba(255,255,255,0.095);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 26px;
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow:
            0 25px 70px rgba(0,0,0,0.22);
        color: white;
    }}

    .glass h2,
    .glass h3 {{
        color: white;
    }}

    /* Upload */

    .upload-title {{
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 6px;
    }}

    .upload-subtitle {{
        color: rgba(255,255,255,0.70);
        margin-bottom: 18px;
    }}

    /* Results */

    .result {{
        background: rgba(255,255,255,0.97);
        border-radius: 26px;
        padding: 32px;
        color: #122016;
        box-shadow:
            0 25px 70px rgba(0,0,0,0.28);
    }}

    .result-label {{
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #65806c;
    }}

    .result-name {{
        font-size: 38px;
        font-weight: 900;
        letter-spacing: -1.5px;
        margin-top: 5px;
        margin-bottom: 15px;
        color: #142519;
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
        font-weight: 700;
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

    /* Footer */

    .footer {{
        text-align: center;
        color: rgba(255,255,255,0.60);
        padding: 45px 10px 15px;
        font-size: 13px;
    }}

    /* Streamlit uploader */

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.07);
        border: 2px dashed rgba(142,226,107,0.55);
        border-radius: 20px;
        padding: 10px;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.04);
    }}

    /* Buttons */

    .stButton > button {{
        width: 100%;
        border-radius: 14px;
        border: none;
        background: #83d95f;
        color: #102411;
        font-weight: 900;
        padding: 14px 20px;
        font-size: 16px;
        transition: 0.2s ease;
    }}

    .stButton > button:hover {{
        background: #a0ed7e;
        transform: translateY(-1px);
    }}

    /* Mobile */

    @media(max-width: 768px) {{

        .hero {{
            padding: 35px 10px 40px;
        }}

        .hero h1 {{
            letter-spacing: -2px;
        }}

        .hero p {{
            font-size: 16px;
        }}

        .glass,
        .result {{
            padding: 22px;
            border-radius: 20px;
        }}

        .result-name {{
            font-size: 30px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
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
    unsafe_allow_html=True
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
    unsafe_allow_html=True
)


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_data(show_spinner=False)
def load_config():

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CONFIG_FILENAME
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


try:

    CONFIG = load_config()

except Exception as e:

    st.error("Unable to load GAIA configuration.")
    st.code(str(e))
    st.stop()


MODEL_NAME = CONFIG.get(
    "model",
    "vit_small_patch16_224"
)

IMAGE_SIZE = int(
    CONFIG.get(
        "image_size",
        224
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
        "The model configuration is inconsistent."
    )

    st.stop()


# ============================================================
# MODEL
# ============================================================

class GaiaTomatoModel(nn.Module):

    def __init__(self, num_classes):

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

            nn.Dropout(0.30),

            nn.Linear(
                1024,
                512
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        return self.head(
            self.backbone(x)
        )


# ============================================================
# LOAD WEIGHTS
# ============================================================

@st.cache_resource(
    show_spinner="Initializing GAIA Tomato AI..."
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


    cleaned = {}

    for key, value in state_dict.items():

        key = key.replace(
            "module.",
            "",
            1
        )

        key = key.replace(
            "model.",
            "",
            1
        )

        cleaned[key] = value


    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        "GAIA could not load the trained model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PREPROCESSING
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
# PREDICTION
# ============================================================

def predict(image):

    tensor = transform(
        image.convert("RGB")
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0
    )

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    return (
        index.item(),
        confidence.item(),
        probabilities
    )


# ============================================================
# UNCERTAINTY
# ============================================================

def uncertainty_score(probabilities):

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

    normalized = (
        entropy / max_entropy
        if max_entropy > 0
        else 0
    )

    return (
        entropy,
        normalized * 100
    )


# ============================================================
# UPLOAD AREA
# ============================================================

st.markdown(
    """
    <div class="glass">

        <div class="upload-title">
            🔬 Analyze a tomato leaf
        </div>

        <div class="upload-subtitle">
            Upload a clear photo. For best results,
            keep the leaf well lit and visible.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drop your image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    image_col, result_col = st.columns(
        [0.9, 1.1],
        gap="large"
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with image_col:

        st.markdown(
            """
            <div class="glass">

                <h3>Uploaded leaf</h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        analyze = st.button(
            "🍅  ANALYZE WITH GAIA",
            use_container_width=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        if analyze:

            with st.spinner(
                "GAIA is examining the leaf..."
            ):

                (
                    index,
                    confidence,
                    probabilities
                ) = predict(image)


            predicted_class = CLASS_NAMES[
                index
            ]

            prediction_name = DISPLAY_NAMES.get(
                predicted_class,
                predicted_class
            )


            entropy, uncertainty = (
                uncertainty_score(
                    probabilities
                )
            )


            confidence_pct = (
                confidence * 100
            )


            is_uncertain = (

                confidence_pct < 60

                or

                uncertainty > 60
            )


            info = DISEASE_INFO.get(
                predicted_class,
                {}
            )


            severity = info.get(
                "severity",
                "Unknown"
            )


            severity_class = (
                "severity-high"
                if severity == "High"
                else
                "severity-moderate"
                if severity == "Moderate"
                else
                "severity-low"
            )


            status_html = (

                '<span class="uncertain">'
                '⚠ REVIEW IMAGE'
                '</span>'

                if is_uncertain

                else

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
                        {confidence_pct:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.progress(
                min(
                    max(
                        float(confidence),
                        0.0
                    ),
                    1.0
                )
            )


            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            m1, m2 = st.columns(2)


            with m1:

                st.markdown(
                    f"""
                    <div class="glass">

                        <div class="metric-label">
                            Uncertainty
                        </div>

                        <div class="big-number"
                             style="color:white;">
                            {uncertainty:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with m2:

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
                    unsafe_allow_html=True
                )


    # ========================================================
    # ANALYSIS RESULTS
    # ========================================================

    if analyze:

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # TOP PREDICTIONS
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="glass">

                <h2>Prediction breakdown</h2>

                <p style="color:rgba(255,255,255,0.7);">
                    GAIA's three strongest predictions for
                    this image.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        for rank, idx in enumerate(
            top_indices,
            1
        ):

            name = DISPLAY_NAMES.get(
                CLASS_NAMES[idx],
                CLASS_NAMES[idx]
            )

            probability = (
                probabilities[idx]
                * 100
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
                    unsafe_allow_html=True
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
                    unsafe_allow_html=True
                )


            st.progress(
                float(
                    probabilities[idx]
                )
            )


        # ----------------------------------------------------
        # RECOMMENDED ACTION
        # ----------------------------------------------------

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="result">

                <div class="result-label">
                    WHAT TO DO
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
                        "Consult a qualified agricultural professional."
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # HEALTHY RESULT
        # ----------------------------------------------------

        if predicted_class == "healthy":

            st.success(
                "🌱 GAIA did not detect one of the "
                "target tomato diseases in this image."
            )


        # ----------------------------------------------------
        # UNCERTAINTY MESSAGE
        # ----------------------------------------------------

        if is_uncertain:

            st.warning(
                "GAIA is uncertain about this image. "
                "Try another photograph with better lighting "
                "and a clearer view of the leaf."
            )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        with st.expander(
            "⚙ Advanced AI analysis"
        ):

            st.write(
                f"Model: `{MODEL_NAME}`"
            )

            st.write(
                f"Input resolution: "
                f"`{IMAGE_SIZE} × {IMAGE_SIZE}`"
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


# ============================================================
# NO IMAGE STATE
# ============================================================

else:

    st.markdown(
        """
        <br><br>

        <div class="glass"
             style="text-align:center; padding:55px 25px;">

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
        unsafe_allow_html=True
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

        <br>

        Results are intended to support agricultural
        decision-making and should not replace assessment
        by a qualified plant-health professional.

    </div>
    """,
    unsafe_allow_html=True
)
