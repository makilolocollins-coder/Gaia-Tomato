import json
import base64
import uuid
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

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
# Production Streamlit application
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html: str):
    """
    Safely render multiline HTML without Python indentation
    turning it into a Markdown code block.
    """
    st.markdown(
        dedent(html).strip(),
        unsafe_allow_html=True,
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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SUPABASE CONFIGURATION
# ============================================================

SUPABASE_URL = None
SUPABASE_KEY = None

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass


SUPABASE_BUCKET = "gaia-images"

# IMPORTANT:
# This is the exact table name you created in Supabase.
SUPABASE_TABLE = "GAIA Diagnosis Database"


# ============================================================
# SUPABASE CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_supabase_client():

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is missing from Streamlit Secrets."
        )

    if not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_KEY is missing from Streamlit Secrets."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
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
# DIAGNOSTIC KNOWLEDGE BASE
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
# BACKGROUND
# ============================================================

def get_background_uri():

    for path in BACKGROUND_FILES:

        if path.exists():

            try:

                encoded = base64.b64encode(
                    path.read_bytes()
                ).decode("utf-8")

                return (
                    "data:image/jpeg;base64,"
                    + encoded
                )

            except Exception:
                pass

    return None


background_uri = get_background_uri()


if background_uri:

    background_css = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(3,18,8,.86),
                rgba(5,25,10,.80)
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


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    f"""
    <style>

    {background_css}

    #MainMenu,
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
        font-size: 28px;
        font-weight: 900;
        color: white;
    }}

    .brand span {{
        color: #91e66d;
    }}

    .nav-right {{
        color: rgba(255,255,255,.70);
        font-size: 12px;
        letter-spacing: 1.5px;
        font-weight: 700;
    }}

    .hero {{
        text-align: center;
        color: white;
        padding: 50px 15px 55px;
    }}

    .badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(145,230,109,.15);
        border: 1px solid rgba(145,230,109,.35);
        color: #a8ee89;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 20px;
    }}

    .hero h1 {{
        font-size: clamp(42px,7vw,75px);
        line-height: .98;
        letter-spacing: -3px;
        margin: 0;
        font-weight: 900;
        color: white;
    }}

    .hero h1 span {{
        color: #91e66d;
    }}

    .hero p {{
        max-width: 700px;
        margin: 24px auto 0;
        font-size: 18px;
        line-height: 1.6;
        color: rgba(255,255,255,.82);
    }}

    .glass {{
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.17);
        border-radius: 25px;
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 25px 70px rgba(0,0,0,.22);
        color: white;
    }}

    .glass h2,
    .glass h3 {{
        color: white;
    }}

    .result {{
        background: rgba(255,255,255,.97);
        border-radius: 25px;
        padding: 30px;
        color: #142519;
        box-shadow: 0 25px 70px rgba(0,0,0,.28);
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

    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,.07);
        border: 2px dashed rgba(142,226,107,.55);
        border-radius: 20px;
        padding: 10px;
    }}

    .stButton>button {{
        width: 100%;
        border-radius: 14px;
        border: none;
        background: #83d95f;
        color: #102411;
        font-weight: 900;
        padding: 14px 20px;
        font-size: 16px;
    }}

    .stButton>button:hover {{
        background: #a0ed7e;
    }}

    .footer {{
        text-align: center;
        color: rgba(255,255,255,.60);
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

render_html("""
<div class="nav">

    <div class="brand">
        GAIA<span>🍅</span>
    </div>

    <div class="nav-right">
        TOMATO HEALTH INTELLIGENCE
    </div>

</div>
""")


# ============================================================
# HERO
# ============================================================

render_html("""
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
""")


# ============================================================
# MODEL CONFIGURATION
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
        "class count does not match class list."
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

    def forward(self, x):

        return self.head(
            self.backbone(x)
        )


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

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):

        value = checkpoint.get(key)

        if isinstance(
            value,
            dict,
        ):

            return value

    return checkpoint


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:

            changed = False

            for prefix in (
                "module.",
                "model.",
                "net.",
            ):

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

        logits = model(tensor)

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
            * np.log(probabilities)
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
# SUPABASE SAVE
# ============================================================

def save_diagnosis_to_supabase(
    uploaded_file,
    crop,
    prediction,
    confidence,
    needs_human_review,
):

    try:

        supabase = get_supabase_client()

        now = datetime.now(
            timezone.utc
        )

        unique_id = str(
            uuid.uuid4()
        )

        original_name = (
            uploaded_file.name
            or "uploaded_image.jpg"
        )

        extension = (
            Path(original_name)
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

        date_folder = now.strftime(
            "%Y/%m/%d"
        )

        storage_path = (
            f"diagnoses/"
            f"{date_folder}/"
            f"{unique_id}{extension}"
        )

        image_bytes = (
            uploaded_file.getvalue()
        )

        if not image_bytes:

            raise RuntimeError(
                "Uploaded image contains no data."
            )

        content_type = (
            uploaded_file.type
            or "image/jpeg"
        )

        # ----------------------------------------------------
        # STORAGE UPLOAD
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
                str(crop),

            "prediction":
                str(prediction),

            "confidence":
                float(confidence),

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

        if not response.data:

            raise RuntimeError(
                "Supabase database insert "
                "returned no record."
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

render_html("""
<div class="glass">

    <h2>
        🔬 Analyze a tomato leaf
    </h2>

    <p>
        Upload a clear JPG, JPEG, PNG or WEBP photograph.
        Good lighting, focus and a visible leaf
        generally provide better screening conditions.
    </p>

</div>
""")


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
# IMAGE ANALYSIS
# ============================================================

if uploaded is not None:

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

    with left:

        render_html("""
        <div class="glass">
            <h3>Uploaded image</h3>
        </div>
        """)

        st.image(
            image,
            use_container_width=True,
        )

        analyze = st.button(
            "🍅 ANALYZE WITH GAIA",
            use_container_width=True,
        )

    with right:

        if not analyze:

            render_html("""
            <div class="result">

                <div class="label">
                    READY
                </div>

                <div class="diagnosis">
                    Ready to analyze
                </div>

                <p>
                    Click <b>Analyze with GAIA</b>
                    to run the trained
                    Vision Transformer.
                </p>

            </div>
            """)

        else:

            try:

                # ====================================================
                # MODEL PREDICTION
                # ====================================================

                with st.spinner(
                    "GAIA is analyzing the leaf..."
                ):

                    model = load_model()

                    idx, conf, probs = predict(
                        model,
                        image,
                    )

                disease = CLASS_NAMES[idx]

                name = DISPLAY_NAMES.get(
                    disease,
                    disease,
                )

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
                # HUMAN REVIEW FLAG
                # ====================================================

                needs_human_review = (
                    confidence_pct < 60
                    or uncertainty > 60
                )

                # ====================================================
                # DIAGNOSTIC INFO
                # ====================================================

                info = DISEASE_INFO.get(

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
                # RESULT BOX
                # ====================================================

                if uncertain:

                    box = (
                        '<div class="warning-box">'
                        '⚠ <b>Lower-confidence result.</b><br>'
                        'For best results, try a clearer close-up '
                        'with good lighting and the affected leaf '
                        'occupying most of the image.'
                        '</div>'
                    )

                else:

                    box = (
                        '<div class="success-box">'
                        '✓ <b>Prediction generated.</b><br>'
                        'GAIA produced a relatively confident '
                        'screening result.'
                        '</div>'
                    )

                render_html(
                    f"""
                    <div class="result">

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

                        {box}

                        <br>

                        <b>Severity:</b>
                        {info["severity"]}

                    </div>
                    """
                )

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

                c1, c2, c3 = st.columns(3)

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

                top_indices = np.argsort(
                    probs
                )[::-1][:3]

                for rank, i in enumerate(
                    top_indices,
                    1,
                ):

                    prediction_name = (
                        DISPLAY_NAMES.get(
                            CLASS_NAMES[i],
                            CLASS_NAMES[i],
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

                render_html(
                    f"""
                    <div class="result">

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
                    """
                )

                # ====================================================
                # HEALTHY MESSAGE
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
                        "Try a clearer close-up with good "
                        "lighting and the affected leaf "
                        "occupying most of the image."
                    )

                # ====================================================
                # DATABASE SAVE RESULT
                # ====================================================

                if save_result["success"]:

                    # Do NOT show internal database information
                    # to the farmer.

                    pass

                else:

                    # Farmer sees a friendly message.
                    # Technical error is kept out of the UI.

                    st.warning(
                        "GAIA completed the analysis, "
                        "but the result could not be saved "
                        "to the diagnostic database."
                    )

                # ====================================================
                # ADVANCED AI INFORMATION
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
                    "Please try the image again."
                )


# ============================================================
# NO IMAGE
# ============================================================

else:

    render_html("""
    <div class="glass"
         style="text-align:center;margin-top:35px;">

        <div style="font-size:52px;">
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
    """)


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

    <strong>
        GAIA Tomato AI
    </strong>

    <br><br>

    AI-assisted tomato crop health screening.

    <br><br>

    Results are intended to support
    agricultural decision-making and
    should not replace assessment by a
    qualified plant-health professional.

</div>
""")
