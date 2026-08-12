import json
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
# Clean production Streamlit app
# ============================================================

st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIG
# ============================================================

HF_REPO_ID = "Makky07/gaiatomato07"

MODEL_FILENAME = "GAIA_TOMATO_VIT_BEST.pt"
CONFIG_FILENAME = "GAIA_TOMATO_CONFIG.json"

SUPABASE_BUCKET = "gaia-images"

# IMPORTANT:
# This is the EXACT table name you created in Supabase.
SUPABASE_TABLE = "GAIA Diagnosis Database"

ROOT_DIR = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SUPABASE SECRETS
# ============================================================

def get_secret(name):
    """
    Safely read a Streamlit secret.

    Put these in:
    Streamlit Cloud
    -> App
    -> Settings
    -> Secrets

    Example:

    SUPABASE_URL = "https://pelxleyfheicfccmprjm.supabase.co"
    SUPABASE_KEY = "your_publishable_key"
    """

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")


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
# MODEL CLASSES
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


DISEASE_INFO = {

    "Late_blight": {
        "description":
            "A destructive disease that can rapidly damage tomato leaves, stems and fruit.",
        "action":
            "Remove severely affected material, improve airflow and avoid prolonged leaf wetness.",
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
# SIMPLE UI
# ============================================================

st.title("🍅 GAIA Tomato AI")

st.caption(
    "AI-powered tomato crop health screening"
)

st.divider()

st.subheader("Know your crop. Grow with confidence.")

st.write(
    "Upload a tomato leaf image and GAIA will "
    "screen it for 11 tomato health and disease "
    "conditions using a Vision Transformer."
)


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

    st.exception(error)

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


    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# CHECKPOINT
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

        value = checkpoint.get(
            key
        )

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

                if new_key.startswith(
                    prefix
                ):

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

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORM
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
    ).to(
        DEVICE
    )

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

    needs_review = (
        confidence_pct < 60
        or uncertainty > 60
    )

    return (
        entropy,
        uncertainty,
        confidence_pct,
        needs_review,
    )


# ============================================================
# SUPABASE SAVE
# ============================================================

def save_to_supabase(
    uploaded_file,
    prediction,
    confidence,
    needs_human_review,
):

    supabase = get_supabase_client()

    now = datetime.now(
        timezone.utc
    )

    unique_id = str(
        uuid.uuid4()
    )

    original_name = (
        uploaded_file.name
        or "tomato.jpg"
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
            "The uploaded image contains no data."
        )


    content_type = (
        uploaded_file.type
        or "image/jpeg"
    )


    # --------------------------------------------------------
    # 1. STORAGE UPLOAD
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 2. DATABASE INSERT
    # --------------------------------------------------------

    record = {

        "image_path":
            storage_path,

        "crop":
            "tomato",

        "prediction":
            prediction,

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
            "Supabase accepted the request "
            "but returned no inserted row."
        )


    return {
        "storage_path":
            storage_path,

        "record":
            response.data[0],
    }


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(

    "Upload tomato leaf image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded is None:

    st.info(
        "Upload a tomato leaf photograph to begin."
    )

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


    st.subheader(
        "Uploaded image"
    )

    st.image(
        image,
        use_container_width=True,
    )


    analyze = st.button(
        "🍅 ANALYZE WITH GAIA",
        type="primary",
        use_container_width=True,
    )


    if analyze:

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            with st.spinner(
                "GAIA is analyzing the leaf..."
            ):

                model = load_model()

                (
                    idx,
                    confidence,
                    probabilities,
                ) = predict(
                    model,
                    image,
                )

        except Exception as error:

            st.error(
                "The AI model could not complete the analysis."
            )

            st.exception(error)

            st.stop()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        disease = CLASS_NAMES[
            idx
        ]

        display_name = DISPLAY_NAMES.get(
            disease,
            disease,
        )

        (
            entropy,
            uncertainty,
            confidence_pct,
            needs_human_review,
        ) = diagnostics(
            probabilities,
            confidence,
        )

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


        st.divider()

        st.subheader(
            "GAIA Detection"
        )

        st.success(
            display_name
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Confidence",
                f"{confidence_pct:.2f}%",
            )

        with col2:

            st.metric(
                "Uncertainty",
                f"{uncertainty:.2f}%",
            )

        with col3:

            st.metric(
                "Entropy",
                f"{entropy:.4f}",
            )


        st.progress(
            min(
                max(
                    confidence,
                    0.0,
                ),
                1.0,
            )
        )


        # ----------------------------------------------------
        # GUIDANCE
        # ----------------------------------------------------

        st.subheader(
            "🌱 Diagnostic guidance"
        )

        st.write(
            info["description"]
        )

        st.write(
            f"**Severity:** {info['severity']}"
        )

        st.write(
            f"**Recommended next step:** "
            f"{info['action']}"
        )


        if disease == "healthy":

            st.info(
                "GAIA did not detect one of its "
                "target tomato diseases in this image."
            )


        if needs_human_review:

            st.warning(
                "This is a lower-confidence result. "
                "For best results, try a clearer close-up "
                "with good lighting and the affected leaf "
                "occupying most of the image."
            )


        # ----------------------------------------------------
        # TOP PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "Top predictions"
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        for rank, class_index in enumerate(
            top_indices,
            1,
        ):

            prediction_name = DISPLAY_NAMES.get(
                CLASS_NAMES[class_index],
                CLASS_NAMES[class_index],
            )

            prediction_probability = float(
                probabilities[class_index]
            )

            st.write(
                f"**{rank}. "
                f"{prediction_name} — "
                f"{prediction_probability * 100:.2f}%**"
            )

            st.progress(
                prediction_probability
            )


        # ----------------------------------------------------
        # SUPABASE
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "☁️ Saving diagnosis"
        )

        try:

            save_result = save_to_supabase(

                uploaded_file=uploaded,

                prediction=disease,

                confidence=confidence,

                needs_human_review=
                    needs_human_review,
            )


            st.success(
                "Saved to Supabase ✓"
            )

            st.write(
                "Image path:"
            )

            st.code(
                save_result["storage_path"]
            )

            st.write(
                "Database row:"
            )

            st.json(
                save_result["record"]
            )


        except Exception as error:

            st.error(
                "The diagnosis was generated, "
                "but saving to Supabase failed."
            )

            st.error(
                str(error)
            )

            st.info(
                "If the image appears in Storage but "
                "there is no database row, the database "
                "INSERT/RLS policy is the part that needs "
                "attention."
            )


        # ----------------------------------------------------
        # ADVANCED
        # ----------------------------------------------------

        with st.expander(
            "⚙ Advanced AI information"
        ):

            st.write(
                f"Model: {MODEL_NAME}"
            )

            st.write(
                f"Input size: "
                f"{IMAGE_SIZE} × {IMAGE_SIZE}"
            )

            st.write(
                f"Device: {DEVICE}"
            )

            st.write(
                f"Classes: {NUM_CLASSES}"
            )

            st.write(
                f"Entropy: {entropy:.6f}"
            )

            st.write(
                f"Normalized uncertainty: "
                f"{uncertainty:.2f}%"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GAIA Tomato AI — AI-assisted tomato crop "
    "health screening. Results should not replace "
    "assessment by a qualified plant-health professional."
)
