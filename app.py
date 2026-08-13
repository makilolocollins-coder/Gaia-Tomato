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


st.set_page_config(
    page_title="GAIA Tomato AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HF_REPO_ID = "Makky07/gaiatomato07"
MODEL_FILENAME = "GAIA_TOMATO_VIT_BEST.pt"
CONFIG_FILENAME = "GAIA_TOMATO_CONFIG.json"

ROOT_DIR = Path(__file__).resolve().parent
BACKGROUND_FILES = [
    ROOT_DIR / "tomato_farmer_africa.jpg",
    ROOT_DIR / "assets" / "tomato_farmer_africa.jpg",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASSES = [
    "Late_blight", "healthy", "Early_blight", "Septoria_leaf_spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Bacterial_spot", "Target_Spot",
    "Tomato_mosaic_virus", "Leaf_Mold",
    "Spider_mites_Two_spotted_spider_mite", "Powdery_Mildew",
]

DISPLAY_NAMES = {
    "Late_blight": "Late Blight",
    "healthy": "Healthy",
    "Early_blight": "Early Blight",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Bacterial_spot": "Bacterial Spot",
    "Target_Spot": "Target Spot",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Leaf_Mold": "Leaf Mold",
    "Spider_mites_Two_spotted_spider_mite": "Two-Spotted Spider Mites",
    "Powdery_Mildew": "Powdery Mildew",
}

DISEASE_INFO = {
    "Late_blight": {
        "description": "A destructive disease that can rapidly damage tomato leaves, stems and fruit.",
        "action": "Remove severely affected material, improve airflow and avoid prolonged leaf wetness. Follow locally approved disease-management practices.",
        "severity": "High",
    },
    "Early_blight": {
        "description": "A fungal disease commonly associated with dark lesions and concentric rings on older leaves.",
        "action": "Remove affected leaves, improve field sanitation and avoid prolonged leaf wetness.",
        "severity": "Moderate",
    },
    "Septoria_leaf_spot": {
        "description": "A fungal disease producing numerous small spots, often beginning on lower leaves.",
        "action": "Remove infected foliage, improve sanitation and reduce prolonged moisture on leaves.",
        "severity": "Moderate",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A viral disease commonly associated with leaf curling, yellowing and reduced plant growth.",
        "action": "Monitor and manage whiteflies, remove severely affected plants and consider resistant varieties.",
        "severity": "High",
    },
    "Bacterial_spot": {
        "description": "A bacterial disease that can produce dark spots on leaves, stems and fruit.",
        "action": "Maintain field sanitation, avoid handling wet plants and remove severely infected material.",
        "severity": "Moderate",
    },
    "Target_Spot": {
        "description": "A fungal disease characterized by circular target-like lesions.",
        "action": "Improve airflow, remove affected leaves and follow locally approved disease-management practices.",
        "severity": "Moderate",
    },
    "Tomato_mosaic_virus": {
        "description": "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant growth.",
        "action": "Remove severely affected plants and disinfect tools to reduce mechanical spread.",
        "severity": "High",
    },
    "Leaf_Mold": {
        "description": "A fungal disease associated with humid conditions and poor ventilation.",
        "action": "Improve ventilation, reduce humidity and minimize prolonged moisture on leaves.",
        "severity": "Moderate",
    },
    "Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites feed on tomato leaves and can cause stippling, yellowing and plant stress.",
        "action": "Inspect the underside of leaves and use an appropriate locally approved management strategy if infestation is confirmed.",
        "severity": "Moderate",
    },
    "Powdery_Mildew": {
        "description": "A fungal disease characterized by powdery white growth on plant surfaces.",
        "action": "Improve airflow, remove severely affected foliage and follow locally approved treatment recommendations.",
        "severity": "Moderate",
    },
    "healthy": {
        "description": "GAIA did not detect one of the target tomato diseases in the uploaded image.",
        "action": "Continue crop monitoring and maintain good irrigation, nutrition and field hygiene.",
        "severity": "Low",
    },
}


def get_background_uri():
    for path in BACKGROUND_FILES:
        if path.exists():
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
            except Exception:
                pass
    return None


bg = get_background_uri()
background_css = (
    f'.stApp {{background-image:linear-gradient(rgba(3,18,8,.86),rgba(5,25,10,.80)),url("{bg}");background-size:cover;background-position:center;background-attachment:fixed;}}'
    if bg else
    ".stApp {background:linear-gradient(135deg,#06140a,#123b1d,#071b0c);}"
)

st.markdown(
    f"""
    <style>
    {background_css}
    #MainMenu,footer{{visibility:hidden}}
    header{{background:transparent!important}}
    .block-container{{max-width:1180px;padding-top:1rem;padding-bottom:4rem}}
    .nav{{display:flex;justify-content:space-between;align-items:center;padding:15px 5px 20px;color:white}}
    .brand{{font-size:28px;font-weight:900}} .brand span,.hero h1 span{{color:#91e66d}}
    .nav-right{{color:rgba(255,255,255,.65);font-size:12px;letter-spacing:1.5px}}
    .hero{{text-align:center;color:white;padding:50px 15px 55px}}
    .badge{{display:inline-block;padding:8px 16px;border-radius:30px;background:rgba(145,230,109,.15);border:1px solid rgba(145,230,109,.35);color:#a8ee89;font-size:13px;font-weight:800;margin-bottom:20px}}
    .hero h1{{font-size:clamp(42px,7vw,75px);line-height:.98;letter-spacing:-3px;margin:0;font-weight:900}}
    .hero p{{max-width:700px;margin:24px auto 0;font-size:18px;line-height:1.6;color:rgba(255,255,255,.82)}}
    .glass{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.17);border-radius:25px;padding:28px;backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);box-shadow:0 25px 70px rgba(0,0,0,.22);color:white}}
    .glass h2,.glass h3{{color:white}}
    .result{{background:rgba(255,255,255,.97);border-radius:25px;padding:30px;color:#142519;box-shadow:0 25px 70px rgba(0,0,0,.28)}}
    .label{{color:#65806c;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px}}
    .diagnosis{{color:#142519;font-size:35px;font-weight:900;margin:6px 0 15px}}
    .confidence{{color:#18351d;font-size:40px;font-weight:900}}
    .warning-box{{background:#fff3d6;border-left:5px solid #d89b19;padding:18px;border-radius:12px;color:#654600;margin-top:18px}}
    .success-box{{background:#e7f8e2;border-left:5px solid #4b9b3f;padding:18px;border-radius:12px;color:#245d20;margin-top:18px}}
    [data-testid="stFileUploader"]{{background:rgba(255,255,255,.07);border:2px dashed rgba(142,226,107,.55);border-radius:20px;padding:10px}}
    .stButton>button{{width:100%;border-radius:14px;border:none;background:#83d95f;color:#102411;font-weight:900;padding:14px 20px;font-size:16px}}
    .stButton>button:hover{{background:#a0ed7e}}
    .footer{{text-align:center;color:rgba(255,255,255,.60);padding:45px 10px 15px;font-size:13px}}
    @media(max-width:768px){{.nav-right{{display:none}}.hero{{padding:35px 10px 45px}}.hero h1{{letter-spacing:-2px}}.result,.glass{{padding:21px}}.diagnosis{{font-size:28px}}}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nav">
      <div class="brand">GAIA<span>🍅</span></div>
      <div class="nav-right">TOMATO HEALTH INTELLIGENCE</div>
    </div>
    <div class="hero">
      <div class="badge">✦ AI-POWERED CROP HEALTH</div>
      <h1>Know your crop.<br><span>Grow with confidence.</span></h1>
      <p>Upload a tomato leaf image and GAIA will screen it for 11 tomato health and disease conditions using a Vision Transformer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_config():
    p = hf_hub_download(repo_id=HF_REPO_ID, filename=CONFIG_FILENAME)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    CONFIG = load_config()
except Exception as e:
    st.error("GAIA could not load the model configuration.")
    st.code(str(e))
    st.stop()

MODEL_NAME = CONFIG.get("model", "vit_small_patch16_224")
IMAGE_SIZE = int(CONFIG.get("image_size", 224))
CLASS_NAMES = CONFIG.get("classes", DEFAULT_CLASSES)
NUM_CLASSES = int(CONFIG.get("num_classes", len(CLASS_NAMES)))

if len(CLASS_NAMES) != NUM_CLASSES:
    st.error("Model configuration error: class count does not match class list.")
    st.stop()


class GaiaTomatoModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0
        )
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(d, 1024),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported model checkpoint format.")
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        k = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "net."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
        cleaned[k] = value
    return cleaned


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=MODEL_FILENAME
    )
    model = GaiaTomatoModel(MODEL_NAME, NUM_CLASSES)
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        clean_state_dict(extract_state_dict(checkpoint)),
        strict=True,
    )
    model.to(DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def predict(model, image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    return index.item(), float(confidence.item()), probs.cpu().numpy()


def diagnostics(probs, confidence):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(p)))
    uncertainty = entropy / max_entropy * 100 if max_entropy > 0 else 0.0
    confidence_pct = confidence * 100
    uncertain = confidence_pct < 60 or uncertainty > 60
    status = (
        "Needs review" if uncertain
        else "High confidence" if confidence_pct >= 90 and uncertainty < 25
        else "Moderate confidence"
    )
    return entropy, uncertainty, confidence_pct, uncertain, status


st.markdown(
    """
    <div class="glass">
      <h2>🔬 Analyze a tomato leaf</h2>
      <p>Upload a clear JPG, JPEG or PNG photograph. Good lighting, focus and a visible leaf generally provide better screening conditions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.markdown(
            '<div class="glass"><h3>Uploaded image</h3></div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        analyze = st.button("🍅 ANALYZE WITH GAIA", use_container_width=True)

    with right:
        if not analyze:
            st.markdown(
                """
                <div class="result">
                  <div class="label">READY</div>
                  <div class="diagnosis">Ready to analyze</div>
                  <p>Click <b>Analyze with GAIA</b> to run the trained Vision Transformer.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                with st.spinner("GAIA is analyzing the leaf..."):
                    model = load_model()
                    idx, conf, probs = predict(model, image)

                disease = CLASS_NAMES[idx]
                name = DISPLAY_NAMES.get(disease, disease)
                entropy, uncertainty, confidence_pct, uncertain, status = diagnostics(probs, conf)
                info = DISEASE_INFO.get(
                    disease,
                    {
                        "description": "GAIA detected a target condition.",
                        "action": "Consult a qualified plant-health professional.",
                        "severity": "Unknown",
                    },
                )

                box = (
                    '<div class="warning-box">⚠ <b>Review recommended.</b><br>GAIA is not sufficiently certain about this image.</div>'
                    if uncertain else
                    '<div class="success-box">✓ <b>Prediction stable.</b><br>The model produced a relatively confident prediction.</div>'
                )

                st.markdown(
                    f"""
                    <div class="result">
                      <div class="label">GAIA DETECTION</div>
                      <div class="diagnosis">{name}</div>
                      <div class="label">CONFIDENCE</div>
                      <div class="confidence">{confidence_pct:.2f}%</div>
                      {box}
                      <br>
                      <b>Status:</b> {status}<br>
                      <b>Severity:</b> {info["severity"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(max(conf, 0.0), 1.0))

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{confidence_pct:.2f}%")
                c2.metric("Uncertainty", f"{uncertainty:.2f}%")
                c3.metric("Entropy", f"{entropy:.4f}")

                st.markdown("### Top predictions")
                for rank, i in enumerate(np.argsort(probs)[::-1][:3], 1):
                    n = DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i])
                    pct = float(probs[i]) * 100
                    st.write(f"**{rank}. {n} — {pct:.2f}%**")
                    st.progress(float(probs[i]))

                st.markdown("### 🌱 Diagnostic guidance")
                st.markdown(
                    f"""
                    <div class="result">
                      <div class="label">WHAT GAIA DETECTED</div>
                      <h2>{name}</h2>
                      <p>{info["description"]}</p>
                      <hr>
                      <div class="label">RECOMMENDED NEXT STEP</div>
                      <p>{info["action"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if disease == "healthy":
                    st.success("🌱 GAIA did not detect one of its target tomato diseases in this image.")

                if uncertain:
                    st.warning(
                        "Try a clearer close-up with good lighting and the affected leaf occupying most of the image."
                    )

                with st.expander("⚙ Advanced AI information"):
                    st.write(f"Model: `{MODEL_NAME}`")
                    st.write(f"Input size: `{IMAGE_SIZE} × {IMAGE_SIZE}`")
                    st.write(f"Device: `{DEVICE}`")
                    st.write(f"Classes: `{NUM_CLASSES}`")
                    st.write(f"Entropy: `{entropy:.6f}`")
                    st.write(f"Normalized uncertainty: `{uncertainty:.2f}%`")

            except Exception as e:
                st.error("GAIA could not complete the analysis.")
                st.code(str(e))
else:
    st.markdown(
        """
        <div class="glass" style="text-align:center;margin-top:35px;">
          <div style="font-size:52px;">🍃</div>
          <h2>Your crop health starts here</h2>
          <p>Upload a tomato leaf photograph above to begin AI-assisted disease screening.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
      <strong>GAIA Tomato AI</strong><br><br>
      AI-assisted tomato crop health screening.<br>
      Results are intended to support agricultural decision-making and should not replace assessment by a qualified plant-health professional.
    </div>
    """,
    unsafe_allow_html=True,
)
