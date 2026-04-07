"""
BoneScan AI — Production App v3
PIPELINE (exact match to your training notebooks):
  Input X-ray
    → Stage 2: Faster R-CNN (Detectron2, conf≥0.30)
        ├─ No tumor → Final: NORMAL  [STOP — no Stage 3]
        └─ Tumor    → Crop ROI (35% padding)
             → Stage 3: ConvNeXt-Tiny-SE → Benign / Malignant
                  → Grad-CAM++ (pytorch-grad-cam, features[-1][-1], γ=0.7)

ARCHITECTURE (exact from explainability notebook Doc4):
  head = Flatten → LayerNorm(768) → Dropout(0.5) → Linear(768,256)
                → GELU → Dropout(0.4) → Linear(256,2)

PREPROCESSING (exact from Doc4 _autocontrast + _pad_resize):
  autocontrast(L, cutoff=1) → letterbox 256×256 → ImageNet normalize

GRAD-CAM++ (exact from Doc4 compute_gradcam_pp):
  target_layers = [model.features[-1][-1]]
  GradCAMPlusPlus → cam[0] → np.power(raw, 0.7)
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image, ImageOps

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

st.set_page_config(
    page_title="BoneScan AI", page_icon="🦴",
    layout="wide", initial_sidebar_state="expanded",
)

# ─── Constants (exact from training notebooks) ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification (Doc4 Config)
IMAGE_SIZE   = 256
CLF_MEAN     = [0.485, 0.456, 0.406]
CLF_STD      = [0.229, 0.224, 0.225]
SE_REDUCTION = 16
CLF_CLASSES  = ["Benign", "Malignant"]   # 0=Benign, 1=Malignant

# Detection (Doc3 pipeline v4)
DET_CONF     = 0.30   # CONF_THRESHOLD = 0.30 (visualisation threshold)
DET_NMS      = 0.40   # NMS_THRESH_TEST = 0.4
DET_PADDING  = 0.35   # PADDING_RATIO = 0.35
D2_CONFIG    = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

DET_PATH = r"C:\Users\mohym\Music\@@@@Bone Tumor all final things\Interface\det_model_best.pth"
CLF_PATH = r"C:\Users\mohym\Music\@@@@Bone Tumor all final things\Interface\final_best.pth"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box}
.main{background:#F4F7FB}
.block-container{padding-top:1.2rem!important}
[data-testid="stSidebar"]{background:linear-gradient(175deg,#0A1628,#0F2040,#112244);border-right:1px solid #1E3A5F}
[data-testid="stSidebar"] *{color:#C8D8F0!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#7BC8E2!important}
[data-testid="stSidebar"] hr{border-color:#1E3A5F!important}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]{background:rgba(0,180,216,.08)!important;border:1.5px dashed #00B4D8!important;border-radius:12px!important}
[data-testid="stSidebar"] .stButton>button{background:linear-gradient(135deg,#0077B6,#00B4D8)!important;color:#fff!important;border:none!important;border-radius:12px!important;font-weight:700!important;padding:.65rem 1rem!important;box-shadow:0 4px 16px rgba(0,180,216,.35)!important}
[data-testid="stSidebar"] .stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(0,180,216,.5)!important}
.hero{background:linear-gradient(135deg,#0A1628,#0D2137,#0A3254);border-radius:20px;padding:26px 34px 20px;margin-bottom:18px;border:1px solid #1E3A5F;box-shadow:0 8px 32px rgba(0,0,0,.2);position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:0;right:0;width:280px;height:100%;background:radial-gradient(ellipse at right,rgba(0,180,216,.12),transparent 70%)}
.hero-title{font-size:1.9em;font-weight:900;color:#fff;margin:0 0 3px}
.hero-title span{color:#00B4D8}
.hero-sub{font-size:.88em;color:#7BC8E2;margin:0}
.hbadge{background:rgba(0,180,216,.14);color:#7BC8E2;border:1px solid rgba(0,180,216,.3);border-radius:20px;padding:3px 11px;font-size:.72em;font-weight:600;display:inline-block;margin:2px 3px 0 0}
.sec-head{display:flex;align-items:center;gap:9px;font-size:.92em;font-weight:700;color:#0A1628;margin:18px 0 9px}
.sec-head::before{content:'';display:block;width:4px;height:19px;background:linear-gradient(180deg,#00B4D8,#0077B6);border-radius:4px}
.card-wrap{border-radius:20px;padding:26px 30px;text-align:center;position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.12);margin:4px 0 16px}
.card-normal{background:linear-gradient(135deg,#D1FAE5,#A7F3D0);border:2px solid #10B981}
.card-benign{background:linear-gradient(135deg,#FEF3C7,#FDE68A);border:2px solid #F59E0B}
.card-malignant{background:linear-gradient(135deg,#FEE2E2,#FECACA);border:2px solid #EF4444}
.card-icon{font-size:2.6em;margin-bottom:2px}
.card-lbl{font-size:.73em;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#374151}
.card-pred{font-size:2.4em;font-weight:900;letter-spacing:4px;color:#111827;margin:3px 0 7px}
.card-conf{font-size:.83em;color:#374151;font-weight:500}
.card-msg{margin-top:10px;font-size:.79em;color:#4B5563;font-style:italic;border-top:1px solid rgba(0,0,0,.1);padding-top:8px}
.mtile{background:#fff;border-radius:14px;padding:14px 18px;margin-bottom:9px;box-shadow:0 2px 10px rgba(0,0,0,.07);border-left:4px solid #00B4D8}
.mtile.orange{border-left-color:#F59E0B}.mtile.red{border-left-color:#EF4444}.mtile.green{border-left-color:#10B981}
.mlbl{font-size:.68em;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:1px}
.mval{font-size:1.45em;font-weight:800;color:#111827;margin-top:2px}
.ipanel{background:#fff;border-radius:14px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,.08);margin-bottom:10px}
.iptitle{font-size:.72em;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;text-align:center}
.prow{background:#fff;border-radius:11px;padding:12px 16px;margin-bottom:6px;box-shadow:0 1px 5px rgba(0,0,0,.06)}
.pname{font-size:.83em;font-weight:600;color:#374151}
.ppct{float:right;font-size:.83em;font-weight:800}
.ptrack{background:#F3F4F6;border-radius:6px;height:9px;margin-top:5px;overflow:hidden}
.pfill{height:100%;border-radius:6px}
.info-panel{background:#EFF6FF;border-left:4px solid #3B82F6;border-radius:11px;padding:12px 16px;margin:9px 0;font-size:.86em;color:#1E3A5F;line-height:1.55}
.warn-panel{background:#FFFBEB;border-left:4px solid #F59E0B;border-radius:11px;padding:12px 16px;margin:9px 0;font-size:.86em;color:#78350F;line-height:1.55}
.ok-panel{background:#ECFDF5;border-left:4px solid #10B981;border-radius:11px;padding:12px 16px;margin:9px 0;font-size:.86em;color:#064E3B;line-height:1.55}
.step-card{background:#fff;border-radius:16px;padding:20px 16px;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,.08);border-top:4px solid #00B4D8}
.sicon{font-size:1.9em;margin-bottom:8px}
.sname{font-size:.86em;font-weight:700;color:#0A1628;margin-bottom:4px}
.sdesc{font-size:.78em;color:#6B7280;line-height:1.5}
.sdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.dg{background:#10B981;box-shadow:0 0 5px #10B981}.dr{background:#EF4444;box-shadow:0 0 5px #EF4444}
[data-testid="stImage"] img{border-radius:11px!important}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE — exact from Doc4 Section 3
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class TumorClassifierConvNeXtTinySE(nn.Module):
    """
    Exact architecture from Doc4 (explainability notebook):
        self.head = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(768),
            nn.Dropout(0.5), nn.Linear(768, 256), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(256, n))
    Grad-CAM++ target: model.features[-1][-1]
    """
    def __init__(self, n=2, r=16):
        super().__init__()
        bb = torchvision.models.convnext_tiny(weights=None)
        self.features = bb.features
        self.se   = SEBlock(768, r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, n),
        )
    def forward(self, x):
        x = self.pool(self.se(self.features(x)))
        return self.head(x)


class _D2CompatPredictor(nn.Module):
    """Fix D2 bbox_pred shape: [fg*4] → prepend bg zeros → [(fg+1)*4]."""
    def __init__(self, in_ch, n_fg):
        super().__init__()
        self.cls_score = nn.Linear(in_ch, n_fg + 1)
        self.bbox_pred = nn.Linear(in_ch, n_fg * 4)
    def forward(self, x):
        cls  = self.cls_score(x)
        bbox = self.bbox_pred(x)
        bg   = torch.zeros(bbox.shape[0], 4, device=bbox.device, dtype=bbox.dtype)
        return cls, torch.cat([bg, bbox], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_prefix(sd):
    return {re.sub(r"^(model\.module\.|module\.model\.|model\.|module\.)", "", k): v
            for k, v in sd.items()}


@st.cache_resource(show_spinner=False)
def load_models():
    det_model = clf_model = None
    det_backend = None
    errors = []

    # ── Detection ──────────────────────────────────────────────────────────
    if not os.path.exists(DET_PATH):
        errors.append(f"Detection checkpoint not found:<br><code>{DET_PATH}</code>")
    else:
        try:  # Detectron2 (exact training framework)
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            cfg_d2 = get_cfg()
            cfg_d2.merge_from_file(model_zoo.get_config_file(D2_CONFIG))
            cfg_d2.MODEL.WEIGHTS                     = DET_PATH
            cfg_d2.MODEL.DEVICE                      = str(DEVICE)
            cfg_d2.MODEL.ROI_HEADS.NUM_CLASSES       = 1
            cfg_d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DET_CONF
            cfg_d2.MODEL.ROI_HEADS.NMS_THRESH_TEST   = DET_NMS
            det_model   = DefaultPredictor(cfg_d2)
            det_backend = "detectron2"
        except ImportError:
            try:  # torchvision fallback with D2 shape fix
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                raw   = torch.load(DET_PATH, map_location=DEVICE, weights_only=False)
                state = _strip_prefix(raw.get("model", raw))
                bk = next((k for k in state if "box_predictor.bbox_pred.weight" in k), None)
                ck = next((k for k in state if "box_predictor.cls_score.weight"  in k), None)
                n_fg  = (state[ck].shape[0] - 1) if ck else 1
                is_d2 = (state[bk].shape[0] == n_fg * 4) if bk else True
                mdl   = fasterrcnn_resnet50_fpn(weights="DEFAULT")
                in_f  = mdl.roi_heads.box_predictor.cls_score.in_features
                if is_d2:
                    mdl.roi_heads.box_predictor = _D2CompatPredictor(in_f, n_fg)
                else:
                    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                    mdl.roi_heads.box_predictor = FastRCNNPredictor(in_f, n_fg + 1)
                mdl.load_state_dict(state, strict=False)
                mdl.roi_heads.score_thresh = DET_CONF
                mdl.to(DEVICE).eval()
                det_model   = mdl
                det_backend = "torchvision"
            except Exception as e:
                errors.append(f"Detection load failed: {e}")
        except Exception as e:
            errors.append(f"Detectron2 load failed: {e}")

    # ── Classification ─────────────────────────────────────────────────────
    if not os.path.exists(CLF_PATH):
        errors.append(f"Classifier not found:<br><code>{CLF_PATH}</code>")
    else:
        try:
            raw   = torch.load(CLF_PATH, map_location=DEVICE, weights_only=False)
            state = _strip_prefix(raw.get("model_state_dict", raw.get("state_dict", raw)))
            clf_model = TumorClassifierConvNeXtTinySE(n=2, r=SE_REDUCTION)
            try:
                clf_model.load_state_dict(state, strict=True)
            except Exception:
                clf_model.load_state_dict(state, strict=False)
            clf_model.to(DEVICE).eval()
        except Exception as e:
            errors.append(f"Classifier load failed: {e}")
            clf_model = None

    return det_model, clf_model, det_backend, errors


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — exact from Doc4 Section 5
# _autocontrast: convert L → autocontrast(cutoff=1) → RGB
# _pad_resize:   aspect-ratio letterbox with black padding
# ═══════════════════════════════════════════════════════════════════════════════

def _autocontrast(img):
    img = img.convert('L')
    img = ImageOps.autocontrast(img, cutoff=1)
    return img.convert('RGB')

def _pad_resize(img, sz=256):
    w, h   = img.size
    r      = min(sz / w, sz / h)
    nw, nh = int(w * r), int(h * r)
    img    = img.resize((nw, nh), Image.BILINEAR)
    out    = Image.new('RGB', (sz, sz), 0)
    out.paste(img, ((sz - nw) // 2, (sz - nh) // 2))
    return out

_TFM = T.Compose([T.ToTensor(), T.Normalize(CLF_MEAN, CLF_STD)])

def preprocess_roi(roi_pil):
    """Exact pipeline from Doc4: autocontrast → letterbox → normalize."""
    img = _autocontrast(roi_pil)
    img = _pad_resize(img, IMAGE_SIZE)
    return _TFM(img).unsqueeze(0).to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DETECTION
# Returns (boxes[N,4], scores[N], is_tumor bool)
# is_tumor=False  →  NORMAL  →  pipeline stops, no Stage 3
# ═══════════════════════════════════════════════════════════════════════════════

def run_detection(image_np, det_model, det_backend):
    if det_model is None:
        return np.empty((0,4)), np.empty((0,)), False

    if det_backend == "detectron2":
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            out = det_model(bgr)
        inst = out["instances"]
        if len(inst) == 0:
            return np.empty((0,4)), np.empty((0,)), False
        boxes  = inst.pred_boxes.tensor.cpu().numpy()
        scores = inst.scores.cpu().numpy()
    else:
        img_t = T.ToTensor()(Image.fromarray(image_np)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = det_model(img_t)[0]
        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        mask   = scores >= DET_CONF
        boxes, scores = boxes[mask], scores[mask]

    if len(scores) == 0:
        return np.empty((0,4)), np.empty((0,)), False
    return boxes, scores, True


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2.5 — ROI CROP (PADDING_RATIO=0.35 from Doc3 pipeline v4)
# ═══════════════════════════════════════════════════════════════════════════════

def crop_roi(image_np, box, padding=DET_PADDING):
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2-x1, y2-y1
    cx1 = max(0, int(x1 - bw*padding))
    cy1 = max(0, int(y1 - bh*padding))
    cx2 = min(W, int(x2 + bw*padding))
    cy2 = min(H, int(y2 + bh*padding))
    if cx2-cx1 < 4 or cy2-cy1 < 4:
        cx1,cy1,cx2,cy2 = max(0,x1),max(0,y1),min(W,x2),min(H,y2)
    return Image.fromarray(image_np[cy1:cy2, cx1:cx2])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_classification(roi_pil, clf_model):
    tensor = preprocess_roi(roi_pil)
    with torch.no_grad():
        probs = F.softmax(clf_model(tensor), dim=1).cpu().numpy()[0]
    label = int(np.argmax(probs))
    return label, CLF_CLASSES[label], probs


# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM++ — exact from Doc4 Section 8 compute_gradcam_pp()
#
#   def _gradcam_layer(model):
#       try:    return [model.features[-1][-1]]
#       except: return [model.features[-1]]
#
#   cam_obj = GradCAMPlusPlus(model=model, target_layers=_gradcam_layer(model))
#   cam = cam_obj(input_tensor=tensor.to(device),
#                 targets=[ClassifierOutputTarget(target)])
#   raw = cam[0].astype(np.float32)
#   return np.power(raw, 0.7)   # gamma correction γ=0.7
#
# Falls back to manual Grad-CAM only if pytorch-grad-cam not installed.
# ═══════════════════════════════════════════════════════════════════════════════

def _gradcam_layer(model):
    try:    return [model.features[-1][-1]]
    except: return [model.features[-1]]


def generate_gradcam_pp(roi_pil, clf_model, target_cls):
    """
    Returns HxWx3 BGR uint8 blend overlay (JET colormap, 50/50).
    Uses pytorch-grad-cam exactly as in your explainability notebook.
    """
    clf_model.eval()
    tensor = preprocess_roi(roi_pil)   # (1,3,256,256)

    cam_map = None  # (H,W) in [0,1] after gamma

    # ── pytorch-grad-cam (exact notebook implementation) ──────────────────
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        cam_obj = GradCAMPlusPlus(
            model        = clf_model,
            target_layers= _gradcam_layer(clf_model),
        )
        cam_out = cam_obj(
            input_tensor = tensor,
            targets      = [ClassifierOutputTarget(target_cls)],
        )
        raw     = cam_out[0].astype(np.float32)   # (H,W) ∈ [0,1]
        cam_map = np.power(raw, 0.7)              # γ=0.7 correction

    except (ImportError, Exception):
        # ── Manual Grad-CAM fallback ──────────────────────────────────────
        acts, grads = {}, {}
        layer = _gradcam_layer(clf_model)[0]
        fh = layer.register_forward_hook(
            lambda *a: acts.update({"f": a[2].detach()}))
        bh = layer.register_full_backward_hook(
            lambda *a: grads.update({"g": a[2][0].detach()}))
        clf_model.zero_grad()
        out = clf_model(tensor)
        out[0, target_cls].backward()
        fh.remove(); bh.remove()
        g = grads["g"][0]
        a = acts["f"][0]
        raw = F.relu((g.mean(dim=(1,2), keepdim=True) * a).sum(0))
        raw = raw.cpu().numpy()
        lo, hi = raw.min(), raw.max()
        cam_map = np.power((raw - lo) / (hi - lo + 1e-8), 0.7)

    # ── Resize to 256×256 ────────────────────────────────────────────────
    cam_up = cv2.resize(
        np.clip(cam_map, 0, 1).astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
    )

    # ── JET heatmap ───────────────────────────────────────────────────────
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)

    # ── Base image: preprocessed ROI (same as classifier sees) ───────────
    base = np.array(_pad_resize(_autocontrast(roi_pil), IMAGE_SIZE)).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    # ── 50/50 blend ───────────────────────────────────────────────────────
    return cv2.addWeighted(base_bgr, 0.5, heatmap, 0.5, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# BBOX DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_bbox(image_np, box, score):
    img = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box.astype(int)
    C  = (0, 200, 255)
    TH = max(2, int(min(image_np.shape[:2]) / 200))
    cv2.rectangle(img, (x1,y1), (x2,y2), C, TH)
    ln = max(16, TH*6)
    for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*ln,cy),C,TH+1)
        cv2.line(img,(cx,cy),(cx,cy+dy*ln),C,TH+1)
    lbl = f"  Tumor  {score:.0%}  "
    sc  = max(0.45, min(image_np.shape[:2])/600)
    (tw,th),bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
    ly = max(y1-6, th+6)
    cv2.rectangle(img,(x1,ly-th-6),(x1+tw,ly+bl),C,-1)
    cv2.putText(img,lbl,(x1,ly),cv2.FONT_HERSHEY_SIMPLEX,sc,(0,0,0),2,cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
#
#  Stage 2: Faster R-CNN on full X-ray
#    └─ No boxes above conf=0.30  →  res["final"]="NORMAL"  [STOP]
#    └─ Best-confidence box found →  crop ROI (35% padding)
#
#  Stage 3: ConvNeXt-SE on cropped ROI
#    └─ label=0 → Benign  |  label=1 → Malignant
#
#  XAI: Grad-CAM++ on predicted class (same as Doc4 notebook)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_pil, det_model, clf_model, det_backend):
    image_np = np.array(image_pil.convert("RGB"))
    res = dict(image_np=image_np, final=None, is_tumor=False,
               det_conf=None, clf_conf=None, clf_class=None, clf_label=None,
               box_xyxy=None, roi_pil=None, image_boxed=None,
               gradcam_bgr=None, probs=None)

    # ── Stage 2: Detection ─────────────────────────────────────────────────
    with st.spinner("🔍  Stage 2 · Faster R-CNN detection…"):
        boxes, scores, is_tumor = run_detection(image_np, det_model, det_backend)

    # ── Decision: NORMAL ──────────────────────────────────────────────────
    # No detections above conf=0.30 → the scan is NORMAL → STOP
    # Stage 3 is NOT called — this is the correct pipeline behaviour
    if not is_tumor:
        res["final"]    = "NORMAL"
        res["is_tumor"] = False
        return res

    # ── Tumor detected: highest-confidence box ────────────────────────────
    best = int(np.argmax(scores))
    box  = boxes[best]
    conf = float(scores[best])

    roi_pil = crop_roi(image_np, box, padding=DET_PADDING)

    res.update(is_tumor=True, box_xyxy=box, det_conf=conf,
               image_boxed=draw_bbox(image_np, box, conf), roi_pil=roi_pil)

    # ── Stage 3: Classification ───────────────────────────────────────────
    with st.spinner("🧬  Stage 3 · ConvNeXt-SE classifying ROI…"):
        lbl, cls_name, probs = run_classification(roi_pil, clf_model)

    res.update(clf_label=lbl, clf_class=cls_name,
               clf_conf=float(probs[lbl]), probs=probs,
               final=cls_name.upper())   # "BENIGN" or "MALIGNANT"

    # ── Grad-CAM++ (pytorch-grad-cam, features[-1][-1], γ=0.7) ───────────
    with st.spinner("🔥  Grad-CAM++ …"):
        res["gradcam_bgr"] = generate_gradcam_pp(roi_pil, clf_model, lbl)

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _prob_bar(name, prob, color):
    pct = max(0, min(100, int(prob*100)))
    st.markdown(f"""<div class="prow"><span class="pname">{name}</span>
    <span class="ppct" style="color:{color}">{pct}%</span>
    <div class="ptrack"><div class="pfill"
    style="width:{pct}%;background:{color}"></div></div></div>""",
    unsafe_allow_html=True)

def _mtile(lbl, val, cls=""):
    st.markdown(f'<div class="mtile {cls}"><div class="mlbl">{lbl}</div>'
                f'<div class="mval">{val}</div></div>', unsafe_allow_html=True)

def _img_panel(img, title, caption=""):
    st.markdown(f'<div class="ipanel"><div class="iptitle">{title}</div>',
                unsafe_allow_html=True)
    st.image(img, use_container_width=True, caption=caption)
    st.markdown("</div>", unsafe_allow_html=True)

def _pred_card(res):
    f = res["final"]
    cfg_map = {
        "NORMAL":    ("card-normal",    "✅", "",
                      "Faster R-CNN found no region above conf=0.30. "
                      "Stage 3 classification was NOT executed."),
        "BENIGN":    ("card-benign",    "⚠️",
                      f"Det: {res.get('det_conf',0):.1%}  ·  "
                      f"Clf: {res.get('clf_conf',0):.1%}",
                      "Low-grade tumour — clinical correlation advised."),
        "MALIGNANT": ("card-malignant", "🚨",
                      f"Det: {res.get('det_conf',0):.1%}  ·  "
                      f"Clf: {res.get('clf_conf',0):.1%}",
                      "High-grade tumour — immediate clinical review required."),
    }
    c, icon, conf_txt, msg = cfg_map[f]
    st.markdown(f"""<div class="card-wrap {c}">
    <div class="card-icon">{icon}</div>
    <div class="card-lbl">Final Prediction</div>
    <div class="card-pred">{f}</div>
    <div class="card-conf"><strong>{conf_txt}</strong></div>
    <div class="card-msg">{msg}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(models_ready):
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:6px 0 12px">
        <div style="font-size:2.2em">🦴</div>
        <div style="font-size:1.05em;font-weight:800;color:#7BC8E2;letter-spacing:1px">
        BoneScan AI</div>
        <div style="font-size:.71em;color:#8AA8C8;margin-top:2px">
        Bone Tumour Classification</div></div>""", unsafe_allow_html=True)
        st.markdown("---")

        det_ok  = os.path.exists(DET_PATH)
        clf_ok  = os.path.exists(CLF_PATH)
        gpu_ok  = torch.cuda.is_available()
        try:    import pytorch_grad_cam; gcam_ok = True
        except: gcam_ok = False

        def dot(ok): return f'<span class="sdot {"dg" if ok else "dr"}"></span>'
        gpu_lbl = f"GPU · {torch.cuda.get_device_name(0)}" if gpu_ok else "CPU"
        st.markdown(f"""**⚙️ System Status**
        <div style="font-size:.78em;line-height:2.2;margin-top:5px">
        {dot(det_ok)} Faster R-CNN (detection)<br>
        {dot(clf_ok)} ConvNeXt-SE (classifier)<br>
        {dot(gcam_ok)} pytorch-grad-cam<br>
        {dot(gpu_ok)} {gpu_lbl}</div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("""**🔬 Decision Pipeline**
        <div style="font-size:.76em;line-height:2.1;color:#8AA8C8;margin-top:5px">
        <b style="color:#7BC8E2">①</b> Pre-process X-ray (AutoContrast · LB)<br>
        <b style="color:#7BC8E2">②</b> Faster R-CNN &nbsp;<span style="color:#AAC8E2">conf≥0.30</span><br>
        &nbsp;&nbsp;&nbsp;<span style="color:#10B981;font-weight:700">▶ No tumor → NORMAL ✓ STOP</span><br>
        &nbsp;&nbsp;&nbsp;<span style="color:#F59E0B">▶ Tumor → crop ROI (35%)</span><br>
        <b style="color:#7BC8E2">③</b> ConvNeXt-SE classify ROI<br>
        &nbsp;&nbsp;&nbsp;→ <b>Benign</b> or <b>Malignant</b><br>
        <b style="color:#7BC8E2">④</b> Grad-CAM++ &nbsp;<span style="color:#AAC8E2">features[-1][-1] · γ=0.7</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("**📂 Upload Bone X-Ray**")
        uploaded = st.file_uploader("JPG · PNG · BMP",
                                     type=["jpg","jpeg","png","bmp"],
                                     label_visibility="collapsed")
        run_btn = False
        if uploaded:
            st.markdown("")
            if models_ready:
                run_btn = st.button("🔬  Run Analysis",
                                     use_container_width=True, type="primary")
            else:
                st.button("🔬  Run Analysis",
                           use_container_width=True, disabled=True)
                st.markdown('<div class="warn-panel" style="margin:5px 0;'
                            'font-size:.76em">Models not loaded.</div>',
                            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""<div style="font-size:.66em;color:#4A6580;text-align:center;line-height:1.6">
        ⚕️ Research use only<br>Not for clinical diagnosis<br>
        Consult a licensed radiologist</div>""", unsafe_allow_html=True)

    return uploaded, run_btn


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def display_results(res):
    _pred_card(res)

    # ── NORMAL path: pipeline stopped ──────────────────────────────────────
    if res["final"] == "NORMAL":
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div class="sec-head">Original X-Ray</div>',
                        unsafe_allow_html=True)
            _img_panel(res["image_np"], "INPUT IMAGE")
        with c2:
            st.markdown('<div class="sec-head">Detection Result</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="ok-panel">
            <strong>✅ No tumour detected</strong><br><br>
            Faster R-CNN found no regions above confidence threshold
            <strong>0.30</strong>.<br><br>
            Final output: <strong>NORMAL</strong><br><br>
            <em>Stage 3 (ConvNeXt-SE classification) was NOT executed —
            no ROI cropping was performed.</em>
            </div>""", unsafe_allow_html=True)
        return

    # ── TUMOR path ──────────────────────────────────────────────────────────
    box = res["box_xyxy"].astype(int)
    x1,y1,x2,y2 = box
    cc = "red" if res["final"] == "MALIGNANT" else "orange"

    # Stage 2 row
    st.markdown('<div class="sec-head">Stage 2 · Faster R-CNN Tumour Detection</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _img_panel(res["image_np"], "ORIGINAL X-RAY")
    with c2:
        _img_panel(res["image_boxed"], "DETECTION + BOUNDING BOX",
                   f"Confidence: {res['det_conf']:.1%}  ·  threshold: {DET_CONF}")

    # Stage 3 row
    st.markdown(f'<div class="sec-head">Stage 3 · ROI Classification · '
                f'Grad-CAM++ Explainability</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        # Show the preprocessed ROI (same as what classifier sees)
        roi_disp = np.array(_pad_resize(_autocontrast(res["roi_pil"]), IMAGE_SIZE))
        _img_panel(roi_disp,
                   "CROPPED ROI  (autocontrast → letterbox → classifier input)",
                   f"35% padding  ·  raw box {x2-x1}×{y2-y1}px")
    with c4:
        gcam_rgb = cv2.cvtColor(res["gradcam_bgr"], cv2.COLOR_BGR2RGB)
        _img_panel(gcam_rgb,
                   "GRAD-CAM++  (pytorch-grad-cam · features[-1][-1] · γ=0.7)",
                   f"Target: {res['clf_class']}  ·  JET 50/50 blend")

    # Metrics
    st.markdown('<div class="sec-head">Quantitative Results</div>',
                unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    with m1: _mtile("Detection Score", f"{res['det_conf']:.1%}", "")
    with m2: _mtile("Classification",  f"{res['clf_conf']:.1%}", cc)
    with m3: _mtile("Prediction",      res["clf_class"],         cc)
    with m4: _mtile("ROI Size",        f"{x2-x1}×{y2-y1} px",  "")

    # Probability bars
    st.markdown('<div class="sec-head">Full Probability Distribution</div>',
                unsafe_allow_html=True)
    pb, disc = st.columns([3, 2])
    p = res["probs"]
    with pb:
        _prob_bar("🟢 Normal  (detection threshold not reached)", 0.0,        "#10B981")
        _prob_bar("🟠 Benign",                                    float(p[0]), "#F59E0B")
        _prob_bar("🔴 Malignant",                                 float(p[1]), "#EF4444")
    with disc:
        st.markdown("""<div class="warn-panel">
        <strong>⚕️ Clinical Disclaimer</strong><br><br>
        AI decision-support only. All predictions must be verified
        by a qualified radiologist before any clinical action.
        </div>""", unsafe_allow_html=True)

    # BBox details
    st.markdown(
        f"""<div class="info-panel">📦 <strong>Detection BBox</strong> &nbsp;·&nbsp;
        ({x1},{y1}) → ({x2},{y2}) &nbsp;|&nbsp;
        W={x2-x1}px H={y2-y1}px &nbsp;|&nbsp;
        padding=35% &nbsp;|&nbsp; conf-thresh=0.30</div>""",
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    with st.spinner("Initialising models…"):
        det_model, clf_model, det_backend, load_errors = load_models()

    models_ready = det_model is not None and clf_model is not None
    uploaded, run_btn = render_sidebar(models_ready)

    # Hero
    st.markdown("""<div class="hero">
    <div class="hero-title">🦴 BoneScan <span>AI</span></div>
    <div class="hero-sub">
    Faster R-CNN Detection &nbsp;→&nbsp; ConvNeXt-SE Classification
    &nbsp;→&nbsp; Grad-CAM++ Explainability
    </div>
    <div style="margin-top:12px">
    <span class="hbadge">Faster R-CNN · Detectron2</span>
    <span class="hbadge">ConvNeXt-Tiny-SE · LN head</span>
    <span class="hbadge">Grad-CAM++ · pytorch-grad-cam</span>
    <span class="hbadge">35% ROI padding</span>
    <span class="hbadge">conf ≥ 0.30</span>
    </div></div>""", unsafe_allow_html=True)

    for err in load_errors:
        st.markdown(f'<div class="warn-panel">⚠️ {err}</div>',
                    unsafe_allow_html=True)

    # Welcome
    if uploaded is None:
        st.markdown("""<div class="info-panel">
        👈 Upload a bone X-ray from the sidebar to begin.<br>
        The pipeline detects tumour regions with Faster R-CNN. If none found →
        <strong>Normal</strong>. If found → crops ROI and classifies as
        <strong>Benign</strong> or <strong>Malignant</strong>.
        </div>""", unsafe_allow_html=True)
        cols = st.columns(4)
        steps = [
            ("🖼️","Pre-processing","AutoContrast(L,cutoff=1) · Letterbox 256² · ImageNet norm"),
            ("🔍","Tumour Detection","Faster R-CNN R50-FPN · conf≥0.30 · NORMAL if empty"),
            ("🧬","ROI Classification","ConvNeXt-SE · Flatten→LN→GELU head · Benign/Malignant"),
            ("🔥","Grad-CAM++","pytorch-grad-cam · features[-1][-1] · γ=0.7 · JET 50/50"),
        ]
        for col,(icon,name,desc) in zip(cols,steps):
            with col:
                st.markdown(f"""<div class="step-card">
                <div class="sicon">{icon}</div>
                <div class="sname">{name}</div>
                <div class="sdesc">{desc}</div>
                </div>""", unsafe_allow_html=True)
        return

    # Load image
    try:
        image_pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Cannot read image: {e}"); return

    st.markdown('<div class="sec-head">Uploaded Image</div>', unsafe_allow_html=True)
    pv1, pv2 = st.columns([3,1])
    with pv1:
        _img_panel(image_pil, "INPUT X-RAY", uploaded.name)
    with pv2:
        w,h = image_pil.size
        nm = uploaded.name[:24]+"…" if len(uploaded.name)>24 else uploaded.name
        _mtile("Filename",   nm)
        _mtile("Dimensions", f"{w}×{h}px")
        _mtile("Mode",       image_pil.mode)

    if run_btn:
        try:
            result = run_pipeline(image_pil, det_model, clf_model, det_backend)
            st.markdown("---")
            display_results(result)
        except torch.cuda.OutOfMemoryError:
            st.error("GPU OOM — try a smaller image or restart.")
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.exception(e)
    else:
        st.markdown("""<div class="ok-panel">
        ✅ Image loaded. Press <strong>🔬 Run Analysis</strong> in the sidebar.
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()