"""
BoneFusionNet — Bone Tumor X-Ray Diagnosis
==========================================
Stage 1 : Faster R-CNN  → detect ROIs
Stage 2 : ConvNeXt-Tiny-SE → Benign / Malignant
Stage 3 : GradCAM++ (+ optional IG)

Edit the two CHECKPOINT constants below, then:
    streamlit run app.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# ██  CHECKPOINT PATHS  ████████████████████████████████████████████████████████
DET_CHECKPOINT = r"D:\python_mastery\machine_learning\model_checkpoints\model_best_fastercnn.pth"
CLS_CHECKPOINT = r"D:\python_mastery\machine_learning\model_checkpoints\final_best_no_aug.pth"

# ██  OOD / INPUT-VALIDATION DEFAULTS  ████████████████████████████████████████
# These reject non-bone-X-ray inputs (random photos, wrong modality, etc.)
# instead of forcing them into Benign/Malignant. Both are exposed as sidebar
# sliders so you can tune them live; the values below are just starting points.
# See the calibration note at the bottom of this file for how to set them
# properly using your own validation set.
GRAY_THRESH_DEFAULT = 6.0   # mean |R-G|+|G-B|+|R-B| per pixel; X-rays ≈ 0
OOD_ENERGY_DEFAULT  = 5.0   # -logsumexp(logits); calibrate to ~99th pct of ID val energy
# ══════════════════════════════════════════════════════════════════════════════

import io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path

# ── optional: GradCAM++ ──────────────────────────────────────────────────────
GRADCAM_OK = False
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_OK = True
except Exception:
    pass

# ── optional: Integrated Gradients ───────────────────────────────────────────
CAPTUM_OK = False
try:
    from captum.attr import IntegratedGradients
    CAPTUM_OK = True
except Exception:
    pass

# scipy gaussian_filter
_GF = None
try:
    from scipy.ndimage import gaussian_filter as _GF
except Exception:
    pass

# ── optional: detectron2 ─────────────────────────────────────────────────────
D2_OK = False
try:
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    D2_OK = True
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BoneFusionNet",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark" not in st.session_state:
    st.session_state.dark = True

# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
def _t():
    if st.session_state.dark:
        return dict(
            bg="090e1a", bg2="0d1526", bg3="111d30",
            brd="1e2d4a", brd2="1e3a5f",
            txt="d4dbe8", dim="4a6280", mid="8baabb", strong="e2f0ff",
            norm_bg="linear-gradient(135deg,#0a1f14,#0d2418)", norm_brd="1a4d30",
            ben_bg="linear-gradient(135deg,#0e1e34,#0a1a30)",  ben_brd="1e4d8a",
            mal_bg="linear-gradient(135deg,#1e0a0a,#280d0d)",  mal_brd="7a1515",
            ood_bg="linear-gradient(135deg,#241c08,#2e2308)",  ood_brd="8a6d1a",
            badge_n="4ade80", badge_b="38bdf8", badge_m="f87171", badge_o="fbbf24",
            ok_bg="081a0e", ok_brd="1a4d2a", ok_txt="4ade80",
            chip="111d30", roi_bg="0d1526",
            bar_na="14532d", bar_nb="4ade80",
            bar_ba="0369a1", bar_bb="38bdf8",
            bar_ma="7f1d1d", bar_mb="f87171",
            bar_oa="78350f", bar_ob="fbbf24",
            sb_th="1e3a5f",
            tog="☀️ Day",
            hdr="linear-gradient(90deg,#e2f0ff 0%,#38bdf8 60%,#818cf8 100%)",
        )
    return dict(
        bg="f0f4f8", bg2="ffffff", bg3="e8eef5",
        brd="c8d4e0", brd2="90aac0",
        txt="1a2a3a", dim="5a7080", mid="2a4a60", strong="0a1828",
        norm_bg="linear-gradient(135deg,#dff5e8,#c8ecd8)", norm_brd="2a9a50",
        ben_bg="linear-gradient(135deg,#deeef8,#c8e0f2)",  ben_brd="2878b8",
        mal_bg="linear-gradient(135deg,#faecec,#f5d5d5)",  mal_brd="c84848",
        ood_bg="linear-gradient(135deg,#fdf3d8,#f7e6b0)",  ood_brd="c9a227",
        badge_n="1a7a3c", badge_b="1a5fa8", badge_m="c0392b", badge_o="b8860b",
        ok_bg="e8faf0", ok_brd="2a8a50", ok_txt="0a5020",
        chip="e8eef5", roi_bg="ffffff",
        bar_na="2a9a50", bar_nb="4ade80",
        bar_ba="2878b8", bar_bb="38bdf8",
        bar_ma="c04040", bar_mb="f87171",
        bar_oa="b8860b", bar_ob="f7c948",
        sb_th="90aac0",
        tog="🌙 Night",
        hdr="linear-gradient(90deg,#0a1828 0%,#1a5fa8 60%,#5048a8 100%)",
    )


def _css(c):
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700&family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
html,body,[class*="css"]{{font-family:'DM Mono',monospace;background:#{c['bg']};color:#{c['txt']};}}
.main{{background:#{c['bg']};}}
html,body{{overflow:auto;height:100vh;}}
.main .block-container{{padding:0.35rem 1.1rem 0 1.1rem!important;max-height:calc(100vh - 40px);overflow:auto;}}
#MainMenu{{visibility:visible!important;}}
footer{{visibility:hidden!important;}}
/* sidebar styling */
[data-testid="stSidebar"]{{background:#{c['bg2']}!important;border-right:1px solid #{c['brd']};}}
[data-testid="stSidebar"] *{{color:#{c['txt']}!important;}}
[data-testid="stSidebarNavSeparator"]{{display:none;}}
.element-container{{margin-bottom:0!important;}}
div[data-testid="stSlider"]>div{{padding-top:0!important;padding-bottom:0!important;}}
div[data-testid="column"]{{padding:0 3px!important;}}
[data-testid="stFileUploaderDropzone"]{{padding:3px 8px!important;min-height:unset!important;}}
[data-testid="stFileUploader"] label{{display:none!important;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-thumb{{background:#{c['sb_th']};border-radius:2px;}}
.dcard{{border-radius:7px;padding:8px 12px;border:1px solid;}}
.d-n{{background:{c['norm_bg']};border-color:#{c['norm_brd']};}}
.d-b{{background:{c['ben_bg']};border-color:#{c['ben_brd']};}}
.d-m{{background:{c['mal_bg']};border-color:#{c['mal_brd']};}}
.d-o{{background:{c['ood_bg']};border-color:#{c['ood_brd']};}}
.badge{{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;line-height:1.2;display:block;}}
.bn{{color:#{c['badge_n']};}} .bb{{color:#{c['badge_b']};}} .bm{{color:#{c['badge_m']};}} .bo{{color:#{c['badge_o']};}}
.dmeta{{font-size:0.58rem;color:#{c['dim']};letter-spacing:0.13em;text-transform:uppercase;}}
.ct{{background:#{c['bg3']};border-radius:3px;height:4px;margin:3px 0 2px 0;overflow:hidden;}}
.fn{{height:100%;background:linear-gradient(90deg,#{c['bar_na']},#{c['bar_nb']});border-radius:3px;}}
.fb{{height:100%;background:linear-gradient(90deg,#{c['bar_ba']},#{c['bar_bb']});border-radius:3px;}}
.fm{{height:100%;background:linear-gradient(90deg,#{c['bar_ma']},#{c['bar_mb']});border-radius:3px;}}
.fo{{height:100%;background:linear-gradient(90deg,#{c['bar_oa']},#{c['bar_ob']});border-radius:3px;}}
.chips{{display:flex;gap:5px;flex-wrap:wrap;margin:3px 0;}}
.chip{{background:#{c['chip']};border:1px solid #{c['brd']};border-radius:5px;
       padding:3px 7px;font-size:0.65rem;color:#{c['mid']};line-height:1.4;}}
.chip strong{{color:#{c['strong']};}}
.rrow{{background:#{c['roi_bg']};border:1px solid #{c['brd']};border-radius:5px;
       padding:4px 8px;margin:2px 0;}}
.rlbl{{font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;}}
.rb{{color:#38bdf8;}} .rm{{color:#f87171;}} .ro{{color:#fbbf24;}}
.rscroll{{max-height:18vh;overflow-y:auto;padding-right:2px;}}
.ok{{background:#{c['ok_bg']};border:1px solid #{c['ok_brd']};border-radius:5px;
     padding:5px 8px;font-size:0.68rem;color:#{c['ok_txt']};margin:3px 0;}}
.cam-scroll{{overflow-y:auto;padding-right:2px;}}
/* fixed-height image box — image always fits in viewport */
.img-box{{
    width:100%; height:calc(100vh - 195px);
    display:flex; align-items:center; justify-content:center;
    overflow:hidden; border-radius:8px;
    background:#000; cursor:zoom-in;
}}
.img-box img{{
    max-width:100%; max-height:100%;
    object-fit:contain; border-radius:6px;
    transform-origin:center center;
    transition:transform 0.2s ease;
}}
/* shorter box when banner is showing */
.img-box-short{{
    width:100%; height:calc(100vh - 285px);
    display:flex; align-items:center; justify-content:center;
    overflow:hidden; border-radius:8px; background:#000;
}}
.img-box-short img{{
    max-width:100%; max-height:100%;
    object-fit:contain; border-radius:6px;
    transform-origin:center center;
    transition:transform 0.2s ease;
}}
/* full-width diagnosis banner */
.diag-banner{{
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; border-radius:10px; padding:10px 18px;
    border:1px solid; margin-bottom:6px;
}}
.diag-banner .badge{{font-size:1.9rem; margin:0;}}
.banner-right{{flex:1; min-width:0;}}
.banner-conf{{font-size:0.72rem; margin-top:3px;}}
/* right-col content scroll for gradcam */
.right-scroll{{max-height:calc(100vh - 285px);overflow-y:auto;padding-right:3px;}}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        m = max(ch // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, m, bias=False), nn.ReLU(True),
            nn.Linear(m, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class ConvNeXtTinySE(nn.Module):
    def __init__(self, n=2, r=16):
        super().__init__()
        bb = torchvision.models.convnext_tiny(weights=None)
        self.features = bb.features
        self.se = SEBlock(768, r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(768), nn.Dropout(0.6),
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, n))

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_TF   = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])


def preprocess(crop):
    img = ImageOps.autocontrast(crop.convert('L'), cutoff=1).convert('RGB')
    w, h = img.size
    r = min(256 / w, 256 / h)
    nw, nh = int(w * r), int(h * r)
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 256), (0, 0, 0))
    canvas.paste(img, ((256 - nw) // 2, (256 - nh) // 2))
    return _TF(canvas).unsqueeze(0), np.array(canvas, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# TIER-1 OOD GATE — whole-image grayscale check (catches color photos for free)
# ══════════════════════════════════════════════════════════════════════════════
def looks_like_xray(pil_img: Image.Image, chan_diff_thresh: float = GRAY_THRESH_DEFAULT) -> bool:
    """Bone X-rays are grayscale even when saved as RGB (R≈G≈B per pixel).
    Natural photos (cars, people, etc.) are not. This is a near-zero-cost
    pre-filter that runs before the detector ever sees the image."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    chan_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
    return chan_diff < chan_diff_thresh


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_det(path, sc, nms):
    if not D2_OK:
        return None, "detectron2 not installed"
    if not Path(path).exists():
        return None, f"Not found: {path}"
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = sc
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms
        cfg.MODEL.WEIGHTS = path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.INPUT.MIN_SIZE_TEST = 640
        cfg.INPUT.MAX_SIZE_TEST = 1024
        cfg.TEST.DETECTIONS_PER_IMAGE = 10
        return DefaultPredictor(cfg), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_cls(path):
    if not Path(path).exists():
        return None, f"Not found: {path}"
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = ConvNeXtTinySE(2, 16)
        ck = torch.load(path, map_location=dev, weights_only=False)
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        m.load_state_dict(sd, strict=True)
        return (m.eval().to(dev), dev), None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def detect(pred, bgr):
    out = pred(bgr)["instances"].to("cpu")
    if len(out) == 0:
        return []
    return [{"box": out.pred_boxes.tensor[i].numpy().astype(int).tolist(),
             "score": float(out.scores[i])} for i in range(len(out))]


def classify(model, dev, tensor, thr):
    with torch.no_grad():
        logits = model(tensor.to(dev))
        energy = -torch.logsumexp(logits, dim=1).item()
        p = F.softmax(logits, 1).squeeze(0).cpu().numpy()
    return ("Malignant" if p[1] >= thr else "Benign"), float(p[0]), float(p[1]), energy


def crop_roi(img, box, pad=0.08):
    W, H = img.size
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    return img.crop((max(0, x1 - pw), max(0, y1 - ph),
                     min(W, x2 + pw), min(H, y2 + ph)))


# ══════════════════════════════════════════════════════════════════════════════
# GRADCAM++
# ══════════════════════════════════════════════════════════════════════════════
def run_gradcam(model, dev, tensor, cls_idx):
    if not GRADCAM_OK:
        return None
    try:
        model.eval()
        cam = GradCAMPlusPlus(model=model,
                              target_layers=[model.features[-1][-1]])
        raw = cam(input_tensor=tensor.to(dev),
                  targets=[ClassifierOutputTarget(cls_idx)])[0].astype(np.float32)
        return np.power(raw, 0.7)
    except Exception:
        return None


def run_ig(model, dev, tensor, cls_idx):
    if not CAPTUM_OK or _GF is None:
        return None
    try:
        t = tensor.to(dev).requires_grad_(True)
        np_img = t.squeeze(0).detach().cpu().numpy()
        blur = np.stack([_GF(np_img[ch], sigma=10.) for ch in range(3)])
        base = (torch.from_numpy(blur.copy())
                .unsqueeze(0).to(dtype=t.dtype, device=t.device))
        ig = IntegratedGradients(model)
        attr, _ = ig.attribute(t, baselines=base, target=cls_idx,
                               n_steps=50, method='gausslegendre',
                               return_convergence_delta=True)
        s = attr.squeeze(0).detach().cpu().numpy().sum(0)
        s = _GF(s, sigma=1.)
        v = np.percentile(np.abs(s), 99)
        if v < 1e-8:
            return None
        return np.clip(s / v, -1, 1).astype(np.float32)
    except Exception:
        return None


def activation_conc(cam, top=25.):
    thr  = np.percentile(cam, 100 - top)
    mask = cam >= thr
    tot  = cam.sum()
    return (float(cam[mask].sum() / tot * 100) if tot > 1e-8 else 0.), mask


def blend(rgb, hm, cmap, alpha=0.45):
    return np.clip((1 - alpha) * rgb + alpha * cmap(hm)[:, :, :3], 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  — single-row compact
# ══════════════════════════════════════════════════════════════════════════════
def make_fig(rgb_u8, cam, ig_map, label, p_mal, alpha=0.45):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CM  = plt.cm.inferno
    rgb = rgb_u8.astype(np.float32) / 255.

    panels = [("ROI", rgb, None)]

    cam_extra = None
    if cam is not None:
        # upsample to the rgb size with cubic for smooth but sharp result
        cr = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
        cr = np.clip(cr, 0, 1)
        cc, mask = activation_conc(cr)
        ov = blend(rgb, cr, CM, alpha)
        panels.append((f"CAM  {cc:.0f}%", ov, (cr, mask)))

    if ig_map is not None:
        igr = cv2.resize(ig_map, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        igabs = np.abs(igr)
        vmax  = np.percentile(igabs, 99)
        ign   = np.clip(igabs / (vmax + 1e-8), 0, 1).astype(np.float32)
        panels.append(("IG", blend(rgb, ign, CM, alpha), None))

    n   = len(panels)
    chx = "#c0392b" if label == "Malignant" else "#1565c0"

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.8), dpi=180, facecolor="white")
    if n == 1:
        axes = [axes]

    for idx, (ax, (title, img_data, extra)) in enumerate(zip(axes, panels)):
        ax.imshow(img_data, aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8, color="#333", pad=3)
        ec = chx if idx == 0 else "#cccccc"
        lw = 1.4 if idx == 0 else 0.4
        for sp in ax.spines.values():
            sp.set_edgecolor(ec); sp.set_linewidth(lw)
        if extra is not None:
            cr2, mask2 = extra
            ax.contour(mask2.astype(float), [0.5],
                       colors=["#FFD700"], linewidths=1.0, alpha=0.95)

    fig.suptitle(f"{label}  P(Mal)={p_mal * 100:.1f}%",
                 fontsize=9, fontweight="bold", color=chx, y=1.02)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
_COLORS = {"Benign": (56, 189, 248), "Malignant": (248, 113, 113), "Uncertain": (251, 191, 36)}


def annotate(img_pil, dets, results):
    img  = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for i, (d, r) in enumerate(zip(dets, results)):
        x1, y1, x2, y2 = d["box"]
        col = _COLORS.get(r["lbl"], (148, 163, 184))
        lw  = max(2, int(min(W, H) / 200))
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        txt = f"#{i+1} {r['lbl']} {max(r['pb'],r['pm'])*100:.0f}%"
        tw = len(txt) * 7; th = 15; pad = 3
        bx1, by1 = x1, max(0, y1 - th - pad * 2)
        bx2, by2 = min(W, x1 + tw + pad * 2), max(0, y1)
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).rectangle([bx1, by1, bx2, by2], fill=(*col, 200))
        img  = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((bx1 + pad, by1 + 1), txt, fill=(255, 255, 255), font=font)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
c = _t()
st.markdown(_css(c), unsafe_allow_html=True)

# ── sidebar — all settings live here ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:8px 0 12px 0;'
        f'border-bottom:1px solid #{c["brd"]};margin-bottom:12px;">'
        f'<span style="font-size:1.2rem">🦴</span>'
        f'<span style="font-family:\'Inter\',sans-serif;font-size:1.05rem;font-weight:700;'
        f'letter-spacing:-0.04em;color:#{c["strong"]};">Bone Fusion Net</span>'
        f'</div>', unsafe_allow_html=True)

    if st.button(c["tog"], key="tog", use_container_width=True):
        st.session_state.dark = not st.session_state.dark
        st.rerun()

    st.markdown(
        f'<div style="font-size:0.60rem;color:#{c["dim"]};margin:6px 0 14px 0;">'
        f'{"🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"} &nbsp;·&nbsp; '
        f'{"CAM ✓" if GRADCAM_OK else "CAM ✗"} &nbsp;·&nbsp; '
        f'{"IG ✓" if CAPTUM_OK else "IG ✗"}</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#{c["dim"]};margin:10px 0 6px 0;">'
        f'Input Validation</div>', unsafe_allow_html=True)
    gray_thresh = st.slider("Grayscale gate strictness", 1.0, 20.0, GRAY_THRESH_DEFAULT, 0.5,
                            help="Rejects whole image before detection if it's not "
                                 "grayscale-like. Lower = stricter (rejects more).")
    ood_thresh = st.slider("Per-ROI OOD energy threshold", 0.0, 15.0, OOD_ENERGY_DEFAULT, 0.5,
                           help="Each detected ROI is scored with -logsumexp(logits). "
                                "ROIs above this are flagged 'Uncertain' instead of being "
                                "forced into Benign/Malignant. Calibrate against your "
                                "validation set — see comment at bottom of app.py.")

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#{c["dim"]};margin:10px 0 6px 0;">'
        f'Detection</div>', unsafe_allow_html=True)
    det_sc = st.slider("Confidence threshold", 0.05, 0.70, 0.15, 0.05)
    nms_t  = st.slider("NMS IoU threshold",    0.20, 0.70, 0.40, 0.05)

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#{c["dim"]};margin:10px 0 6px 0;">'
        f'Classification</div>', unsafe_allow_html=True)
    cls_t = st.slider("Malignancy threshold", 0.20, 0.80, 0.50, 0.05)
    roi_p = st.slider("ROI crop padding %",   0, 25, 8, 1) / 100.
    agg   = st.selectbox("Aggregation mode",
                         ["Any Malignant", "Majority Vote", "Max Probability"], 0)

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#{c["dim"]};margin:10px 0 6px 0;">'
        f'Explainability</div>', unsafe_allow_html=True)
    do_cam = st.toggle("GradCAM++",          value=True)
    do_ig  = st.toggle("Integrated Gradients", value=False,
                       disabled=not (CAPTUM_OK and _GF),
                       help="Requires: pip install captum --no-deps")

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#{c["dim"]};margin:10px 0 6px 0;">'
        f'Image Zoom</div>', unsafe_allow_html=True)
    zoom = st.slider("Zoom level", 1.0, 4.0, 1.0, 0.25,
                     format="%.2fx")

# ── topbar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 4px 10px 4px;
    border-bottom:1px solid #{c['brd']};
    margin-bottom:2px;
">
  <!-- left: icon + wordmark -->
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="
        width:36px;height:36px;border-radius:10px;
        background:linear-gradient(135deg,#0ea5e9,#6366f1);
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 0 14px #38bdf855;
        font-size:1.1rem;line-height:1;flex-shrink:0;">🦴</div>
    <div>
      <div style="
          font-family:'Inter',sans-serif;font-size:1.45rem;font-weight:700;
          color:#{c['strong']};
          line-height:1.1;letter-spacing:-0.04em;">Bone Fusion Net</div>
      <div style="
          font-size:0.52rem;color:#{c['dim']};letter-spacing:0.20em;
          text-transform:uppercase;margin-top:1px;">
        Bone Tumor Detection &amp; Classification
      </div>
    </div>
  </div>
  <!-- right: pill badges -->
  <div style="display:flex;align-items:center;gap:6px;">
    <span style="
        background:rgba(56,189,248,0.10);border:1px solid rgba(56,189,248,0.25);
        color:#38bdf8;border-radius:20px;padding:3px 10px;
        font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;">
      Detection
    </span>
    <span style="color:#{c['dim']};font-size:0.65rem;">›</span>
    <span style="
        background:rgba(129,140,248,0.10);border:1px solid rgba(129,140,248,0.25);
        color:#818cf8;border-radius:20px;padding:3px 10px;
        font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;">
      Classification
    </span>
    <span style="color:#{c['dim']};font-size:0.65rem;">›</span>
    <span style="
        background:rgba(251,191,36,0.10);border:1px solid rgba(251,191,36,0.25);
        color:#fbbf24;border-radius:20px;padding:3px 10px;
        font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;">
      Explainability
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── upload ────────────────────────────────────────────────────────────────────
up = st.file_uploader("x", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                      label_visibility="collapsed")

if up is None:
    st.markdown(
        f'<div style="text-align:center;padding:50px 0;'
        f'color:#{c["dim"]};font-size:0.78rem;">'
        f'Drop a bone X-ray above to begin<br>'
        f'<span style="font-size:0.62rem;letter-spacing:0.10em;">'
        f'PNG · JPG · BMP · TIFF</span></div>',
        unsafe_allow_html=True)
    st.stop()

# ── decode ────────────────────────────────────────────────────────────────────
pil  = Image.open(io.BytesIO(up.read())).convert("RGB")

# ── Tier-1 OOD gate: reject non-grayscale (i.e. non-X-ray) inputs up front ────
if not looks_like_xray(pil, gray_thresh):
    st.markdown(
        f'<div class="dcard d-o" style="margin-top:24px;">'
        f'<div class="dmeta">Input Validation Failed</div>'
        f'<span class="badge bo">NOT A VALID X-RAY</span>'
        f'<div style="font-size:0.72rem;color:#{c["mid"]};margin-top:6px;">'
        f'This image does not look like a grayscale radiograph (color-channel '
        f'divergence is too high). BoneFusionNet only operates on bone X-ray '
        f'images — please upload a valid radiograph.'
        f'</div></div>', unsafe_allow_html=True)
    st.stop()

bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
W, H = pil.size

# ── load models ───────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    det_m, det_e = (load_det(DET_CHECKPOINT, det_sc, nms_t)
                    if D2_OK else (None, "detectron2 not installed"))
    cls_pack, cls_e = load_cls(CLS_CHECKPOINT)

if det_m is None or cls_pack is None:
    if det_e: st.error(f"Detection: {det_e}")
    if cls_e: st.error(f"Classifier: {cls_e}")
    st.stop()

mdl, dev = cls_pack

with st.spinner("Detecting…"):
    dets = detect(det_m, bgr)

# ══════════════════════════════════════════════════════════════════════════════
# NO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
if not dets:
    ci, cr = st.columns([1, 1], gap="medium")
    with ci:
        import base64, io as _io
        _buf = _io.BytesIO(); pil.save(_buf, format="JPEG", quality=88)
        _b64 = base64.b64encode(_buf.getvalue()).decode()
        zoom_style = f"transform:scale({zoom});transform-origin:center center;"
        st.markdown(
            f'<div class="img-box" style="overflow:{"hidden" if zoom==1.0 else "auto"};">'
            f'<img src="data:image/jpeg;base64,{_b64}" '
            f'style="{zoom_style}max-width:{"none" if zoom>1 else "100%"};'
            f'max-height:{"none" if zoom>1 else "100%"};object-fit:contain;border-radius:6px;"/>'
            f'</div>', unsafe_allow_html=True)
    with cr:
        st.markdown(
            f'<div class="dcard d-n" style="margin-top:6px;">'
            f'<div class="dmeta">Final Diagnosis</div>'
            f'<span class="badge bn">NORMAL</span>'
            f'<div class="ct"><div class="fn" style="width:100%"></div></div>'
            f'<div style="font-size:0.70rem;color:#{c["mid"]};margin-top:3px;">'
            f'No tumor ROI found at threshold {det_sc:.2f}</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chips">'
            f'<div class="chip"><strong>0</strong><br>Detections</div>'
            f'<div class="chip"><strong>{W}×{H}</strong><br>Resolution</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown('<div class="ok">✅ No suspicious regions found.</div>',
                    unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY + EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Classifying {len(dets)} ROI(s)…"):
    rois = []
    for d in dets:
        cr2 = crop_roi(pil, d["box"], roi_p)
        ten, rgb8 = preprocess(cr2)
        lbl, pb, pm, energy = classify(mdl, dev, ten, cls_t)
        is_ood = energy > ood_thresh
        if is_ood:
            lbl = "Uncertain"
        rois.append(dict(lbl=lbl, pb=pb, pm=pm, ten=ten, rgb8=rgb8,
                         energy=energy, is_ood=is_ood))

if do_cam:
    with st.spinner("Computing GradCAM++…"):
        for r in rois:
            if r["is_ood"]:
                # GradCAM on an unfamiliar input isn't meaningful — skip it
                r["cam"] = r["ig"] = r["fig"] = None
                continue
            r["cam"] = run_gradcam(mdl, dev, r["ten"], 1)  # explain malignant neuron
            r["ig"]  = run_ig(mdl, dev, r["ten"], 1) if do_ig else None
            r["fig"] = make_fig(r["rgb8"], r["cam"], r["ig"], r["lbl"], r["pm"])
else:
    for r in rois:
        r["cam"] = r["ig"] = r["fig"] = None

# ── aggregate ─────────────────────────────────────────────────────────────────
valid_rois = [r for r in rois if not r["is_ood"]]
n_mal  = sum(1 for r in valid_rois if r["lbl"] == "Malignant")
n_ben  = sum(1 for r in valid_rois if r["lbl"] == "Benign")
n_ood  = len(rois) - len(valid_rois)
max_pm = max((r["pm"] for r in valid_rois), default=0.0)
max_pb = max((r["pb"] for r in valid_rois), default=0.0)

if not valid_rois:
    final = "Uncertain"   # every detected ROI failed the OOD energy check
elif agg == "Any Malignant":  final = "Malignant" if n_mal > 0        else "Benign"
elif agg == "Majority Vote":  final = "Malignant" if n_mal >= n_ben   else "Benign"
else:                         final = "Malignant" if max_pm >= cls_t  else "Benign"

ann = annotate(pil, dets, rois)

import base64 as _b64mod, io as _io2

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT  — full-width diagnosis banner, then xray | gradcam side by side
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Full-width diagnosis banner ───────────────────────────────────────────
if   final == "Malignant":  dc, bc, fc, cv = "d-m", "bm", "fm", max_pm
elif final == "Benign":     dc, bc, fc, cv = "d-b", "bb", "fb", max_pb
elif final == "Uncertain":  dc, bc, fc, cv = "d-o", "bo", "fo", 1.0
else:                       dc, bc, fc, cv = "d-n", "bn", "fn", 1.0
cp = int(cv * 100)

# ROI summary inline
roi_pills = ""
for i, r in enumerate(rois):
    if r["lbl"] == "Malignant":
        bg_col, txt_col = "rgba(248,113,113,0.15)", "#f87171"
    elif r["lbl"] == "Benign":
        bg_col, txt_col = "rgba(56,189,248,0.12)", "#38bdf8"
    else:  # Uncertain
        bg_col, txt_col = "rgba(251,191,36,0.15)", "#fbbf24"
    conf = max(r["pm"], r["pb"]) * 100
    roi_pills += (
        f'<span style="font-size:0.62rem;padding:1px 7px;border-radius:10px;'
        f'background:{bg_col};color:{txt_col};">'
        f'#{i+1} {r["lbl"]} {conf:.0f}%</span> &nbsp;'
    )

if final == "Uncertain":
    st.markdown(f"""<div class="diag-banner {dc}">
<div>
<div class="dmeta" style="margin-bottom:4px;">Diagnosis · {agg}</div>
<span class="badge {bc}">UNCERTAIN</span>
</div>
<div class="banner-right">
<div class="banner-conf" style="color:#{c['mid']};">
All <strong>{len(dets)}</strong> detected region{'s' if len(dets)!=1 else ''}
failed the OOD energy check (threshold {ood_thresh:.1f}).
This image may not contain recognizable bone tumor tissue, or may not be
a valid bone X-ray at all — treat any Benign/Malignant call here with caution.
</div>
<div style="margin-top:5px;">{roi_pills}</div>
</div>
</div>""", unsafe_allow_html=True)
else:
    n_ood_html = f'&nbsp;<strong style="color:#fbbf24">{n_ood}U</strong>' if n_ood else ''
    st.markdown(f"""<div class="diag-banner {dc}">
<div>
<div class="dmeta" style="margin-bottom:4px;">Diagnosis · {agg}</div>
<span class="badge {bc}">{final.upper()}</span>
</div>
<div class="banner-right">
<div class="ct" style="margin:6px 0 4px 0;">
<div class="{fc}" style="width:{cp}%"></div>
</div>
<div class="banner-conf" style="color:#{c['mid']};">
Confidence <strong style="color:#{c['strong']}">{cp}%</strong>
&nbsp;·&nbsp;
<strong>{len(dets)}</strong> ROI{'s' if len(dets)!=1 else ''}
&nbsp;·&nbsp;
<strong style="color:#f87171">{n_mal}M</strong>
<strong style="color:#38bdf8"> {n_ben}B</strong>
{n_ood_html}
&nbsp;·&nbsp; MaxP(M) <strong>{max_pm*100:.1f}%</strong>
</div>
<div style="margin-top:5px;">{roi_pills}</div>
</div>
</div>""", unsafe_allow_html=True)

# ── 2. Side-by-side: X-ray | GradCAM ─────────────────────────────────────────
col_img, col_cam = st.columns([1, 1], gap="medium")

with col_img:
    _buf2 = _io2.BytesIO(); ann.save(_buf2, format="JPEG", quality=90)
    _b64v = _b64mod.b64encode(_buf2.getvalue()).decode()
    zoom_style = f"transform:scale({zoom});transform-origin:center center;"
    st.markdown(
        f'<div class="img-box-short" style="overflow:{"hidden" if zoom==1.0 else "auto"};">'
        f'<img src="data:image/jpeg;base64,{_b64v}" '
        f'style="{zoom_style}max-width:{"none" if zoom>1 else "100%"};'
        f'max-height:{"none" if zoom>1 else "100%"};object-fit:contain;border-radius:6px;"/>'
        f'</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.58rem;color:#{c["dim"]};text-align:center;'
        f'margin-top:3px;">{up.name} · {W}×{H} · zoom {zoom:.2f}×</div>',
        unsafe_allow_html=True)

with col_cam:
    if do_cam:
        if not GRADCAM_OK:
            st.markdown(
                f'<div style="font-size:0.68rem;color:#c8b060;padding:8px 0;">'
                f'⚠ <code>pip install grad-cam</code></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.11em;'
                f'text-transform:uppercase;color:#{c["dim"]};margin-bottom:4px;">'
                f'🔍 GradCAM++'
                + (" · IG" if do_ig and CAPTUM_OK else "")
                + " · Malignant neuron</div>",
                unsafe_allow_html=True)
            st.markdown('<div class="right-scroll">', unsafe_allow_html=True)
            for r in rois:
                if r.get("fig"):
                    st.image(r["fig"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No GradCAM — show ROI detail cards instead
        st.markdown(
            f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:#{c["dim"]};margin-bottom:4px;">ROI Details</div>',
            unsafe_allow_html=True)
        rows_html = ""
        for i, (d, r) in enumerate(zip(dets, rois)):
            if r["lbl"] == "Malignant":
                rc, fc2 = "rm", "fm"
            elif r["lbl"] == "Benign":
                rc, fc2 = "rb", "fb"
            else:
                rc, fc2 = "ro", "fo"
            bw  = int(max(r["pm"], r["pb"]) * 100)
            rows_html += (
                f'<div class="rrow">'
                f'<span class="rlbl {rc}">#{i+1} {r["lbl"]}</span>'
                f'<div class="ct"><div class="{fc2}" style="width:{bw}%"></div></div>'
                f'<span style="font-size:0.60rem;color:#{c["dim"]};">'
                f'P(B)={r["pb"]*100:.0f}% P(M)={r["pm"]*100:.0f}% '
                f'det={d["score"]*100:.0f}% energy={r["energy"]:.2f}'
                f'</span></div>')
        st.markdown(f'<div class="rscroll">{rows_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION NOTE — how to set GRAY_THRESH_DEFAULT and OOD_ENERGY_DEFAULT
# (not executed by Streamlit — run this logic separately, e.g. in a notebook)
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Grayscale gate (GRAY_THRESH_DEFAULT):
#    Run looks_like_xray()'s chan_diff computation over a batch of your real
#    BTXRD images and confirm they all land near 0. The current 6.0 default
#    has generous headroom; tighten it only if you see false rejects on
#    legitimate (but noisy/colorized) X-rays.
#
# 2. OOD energy threshold (OOD_ENERGY_DEFAULT):
#    energies = []
#    for crop_tensor in val_roi_tensors:        # held-out, real bone-tumor ROI crops
#        with torch.no_grad():
#            logits = mdl(crop_tensor.to(dev))
#        energies.append(-torch.logsumexp(logits, dim=1).item())
#    print(np.percentile(energies, 99))          # use this as OOD_ENERGY_DEFAULT
#
#    This sets the threshold so you reject <1% of legitimate ROIs as false
#    positives. Sanity-check it by also running the same computation over a
#    handful of clearly non-bone images (natural photos, other modalities) —
#    their energy should sit well above the threshold.
#
# 3. (Optional, more rigorous) Mahalanobis distance on ConvNeXtTinySE features:
#    For a stronger OOD detector — worth it if this becomes part of a paper's
#    robustness/open-set section — add a forward hook to return the 768-d
#    pooled feature vector (after self.se + self.pool, before self.head),
#    fit per-class mean + shared covariance over training ROI features, and
#    threshold the Mahalanobis distance the same way as the energy score above.
# ══════════════════════════════════════════════════════════════════════════════