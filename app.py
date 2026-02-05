import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Klasifikasi Penyakit Kulit", page_icon="🩺", layout="wide")

# ---------- Model load ----------
@st.cache_resource
def load_model_and_cfg():
    cfg = json.load(open("models/config.json", "r"))
    num_classes = int(cfg["num_classes"])

    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=num_classes)
    state = torch.load("models/best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize((int(cfg["img_size"]), int(cfg["img_size"]))),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    idx_to_class = {int(k): v for k, v in cfg["idx_to_class"].items()}
    return model, tfm, idx_to_class

def preprocess(pil_img, tfm):
    return tfm(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_top3(model, x, idx_to_class):
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idxs = torch.topk(probs, k=3)

    results = []
    for p, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        results.append({"label": idx_to_class[idx], "confidence": float(p)})
    return results

# ---------- Grad-CAM ----------
def find_target_layer(model):
    if hasattr(model, "conv_head"):
        return model.conv_head
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def gradcam(model, x, class_idx=None):
    model.eval()
    target = find_target_layer(model)
    if target is None:
        raise RuntimeError("Target conv layer tidak ditemukan untuk Grad-CAM.")

    activations = None
    gradients = None

    def fwd_hook(_, __, out):
        nonlocal activations
        activations = out

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    x = x.requires_grad_(True)
    logits = model(x)

    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())

    score = logits[0, class_idx]
    model.zero_grad()
    score.backward()

    h1.remove(); h2.remove()

    w = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (w * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_heatmap(pil_img, cam):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.62, heatmap, 0.38, 0)
    return Image.fromarray(overlay)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
  background: rgba(255,255,255,0.03);
}
.hr {margin: 0.7rem 0 0.9rem 0; border-bottom: 1px solid rgba(255,255,255,0.12);}
.badge {
  display: inline-block;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🩺 Klasifikasi Penyakit/Lesi Kulit")
st.caption("⚠️ Sistem pendukung keputusan berbasis citra, bukan pengganti diagnosis dokter.")

model, tfm, idx_to_class = load_model_and_cfg()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 Upload Gambar")
uploaded = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("Silakan upload gambar untuk mulai prediksi.")
    st.stop()

img = Image.open(uploaded)

colL, colR = st.columns([1.05, 0.95], gap="large")

with colL:
    st.subheader("📷 Preview Gambar")
    st.image(img, use_container_width=True)

with colR:
    st.subheader("📌 Hasil Prediksi")

    x = preprocess(img, tfm)
    top3 = predict_top3(model, x, idx_to_class)
    best = top3[0]

    st.markdown(f"### **{best['label']}**")
    st.write(f"Confidence: **{best['confidence']*100:.2f}%**")
    st.progress(int(best["confidence"] * 100))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("#### Top-3 Prediksi")
    for i, r in enumerate(top3, start=1):
        st.write(f"{i}. **{r['label']}** — {r['confidence']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧠 Explainability (Grad-CAM)")
st.caption("Heatmap menunjukkan area yang paling memengaruhi keputusan model.")
try:
    cam_map = gradcam(model, x, class_idx=None)
    overlay = overlay_heatmap(img, cam_map)
    st.image(overlay, use_container_width=True)
except Exception as e:
    st.error(f"Grad-CAM gagal: {e}")
    st.info("Kalau ini terjadi, kita bisa set target layer manual sesuai model yang kamu pakai.")
st.markdown("</div>", unsafe_allow_html=True)