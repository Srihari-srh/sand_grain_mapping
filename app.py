
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

st.set_page_config(page_title="Sand Grain Mapping Dashboard", layout="wide")

# ---- Udden-Wentworth sand grain-size scale (Wentworth, 1922) ----
WENTWORTH_SAND_CLASSES = [
    ("Very fine sand",   0.0625, 0.125),
    ("Fine sand",        0.125,  0.25),
    ("Medium sand",      0.25,   0.5),
    ("Coarse sand",      0.5,    1.0),
    ("Very coarse sand", 1.0,    2.0),
]

def classify_grain_size_mm(diameter_mm):
    if diameter_mm < WENTWORTH_SAND_CLASSES[0][1]:
        return "Silt / clay (finer than sand)"
    for name, lo, hi in WENTWORTH_SAND_CLASSES:
        if lo <= diameter_mm < hi:
            return name
    return "Granule / gravel (coarser than sand)"

def simplify_class(name):
    if "Very fine" in name or name == "Fine sand" or "Silt" in name:
        return "Fine"
    if name == "Medium sand":
        return "Medium"
    return "Coarse"

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Sand Grain Mapping Dashboard")
st.write("Upload an image to detect, count, measure, and classify sand grains (Fine / Medium / Coarse) using a trained YOLOv8 model.")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.selectbox("Inference image size", [640, 960, 1280], index=1)
    st.markdown("---")
    st.subheader("Calibration")
    st.caption("Enter how many millimeters one pixel represents, e.g. from a ruler/scale bar in your photo. Leave at 0 to skip and only see pixel sizes.")
    mm_per_pixel = st.number_input("mm per pixel", min_value=0.0, value=0.0, step=0.001, format="%.4f")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model.predict(source=img_array, imgsz=imgsz, conf=conf)
    r = results[0]
    annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

    n_grains = len(r.boxes)
    diameters_px = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        diameters_px.append(((x2 - x1) + (y2 - y1)) / 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    st.metric("Grains detected", n_grains)

    if mm_per_pixel and mm_per_pixel > 0 and n_grains > 0:
        diameters_mm = [d * mm_per_pixel for d in diameters_px]
        wentworth_classes = [classify_grain_size_mm(d) for d in diameters_mm]
        buckets = [simplify_class(c) for c in wentworth_classes]

        df = pd.DataFrame({
            "diameter_px": diameters_px,
            "diameter_mm": diameters_mm,
            "wentworth_class": wentworth_classes,
            "size_bucket": buckets,
        })

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg diameter (mm)", round(df["diameter_mm"].mean(), 4))
        m2.metric("Fine", int((df["size_bucket"] == "Fine").sum()))
        m3.metric("Medium", int((df["size_bucket"] == "Medium").sum()))
        m4.metric("Coarse", int((df["size_bucket"] == "Coarse").sum()))

        st.subheader("Fine / Medium / Coarse distribution")
        bucket_counts = df["size_bucket"].value_counts().reindex(["Fine", "Medium", "Coarse"]).fillna(0)
        st.bar_chart(bucket_counts)

        st.subheader("Detailed Wentworth classes")
        st.bar_chart(df["wentworth_class"].value_counts())

        with st.expander("Show raw grain measurements"):
            st.dataframe(df)
    elif n_grains > 0:
        avg_px = round(sum(diameters_px) / len(diameters_px), 2)
        st.metric("Average grain size (px)", avg_px)
        st.info("Set mm-per-pixel in the sidebar to classify grains as Fine / Medium / Coarse.")
else:
    st.info("Upload an image to get started.")
