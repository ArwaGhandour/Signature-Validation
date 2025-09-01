import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

st.title("Signature Comparison App 🖊️")

# رفع الصور
uploaded_files = st.file_uploader(
    "Upload EXACTLY TWO signature images (jpg/png)", 
    type=["jpg","png"], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 2:
    img1_file, img2_file = uploaded_files[0], uploaded_files[1]

    def load_and_preprocess(file, size=(300,300)):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        return img

    img1 = load_and_preprocess(img1_file)
    img2 = load_and_preprocess(img2_file)

    # ORB feature matching
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 60]
    similarity_score_orb = len(good_matches)
    max_possible_matches = min(len(kp1), len(kp2))
    orb_percent = (similarity_score_orb / max_possible_matches) * 100 if max_possible_matches > 0 else 0
