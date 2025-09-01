import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import tempfile
import os

# -----------------------
# Helper: ORB matching
# -----------------------
def orb_match(img1, img2):
    orb = cv2.ORB_create()

    # keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, None, kp1, kp2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if not matches:
        return 0, None, kp1, kp2

    matches = sorted(matches, key=lambda x: x.distance)
    score = sum([1 - (m.distance / 100) for m in matches]) / len(matches)
    return score, matches, kp1, kp2

# -----------------------
# Helper: SSIM
# -----------------------
def ssim_score(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    s, _ = ssim(img1_gray, img2_gray, full=True)
    return s

# -----------------------
# Visualization of matches
# -----------------------
def make_match_viz(img1, img2, kp1, kp2, matches, max_matches=20):
    return cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

# -----------------------
# Streamlit app
# -----------------------
st.title("Signature Verification ✍️")
st.write("Upload two signature images to compare if they match.")

# Upload images
img1_file = st.file_uploader("Upload first signature", type=["jpg", "png", "jpeg"])
img2_file = st.file_uploader("Upload second signature", type=["jpg", "png", "jpeg"])

if img1_file and img2_file:
    # OpenCV images
    img1 = np.array(Image.open(img1_file).convert("RGB"))
    img2 = np.array(Image.open(img2_file).convert("RGB"))

    img1_cv = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # ORB
    orb_score, matches, kp1, kp2 = orb_match(img1_cv, img2_cv)

    # SSIM
    ssim_val = ssim_score(img1_cv, img2_cv)

    # Results
    st.subheader("Results")
    st.write(f"🔍 **ORB similarity score:** {orb_score:.2f}")
    st.write(f"🖼️ **SSIM score:** {ssim_val:.2f}")

    if orb_score > 0.5 and ssim_val > 0.5:
        st.success("✅ Signatures match")
    else:
        st.error("❌ Signatures do not match")

    # Show visualizations
    if matches:
        match_img = make_match_viz(img1_cv, img2_cv, kp1, kp2, matches)
        st.image(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB), caption="Feature Matching")
