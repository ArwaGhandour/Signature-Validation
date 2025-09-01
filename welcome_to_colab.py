import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Signature Validation", layout="centered")
st.title("Signature Comparison App 🖊️")

st.markdown(
    "ارفع **صورتين توقيع** (JPG/PNG). هنحسب تشابه ORB + SSIM ونطلع قرار نهائي."
)

# ============= Helpers =============
def load_and_preprocess(uploaded_file, size=(300, 300)):
    """
    يقرأ ملف Streamlit UploadedFile بشكل آمن (getvalue) عشان re-run ما يبوّظش البوفر،
    ويحوّله لصورة رمادية ومقاس ثابت.
    """
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("تعذّر قراءة الصورة. تأكدي من أن الملف صورة JPG/PNG سليمة.")
    img = cv2.resize(img, size)
    return img

def compute_orb_similarity(img1, img2, max_features=5000, dist_thresh=60):
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # لو مفيش ديسكربتورز، نخلي النسبة 0 ونرجّع لستريملِت رسالة هادية
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, [], kp1 or [], kp2 or []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good = [m for m in matches if m.distance < dist_thresh]
    max_possible = min(len(kp1), len(kp2))
    orb_percent = (len(good) / max_possible) * 100 if max_possible > 0 else 0.0

    return orb_percent, good, kp1, kp2

def make_match_viz(img1, img2, kp1, kp2,
