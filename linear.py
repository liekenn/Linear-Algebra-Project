import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

st.set_page_config(page_title="Matrix Transformations App", layout="centered")

# =========================
# LANGUAGE SYSTEM
# =========================

LANGUAGES = {
    "INDONESIA": {
        "title": "Aplikasi Transformasi Matriks",
        "upload": "Unggah Gambar",
        "feature": "Pilih Fitur",
        "translation": "Translasi",
        "scaling": "Skala",
        "rotation": "Rotasi",
        "shearing": "Shearing",
        "reflection": "Refleksi",
        "blur": "Blur",
        "sharpen": "Sharpen",
        "processed": "Hasil",
        "original": "Gambar Asli",
        "matrix": "Matriks Transformasi",
        "download": "Unduh Gambar Hasil Proses (.JPG)", # Diubah ke JPG
    },
    "ENGLISH": {
        "title": "Matrix Transformation App",
        "upload": "Upload Image",
        "feature": "Choose Feature",
        "translation": "Translation",
        "scaling": "Scaling",
        "rotation": "Rotation",
        "shearing": "Shearing",
        "reflection": "Reflection",
        "blur": "Blur",
        "sharpen": "Sharpen",
        "processed": "Processed Image",
        "original": "Original Image",
        "matrix": "Transformation Matrix",
        "download": "Download Processed Image (.JPG)", # Diubah ke JPG
    },
    "CHINA": {
        "title": "矩阵变换应用",
        "upload": "上传图片",
        "feature": "选择功能",
        "translation": "平移",
        "scaling": "缩放",
        "rotation": "旋转",
        "shearing": "错切",
        "reflection": "反射",
        "blur": "模糊",
        "sharpen": "锐化",
        "processed": "处理后图像",
        "original": "原始图像",
        "matrix": "变换矩阵",
        "download": "下载处理后图像 (.JPG)",
    },
    "JAPAN": {
        "title": "行列変換アプリ",
        "upload": "画像をアップロード",
        "feature": "機能を選択",
        "translation": "平行移動",
        "scaling": "拡大縮小",
        "rotation": "回転",
        "shearing": "シアー",
        "reflection": "反射",
        "blur": "ぼかし",
        "sharpen": "シャープ",
        "processed": "処理画像",
        "original": "元画像",
        "matrix": "変換行列",
        "download": "処理画像をダウンロード (.JPG)",
    },
    "KOREA": {
        "title": "행렬 변환 앱",
        "upload": "이미지 업로드",
        "feature": "기능 선택",
        "translation": "이동",
        "scaling": "스케일링",
        "rotation": "회전",
        "shearing": "전단",
        "reflection": "반사",
        "blur": "블러",
        "sharpen": "샤픈",
        "processed": "처리된 이미지",
        "original": "원본 이미지",
        "matrix": "변환 행렬",
        "download": "처리된 이미지 다운로드 (.JPG)",
    }
}

# pilih bahasa
choice = st.sidebar.selectbox("LANGUAGE", ["INDONESIA", "ENGLISH", "JAPAN", "CHINA", "KOREA"])
TXT = LANGUAGES[choice]

# =========================
# IMAGE HELPERS
# =========================

def to_numpy(img):
    return np.array(img.convert("RGB"))

def to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

# --- PERUBAHAN DI SINI ---
def get_image_download_link(img_pil):
    """Menyimpan gambar PIL ke byte stream dalam format JPEG."""
    buf = io.BytesIO()
    # Menyimpan sebagai JPEG. Quality 95 adalah kualitas tinggi.
    img_pil.save(buf, format="JPEG", quality=95) 
    byte_im = buf.getvalue()
    return byte_im
# -------------------------

# =========================
# MATRIX OPS
# =========================
def translation_matrix(tx, ty):
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], float)

def scaling_matrix(sx, sy):
    return np.array([[sx,0,0],[0,sy,0],[0,0,1]], float)

def rotation_matrix(theta):
    rad = np.deg2rad(theta)
    c,s = np.cos(rad), np.sin(rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def shearing_matrix(shx, shy):
    return np.array([[1,shx,0],[shy,1,0],[0,0,1]], float)

def reflection_matrix(axis):
    if axis=="x": return np.array([[1,0,0],[0,-1,0],[0,0,1]],float)
    if axis=="y": return np.array([[-1,0,0],[0,1,0],[0,0,1]],float)
    if axis=="both": return np.array([[-1,0,0],[0,-1,0],[0,0,1]],float)
    return np.eye(3)

# =========================
# APPLY MATRIX
# =========================
def apply_transform(img_arr, M):
    h, w = img_arr.shape[:2]
    ys, xs = np.indices((h,w))
    ones = np.ones_like(xs)
    coords = np.stack([xs.ravel(), ys.ravel(), ones.ravel()])
    Minv = np.linalg.inv(M)
    mapped = Minv @ coords
    xmap = mapped[0].reshape(h,w).astype("float32")
    ymap = mapped[1].reshape(h,w).astype("float32")
    out = cv2.remap(img_arr, xmap, ymap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out

# =========================
# MANUAL CONVOLUTION
# =========================
def apply_convolution(img_arr, kernel):
    img = img_arr.astype(float)
    h, w, ch = img.shape
    k = np.array(kernel)
    pad = k.shape[0] // 2
    
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='edge')
    out = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+3, x:x+3]
            for c in range(ch):
                out[y,x,c] = np.sum(region[:,:,c] * k)

    return np.clip(out, 0, 255).astype("uint8")

# =========================
# UI SECTION
# =========================
st.title(TXT["title"])

uploaded = st.file_uploader(TXT["upload"], type=["jpg","png","jpeg"])
if uploaded is None:
    st.stop()

image = Image.open(uploaded)
img_arr = to_numpy(image)

feature = st.sidebar.selectbox(TXT["feature"], [
    TXT["translation"], TXT["scaling"], TXT["rotation"], TXT["shearing"],
    TXT["reflection"], TXT["blur"], TXT["sharpen"]
])

processed = img_arr.copy()
M = np.eye(3)

# =========================
# APPLY SELECTED FEATURE
# =========================

if feature == TXT["translation"]:
    tx = st.sidebar.slider("tx", -300,300,0)
    ty = st.sidebar.slider("ty", -300,300,0)
    M = translation_matrix(tx,ty)
    processed = apply_transform(img_arr, M)

elif feature == TXT["scaling"]:
    sx = st.sidebar.slider("sx", 0.1,3.0,1.0)
    sy = st.sidebar.slider("sy", 0.1,3.0,1.0)
    cx, cy = img_arr.shape[1]/2, img_arr.shape[0]/2
    M = translation_matrix(cx,cy) @ scaling_matrix(sx,sy) @ translation_matrix(-cx,-cy)
    processed = apply_transform(img_arr, M)

elif feature == TXT["rotation"]:
    angle = st.sidebar.slider("angle", -180,180,0)
    cx, cy = img_arr.shape[1]/2, img_arr.shape[0]/2
    M = translation_matrix(cx,cy) @ rotation_matrix(angle) @ translation_matrix(-cx,-cy)
    processed = apply_transform(img_arr, M)

elif feature == TXT["shearing"]:
    shx = st.sidebar.slider("shx", -1.0,1.0,0.0)
    shy = st.sidebar.slider("shy", -1.0,1.0,0.0)
    cx, cy = img_arr.shape[1]/2, img_arr.shape[0]/2
    M = translation_matrix(cx,cy) @ shearing_matrix(shx,shy) @ translation_matrix(-cx,-cy)
    processed = apply_transform(img_arr, M)

elif feature == TXT["reflection"]:
    axis = st.sidebar.selectbox("Axis", ["x","y","both"])
    cx, cy = img_arr.shape[1]/2, img_arr.shape[0]/2
    M = translation_matrix(cx,cy) @ reflection_matrix(axis) @ translation_matrix(-cx,-cy)
    processed = apply_transform(img_arr, M)

elif feature == TXT["blur"]:
    kernel = np.ones((3,3))/9
    processed = apply_convolution(img_arr, kernel)
    M = np.eye(3) 

elif feature == TXT["sharpen"]:
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    processed = apply_convolution(img_arr, kernel)
    M = np.eye(3) 

# =========================
# SHOW IMAGES
# =========================

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption=TXT["original"], use_container_width=True)
with col2:
    processed_pil = to_pil(processed)
    st.image(processed_pil, caption=TXT["processed"], use_container_width=True)

st.sidebar.subheader(TXT["matrix"])
if feature not in [TXT["blur"], TXT["sharpen"]]:
    matrix_str = ' \\\\ '.join(' & '.join(f'{x:.2f}' for x in row) for row in M)
    st.sidebar.latex(f"\\mathbf{{M}} = \\begin{{pmatrix}} {matrix_str} \\end{{pmatrix}}")
else:
    st.sidebar.markdown("Matriks Konvolusi (Kernel) yang Digunakan:")
    st.sidebar.code(str(kernel).replace(' ', '  ')) 

# =========================
# DOWNLOAD FEATURE (SEKARANG JPG)
# =========================

st.markdown("---")
st.download_button(
    label=TXT["download"],
    data=get_image_download_link(processed_pil),
    file_name="processed_image.jpg", # Nama file diubah ke .jpg
    mime="image/jpeg" # MIME type diubah ke image/jpeg
)