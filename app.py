import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="Image Similarity Checker", page_icon="🔍", layout="wide")

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_search = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_success = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_yjhbzzrc.json")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-title {
        font-size: 3em;
        color: #ff4b4b;
        text-align: center;
    }
    .upload-section {
        border: 2px dashed #ff4b4b;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .image-container {
        display: flex;
        justify-content: center;
    }
    .image-container img {
        width: 40%;
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to remove background using GrabCut
def remove_background(image):
    img = np.array(image)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1] - 30, img.shape[0] - 30)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    background = np.ones_like(img, dtype=np.uint8) * 255
    no_bg_img = np.where(img == 0, background, img)
    return Image.fromarray(no_bg_img)

# Function to extract features using VGG16
def extract_features_vgg16(img):
    model = VGG16(weights='imagenet', include_top=False)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Function to extract color histograms
def extract_color_histogram(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to extract LBP features
def extract_lbp_features(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def local_binary_pattern(image, P, R, method="default"):
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(R, image.shape[0] - R):
        for j in range(R, image.shape[1] - R):
            center = image[i, j]
            binary_string = ""
            for p in range(P):
                angle = 2 * np.pi * p / P
                x = i + R * np.sin(angle)
                y = j - R * np.cos(angle)
                if image[int(x), int(y)] > center:
                    binary_string += '1'
                else:
                    binary_string += '0'
            lbp[i, j] = int(binary_string, 2)
    return lbp

# Function to compare images using multiple features
def compare_images(image1, image2):
    features1_vgg = extract_features_vgg16(image1)
    features2_vgg = extract_features_vgg16(image2)
    hist1 = extract_color_histogram(image1)
    hist2 = extract_color_histogram(image2)
    lbp1 = extract_lbp_features(image1)
    lbp2 = extract_lbp_features(image2)
    similarity_vgg = cosine_similarity([features1_vgg], [features2_vgg])[0][0]
    similarity_hist = cosine_similarity([hist1], [hist2])[0][0]
    similarity_lbp = cosine_similarity([lbp1], [lbp2])[0][0]
    total_similarity = (0.5 * similarity_vgg) + (0.3 * similarity_hist) + (0.2 * similarity_lbp)
    return total_similarity

st.markdown("<h1 class='main-title'> Customer Image Review Similarity Checker </h1>", unsafe_allow_html=True)

uploaded_catalog_image = st.file_uploader("Upload Catalog Image", type=["jpg", "jpeg", "png"])
uploaded_customer_image = st.file_uploader("Upload Customer Image", type=["jpg", "jpeg", "png"])

if uploaded_catalog_image is not None and uploaded_customer_image is not None:
    catalog_image = Image.open(uploaded_catalog_image).convert('RGB')
    customer_image = Image.open(uploaded_customer_image).convert('RGB')

    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(catalog_image, caption="Catalog Image", use_column_width=False)
    st.image(customer_image, caption="Customer Image", use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner('Removing background from images...'):
        catalog_image_no_bg = remove_background(catalog_image)
        customer_image_no_bg = remove_background(customer_image)

    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(catalog_image_no_bg, caption="Catalog Image without Background", use_column_width=False)
    st.image(customer_image_no_bg, caption="Customer Image without Background", use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner('Comparing images...'):
        similarity = compare_images(catalog_image_no_bg, customer_image_no_bg)

    if similarity > 0.75:
        st.markdown("""
            <div style="text-align: center;">
                <h2>Congratulations! You have won a reward!</h2>
                <div style="width: 100%; display: flex; justify-content: center;">
                    # <iframe src="https://embed.lottiefiles.com/animation/42625" width="300" height="300"></iframe>
                </div>
                <p>Your images matched successfully. Enjoy your reward and keep sharing your style!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: center;">
                <h2>Sorry, the images are not similar enough.</h2>
                <div style="width: 100%; display: flex; justify-content: center;">
                    <iframe src="https://embed.lottiefiles.com/animation/81463" width="300" height="300"></iframe>
                </div>
                <p>Unfortunately, the images didn't match. Please try again with a different image.</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="text-align: center;">
            <h3>Upload both Catalog and Customer Images to start comparing!</h3>
            <div style="width: 100%; display: flex; justify-content: center;">
                <iframe src="https://embed.lottiefiles.com/animation/53909" width="300" height="300"></iframe>
            </div>
        </div>
    """, unsafe_allow_html=True)
