
import streamlit as st
import numpy as np
import cv2
import joblib

from PIL import Image
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

# Chargement du modèle sauvegardé
model = joblib.load("car_ship_svm.joblib")

st.set_page_config(
    page_title="Car vs Ship Classifier",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car vs Ship Classifier")
st.write(
    "Classification supervisée d'images avec "
    "RGB, HSV, GLCM, LBP, Hu et SVM RBF."
)


def extract_features(image):
    # Préparation : RGB, taille 32x32, pixels [0, 1]
    image = image.convert("RGB").resize((32, 32))
    img = np.asarray(image, dtype=np.float32) / 255.0

    # 1. RGB : moyenne et écart-type par canal
    rgb = []
    for c in range(3):
        rgb.extend([
            np.mean(img[:, :, c]),
            np.std(img[:, :, c])
        ])

    # 2. HSV : histogramme 3D (8, 2, 2)
    hsv_img = color.rgb2hsv(img)
    hsv_hist, _ = np.histogramdd(
        hsv_img.reshape(-1, 3),
        bins=(8, 2, 2),
        range=((0, 1), (0, 1), (0, 1))
    )
    hsv_hist = hsv_hist.astype(np.float32)
    hsv_hist /= hsv_hist.sum()

    # Niveaux de gris utilisés par GLCM, LBP et Hu
    gray = color.rgb2gray(img)
    gray_uint8 = (gray * 255).astype(np.uint8)

    # 3. GLCM : quatre propriétés
    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_features = [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0]
    ]

    # 4. LBP : histogramme de 10 valeurs
    lbp = local_binary_pattern(
        gray_uint8, 8, 1, method="uniform"
    )
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= lbp_hist.sum()

    # 5. Moments de Hu : sept valeurs transformées
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    # Fusion dans le même ordre que pendant l'apprentissage
    features = np.concatenate([
        np.asarray(rgb, dtype=np.float32),
        hsv_hist.flatten(),
        np.asarray(glcm_features, dtype=np.float32),
        lbp_hist,
        hu.astype(np.float32)
    ])

    return features.reshape(1, -1)


uploaded_file = st.file_uploader(
    "Importer une image de voiture ou de bateau",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Image importée", width=350)

    if st.button("Classifier l'image", type="primary"):
        features = extract_features(image)

        if features.shape[1] != 59:
            st.error("Erreur : le vecteur doit contenir 59 caractéristiques.")
        else:
            prediction = int(model.predict(features)[0])

            class_names = {
                0: "🚗 Voiture",
                1: "🚢 Bateau"
            }

            st.success(
                f"Classe prédite : {class_names[prediction]}"
            )

            st.caption(
                "Résultat produit par le modèle SVM RBF "
                "entraîné sur le dataset DB2C."
            )
