import streamlit as st
from PIL import Image
import random
from AI_Generated_Image_Detector.app import predict


# --------- Données et modèle fictif ---------
STYLES = ["🎥 Ghibli", "💛 Simpsons", "🧙 Arcane", "🎩 JoJo", "🌀 AutreStyle"]
def predict_style(image):
    return random.choice(STYLES)

# --------- Gestion de session ---------


if "page" not in st.session_state:
    st.session_state.page = "intro"

# --------- PAGE 1 : INTRO ---------
if st.session_state.page == "intro":
    st.markdown("<h1 style='text-align: center;'>🎬 Bienvenue sur <span style='color:#FF4B4B;'>StyleVision</span> !</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Détectez automatiquement le style d'une image avec notre outil IA !</p>", unsafe_allow_html=True)
    st.image("https://media.giphy.com/media/QxkfTjJ84zc2c/giphy.gif", use_column_width=True)

    st.markdown("---")
    if st.button("🚀 Continuer vers la connexion"):
        st.session_state.page = "login"
        st.rerun()

# --------- PAGE 2 : CONNEXION ---------
elif st.session_state.page == "login":
    st.markdown("<h2 style='text-align: center;'>🔐 Connexion à l’espace utilisateur</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Veuillez saisir vos identifiants :</p>", unsafe_allow_html=True)

    username = st.text_input("👤 Nom d'utilisateur")
    password = st.text_input("🔒 Mot de passe", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Se connecter ✅"):
            if username == "admin" and password == "admin":
                st.success("Connexion réussie !")
                st.session_state.page = "interface"
                st.rerun()
            else:
                st.error("Identifiants incorrects ❌")
    with col2:
        if st.button("⬅️ Retour à l'accueil"):
            st.session_state.page = "intro"
            st.rerun()

# --------- PAGE 3 : INTERFACE STYLE ---------
elif st.session_state.page == "interface":
    
    st.markdown("<h2 style='text-align: center;'>🎨 Détecteur de Style d'image</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Importez une image et découvrez son style !</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📤 Importer une image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Image importée", use_column_width=True)
        #gen_ai, prediction, emissions, inf_time= predict(image)  ## Il faut mettre ça après le bouton c'est pour faire l'analyse

        if st.button("🔍 Lancer la détection"):
            style = predict_style(image)
            st.success(f"✅ Style détecté : **{style}**")
    else:
        st.info("🕹️ Veuillez déposer une image pour commencer.")

    st.markdown("---")
    if st.button("🔒 Se déconnecter"):
        st.session_state.page = "intro"
        st.rerun()