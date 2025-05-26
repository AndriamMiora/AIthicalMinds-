<<<<<<< HEAD
import streamlit as st
from PIL import Image
import random

# --------- Données et modèle fictif ---------
STYLES = ["🎥 Ghibli", "💛 Simpsons", "🧙 Arcane", "🎩 JoJo", "🌀 AutreStyle"]


import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def predict_style(image):
    # Valeurs fixes pour la démo
    style = "🌀 AutreStyle"  # Style fixe pour la démo
    proba_style = 0.92   # 92% de certitude
    carbone_style = "0.25g CO₂"
    temps_style = "1.2s"
    return style, proba_style, carbone_style, temps_style

def predict_AI(image):
    # Valeurs fixes pour la démo
    is_ai = False         # Toujours détecter comme IA pour cet exemple
    proba = 0.95         # 95% de certitude
    carbon = "0.18g CO₂" # Émission carbone fixe
    inference_time = "0.8s" # Temps d'inférence fixe
    return is_ai, proba, carbon, inference_time

st.markdown("""
    <style>
    /* Met tout le fond principal en blanc */
    .stApp {
        background-color: white !important;
        color: black !important;
    }

    /* Conteneur principal */
    .block-container {
        background-color: white !important;
        color: black !important;
    }

    /* Titres et paragraphes */
    h1, h2, p, label {
        color: black !important;
        text-align: center;
    }

    /* Boutons Streamlit */
    .stButton > button {
        background: linear-gradient(to right, #74ebd5, #acb6e5);
        color: black !important;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 12px;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #acb6e5, #74ebd5);
        color: black !important;
    }

    /* Uploader */
    .uploadedFile {
        background-color: rgba(0, 0, 0, 0.05);
        color: black !important;
    }

    /* Alertes (success, error, info...) */
    .stAlert {
        background-color: rgba(0, 0, 0, 0.05) !important;
        border-left: 4px solid #FF4B4B;
    }
    
    /* Ajoute le logo en haut à gauche */
.logo-container {
    position: fixed;
    top: 15px;
    left: 15px;
    z-index: 999;
}

.logo-container img {
    max-height: 100px;  /* Ajuste ici selon ta préférence */
    height: auto;
    width: auto;
}

    /* Fond blanc et texte noir */
    .stApp, .block-container {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, p, label {
        color: black !important;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(to right, #74ebd5, #acb6e5);
        color: black !important;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 12px;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #acb6e5, #74ebd5);
        color: black !important;
    }

    .uploadedFile {
        background-color: rgba(0, 0, 0, 0.05);
        color: black !important;
    }

    .stAlert {
        background-color: rgba(0, 0, 0, 0.05) !important;
        border-left: 4px solid #FF4B4B;
    }
    
        .stApp {
        background-image: url("background.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.8); /* Optionnel : fond blanc semi-transparent pour les blocs */
        padding: 2rem;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)
bg_img = get_base64_of_bin_file("LOGO.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    .block-container {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="logo-container">
    <img src="LOGO.png" style="height: auto; max-height: 100px; width: auto;">
</div>

""", unsafe_allow_html=True)


# --------- Données et modèle fictif ---------
STYLES = ["🎥 Ghibli", "💛 Simpsons", "🧙 Arcane", "🎩 JoJo", "🌀 AutreStyle"]
def predict_style(image):
    return random.choice(STYLES)

# --------- Gestion de session ---------


if "page" not in st.session_state:
    st.session_state.page = "intro"

# --------- PAGE 1 : INTRO ---------
if st.session_state.page == "intro":
    st.markdown("<h1 style='text-align: center;'> Bienvenue sur <span style='color:#FF4B4B;'>StyleVision</span> !</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Détectez automatiquement le style d’une image avec notre outil IA !</p>", unsafe_allow_html=True)
    st.image("GIF_ACCUEIL.gif", use_container_width=True)

    
    
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("Continuer vers la connexion ⮕"):
            st.session_state.page = "login"
            st.rerun()
            
    st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 30px;">

<p style='text-align: center; font-size: 16px; color: gray;'>
Nous sommes <strong>Ashley</strong>, <strong>Miora</strong>, <strong>Cloé</strong> et <strong>Linda</strong>, étudiantes 5ème année <strong>Polytech Sorbonne</strong>, et nous participons au hackathon organisé par BEINK .<br><br>
Notre produit, <em>StyleVision</em>, permet d'analyser automatiquement le style visuel d'une image grâce à une IA conviviale et intuitive.<br><br>
Nous vous remercions chaleureusement pour votre attention, et nous vous souhaitons une excellente session d’imagerie !
</p>
""", unsafe_allow_html=True)




# --------- PAGE 2 : CONNEXION ---------
elif st.session_state.page == "login":
    st.markdown("<h2 style='text-align: center;'>Espace utilisateur</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Veuillez saisir vos identifiants :</p>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Se connecter"):
            if username == "admin" and password == "admin":
                st.success("Connexion réussie !")
                st.session_state.page = "interface"
                st.rerun()
            else:
                st.error("Identifiants incorrects, veuillez vérifier vos identifiants")
    with col1:
        if st.button("⬅ Retour à l'accueil"):
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
        st.image(image, caption="🖼️ Image importée", use_container_width=True)

        if st.button("🔍 Lancer la détection"):
            # D'abord vérifier si c'est une IA
            is_ai, proba, carbon, inference_time = predict_AI(image)

            if not is_ai:
                st.warning("L'image n'a pas été générée par une IA. Fin du traitement")
                st.write(f"🔎 Certitude (Accuracy) : {proba*100:.2f}%")
                st.write(f"🌱 Code carbone estimé : {carbon}")
                st.write(f"⏱️ Temps d'inférence : {inference_time}")
            else:
                st.success("L'image a été générée par une IA.")
                st.write(f"🔎 Certitude (Accuracy) : {proba*100:.2f}%")
                st.write(f"🌱 Code carbone estimé : {carbon}")
                st.write(f"⏱️ Temps d'inférence : {inference_time}")

                # Si c'est une IA, détecter le style
                style, proba_style, carbone_style, temps_style = predict_style(image)

                if style == "🌀 AutreStyle":
                    st.info("Aucun style détecté dans notre base de données.")
                else:
                    st.success(f"✅ Style détecté : **{style}**")
                    st.write(f"🔎 Certitude (Accuracy) : {proba_style*100:.2f} %")
                    st.write(f"🌱 Code carbone estimé : {carbone_style}")
                    st.write(f"⏱️ Temps d'inférence : {temps_style}")
    else:
        st.info("🕹️ Veuillez déposer une image pour commencer.")

    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("🔒 Se déconnecter"):
            st.session_state.page = "intro"
            st.rerun()