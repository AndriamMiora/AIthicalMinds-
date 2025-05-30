import streamlit as st
from PIL import Image
import random
from AI_Generated_Image_Detector.app import predict
from Predict_style.predict_style import predict_image_with_onnx

# --------- Démo IA ---------
def predict_style(image):
    result = predict_image_with_onnx(image, onnx_model_path=MODEL)
    #classe_predite": predicted_class,
        # "probabilite_predite": predicted_prob,
        # "temps_inference_s": end_inference - start_inference,
        # "emissions_CO2_kg": emissions,
        # "temps_total_s": end_global - start_global,
        # "probabilites_par_classe": dict(zip(class_names, probabilities))
    # Valeurs fixes pour la démo
    style = result["classe_predite"]  # Style fixe pour la démo
    proba_style = result["probabilite_predite"] # 92% de certitude
    carbone_style =result["emissions_CO2_kg"] # "0.25g CO₂"
    temps_style = result["temps_total_s"] #"1.2s"
    prob_par_class = result["probabilites_par_classe"] 
    return style, proba_style, carbone_style, temps_style, prob_par_class

def predict_AI(image):
    return predict(image)


#SI PB AVEC predict_AI faire : 

#def predict_AI(image):
#    result = predict(image)
#    return result["est_IA"], result["probabilite"], result["emissions_CO2_kg"], result["temps_inference_s"]


# --------- CSS global (à mettre en haut) ---------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #d9f3ff, #0072ff);
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    background-color: white;
    border-radius: 12px;
    padding: 2rem;
    margin: 2rem;
}
h1, h2, h3 {
    font-weight: 800;
    color: black;
    text-align: center;
}
p, label, span {
    font-size: 16px;
    color: #333;
    text-align: center;
}
.stButton > button {
    background: linear-gradient(to right, #12c2e9, #0072ff);
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px 24px;
    border-radius: 20px;
}
.stButton > button:hover {
    background: linear-gradient(to right, #0072ff, #12c2e9);
    color: white;
}
input[type="text"], input[type="password"] {
    background-color: #111;
    color: white;
    border: none;
    border-radius: 5px;
}
.css-1xarl3l {
    background-color: #f5f5f5;
    border: 2px dashed #0072ff;
    border-radius: 10px;
    padding: 1em;
}
.stAlert {
    border-left: 5px solid #FF4B4B;
    background-color: #fff5f5;
}
</style>
""", unsafe_allow_html=True)

# bouton JS simulé à la place de onclick (sinon ne fonctionne pas en Streamlit)
if st.session_state.get("page_trigger") == "team":
    st.session_state.page = "team"
    st.session_state.page_trigger = None
    st.rerun()
elif st.session_state.get("page_trigger") == "login":
    st.session_state.page = "login"
    st.session_state.page_trigger = None
    st.rerun()

# --------- Navigation ---------
if "connected" not in st.session_state:
    st.session_state.connected = False

if "page" not in st.session_state:
    st.session_state.page = "intro"

# --------- PAGE 1 : Accueil (intro) ---------
if st.session_state.page == "intro":
    # NAVIGATION (visible sur toutes les pages)
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.image("LOGO.png", width=80)
    with col2:
        st.markdown("### Home")
    with col3:
        if st.button("About Us"):
            st.session_state.page = "team"
            st.rerun()
    with col4:
        if st.session_state.get("connected"):
            if st.button("Sign Out"):
                st.session_state.connected = False
                st.success("Déconnexion réussie.")
                st.session_state.page = "intro"
                st.rerun()


    # STYLE DE FOND GRADIENT BLANC ➜ BLEU
    st.markdown("""
    <style>
    .main-container {
        display: flex;
        background: linear-gradient(to right, white 50%, #00c6ff, #0072ff);
        border-radius: 10px;
        padding: 30px;
        align-items: center;
        justify-content: space-between;
        min-height: 450px;
    }
    .left-content {
        width: 45%;
        padding-left: 30px;
    }
    .left-content h1 {
        font-size: 40px;
        font-weight: 900;
        color: black;
    }
    .left-content p {
        font-size: 16px;
        color: black;
        margin-bottom: 20px;
    }
    .right-content {
        width: 50%;
        text-align: center;
    }
    .right-content img {
        width: 80%;
        max-height: 300px;
    }
    .brand-title {
        font-size: 30px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # CONTENU PRINCIPAL
    # CONTENU PRINCIPAL AVEC IMAGE STREAMLIT (centrée dans la section droite)

    st.markdown("""
    <div class="main-container">
        <div class="left-content">
            <h1>Welcome to<br>StyleVision!</h1>
            <p>Automatically detect the style of an image with our AI tool!</p>
        </div>
        <div class="right-content">
            <div class="brand-title">StyleVision</div>
    """, unsafe_allow_html=True)

    # ⚠️ L’image doit être en dehors du HTML
    st.image("GIF_ACCUEIL.gif", use_container_width=True)

    # 🔚 On ferme le HTML après l’image
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)


    # BOUTON CONNECT EN BAS (visible aussi en mobile)
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if not st.session_state.connected:
            if st.button("Connect"):
                st.session_state.page = "login"
                st.rerun()
        else:
            if st.button("Modèle"):
                st.session_state.page = "interface"
                st.rerun()


# --------- PAGE 2 : Login ---------
elif st.session_state.page == "login":
    # NAVIGATION (visible sur toutes les pages)
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.image("LOGO.png", width=80)
    with col2:
        st.markdown("### Home")
    with col3:
        if st.button("About Us"):
            st.session_state.page = "team"
            st.rerun()
    with col4:
        if st.button("Home"):
            st.session_state.page = "intro"
            st.rerun()
    st.markdown("<h2 style='text-align: center; color: black; font-size: 32px;'>Espace utilisateur</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Veuillez saisir vos identifiants :</p>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Se connecter"):
            if username == "admin" and password == "admin":
                st.session_state.connected = True
                st.success("Connexion réussie !")
                st.session_state.page = "interface"
                st.rerun()

            else:
                st.error("Identifiants incorrects, veuillez vérifier vos identifiants")

# --------- PAGE 3 : Interface (upload) ---------

elif st.session_state.page == "interface":
        # NAVIGATION (visible sur toutes les pages)
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.image("LOGO.png", width=80)
    with col2:
        st.markdown("### Home")
    with col3:
        if st.button("About Us"):
            st.session_state.page = "team"
            st.rerun()
    with col4:
        if st.button("Home"):
            st.session_state.page = "intro"
            st.rerun()
    if not st.session_state.connected:
        st.warning("Vous devez être connecté pour accéder à cette page.")
        st.session_state.page = "login"
        st.rerun()

    st.markdown("<h1 style='color: black; text-align: center;'>Détecteur de Style d'image</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée succesfully", use_container_width=True)
        if st.button("START!"):
            st.session_state.image = image
            st.session_state.page = "resultat"
            st.rerun()
    else:
        st.info("Veuillez déposer une image pour commencer.")


# --------- PAGE 4 : Résultat ---------
elif st.session_state.page == "resultat":
            # NAVIGATION (visible sur toutes les pages)
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.image("LOGO.png", width=80)
    with col2:
        st.markdown("### Home")
    with col3:
        if st.button("About Us"):
            st.session_state.page = "team"
            st.rerun()
    with col4:
        if st.button("Home"):
            st.session_state.page = "intro"
            st.rerun()
    st.markdown("<h1 style='color: black; text-align: center;'>Résultats de l’analyse</h1>", unsafe_allow_html=True)

    image = st.session_state.get("image", None)

    if image:
        st.image(image, caption="Image analysée", use_container_width=True)
        is_ai, proba, carbon, inference_time = predict_AI(image)
        if not is_ai:
            st.warning("L'image n'a pas été générée par une IA.")
        else:
            st.success("L'image a été générée par une IA.")
            style, proba_style, carbone_style, temps_style ,prob_par_class = predict_style(image)
            if style == "🌀 AutreStyle":
                st.info("Aucun style détecté.")
            else:
                st.success(f"Style détecté : **{style}**")
                st.markdown(f" **Certitude style** : {proba_style*100:.2f}%")
                st.markdown(f" **Carbone style** : {carbone_style}")
                st.markdown(f" **Temps IA style** : {temps_style}")

        st.markdown(f"**Certitude IA** : {proba*100:.2f}%")
        st.markdown(f"**Carbone IA** : {carbon}")
        st.markdown(f" **Temps IA** : {inference_time}")
        
        if image and is_ai:
            st.markdown("### Probabilités par style détecté :")
            for classe, prob in prob_par_class.items():
                st.markdown(f"- **{classe}** : {prob*100:.2f}%")


    if st.button(" Retour"):
        st.session_state.page = "interface"
        st.rerun()

# --------- PAGE 5 : Équipe ---------
elif st.session_state.page == "team":
    # --- Barre de navigation ---
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col1:
        st.image("LOGO.png", width=80)
    with col2:
        st.markdown("### About Us")
    with col3:
        if st.button("Home"):
            st.session_state.page = "intro"
            st.rerun()
    with col4:
        if st.button("StyleVision"):
            st.session_state.page = "interface"
            st.rerun()
    # Colonnes : Texte + Image + Titre bleu
    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("""
        <div style='padding: 2rem; font-size: 16px; text-align: justify;'>
        <p>Nous sommes Ashley, Miora, Cloé et Linda, étudiantes 5ème année Polytech Sorbonne en Mathematiques Appliquées et Informatique, et nous participons au hackathon organisé par BEINK.</p>
        <p>Notre produit, <strong>StyleVision</strong>, permet d'analyser automatiquement le style visuel d'une image grâce à une IA conviviale et intuitive.</p>
        <p>Nous vous remercions chaleureusement pour votre attention, et nous vous souhaitons une excellente session d’imagerie !</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style='
            background: linear-gradient(to right, #00c6ff, #0072ff);
            border-radius: 15px;
            padding: 10px 10px;
            text-align: center;
            margin-bottom: 10px;
        '>
        <p style='color: white; font-size: 20px; font-weight: bold; margin: 5px;'>StyleVision</p>
        </div>
        """, unsafe_allow_html=True)

        st.image("photo_equipe.jpg", width=250)

