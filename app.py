import streamlit as st
import joblib
import pickle
import os
import re
import string
import time
import hashlib
from cryptography.fernet import Fernet
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Téléchargement des ressources nécessaires de nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------------- CONFIGURATION DE LA PAGE ------------------------
st.set_page_config(page_title="SMS Classifier", layout="centered")

# Signature personnalisée
st.markdown("<div style='text-align: left; color: gray; font-size: 14px;'>👨‍💻 by Stéphane & Sanogo</div>", unsafe_allow_html=True)

# ---------------------- UTILISATEURS SIMULÉS ----------------------------
USERS = {
    "Stephane": {"password": "1234", "role": "data_scientist"},
    "Lionnel": {"password": "5678", "role": "analyst"}
}

# ---------------------- NETTOYAGE DU TEXTE -----------------------------
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r'http\S+|www\S+', '', texte)
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    tokens = texte.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ---------------------- CHARGEMENT DES MODÈLES --------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open("vectorizer.pkl", "rb") as vfile:
        vectorizer = pickle.load(vfile)
    model = joblib.load("spam_model.joblib")
    return model, vectorizer

@st.cache_resource
def load_fernet():
    if os.path.exists("fernet.key"):
        with open("fernet.key", "rb") as f:
            key = f.read()
        return Fernet(key)
    return None

# ---------------------- INTÉGRITÉ DU MODÈLE ----------------------------
def verifier_hash_model(filepath, hash_attendu):
    with open(filepath, "rb") as file:
        file_hash = hashlib.sha256(file.read()).hexdigest()
    return file_hash == hash_attendu

HASH_MODELE_ATTENDU = "votre_hash_attendu_ici"

# ---------------------- THÈME CLAIR/SOMBRE -----------------------------
thèmes = {"Clair": "#f5f5f5", "Sombre": "#2e2e2e"}
thème_choisi = st.sidebar.selectbox("Choisissez un thème", list(thèmes.keys()))
st.markdown(f"""
    <style>
        .main {{ background-color: {thèmes[thème_choisi]}; }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- CONNEXION UTILISATEUR --------------------------
st.sidebar.header("Connexion utilisateur")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if username in USERS and USERS[username]["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = USERS[username]["role"]
        st.sidebar.success(f"Connecté en tant que {username} ({st.session_state.role})")
    elif username and password:
        st.sidebar.error("Identifiants incorrects")
else:
    st.sidebar.success(f"Connecté en tant que {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Déconnexion"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------------- INTERFACE PRINCIPALE ---------------------------
if st.session_state.logged_in:
    model, vectorizer = load_model_and_vectorizer()
    fernet = load_fernet()

    if not verifier_hash_model("spam_model.joblib", HASH_MODELE_ATTENDU):
        st.error("🚨 L'intégrité du modèle n'est pas vérifiée !")

    st.subheader("📨 Analyse d'un SMS")
    sms_input = st.text_area("Entrez le message SMS à analyser")

    if "historique" not in st.session_state:
        st.session_state.historique = []

    if sms_input:
        langue = detect(sms_input)
        st.info(f"Langue détectée : **{langue.upper()}**")

        if langue != "en":
            sms_input = GoogleTranslator(source='auto', target='en').translate(sms_input)
            st.write("✏️ Message traduit en anglais pour l'analyse.")

        clean_text = nettoyer_texte(sms_input)
        vecteur = vectorizer.transform([clean_text])
        prediction = model.predict(vecteur)[0]
        label = "Spam" if prediction == 1 else "Ham"

        st.session_state.historique.append((sms_input, label))

        if st.session_state.role == "data_scientist":
            st.write("### Résultat complet")
            st.write(f"Vecteur TF-IDF : {vecteur.toarray()}")
            st.success(f"Prédiction : {label}")
        elif st.session_state.role == "analyst":
            st.write("### 🔎 Résultat")
            st.success(f"Prédiction : {label}")

    # 🔄 Afficher historique des prédictions
    if st.checkbox("Afficher l'historique des prédictions"):
        for i, (msg, lbl) in enumerate(st.session_state.historique[::-1]):
            st.write(f"{i+1}. **{lbl}** — {msg}")

    # 📊 Statistiques du modèle
    if st.checkbox("Afficher les statistiques du modèle"):
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/your-username/your-repo/main/spam.csv", encoding='latin-1')
            df = df.rename(columns={"v1": "label", "v2": "message"})
            df['label_num'] = df['label'].map({"ham": 0, "spam": 1})

            X = df['message'].apply(nettoyer_texte)
            X_vect = vectorizer.transform(X)
            y_true = df['label_num']
            y_pred = model.predict(X_vect)

            precision = precision_score(y_true, y_pred)
            rappel = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            st.write(f"**Taille du dataset** : {len(df)} messages")
            st.write(f"**Précision** : {precision:.2f}")
            st.write(f"**Rappel** : {rappel:.2f}")
            st.write(f"**F1-score** : {f1:.2f}")

        except Exception as e:
            st.error("Erreur lors du calcul des statistiques : " + str(e))
else:
    st.warning("Veuillez vous connecter pour accéder à l'application.")
