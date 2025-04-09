import streamlit as st
import joblib
import pickle
import os
import re
import string
from cryptography.fernet import Fernet
import hashlib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException

# Téléchargement des ressources nécessaires de nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Utilisateurs simulés
USERS = {
    "Stephane": {"password": "1234", "role": "data_scientist"},
    "Lionnel": {"password": "5678", "role": "analyst"}
}

# Fonction de nettoyage
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

# Détection de langue
def detecter_langue(texte):
    try:
        return detect(texte)
    except LangDetectException:
        return "unknown"

# Chargement des modèles
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

# Interface utilisateur
st.set_page_config(page_title="Détecteur de SMS", page_icon="📨")
st.title(" Interface sécurisée de classification des SMS")
st.sidebar.header("Connexion utilisateur")

username = st.sidebar.text_input("Nom d'utilisateur")
password = st.sidebar.text_input("Mot de passe", type="password")

authenticated = False
role = None

if username in USERS and USERS[username]["password"] == password:
    authenticated = True
    role = USERS[username]["role"]
    st.sidebar.success(f"Connecté en tant que {username} ({role})")
else:
    if username and password:
        st.sidebar.error("Identifiants incorrects")

# Interface principale après authentification
if authenticated:
    model, vectorizer = load_model_and_vectorizer()
    fernet = load_fernet()

    st.subheader(" Analyse d'un SMS")
    sms_input = st.text_area("Entrez le message SMS à analyser")

    if sms_input:
        langue = detecter_langue(sms_input)
        st.info(f"🌍 Langue détectée : **{langue}**")

        if langue != "en":
            st.warning("⚠️ Le modèle est entraîné uniquement sur des messages en **anglais**. Les résultats peuvent être incorrects.")
        else:
            clean_text = nettoyer_texte(sms_input)
            vecteur = vectorizer.transform([clean_text])
            prediction = model.predict(vecteur)[0]
            label = "Spam" if prediction == 1 else "Ham"

            if role == "data_scientist":
                st.write("### 🔬 Résultat complet")
                st.write(f"Vecteur TF-IDF : {vecteur.toarray()}")
                st.success(f"✅ Prédiction : **{label}**")
            elif role == "analyst":
                st.write("### 🔎 Résultat")
                st.success(f"✅ Prédiction : **{label}**")
else:
    st.warning("Veuillez vous connecter pour accéder à l'application.")
