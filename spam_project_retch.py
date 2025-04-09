# 📊 Statistiques du modèle
if st.checkbox("Afficher les statistiques du modèle"):
    try:
        # Lire le fichier CSV
        df = pd.read_csv("spam.csv", encoding='latin-1')

        # Afficher les colonnes du fichier pour vérifier
        st.write("Colonnes du fichier CSV:", df.columns)

        # Si les colonnes ne sont pas celles attendues, on les renomme
        if 'label' not in df.columns or 'message' not in df.columns:
            st.error("Le fichier CSV ne contient pas les colonnes attendues : 'label' et 'message'.")
        else:
            # Renommer les colonnes si nécessaire
            df = df.rename(columns={"v1": "label", "v2": "message"})
            
            # Ajouter une colonne numérique pour 'label' (ham=0, spam=1)
            df['label_num'] = df['label'].map({"ham": 0, "spam": 1})

            # Nettoyage du texte et vectorisation
            X = df['message'].apply(nettoyer_texte)
            X_vect = vectorizer.transform(X)
            y_true = df['label_num']
            y_pred = model.predict(X_vect)

            # Calcul des statistiques
            precision = precision_score(y_true, y_pred)
            rappel = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            # Affichage des résultats
            st.write(f"**Taille du dataset** : {len(df)} messages")
            st.write(f"**Précision** : {precision:.2f}")
            st.write(f"**Rappel** : {rappel:.2f}")
            st.write(f"**F1-score** : {f1:.2f}")

    except Exception as e:
        st.error("Erreur lors du calcul des statistiques : " + str(e))
