

````markdown
# 🎭 Age, Gender & Emotion Detection

## 📌 Description
Ce projet propose un modèle intelligent de reconnaissance faciale en temps réel, capable de détecter **l’âge, le genre et l’émotion** d’une personne à partir d’une image ou d’une vidéo.  
Il combine **MTCNN** pour la détection de visages et des **CNNs spécialisés** pour la classification des attributs (âge, genre et émotions).  

📄 Le projet est détaillé dans l’article scientifique suivant :  
👉 [Toward Affective Intelligence: Real-Time Age, Gender, and Emotion Detection via Multi-CNN and MTCNN Integration](./output_47.pdf)

---

## 🚀 Fonctionnalités
- Détection en temps réel des visages.
- Classification de l’âge, du genre et de l’émotion.
- Interface simple pour tester avec des images.
- Résultats robustes même sous des conditions difficiles (éclairage faible, angles variés, occlusions partielles).

---

## 🧠 Modèles & Jeux de Données
- **MTCNN** : détection et alignement des visages.
- **CNN spécialisés** :
- **Âge** : UTKFace, Combined_Faces
- **Genre** : UTKFace
 - **Émotions** : CK+48 (7 émotions : joie, tristesse, colère, dégoût, surprise, peur, neutralité)

**Performances obtenues :**
- Âge : 89% de précision
- Genre : 88% de précision
- Émotions : 99% de précision
- Moyenne globale : ~92%

---

## 📊 Architecture
```mermaid
flowchart TD
    A[Image/Vidéo] --> B[MTCNN - Détection du visage]
    B --> C[Prétraitement (resize, normalisation...)]
    C --> D1[Âge - CNN]
    C --> D2[Genre - CNN]
    C --> D3[Émotions - CNN]
    D1 --> E[Fusion & Résultats]
    D2 --> E
    D3 --> E
    E --> F[Affichage : âge, genre, émotion]
````

---

## ⚙️ Installation

```bash
# Cloner le projet
git clone https://github.com/benchekchou/age-emtion-genre-detection.git
cd age-emtion-genre-detection

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # sous Linux/Mac
venv\Scripts\activate      # sous Windows

# Installer les dépendances
pip install -r requirements.txt
```

---


---

## 📂 Structure du Projet

```
age-emtion-genre-detection/
│── models/           # Modèles CNN pré-entraînés
│── datasets/         # Jeux de données (si disponibles localement)
│── detect.py         # Script principal de détection
│── utils/            # Fonctions utilitaires (prétraitement, affichage...)
│── requirements.txt  # Dépendances
│── output_47.pdf     # Article scientifique du projet
│── README.md         # Ce fichier
```

---

## 👥 Auteurs

* Hamza Benchekchou – [GitHub](https://github.com/benchekchou)
* Halima Rissouni

---

## 📜 Licence

Ce projet est publié sous licence **MIT** – libre pour utilisation, modification et distribution.

---

✨ N’hésitez pas à forker le projet, proposer des améliorations ou l’utiliser dans vos propres applications liées à l’IA et à la reconnaissance faciale !

```

---

Veux-tu que je prépare aussi un **badge Shields.io** (par ex. Python, TensorFlow, Licence MIT) et un **aperçu visuel (screenshot ou GIF de démo)** dans le README pour le rendre plus attractif ?
```
