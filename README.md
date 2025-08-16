

# 🎭 Age, Gender & Emotion Detection

## 📖 Aperçu
Ce projet implémente un système basé sur l’intelligence artificielle permettant de détecter en **temps réel** l’âge, le genre et l’émotion d’une personne à partir d’une image ou d’une vidéo.  
Il s’appuie sur **MTCNN** pour la détection des visages et sur des **CNN spécialisés** pour la classification.  

📄 Plus d’informations sont disponibles dans l’article associé :  
👉 [Lire l’article (PDF)]("https://github.com/benchekchou/age-emtion-genre-detection/blob/main/article.pdf")

---

## ✨ Fonctionnalités
- Détection de visages en conditions réelles (éclairage faible, angles variés, occlusions).  
- Prédiction simultanée de l’âge, du genre et de l’émotion.  
- Interface simple d’utilisation pour images et vidéos.  
- Architecture modulaire et extensible.  

---

## 📊 Résultats
- Âge : **89%** de précision  
- Genre : **88%** de précision  
- Émotions : **99%** de précision  
- Moyenne globale : **≈92%**  

---

## ⚙️ Installation
```bash
git clone https://github.com/benchekchou/age-emtion-genre-detection.git
cd age-emtion-genre-detection
pip install -r requirements.txt
````

---

## ▶️ Utilisation

```bash
# Pour une image
python detect.py --image chemin/vers/image.jpg

# Pour une vidéo
python detect.py --video chemin/vers/video.mp4
```

---

## 📜 Licence

Projet distribué sous licence **MIT**.

```

---

Veux-tu que je garde aussi une **section schéma d’architecture** (visuel ou mermaid) ou tu préfères laisser ce README très minimaliste comme ça ?
```
