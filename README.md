# 💊 MED+ — A Disease Prediction System Using Machine Learning

> *Get the feels before the fever. Diagnose smarter, not harder.* 😎🩺

MED+ is your AI-powered sidekick for early-stage disease prediction. Just select your symptoms, and this machine-learning-powered web app will spit out a full diagnosis report — predicted disease, probability score, home remedies, and even specialist suggestions with nearby doctor info.

⚠️ *Disclaimer: MED+ is for educational and awareness purposes only. Not a substitute for professional medical advice. Don't sue us, bro.* 😅

---

## 🚀 Features

- 🔍 **Symptom-Based Prediction** — Drop down your pain points and we handle the logic.
- 🤖 **ML-Powered Diagnosis** — Powered by Random Forest, Decision Tree, and SVM models.
- 📊 **Probabilistic Analysis** — See how likely your condition is (based on your inputs).
- 🏡 **Home Remedies** — Temporary care tips before you sprint to the clinic.
- 🧑‍⚕️ **Doctor Suggestions** — Specialist type + contact info of nearby pros.
- 🌐 **Web App Interface** — Built with Flask, templates, and clean CSS vibes.

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| 💬 Language | Python |
| 📦 Backend | Flask |
| 🧠 ML Models | Scikit-learn (RF, DT, SVM) |
| 📊 Data | Handcrafted datasets (Kaggle who? 😤) |
| 🎨 Frontend | HTML, CSS, Jinja2 Templates |

---

## 🛠️ How It Works

1. **Pick your symptoms** from a dropdown list.
2. **The model predicts** the most probable disease.
3. You get a full report including:
   - 🧬 Disease Name
   - 📈 Prediction Probability
   - 🛏️ Home Remedy Suggestions
   - 🩺 Specialist Type
   - 📍 Nearby Doctor/Hospital Info

---

## 🧪 Model Info

- **Random Forest**: Our main MVP for accuracy.
- **Decision Tree**: Quick & interpretable.
- **SVM**: Great for sharp boundaries (and sharp code).

---

## 🧰 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/KunalBadave07/MED-.git
cd MED-

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

# Open your browser
localhost:5000
