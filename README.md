# 🚢 Titanic Survival Predictor - Machine Learning Web App

Predict the survival of Titanic passengers using a trained ML model and a clean Gradio interface. Built with Python, deployed on Hugging Face Spaces.

[![Hugging Face Spaces](https://img.shields.io/badge/Live-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/praveenadi2005/titanic-gradio-app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/pravee-n2005/titanic-gradio-app)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Demo

👉 **Try the App Here:**  
https://huggingface.co/spaces/praveenadi2005/titanic-gradio-app

---

## 🧠 Project Overview

This app uses demographic and ticketing data to predict whether a Titanic passenger would survive.

### ✨ Features:
- Clean Gradio UI for easy prediction
- Feature engineering (Title, FamilySize, IsAlone, FarePerPerson)
- Model training using Random Forest (and more)
- Visualizations with Plotly (EDA)
- Deployed on Hugging Face Spaces

---

## 🛠 Tech Stack

- **Frontend/UI**: Gradio
- **ML/Backend**: Python, scikit-learn
- **Visualization**: Plotly
- **Deployment**: Hugging Face Spaces
- **Data Source**: Titanic dataset (Kaggle)

---

## 🧪 Local Setup

```bash
# Clone the repo
git clone https://github.com/pravee-n2005/titanic-gradio-app.git
cd titanic-gradio-app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
