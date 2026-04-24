import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Cargar modelos


@st.cache_resource
def load_models():
    with open("notebooks/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("notebooks/models/final_model.pkl", "rb") as f:  # ← final_model
        modelo = pickle.load(f)
    return scaler, modelo


scaler, modelo = load_models()

# Cargar datos de referencia para SHAP


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop("target", axis=1)
    return X


X_ref = load_reference_data()
X_ref_scaled = scaler.transform(X_ref)

# Título
st.title("🔬 Breast Cancer Classifier")
st.markdown("""
This tool predicts whether a breast tumor is **malignant or benign** based on 
cell nucleus measurements from a Fine Needle Aspiration (FNA) biopsy.

> ⚠️ **Disclaimer:** This model is a portfolio project and should never replace 
> the clinical judgment of a medical professional.
""")

st.divider()

# Sidebar, inputs del usuario
st.sidebar.header("🧬 Cell Nucleus Measurements")
st.sidebar.markdown("Adjust the values based on the biopsy report.")


def user_input():
    data = {}
    features = X_ref.columns.tolist()
    for feature in features:
        min_val = float(X_ref[feature].min())
        max_val = float(X_ref[feature].max())
        mean_val = float(X_ref[feature].mean())
        data[feature] = st.sidebar.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            format="%.4f"
        )
    return pd.DataFrame([data])


input_df = user_input()

# Predicción
input_scaled = scaler.transform(input_df)
prediction = modelo.predict(input_scaled)[0]
probability = modelo.predict_proba(input_scaled)[0]

st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 0:
        st.error("🔴 **Malignant** tumor predicted")
    else:
        st.success("🟢 **Benign** tumor predicted")

with col2:
    st.metric("Probability of Malignant", f"{probability[0]:.2%}")
    st.metric("Probability of Benign", f"{probability[1]:.2%}")

st.divider()

# SHAP local
st.subheader("🔍 Why did the model predict this?")
st.markdown(
    "The chart below shows which features pushed the prediction toward malignant or benign.")

explainer = shap.LinearExplainer(modelo, X_ref_scaled)
shap_values = explainer.shap_values(input_scaled)

fig, ax = plt.subplots(figsize=(9, 4))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=X_ref.columns.tolist()
    ),
    show=False
)
plt.tight_layout()
st.pyplot(fig)
plt.close()
