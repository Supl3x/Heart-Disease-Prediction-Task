import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="💟 Heart Disease Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")
    return df

df = load_data()

# --- TITLE ---
st.title("💟 Heart Disease Predictor")
st.markdown("Using Logistic Regression to predict heart disease based on patient data.")

# --- USER INPUT ---
st.sidebar.header("Enter Patient Data")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 45)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 240)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

    data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    return pd.DataFrame([data])

user_data = user_input_features()

# --- TRAIN MODEL ---
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# --- PREDICTION ---
user_scaled = scaler.transform(user_data)
prediction = model.predict(user_scaled)
probability = model.predict_proba(user_scaled)

# --- DISPLAY RESULTS ---
st.subheader("Prediction Result")
result = "🟢 No Heart Disease" if prediction[0] == 0 else "🔴 Heart Disease Detected"
st.markdown(f"### {result}")
st.markdown(f"**Probability of Disease:** `{round(probability[0][1] * 100, 2)}%`")

# --- BAR CHART ---
st.subheader("Prediction Probability")
bar_data = pd.DataFrame({
    'Condition': ['No Disease', 'Heart Disease'],
    'Probability': probability[0]
})

fig1, ax1 = plt.subplots()
sns.barplot(data=bar_data, x='Condition', y='Probability', palette='magma', ax=ax1)
ax1.set_ylim(0, 1)
st.pyplot(fig1)

# --- FEATURE COMPARISON ---
st.subheader("Comparison with Dataset")

compare_df = pd.concat([user_data, pd.DataFrame(X, columns=X.columns).mean().to_frame().T])
compare_df.index = ["You", "Average Patient"]
st.dataframe(compare_df.style.highlight_max(axis=0, color='lightgreen'))

# Footer
st.markdown("---")
st.caption("🔬 Model trained using Random Forest & Logistic Regression | Made by Muhammad Feras Malik | Data Source: UCI Heart Disease Dataset")