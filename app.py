import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
}

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    return df


st.title(" Loan Credit Risk")
st.write(
    "Enter applicant details below to predict whether a loan application "
    "would be **Approved** or **Rejected**."
)
 
st.info(
    " **Dataset range:** This model was trained on applicant incomes between "
    "₹2,009–₹19,988, loan amounts between ₹1,015–₹39,995, credit scores "
    "550–799, and collateral/savings up to ~₹50,000. Predictions are most "
    "reliable for inputs within these ranges — values far outside them "
    "(e.g. loan amounts in lakhs) are out-of-distribution for the model "
    "and may produce unreliable results."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

    metrics = load_metrics()
    if metrics:
        st.subheader("Model performance")
        key_map = {
            "Logistic Regression": "logistic_regression",
            "K-Nearest Neighbors": "knn",
            "Naive Bayes": "naive_bayes",
        }
        m = next((x for x in metrics if x["model"] == key_map[model_choice]), None)
        if m:
            st.metric("Precision", f"{m['precision']*100:.1f}%")
            st.metric("Recall", f"{m['recall']*100:.1f}%")
            st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")

    st.caption(
        "Naive Bayes has the highest precision (fewer risky loans wrongly "
        "approved). Logistic Regression gives the best precision/recall balance."
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=15000.0, step=500.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0, step=500.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
    dti_ratio = st.slider("DTI Ratio (Debt-to-Income)", 0.0, 1.0, 0.35, 0.01)
    savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)

with col2:
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=30000.0, step=500.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0, step=500.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120], index=2)
    loan_purpose = st.selectbox(
        "Loan Purpose", ["Personal", "Home", "Education", "Car", "Business"]
    )
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    employer_category = st.selectbox(
        "Employer Category", ["Government", "Private", "MNC", "Startup", "Self-Employed"]
    )
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed", "Business"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])

st.divider()

if st.button("Predict Loan Approval", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment_status,
                "Age": age,
                "Marital_Status": marital_status,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education_level,
                "Gender": gender,
                "Employer_Category": employer_category,
            }
        ]
    )
    input_df = add_engineered_features(input_df)

    model = load_model(MODEL_FILES[model_choice])
    prediction = model.predict(input_df)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        proba = model.predict_proba(input_df)[0][classes.index("Yes")]

    if prediction == "Yes":
        st.success(f"✅ Loan likely **Approved** (using {model_choice})")
    else:
        st.error(f"❌ Loan likely **Rejected** (using {model_choice})")

    if proba is not None:
        st.write(f"Model confidence of approval: **{proba*100:.1f}%**")
        st.progress(float(proba))

st.divider()
st.caption("Built with the Logistic Regression / KNN / Naive Bayes models trained in loan_approval.ipynb")
with st.expander(" Future Scope"):
    st.markdown(
        "- **Expand the training data range** to cover real-world loan amounts "
        "(up to several lakhs/crores) and incomes, so the model generalizes "
        "beyond the current ₹2K–₹40K sample range.\n"
        "- Add more training data across diverse applicant profiles to improve "
        "generalization and reduce bias.\n"
        "- Include out-of-range input detection/warnings directly on each field.\n"
        "- Experiment with ensemble or gradient-boosted models (XGBoost, "
        "LightGBM) for potentially better accuracy.\n"
        "- Add model explainability (e.g. SHAP values) to show *why* a loan "
        "was approved or rejected."
    )