
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import random, json
from streamlit_lottie import st_lottie

# ---------------------------
# Load Lottie animation
# ---------------------------
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottiefile("animation.json")

# ---------------------------
# Streamlit UI Config
# ---------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Dataset Overview", "EDA", "Model Training", "Prediction"])


# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("onlinefraud.csv")
    df.drop("isFlaggedFraud", axis=1, inplace=True)
    return df

df = load_data()

# ---------------------------
# Home Page
# ---------------------------
if menu == "Home":
    st_lottie(lottie_animation, height=250, key="fraud_anim")
    st.markdown(
        "<h1 style='text-align: center; color: white;'>💳 Online Payment Fraud Detection</h1>",
        unsafe_allow_html=True
    )
    st.write("Welcome to the Fraud Detection Dashboard 🚀")
    st.write("Use the sidebar to explore the dataset, view EDA, and train models.")

# ---------------------------
# Dataset Overview
# ---------------------------
elif menu == "Dataset Overview":
   
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    st.write("Columns:", list(df.columns))

    # Fraud vs Non-Fraud Summary
    st.subheader("Fraud vs Non-Fraud Summary")

    fraud_counts = df["isFraud"].value_counts().rename({0: "Non-Fraud", 1: "Fraud"})
    st.write(fraud_counts)

    # Bar chart
    fig, ax = plt.subplots()
    sns.countplot(x="isFraud", data=df, palette="coolwarm", ax=ax)
    ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Fraud vs Non-Fraud Count")
    st.pyplot(fig)

    # Pie chart
    fig, ax = plt.subplots()
    fraud_counts.plot.pie(
        autopct="%1.2f%%",
        colors=["skyblue", "lightcoral"],
        explode=[0, 0.1],
        ax=ax,
        shadow=True
    )
    ax.set_ylabel("")
    ax.set_title("Fraud vs Non-Fraud Ratio")
    st.pyplot(fig)


# ---------------------------
# Exploratory Data Analysis
# ---------------------------
elif menu == "EDA":
    st.subheader("Distribution of Transaction Amounts")
    fig, ax = plt.subplots()
    sns.kdeplot(df["amount"], linewidth=3, ax=ax)
    st.pyplot(fig)
    st.subheader("Fraud vs Non-Fraud Transactions")
    fig, ax = plt.subplots()
    sns.countplot(x="isFraud", data=df, palette="PuBu", ax=ax)

    # Add counts & percentages on bars
    total = len(df)
    for p in ax.patches:
    	count = p.get_height()
    	percentage = 100 * count / total
    	ax.annotate(f'{count:,}\n({percentage:.4f}%)', 
                (p.get_x() + p.get_width() / 2., count), 
                ha='center', va='bottom', fontsize=10, color="black")

    ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    st.pyplot(fig)
    st.subheader("Transaction Types")
    fig, ax = plt.subplots()
    sns.countplot(x="type", data=df, palette="PuBu", ax=ax)
    st.pyplot(fig)


# ---------------------------
# Model Training
# ---------------------------

# ---------------------------
# Model Training
# ---------------------------
elif menu == "Model Training":
    st.subheader("⚡ Model Training & Evaluation")

    # Encode transaction type
    df["type"] = df["type"].map(
        {"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4}
    )

    # Features and target
    X = df.copy()
    X.drop(["nameOrig", "nameDest"], axis=1, inplace=True)  # keep balance columns
    y = X.pop("isFraud")

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Stratified Split
    from sklearn.model_selection import StratifiedKFold
    skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, test_idx in skfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Standardize only numeric features
    feature_cols = ["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train[feature_cols])
    X_test_scaled = sc.transform(X_test[feature_cols])

    # Handle class imbalance
    X_train_res, y_train_res = RandomUnderSampler(sampling_strategy="majority").fit_resample(
        X_train_scaled, y_train
    )

    # Train Random Forest
    model = RandomForestClassifier(class_weight="balanced", random_state=seed)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_pred_score = model.predict_proba(X_test_scaled)[:, 1]

    # Show classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", c="steelblue")
    ax.plot([0, 1], [0, 1], "--", c="lightgray")
    ax.set_title("ROC Curve - Random Forest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Save model and scaler
    import joblib
    joblib.dump(model, "rf_fraud_model.pkl")
    joblib.dump(sc, "scaler.pkl")
    st.success("✅ Model and scaler saved successfully!")

# ---------------------------
# Prediction Form
# ---------------------------
# ---------------------------
# Prediction Form
# ---------------------------
elif menu == "Prediction":
    st.subheader("🔮 Fraud Prediction Tool")
    st.write("Enter transaction details below:")

    # User Inputs
    amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, step=100.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, step=100.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, step=100.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, step=100.0)

    type_map = {"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4}
    type_choice = st.selectbox("Transaction Type", list(type_map.keys()))

    if st.button("Predict Fraud"):
        # Prepare input
        user_input = pd.DataFrame([{
            "step": 1,
            "type": type_map[type_choice],
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        # Load saved model and scaler
        import joblib
        model = joblib.load("rf_fraud_model.pkl")
        sc = joblib.load("scaler.pkl")

        # Scale input
        feature_cols = ["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        user_input_scaled = sc.transform(user_input[feature_cols])

        # Predict
        prediction = model.predict(user_input_scaled)[0]
        prediction_proba = model.predict_proba(user_input_scaled)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Probability of Fraud: {prediction_proba:.2f})")

