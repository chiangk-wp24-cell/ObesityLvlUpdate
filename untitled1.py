import streamlit as st
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

# --- SIDEBAR (Global Controls) ---
with st.sidebar:
    st.title("Settings")
    selected_model = st.selectbox(
        "Choose Model for Live Prediction",
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN"]
    )
    st.info(f"Currently viewing results for: **{selected_model}**")

# --- MAIN INTERFACE ---
st.title("Model Findings & Analysis")

# Define the tabs as per your requirements
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Comparisons & Params", 
    "📉 Confusion Matrices", 
    "🔮 Single Prediction",
    "📂 Batch Prediction"
])

# --- TAB 1: COMPARISONS ---
with tab1:
    st.header("Model Performance Summary")
    st.info("Below is a side-by-side comparison of all models after final tuning.")

    # KEY STEP: Manually enter your final metrics here
    comparison_data = {
        "Model": ["Random Forest", "Decision Tree", "Logistic Regression", "KNN"],
        "Accuracy": [98.09, 97.37, 96.41, 87.56],      # <-- Replace with your real data
        "Precision": [98.12, 97.24, 96.26, 86.86],     # <-- Replace with your real data
        "Recall": [97.96, 97.24, 96.26, 87.09],        # <-- Replace with your real data
        "F1-Score": [98, 97.31, 96.29, 86.89]       # <-- Replace with your real data
    }
    
    # Convert to DataFrame and display
    df_metrics = pd.DataFrame(comparison_data)
    
    # Display the static table
    st.table(df_metrics)

    st.divider()

    # Model Parameters (Hyperparameters)
    st.subheader("Final Hyperparameters")
    
    # Use columns to keep the UI from becoming too long
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        with st.expander("🌲 Random Forest Details"):
            st.json({"criterion": "entropy","max_depth": "None", "min_samples_leaf": 2, "min_samples_split": 2, "n_estimators": 200, }) # <-- Enter your params here
        with st.expander("🌿 Decision Tree Details"):
            st.json({"criterion": "entropy", "max_depth": "None", "min_samples_leaf": 2, "min_samples_split": 2}) 
            
    with p_col2:
        with st.expander("📈 Logistic Regression Details"):
            st.json({"C": 100, "max_iter": "1000", "solver": "lbfg"})
        with st.expander("👥 KNN Details"):
            st.json({"metric": "manhattan", "n_neighbors": 5, "weights": "distance"})

# --- TAB 2: CONFUSION MATRICES ---
with tab2:
    st.header("Confusion Matrix Comparison")
    st.write("Visualizing the performance across all models in one view.")

    # Display the saved image
    # use_container_width=True is important so the 2x2 grid is readable
    st.image("matrix.png", caption="Model Comparison Matrix", use_container_width=True)

    st.divider()

    # The Analysis Box
    st.subheader("📝 Key Findings & Analysis")
    st.markdown('''1. Random Forest (top-left)

Nearly perfect classification.
Strong diagonal dominance (all predictions correct except a few in classes 5 and 6).
Very minimal confusion → best overall performer.

2. Decision Tree (top-right)

Also very strong, but slightly more errors than Random Forest.
Some confusion:
Class 3 → misclassified as class 2 (2 cases).
Class 5 and 6 show small spillover errors.
Still highly accurate, but less robust than Random Forest.

3. Logistic Regression (bottom-left)

Performance is close to Decision Tree.
Small but noticeable misclassifications:
Class 1 → confused with class 0.
Class 5 and 6 → minor mix-ups.
Suggests linear decision boundaries work fairly well but not perfectly.

4. KNN (bottom-right)

Clearly the weakest model here.
Much more spread outside the diagonal:
Class 1 is heavily confused (predicted as 0, 5, and 6).
Class 5 and 6 also show higher misclassification rates.
Indicates sensitivity to feature space overlap and possibly poor scaling or distance effects.''')

# --- TAB 3: SINGLE PREDICTION ---
with tab3:
    st.header(f"Live Prediction: {selected_model}")
    st.write("Placeholder: Manual input fields (sliders/number inputs) go here.")
    if st.button("Predict Single"):
        st.write("Prediction logic will trigger here.")

# --- TAB 4: BATCH PREDICTION ---
with tab4:
    st.header("CSV Batch Processing")
    st.write("Placeholder: File uploader and results table go here.")
    uploaded_file = st.file_uploader("Upload your data", type="csv")
