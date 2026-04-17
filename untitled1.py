import streamlit as st
import pandas as pd
import numpy as np
import joblib  
import pickle

@st.cache_resource
def load_assets():
    """
    Loads assets from a mix of Pickle and Joblib formats.
    """
    # 1. Load Preprocessing (Pickle)
    # Note: Pickle requires the 'rb' (read binary) mode
    with open('scaler (2).pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder (2).pkl', 'rb') as f:
        le = pickle.load(f)
    
    # 2. Load Models (Joblib)
    # Joblib handles the file opening for you automatically
    models = {
        "Random Forest": joblib.load('rf_model.joblib'),
        "Decision Tree": joblib.load('dt_model.joblib'),
        "Logistic Regression": joblib.load('lr_model.joblib'),
        "KNN": joblib.load('knn_model.joblib')
    }
    
    return models, scaler, le

# Run the loader
try:
    models, scaler, le = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()
# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
#define
train_cols = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI',
    'Gender_Male', 'family_history_with_overweight_yes', 'FAVC_yes',
    'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes',
    'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
    'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
]
# --- SIDEBAR (Global Controls) ---
with st.sidebar:
    st.title("Settings")
    selected_model = st.selectbox(
        "Choose Model for Live Prediction",
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN"]
    )
    current_model = models[selected_model]
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
    st.subheader("Sample dataset used for current project")
    try:
            # Load the small CSV you just created
            sample_data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
            # Display as an interactive table
            st.dataframe(
                df, 
                use_container_width=True, 
                height=300)
        
    except FileNotFoundError:
                                st.warning("ObesityDataSet_raw_and_data_sinthetic not found. Please upload it to your repository.")

                                st.divider
    
    st.info("Below is a side-by-side comparison of all models after final tuning.")
    st.image("Accuraccy.png", caption="Model Accuracy Comparison chart")
    st.image("precision.png", caption="Model precision comparison chart")
    st.image("Recall.png", caption="Model Recall Comparison chart")
    st.image("F1-score.png", caption="Model F1-score Comparison chart")
    
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
    st.markdown("""
1. Random Forest (top-left)

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
Indicates sensitivity to feature space overlap and possibly poor scaling or distance effects.
""")

# --- TAB 3: LIVE PREDICTION ---
with tab3:
    st.header(f"Predicting with: {selected_model}")
    st.write("Enter your health and lifestyle details below.")

    # We use a form to prevent the app from refreshing every time a slider moves
    with st.form("obesity_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", 1.0, 100.0, 25.0)
            height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
            weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0)
            family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
            favc = st.selectbox("Frequent high caloric food?", ["yes", "no"])
            fcvc = st.slider("Vegetable consumption frequency", 1.0, 3.0, 2.0)
            ncp = st.slider("Number of main meals", 1.0, 4.0, 3.0)

        with col2:
            caec = st.selectbox("Food consumption between meals", ["Sometimes", "Frequently", "Always", "no"])
            smoke = st.selectbox("Do you smoke?", ["yes", "no"])
            ch2o = st.slider("Daily water intake (L)", 1.0, 3.0, 2.0)
            scc = st.selectbox("Do you monitor calories?", ["yes", "no"])
            faf = st.slider("Physical activity frequency", 0.0, 3.0, 1.0)
            tue = st.slider("Time using technology devices", 0.0, 2.0, 1.0)
            calc = st.selectbox("Alcohol consumption", ["Sometimes", "Frequently", "Always", "no"])
            mtrans = st.selectbox("Main transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

        # The form submit button
        submit_btn = st.form_submit_button("Predict Weight Category")

    # 4. Prediction Logic
    if submit_btn:
        # Calculate BMI exactly like your old code
        bmi_val = weight / (height ** 2)

        # Mapping inputs to the One-Hot-Encoded format the model expects
        input_dict = {
            'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp,
            'CH2O': ch2o, 'FAF': faf, 'TUE': tue, 'BMI': bmi_val,
            'Gender_Male': 1 if gender == 'Male' else 0,
            'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
            'FAVC_yes': 1 if favc == 'yes' else 0,
            'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
            'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
            'CAEC_no': 1 if caec == 'no' else 0,
            'SMOKE_yes': 1 if smoke == 'yes' else 0,
            'SCC_yes': 1 if scc == 'yes' else 0,
            'CALC_Frequently': 1 if calc == 'Frequently' else 0,
            'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
            'CALC_no': 1 if calc == 'no' else 0,
            'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
            'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
            'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
            'MTRANS_Walking': 1 if mtrans == 'Walking' else 0
        }

        # Create DataFrame and ensure training order (using the global train_cols list)
        input_df = pd.DataFrame([input_dict])[train_cols]
        
        # Scale and Predict
        scaled_data = scaler.transform(input_df.values)
        
        # Use 'current_model' which is selected from the sidebar
        prediction = current_model.predict(scaled_data)
        
        # Turn the number back into a readable name (e.g., 0 -> "Normal_Weight")
        final_label = le.inverse_transform(prediction)

        st.markdown("---")
        st.success(f"Predicted Category by **{selected_model}**: **{final_label[0]}**")
        st.info(f"Calculated BMI: **{bmi_val:.2f}**")
# --- TAB 4: BATCH PREDICTION ---
with tab4:
    st.header("CSV Batch Processing")
    st.write("Placeholder: File uploader and results table go here.")
    uploaded_file = st.file_uploader("Upload your data", type="csv")
