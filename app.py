# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import plotly.express as px

# # --- Load Models ---
# MODEL_FILES = {
#     "Logistic Regression": "models/Logistic_Regression.pkl",
#     "Random Forest": "models/Random_Forest.pkl",
#     "SVM": "models/SVM.pkl",
#     "KNN": "models/KNN.pkl",
#     "Decision Tree": "models/Decision_Tree.pkl",
#     "XGBoost": "models/XGBoost.pkl",
#     "Extra Tuned XGBoost": "models/Extra_Tuned_XGBoost.pkl"
# }

# models = {}
# for name, path in MODEL_FILES.items():
#     if os.path.exists(path):
#         models[name] = joblib.load(path)

# if not models:
#     st.error("No model files found. Please check the paths.")

# # --- Load Dataset for EDA ---
# DATA_FILE = "data/cardio_train.csv"
# if os.path.exists(DATA_FILE):
#     df = pd.read_csv(DATA_FILE)
# else:
#     st.warning("Dataset file not found. EDA will not work.")
#     df = None

# # --- Sidebar ---
# st.sidebar.title("Heart Disease Predictor")
# menu = st.sidebar.radio("Menu", ["Home", "EDA", "Predict"])

# # --- Home Page ---
# if menu == "Home":
#     st.title("❤️ Heart Disease Prediction App")
#     st.markdown("""
#     This app allows you to:
#     - Explore heart disease dataset interactively.
#     - Predict heart disease risk using multiple ML models.
#     - Compare models and probabilities.
#     """)
#     st.image("https://cdn.pixabay.com/photo/2017/03/26/17/48/heart-2177753_1280.png", width=300)

# # --- EDA Page ---
# elif menu == "EDA":
#     if df is not None:
#         st.header("Exploratory Data Analysis")

#         # --- Interactive Filters ---
#         st.subheader("Filter Data")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))
#         with col2:
#             chol_range = st.slider("Cholesterol Range", int(df.chol.min()), int(df.chol.max()), (int(df.chol.min()), int(df.chol.max())))
#         with col3:
#             thalach_range = st.slider("Max Heart Rate Range", int(df.thalach.min()), int(df.thalach.max()), (int(df.thalach.min()), int(df.thalach.max())))

#         filtered_df = df[(df.age >= age_range[0]) & (df.age <= age_range[1]) &
#                          (df.chol >= chol_range[0]) & (df.chol <= chol_range[1]) &
#                          (df.thalach >= thalach_range[0]) & (df.thalach <= thalach_range[1])]
#         st.write(f"Filtered dataset: {filtered_df.shape[0]} rows")
#         st.dataframe(filtered_df)

#         # --- Interactive Charts ---
#         st.subheader("Visualizations")

#         # Heart Disease Count
#         st.write("Heart Disease Count by Sex")
#         fig1 = px.histogram(filtered_df, x='sex', color='target', barmode='group',
#                             labels={'sex': 'Sex', 'target': 'Heart Disease'}, 
#                             color_discrete_map={0:'green',1:'red'})
#         st.plotly_chart(fig1, use_container_width=True)

#         # Age vs Max Heart Rate
#         st.write("Age vs Max Heart Rate")
#         fig2 = px.scatter(filtered_df, x='age', y='thalach', color='target', 
#                           labels={'age':'Age','thalach':'Max Heart Rate','target':'Heart Disease'},
#                           color_discrete_map={0:'green',1:'red'},
#                           hover_data=['chol','cp'])
#         st.plotly_chart(fig2, use_container_width=True)

#         # Correlation Heatmap
#         st.write("Correlation Heatmap")
#         corr = filtered_df.corr()
#         fig3 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
#         st.plotly_chart(fig3, use_container_width=True)

#         # Chest Pain Distribution
#         st.write("Chest Pain Type Distribution")
#         fig4 = px.histogram(filtered_df, x='cp', color='target', barmode='group',
#                             labels={'cp':'Chest Pain Type','target':'Heart Disease'}, 
#                             color_discrete_map={0:'green',1:'red'})
#         st.plotly_chart(fig4, use_container_width=True)

#     else:
#         st.warning("EDA is unavailable. Dataset not loaded.")

# # --- Prediction Page ---
# elif menu == "Predict":
#     st.header("Input Patient Data for Prediction")

#     # --- Input Form with Sliders ---
#     age = st.slider("Age", 1, 120, 50)
#     sex = st.select_slider("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
#     cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
#     trestbps = st.slider("Resting Blood Pressure", 50, 250, 120)
#     chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
#     fbs = st.select_slider("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
#     restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)
#     thalach = st.slider("Max Heart Rate Achieved", 50, 250, 150)
#     exang = st.select_slider("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
#     oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
#     slope = st.slider("Slope of ST Segment (0-2)", 0, 2, 1)
#     ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
#     thal = st.slider("Thalassemia (1-3)", 1, 3, 2)

#     # --- Prepare Input DataFrame ---
#     input_data = pd.DataFrame([[
#         age, sex, cp, trestbps, chol, fbs, restecg,
#         thalach, exang, oldpeak, slope, ca, thal
#     ]], columns=[
#         'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
#     ])

#     # --- Select Model ---
#     selected_model_name = st.selectbox("Select a Model", list(models.keys()))
#     selected_model = models[selected_model_name]

#     # --- Prediction ---
#     if st.button("Predict Heart Disease"):
#         prediction = selected_model.predict(input_data)[0]
#         probability = selected_model.predict_proba(input_data)[0][1] if hasattr(selected_model, "predict_proba") else None

#         if prediction == 1:
#             st.error(f"⚠️ High Risk of Heart Disease")
#         else:
#             st.success(f"✅ Low Risk of Heart Disease")

#         if probability is not None:
#             st.info(f"Prediction Probability: {probability:.2f}")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from datetime import date

# ---------------- Theme Colors ----------------
PRIMARY_COLOR = "#0F4C81"  # Dark Blue
SECONDARY_COLOR = "#00B894"  # Green Accent
CARD_BG_COLOR = "#E8F6F3"  # Light card background

# ---------------- Load Models ----------------
# MODEL_FILES = {
#     "Logistic Regression": "models/GS_Logistic_Regression.pkl",
#     "Random Forest": "models/RS_Random_Forest.pkl",
#     "SVM": "models/Tuned_SVM.pkl",
#     "KNN": "models/GS_KNN.pkl",
#     "Decision Tree": "models/RS_Decision_Tree.pkl",
#     "XGBoost": "models/XGBoost.pkl",
#     "Extra Tuned XGBoost": "models/Extra_Tuned_XGBoost.pkl"
# }
MODEL_FILES = {
    "Logistic Regression":       "models/GS_Logistic_Regression.pkl",
    "Random Forest":             "models/RS_Random_Forest.pkl",
    "SVM":                       "models/Tuned_SVM.pkl",
    "KNN":                       "models/GS_KNN.pkl",
    "Decision Tree":             "models/RS_Decision_Tree.pkl",
    "XGBoost":                   "models/XGBoost.pkl",
    "Extra Tuned XGBoost":       "models/Extra_Tuned_XGBoost.pkl"
}

models = {}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)

if not models:
    st.error("No model files found. Please check the paths.")

# ---------------- Load Dataset ----------------
DATA_FILE = "data/cardio_train.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, sep=';')
else:
    st.warning("Dataset file not found. EDA will not work.")
    df = None

# ---------------- Sidebar ----------------
st.sidebar.title("❤️ Heart Disease Predictor")
menu = st.sidebar.radio("Menu", ["Home", "EDA", "Predict"])

# ---------------- HOME PAGE ----------------
if menu == "Home":
    st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>❤️ Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #0F4C81'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:{CARD_BG_COLOR}; padding:20px; border-radius:10px'>
        <h3>Welcome to the Heart Disease Prediction App!</h3>
        <ul>
            <li>Explore heart disease dataset interactively.</li>
            <li>Predict heart disease risk using multiple ML models.</li>
            <li>Compare models and probabilities to assess risk.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2017/03/26/17/48/heart-2177753_1280.png", use_container_width=True)

# ---------------- EDA PAGE ----------------
elif menu == "EDA":
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #0F4C81'>", unsafe_allow_html=True)

    if df is not None:
        # ---------------- Filter Data ----------------
        with st.expander("Filter Data", expanded=True):
            filters = {}
            for col in ['age','cholesterol','ap_hi','ap_lo','weight','height']:
                if col in df.columns:
                    min_val, max_val = int(df[col].min()), int(df[col].max())
                    filters[col] = st.slider(f"{col} range", min_val, max_val, (min_val, max_val))

            filtered_df = df.copy()
            for col, val in filters.items():
                filtered_df = filtered_df[(filtered_df[col]>=val[0]) & (filtered_df[col]<=val[1])]
            st.write(f"Filtered dataset: {filtered_df.shape[0]} rows")
            st.dataframe(filtered_df)

        # ---------------- Visualizations ----------------
        st.subheader("Visualizations")
        st.markdown(f"<div style='background-color:{CARD_BG_COLOR}; padding:10px; border-radius:10px'>", unsafe_allow_html=True)
        
        # Gender vs Heart Disease
        if 'cardio' in filtered_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(
                    filtered_df, x='gender', color='cardio', barmode='group',
                    labels={'gender':'Gender','cardio':'Heart Disease'},
                    color_discrete_map={0:'green',1:'red'},
                    title="Heart Disease Count by Gender"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                hover_cols = [c for c in ['cholesterol','weight','height'] if c in filtered_df.columns]
                fig2 = px.scatter(
                    filtered_df, x='age', y='ap_hi', color='cardio',
                    labels={'age':'Age','ap_hi':'Systolic BP','cardio':'Heart Disease'},
                    color_discrete_map={0:'green',1:'red'},
                    hover_data=hover_cols,
                    title="Age vs Systolic Blood Pressure"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Correlation Heatmap
            corr = filtered_df.corr()
            st.subheader("Correlation Heatmap")
            fig3 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Dataset not loaded. EDA unavailable.")

# ---------------- PREDICTION PAGE ----------------
elif menu == "Predict":
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}'>💓 Heart Disease Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #0F4C81'>", unsafe_allow_html=True)

    st.subheader("Patient Information Input")
    col1, col2 = st.columns(2)

    default_values = {
        "dob": pd.to_datetime("1980-01-01"), "gender": "Male", "height": 165, "weight": 70,
        "cp": 1, "trestbps": 120, "chol": 200, "fbs_val": 0, "restecg": 1,
        "thalach": 150, "exang_val": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
    }

    # ---------------- Inputs ----------------
    with col1:
        dob = st.date_input("Date of Birth", default_values["dob"])
        today = date.today()
        age = (today - dob).days // 365
        gender = st.radio("Gender", ["Male","Female"], index=0 if default_values["gender"]=="Male" else 1)
        sex = 1 if gender=="Male" else 0
        height = st.number_input("Height (cm)", 140, 210, default_values["height"])
        weight = st.number_input("Weight (kg)", 40, 150, default_values["weight"])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3], index=default_values["cp"],
                          format_func=lambda x:["Typical Angina","Atypical","Non-Anginal","Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, default_values["trestbps"])
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, default_values["chol"])

    with col2:
        fbs_val = st.radio("Fasting Blood Sugar >120 mg/dl?", [1,0], index=default_values["fbs_val"], format_func=lambda x:"Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG", [0,1,2], index=default_values["restecg"])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, default_values["thalach"])
        exang_val = st.radio("Exercise Induced Angina?", [1,0], index=default_values["exang_val"], format_func=lambda x:"Yes" if x==1 else "No")
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, step=0.1, value=default_values["oldpeak"])
        slope = st.selectbox("Slope of ST Segment", [0,1,2], index=default_values["slope"])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3], index=default_values["ca"])
        thal = st.selectbox("Thalassemia", [1,2,3], index=default_values["thal"]-1, format_func=lambda x:["Normal","Fixed Defect","Reversable Defect"][x-1])

    # ---------------- Reset Button ----------------
    if st.button("Reset to Default Values"):
        st.rerun()

    # ---------------- BMI ----------------
    bmi = weight / ((height/100)**2)
    if bmi < 18.5: bmi_status, bmi_color = "Underweight", "blue"
    elif bmi < 25: bmi_status, bmi_color = "Normal", "green"
    elif bmi < 30: bmi_status, bmi_color = "Overweight", "orange"
    else: bmi_status, bmi_color = "Obese", "red"

    st.markdown(f"<b>BMI:</b> {bmi:.2f} - <span style='color:{bmi_color}'>{bmi_status}</span>", unsafe_allow_html=True)

    # ---------------- Prepare Input DataFrame ----------------
    input_df = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs_val, restecg,
        thalach, exang_val, oldpeak, slope, ca, thal
    ]], columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

    # ---------------- Model Selection ----------------
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[selected_model_name]

    if st.button("Predict Heart Disease"):
        pred = selected_model.predict(input_df)[0]
        proba = selected_model.predict_proba(input_df)[0][1] if hasattr(selected_model,"predict_proba") else None
        color = 'red' if pred==1 else 'green'
        st.markdown(f"<b>Prediction:</b> <span style='color:{color}'>{'High Risk' if pred else 'Low Risk'}</span>", unsafe_allow_html=True)
        if proba is not None: st.info(f"Probability: {proba:.2f}")

        # ---------------- Overall Probability Comparison ----------------
        results = {}
        for name, model in models.items():
            try: results[name] = model.predict_proba(input_df)[0][1]
            except: continue

        if results:
            avg_proba = np.mean(list(results.values()))
            if avg_proba < 0.33: overall_risk, risk_color = "Low", "green"
            elif avg_proba < 0.66: overall_risk, risk_color = "Medium", "orange"
            else: overall_risk, risk_color = "High", "red"

            st.subheader("Overall Risk Level")
            st.markdown(f"<span style='color:{risk_color}; font-size:20px'><b>{overall_risk}</b></span>", unsafe_allow_html=True)
            st.markdown(f"**Average Probability:** {avg_proba:.2f}")

            fig = px.bar(
                x=list(results.keys()),
                y=list(results.values()),
                labels={'x':'Model','y':'Probability of Heart Disease'},
                color=list(results.values()),
                color_continuous_scale='RdYlGn_r',
                title="Heart Disease Probability Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
