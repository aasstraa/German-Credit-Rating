import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

@st.cache_resource
def load_model():
    # Native CatBoost loading bypasses Python serialization issues completely
    model = CatBoostClassifier()
    model.load_model("german_credit_catboost.cbm")
    return model


try:
    model = load_model()
except Exception as e:
    st.error("Error: Could not locate 'german_credit_model.pkl'. Ensure the file is downloaded from Kaggle into this exact directory.")
    st.stop()

# 3. Exactly Matched 1-Indexed English Mappings From Your Key
mappings = {
    'laufkont': {
        "No checking account": 1,
        "Balance < 0 DM": 2,
        "0 <= Balance < 200 DM": 3,
        "Balance >= 200 DM / Salary for at least 1 year": 4
    },
    'moral': {
        "Delay in paying off in the past": 0,
        "Critical account / other credits elsewhere": 1,
        "No credits taken / all credits paid back duly": 2,
        "Existing credits paid back duly till now": 3,
        "All credits at this bank paid back duly": 4
    },
    'verw': {
        "Others": 0, "Car (new)": 1, "Car (used)": 2, 
        "Furniture/equipment": 3, "Radio/television": 4, 
        "Domestic appliances": 5, "Repairs": 6, "Education": 7, 
        "Vacation": 8, "Retraining": 9, "Business": 10
    },
    'sparkont': {
        "Unknown / no savings account": 1,
        "Balance < 100 DM": 2,
        "100 <= Balance < 500 DM": 3,
        "500 <= Balance < 1000 DM": 4,
        "Balance >= 1000 DM": 5
    },
    'beszeit': {
        "Unemployed": 1,
        "Duration < 1 year": 2,
        "1 <= Duration < 4 years": 3,
        "4 <= Duration < 7 years": 4,
        "Duration >= 7 years": 5
    },
    'rate': {
        "Ratio >= 35%": 1,
        "25% <= Ratio < 35%": 2,
        "20% <= Ratio < 25%": 3,
        "Ratio < 20%": 4
    },
    'famges': {
        "Male: Divorced/separated": 1,
        "Female: Non-single OR Male: Single": 2,
        "Male: Married/widowed": 3,
        "Female: Single": 4
    },
    'buerge': {
        "None": 1,
        "Co-applicant": 2,
        "Guarantor": 3
    },
    'wohnzeit': {
        "Duration < 1 year": 1,
        "1 <= Duration < 4 years": 2,
        "4 <= Duration < 7 years": 3,
        "Duration >= 7 years": 4
    },
    'verm': {
        "Unknown / no property": 1,
        "Car or other vehicle": 2,
        "Building society savings / life insurance": 3,
        "Real estate": 4
    },
    'weitkred': {
        "Bank": 1,
        "Stores": 2,
        "None": 3
    },
    'wohn': {
        "For free": 1,
        "Rent": 2,
        "Own": 3
    },
    'bishkred': {
        "1 existing credit": 1,
        "2 to 3 existing credits": 2,
        "4 to 5 existing credits": 3,
        "6 or more existing credits": 4
    },
    'beruf': {
        "Unemployed / unskilled non-resident": 1,
        "Unskilled resident": 2,
        "Skilled employee / official": 3,
        "Manager / self-employed / highly qualified employee": 4
    },
    'pers': {
        "3 or more people": 1,
        "0 to 2 people": 2
    },
    'telef': {
        "No": 1,
        "Yes (under customer name)": 2
    },
    'gastarb': {
        "Yes": 1,
        "No": 2
    }
}

# 4. Construct Layout Input Sections
st.subheader("📋 Continuous Applicant Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    laufzeit = st.number_input("Duration (Months) [laufzeit]", min_value=1, max_value=72, value=24)
with col2:
    hoehe = st.number_input("Credit Amount (DM) [hoehe]", min_value=100, max_value=20000, value=3000)
with col3:
    alter = st.number_input("Age (Years) [alter]", min_value=18, max_value=100, value=35)


# Section: Categorical Dropdowns Layout Matrix
st.subheader("🔍 Financial & Qualitative Selection Dropdowns")

cat_inputs = {}
dropdown_cols = st.columns(2)  # This creates a list: [Column 0, Column 1]

for i, (key, options_dict) in enumerate(mappings.items()):
    # FIX: Use [0] to select the first column or [1] to select the second column
    target_col = dropdown_cols[0] if i % 2 == 0 else dropdown_cols[1]
    
    with target_col:
        chosen_label = st.selectbox(f"{key.upper()} (Status/Category):", options=list(options_dict.keys()), key=key)
        cat_inputs[key] = options_dict[chosen_label]

# 5. Build the Input DataFrame matching column tracking indices exactly
input_data = pd.DataFrame([{
    'laufkont': int(cat_inputs['laufkont']),
    'laufzeit': int(laufzeit),
    'moral': int(cat_inputs['moral']),
    'verw': int(cat_inputs['verw']),
    'hoehe': int(hoehe),
    'sparkont': int(cat_inputs['sparkont']),
    'beszeit': int(cat_inputs['beszeit']),
    'rate': int(cat_inputs['rate']),
    'famges': int(cat_inputs['famges']),
    'buerge': int(cat_inputs['buerge']),
    'wohnzeit': int(cat_inputs['wohnzeit']),
    'verm': int(cat_inputs['verm']),
    'alter': int(alter),
    'weitkred': int(cat_inputs['weitkred']),
    'wohn': int(cat_inputs['wohn']),
    'bishkred': int(cat_inputs['bishkred']),
    'beruf': int(cat_inputs['beruf']),
    'pers': int(cat_inputs['pers']),
    'telef': int(cat_inputs['telef']),
    'gastarb': int(cat_inputs['gastarb'])
}])

# 6. Prediction Logic Execution
st.markdown("---")
if st.button("🚀 Calculate Risk Evaluation", use_container_width=True):
    probabilities = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)
    
    # Extract prediction value safely out of potential array wrappers
    if isinstance(prediction, (np.ndarray, list)):
        pred_value = prediction[0]
    else:
        pred_value = prediction

    # 0 = Bad Credit Risk, 1 = Good Credit Risk
    if int(pred_value) == 1:
        st.success(f"### Result: ✅ Credit Risk is GOOD")
        st.metric(label="Model Confidence Score", value=f"{probabilities[1]*100:.2f}%")
    else:
        st.error(f"### Result: ⚠️ Credit Risk is BAD")
        st.metric(label="Model Confidence Score", value=f"{probabilities[0]*100:.2f}%")
        
    with st.expander("View Encoded Input Vector Passed to Model"):
        st.dataframe(input_data)
