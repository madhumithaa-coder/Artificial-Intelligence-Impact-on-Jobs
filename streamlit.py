import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\madhu\Downloads\AI_Impact_on_Jobs_2030.csv")
    return data

data = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
le_job = LabelEncoder()
le_risk = LabelEncoder()

data["Job_Title"] = le_job.fit_transform(data["Job_Title"])
data["Risk_Category"] = le_risk.fit_transform(data["Risk_Category"])

X = data[["Job_Title", "Years_Experience", "AI_Exposure_Index"]]
y = data["Risk_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Train Decision Tree Model
# -------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🤖 AI Job Risk Prediction System")
st.write("Decision Tree Model")

st.subheader("Enter Job Details")

job_title = st.text_input("Job Title")
years_exp = st.text_input("Years of Experience")
ai_exposure = st.text_input("AI Exposure Index")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk Category"):
    if job_title == "" or years_exp == "" or ai_exposure == "":
        st.warning("Please fill all input fields")
    else:
        try:
            job_encoded = le_job.transform([job_title])[0]
            years_exp = float(years_exp)
            ai_exposure = float(ai_exposure)

            input_data = [[job_encoded, years_exp, ai_exposure]]
            prediction = model.predict(input_data)

            risk_result = le_risk.inverse_transform(prediction)[0]

            st.success(f"Predicted Risk Category: **{risk_result}**")

        except ValueError:
            st.error("Please enter valid numeric values")
        except:
            st.error("Job Title not found in training data")
