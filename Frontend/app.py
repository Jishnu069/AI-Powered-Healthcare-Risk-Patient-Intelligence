import os

import streamlit as st
import requests

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Healthcare Risk Intelligence",
    page_icon="🏥",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
HEALTH_URL = "http://127.0.0.1:8000/"


# =====================================================
# 2. PAGE HEADER
# =====================================================

st.title("🏥 AI-Powered Healthcare Risk & Patient Intelligence")

st.write(
    "Enter patient and hospital information to request "
    "a readmission prediction from the machine learning backend."
)

st.caption(
    "Educational project. Predictions are not a substitute "
    "for professional clinical assessment."
)

st.divider()


# =====================================================
# 3. BACKEND STATUS
# =====================================================

with st.expander("Backend connection status", expanded=False):
    if st.button("Check FastAPI connection"):
        try:
            response = requests.get(HEALTH_URL, timeout=5)

            if response.ok:
                result = response.json()

                st.success(result.get("message", "API is running."))

                if result.get("model_loaded"):
                    st.success("MLflow model is loaded.")
                else:
                    st.error("MLflow model is not loaded.")

                if result.get("preprocessor_loaded"):
                    st.success("Preprocessor is loaded.")
                else:
                    st.error("Preprocessor is not loaded.")

            else:
                st.error(f"API returned HTTP {response.status_code}.")

        except requests.RequestException as exc:
            st.error(f"Cannot connect to FastAPI: {exc}")


# =====================================================
# 4. INPUT HELPERS
# =====================================================

def text_field(label, key, default=""):
    return st.text_input(
        label,
        value=default,
        key=key,
    )


def number_field(label, key, default=0, minimum=0):
    return st.number_input(
        label,
        min_value=minimum,
        value=default,
        step=1,
        key=key,
    )


# =====================================================
# 5. PATIENT FORM
# =====================================================

with st.form("patient_form"):

    # -------------------------------------------------
    # A. DEMOGRAPHIC INFORMATION
    # -------------------------------------------------

    st.header("1. Patient Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        race = st.selectbox(
            "Race",
            ["Caucasian", "AfricanAmerican", "Asian",
             "Hispanic", "Other", "?"],
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Unknown/Invalid"],
        )

    with col3:
        age = number_field(
            "Age (integer, as required by your API)",
            "age",
            default=50,
        )

    # -------------------------------------------------
    # B. HOSPITAL INFORMATION
    # -------------------------------------------------

    st.header("2. Hospital Admission")

    col1, col2, col3 = st.columns(3)

    with col1:
        admission_type_id = number_field(
            "Admission Type ID",
            "admission_type_id",
            default=1,
        )

    with col2:
        discharge_disposition_id = number_field(
            "Discharge Disposition ID",
            "discharge_disposition_id",
            default=1,
        )

    with col3:
        admission_source_id = number_field(
            "Admission Source ID",
            "admission_source_id",
            default=7,
        )

    col1, col2 = st.columns(2)

    with col1:
        time_in_hospital = number_field(
            "Time in Hospital (days)",
            "time_in_hospital",
            default=1,
        )

    with col2:
        medical_specialty = text_field(
            "Medical Specialty",
            "medical_specialty",
            "InternalMedicine",
        )

    # -------------------------------------------------
    # C. PROCEDURES AND VISITS
    # -------------------------------------------------

    st.header("3. Procedures and Previous Visits")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_lab_procedures = number_field(
            "Number of Lab Procedures",
            "num_lab_procedures",
        )

        num_procedures = number_field(
            "Number of Procedures",
            "num_procedures",
        )

    with col2:
        num_medications = number_field(
            "Number of Medications",
            "num_medications",
        )

        number_outpatient = number_field(
            "Outpatient Visits",
            "number_outpatient",
        )

    with col3:
        number_emergency = number_field(
            "Emergency Visits",
            "number_emergency",
        )

        number_inpatient = number_field(
            "Inpatient Visits",
            "number_inpatient",
        )

    number_diagnoses = number_field(
        "Number of Diagnoses",
        "number_diagnoses",
        default=1,
    )

    # -------------------------------------------------
    # D. MEDICATIONS
    # -------------------------------------------------

    st.header("4. Medication Information")

    st.caption(
        "Select values that match the categories used during model training."
    )

    medication_options = [
        "No",
        "Steady",
        "Up",
        "Down",
    ]

    medication_fields = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide_metformin",
        "glipizide_metformin",
        "glimepiride_pioglitazone",
        "metformin_rosiglitazone",
        "metformin_pioglitazone",
    ]

    medication_labels = {
        "glyburide_metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
    }

    medication_values = {}

    with st.expander("Enter medication details", expanded=True):

        medication_columns = st.columns(3)

        for index, field in enumerate(medication_fields):
            with medication_columns[index % 3]:
                medication_values[field] = st.selectbox(
                    medication_labels.get(field, field),
                    medication_options,
                    key=f"med_{field}",
                )

    col1, col2 = st.columns(2)

    with col1:
        change = st.selectbox(
            "Medication Changed?",
            ["No", "Ch"],
        )

    with col2:
        diabetesMed = st.selectbox(
            "Diabetes Medication Prescribed?",
            ["Yes", "No"],
        )

    # -------------------------------------------------
    # E. WEIGHT AND DIAGNOSIS
    # -------------------------------------------------

    st.header("5. Weight and Diagnosis Categories")

    weight = number_field(
        "Weight (integer as required by the API)",
        "weight",
        default=0,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        diag_1_category = text_field(
            "Primary Diagnosis Category",
            "diag_1_category",
            "Circulatory",
        )

    with col2:
        diag_2_category = text_field(
            "Secondary Diagnosis Category",
            "diag_2_category",
            "Diabetes",
        )

    with col3:
        diag_3_category = text_field(
            "Third Diagnosis Category",
            "diag_3_category",
            "Respiratory",
        )

    # -------------------------------------------------
    # F. ENGINEERED FEATURES
    # -------------------------------------------------

    st.header("6. Additional Model Features")

    st.caption(
        "These fields must be calculated or entered according to "
        "the feature-engineering rules used during training."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        num_unique_diagnosis_categories = number_field(
            "Unique Diagnosis Categories",
            "num_unique_diagnosis_categories",
        )

        total_previous_visits = number_field(
            "Total Previous Visits",
            "total_previous_visits",
        )

    with col2:
        total_medications = number_field(
            "Total Medications",
            "total_medications",
        )

        num_medications_changed = number_field(
            "Medications Changed Count",
            "num_medications_changed",
        )

    with col3:
        age_group = text_field(
            "Age Group",
            "age_group",
            "Senior",
        )

        stay_category = text_field(
            "Hospital Stay Category",
            "stay_category",
            "Medium",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        has_previous_visits = st.selectbox(
            "Has Previous Visits?",
            [0, 1],
        )

        num_diagnoses_present = number_field(
            "Diagnoses Present",
            "num_diagnoses_present",
        )

    with col2:
        has_medication_change = st.selectbox(
            "Has Medication Change?",
            [0, 1],
        )

    with col3:
        medical_specialty_grouped = text_field(
            "Grouped Medical Specialty",
            "medical_specialty_grouped",
            "InternalMedicine",
        )

    # -------------------------------------------------
    # G. SUBMIT
    # -------------------------------------------------

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Readmission",
        type="primary",
        use_container_width=True,
    )


# =====================================================
# 6. SEND PATIENT DATA TO FASTAPI
# =====================================================

if submitted:

    patient_data = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "medical_specialty": medical_specialty,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        **medication_values,
        "change": change,
        "diabetesMed": diabetesMed,
        "weight": weight,
        "diag_1_category": diag_1_category,
        "diag_2_category": diag_2_category,
        "diag_3_category": diag_3_category,
        "num_unique_diagnosis_categories": (
            num_unique_diagnosis_categories
        ),
        "total_previous_visits": total_previous_visits,
        "total_medications": total_medications,
        "num_medications_changed": num_medications_changed,
        "age_group": age_group,
        "stay_category": stay_category,
        "has_previous_visits": has_previous_visits,
        "num_diagnoses_present": num_diagnoses_present,
        "has_medication_change": has_medication_change,
        "medical_specialty_grouped": medical_specialty_grouped,
    }

    try:
        with st.spinner("Requesting prediction from the ML model..."):

            response = requests.post(
                API_URL,
                json=patient_data,
                timeout=60,
            )

        if response.ok:

            result = response.json()

            st.divider()
            st.header("7. Prediction Result")

            category = result.get("readmission_category")
            probability = result.get("prediction_probability")

            if category:
                st.success(category)
            else:
                st.warning("The API did not return a readmission category.")

            if probability is not None:
                st.metric(
                    "Model Probability for Predicted Class",
                    probability,
                )

            st.caption(
                "This output is a model prediction, not a diagnosis "
                "or a guarantee of future readmission."
            )

        else:
            st.error(f"Prediction failed (HTTP {response.status_code}).")

            try:
                error_details = response.json()
                st.json(error_details)
            except ValueError:
                st.code(response.text)

    except requests.RequestException as exc:
        st.error(
            "Could not contact FastAPI. Make sure the backend is running."
        )
        st.code(str(exc))