from pathlib import Path

import joblib
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AI-Powered Healthcare Risk & Patient Intelligence API",
    description="API connected to the registered MLflow model.",
    version="1.0.0",
)


# --------------------------------------------------
# 2. CONFIGURE PROJECT PATHS AND MLFLOW
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

mlflow.set_tracking_uri(TRACKING_URI)

MODEL_PATH = (
    "/app/mlruns/1/models/"
    "m-5f804e03f1584f14ad5bf1001f048d28/artifacts"
)

model = None
preprocessor = None

try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("MLflow scikit-learn model loaded successfully!")
    print("Model type:", type(model).__name__)

    if hasattr(model, "n_features_in_"):
        print("Model feature count:", model.n_features_in_)

except Exception as exc:
    print(f"Model loading failed: {exc}")


try:
    # Load the fitted preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Preprocessor loaded successfully!")

    if hasattr(preprocessor, "feature_names_in_"):
        print(
            "Preprocessor input feature count:",
            len(preprocessor.feature_names_in_),
        )

    if hasattr(preprocessor, "get_feature_names_out"):
        print(
            "Preprocessor output feature count:",
            len(preprocessor.get_feature_names_out()),
        )

except Exception as exc:
    print(f"Preprocessor loading failed: {exc}")


# --------------------------------------------------
# 4. DEFINE PATIENT INPUT SCHEMA
# --------------------------------------------------

class Patient(BaseModel):

    race: str
    gender: str
    age: int

    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int

    medical_specialty: str

    num_lab_procedures: int
    num_procedures: int
    num_medications: int

    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int

    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str

    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str

    change: str
    diabetesMed: str
    weight: int

    diag_1_category: str
    diag_2_category: str
    diag_3_category: str

    num_unique_diagnosis_categories: int
    total_previous_visits: int
    total_medications: int
    num_medications_changed: int

    age_group: str
    stay_category: str
    has_previous_visits: int
    num_diagnoses_present: int
    has_medication_change: int
    medical_specialty_grouped: str


# --------------------------------------------------
# 5. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Healthcare Risk Intelligence API is running",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_uri": MODEL_URI,
        "registered_model_uri": REGISTERED_MODEL_URI,
    }



# --------------------------------------------------
# 6. MODEL INFORMATION
# --------------------------------------------------

@app.post("/predict")
def predict(patient: Patient):

    # Check whether the model and preprocessor loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow model is not available.",
        )

    if preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor is not available.",
        )

    try:
        # ------------------------------------------
        # STEP 1: CONVERT JSON INTO A DATAFRAME
        # ------------------------------------------

        patient_data = pd.DataFrame(
            [patient.model_dump()]
        )

        # ------------------------------------------
        # STEP 2: RENAME COLUMNS
        # Match original training column names
        # ------------------------------------------

        column_mapping = {
            "glyburide_metformin": "glyburide-metformin",
            "glipizide_metformin": "glipizide-metformin",
            "glimepiride_pioglitazone": "glimepiride-pioglitazone",
            "metformin_rosiglitazone": "metformin-rosiglitazone",
            "metformin_pioglitazone": "metformin-pioglitazone",
        }

        patient_data = patient_data.rename(
            columns=column_mapping
        )

        # ------------------------------------------
        # STEP 3: VALIDATE RAW INPUT COLUMNS
        # ------------------------------------------

        expected_input_columns = list(
            preprocessor.feature_names_in_
        )

        missing_columns = [
            column
            for column in expected_input_columns
            if column not in patient_data.columns
        ]

        extra_columns = [
            column
            for column in patient_data.columns
            if column not in expected_input_columns
        ]

        if missing_columns or extra_columns:
            raise ValueError(
                f"Missing input columns: {missing_columns}; "
                f"unexpected input columns: {extra_columns}"
            )

        # ------------------------------------------
        # STEP 4: MATCH TRAINING COLUMN ORDER
        # ------------------------------------------

        patient_data = patient_data[
            expected_input_columns
        ]

        # ------------------------------------------
        # STEP 5: APPLY FITTED PREPROCESSOR
        # 54 raw columns -> 256 transformed columns
        # ------------------------------------------

        transformed_data = preprocessor.transform(
            patient_data
        )

        all_feature_names = (
            preprocessor.get_feature_names_out()
        )

        processed_patient = pd.DataFrame(
            transformed_data,
            columns=all_feature_names,
        )

        # ------------------------------------------
        # STEP 6: GET THE MODEL'S EXPECTED FEATURES
        # ------------------------------------------

        expected_features = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if expected_features is None:
            raise ValueError(
                "The model does not expose feature_names_in_. "
                "Check how the model was trained and saved."
            )

        expected_features = list(expected_features)

        # ------------------------------------------
        # STEP 7: CHECK REQUIRED FEATURES
        # ------------------------------------------

        missing_features = [
            feature
            for feature in expected_features
            if feature not in processed_patient.columns
        ]

        if missing_features:
            raise ValueError(
                "Preprocessor output is missing model features: "
                f"{missing_features}"
            )

        # ------------------------------------------
        # STEP 8: SELECT AND ORDER MODEL FEATURES
        # 256 transformed columns -> 123 model columns
        # ------------------------------------------

        processed_patient = processed_patient[
            expected_features
        ]

        # ------------------------------------------
        # STEP 9: MAKE PREDICTION
        # ------------------------------------------

        prediction = model.predict(
            processed_patient
        )

        predicted_class = str(prediction[0])

        # ------------------------------------------
        # STEP 10: GET THE MATCHING DESCRIPTION
        # ------------------------------------------

        readmission_categories = {
            "<30": "Readmitted within 30 days.",
            ">30": "Readmitted after 30 days.",
            "NO": (
                "No readmission recorded during "
                "the observation period."
            ),
        }

        readmission_category = readmission_categories.get(
            predicted_class
        )

        if readmission_category is None:
            raise ValueError(
                f"Unexpected prediction class: {predicted_class}"
            )

        # ------------------------------------------
        # STEP 11: GET PROBABILITY FOR THE PREDICTED CLASS
        # ------------------------------------------

        prediction_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                processed_patient
            )[0]

            model_classes = [
                str(class_name)
                for class_name in model.classes_
            ]

            class_probabilities = dict(
                zip(model_classes, probabilities)
            )

            prediction_probability = float(
                class_probabilities[predicted_class]
            )

        # ------------------------------------------
        # STEP 12: FORMAT PREDICTION LABEL
        # ------------------------------------------

        if predicted_class == "NO":
            predicted_readmission = "NO"
        else:
            predicted_readmission = (
                f"{predicted_class} days"
            )

        # ------------------------------------------
        # STEP 13: RETURN API RESPONSE
        # ------------------------------------------

        return {
            "readmission_category": readmission_category,
            "prediction_probability": (
                f"{prediction_probability:.2%}"
                if prediction_probability is not None
                else None
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc