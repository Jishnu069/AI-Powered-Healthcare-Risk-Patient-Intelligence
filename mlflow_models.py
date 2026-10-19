import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AI-Healthcare-Risk-Intelligence")

# Load test data
X_test = pd.read_csv("data/ModelDATA/X_test_selected.csv")
y_test = pd.read_csv("data/ModelDATA/y_test_selected.csv").squeeze()

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl"
}

scaler = joblib.load("models/scaler.pkl")

for name, path in models.items():

    model = joblib.load(path)

    # Scale only Logistic Regression
    if name == "Logistic Regression":
        X_test_model = scaler.transform(X_test)
    else:
        X_test_model = X_test

    y_pred = model.predict(X_test_model)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run(run_name=name):

        mlflow.log_param("model", name)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"{name} logged successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")