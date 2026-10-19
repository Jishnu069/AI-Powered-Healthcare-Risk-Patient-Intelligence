import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AI-Healthcare-Risk-Intelligence")

print("MLflow tracking configured successfully!")