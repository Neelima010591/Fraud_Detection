import os
import pickle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


def train_model(X_train, y_train, scaler, columns):

    print("Training started...")

    # ==============================
    # Step 1: Apply SMOTE
    # ==============================
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("SMOTE applied")

    # ==============================
    # Step 2: Train Model
    # ==============================
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )


    model.fit(X_train_res, y_train_res)

    print("Model training completed")

    # ==============================
    # Step 3: Save Model
    # ==============================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "fraud_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

# Save columns
    with open(os.path.join(models_dir, "columns.pkl"), "wb") as f:
       pickle.dump(columns, f)
       print("Model and preprocessing files saved in:", models_dir)
       return model