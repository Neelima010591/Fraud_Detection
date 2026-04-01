from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model


def main():

    print("===== FRAUD DETECTION PROJECT STARTED =====\n")

    # ==============================
    # Step 1: Load Data
    # ==============================
    df = load_data()
    print("Data loaded successfully\n")

    # ==============================
    # Step 2: Preprocess Data
    # ==============================
    X_train, X_test, y_train, y_test,scaler,columns = preprocess_data(df) 
    print("Data preprocessing completed\n")
    print("\nSample input for API:\n", X_train.iloc[0].tolist())
   

    # ==============================
    # Step 3: Train Model
    # ==============================
    model = train_model(X_train, y_train, scaler, columns)

    # ==============================
    # Step 4: Evaluate Model
    # ==============================
    evaluate_model(model, X_test, y_test)

    print("\n===== PROCESS COMPLETED SUCCESSFULLY =====")



if __name__ == "__main__":
    main()