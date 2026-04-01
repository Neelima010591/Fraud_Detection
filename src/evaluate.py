from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("\n===== MODEL EVALUATION =====\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    try:
        roc_score = roc_auc_score(y_test, y_pred)
        print("\nROC AUC Score:", roc_score)
    except:
        print("\nROC AUC could not be calculated")

    print("\n============================\n")