from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.ml_models import train_models
from src.analysis import evaluate_model


def main():

    print("CardioMap Pipeline Starting...\n")

    # Step 1: Load data
    X, y = load_data(
        "data/EMS.xlsx",
        "data/Hospitals.xlsx"
    )

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Step 4: Evaluate
    evaluate_model(y_test, results["rf_preds"], "Random Forest")
    evaluate_model(y_test, results["lr_preds"], "Logistic Regression")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
