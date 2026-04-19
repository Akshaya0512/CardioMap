from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(y_test, preds, name):

    print("\n====================")
    print(name)
    print("====================")

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
